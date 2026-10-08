"""Shared adapter helpers and the external-root policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from .contracts import RasterAsset, TargetGridMetadata
from .errors import DatasetAdapterError, DatasetDiscoveryError, DatasetIntegrityError

__all__ = [
    "DatasetAdapter",
    "asset_from_path",
    "derive_target_grid",
    "external_root",
    "require_equal_metadata",
]


class DatasetAdapter:
    """Minimal protocol-like base class for lazy dataset adapters."""

    dataset_id: str

    def discover(self):  # pragma: no cover - concrete adapters implement this
        raise NotImplementedError

    def load_sample(self, sample_id: str):  # pragma: no cover
        raise NotImplementedError


def external_root(root: Optional[Path], env_name: str) -> Path:
    """Resolve a caller-supplied external dataset root.

    No dataset location is embedded in production code.  Callers may pass a
    path explicitly or set the dataset-specific environment variable.
    """

    candidate = root if root is not None else os.environ.get(env_name)
    if not candidate:
        raise DatasetDiscoveryError(
            f"{env_name} is unset; pass an external dataset root explicitly or set it"
        )
    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise DatasetDiscoveryError(f"external dataset root does not exist: {path}")
    return path


def asset_from_path(path: Path) -> RasterAsset:
    if not path.is_file():
        raise DatasetDiscoveryError(f"required raster is missing: {path}")
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise DatasetIntegrityError(f"raster has no CRS: {path}")
            return RasterAsset(
                path=path,
                crs=ds.crs.to_string(),
                transform=tuple(float(v) for v in ds.transform[:6]),
                shape=(int(ds.height), int(ds.width)),
                dtype=str(ds.dtypes[0]),
                count=int(ds.count),
                nodata=None if ds.nodata is None else float(ds.nodata),
                descriptions=tuple(ds.descriptions),
            )
    except DatasetAdapterError:
        raise
    except Exception as exc:
        raise DatasetIntegrityError(f"could not read raster metadata: {path}") from exc


def require_equal_metadata(
    assets: Iterable[RasterAsset], *, transform_tolerance: float = 1e-10
) -> RasterAsset:
    """Require CRS, shape and affine alignment for a group of source rasters."""

    items = list(assets)
    if not items:
        raise DatasetIntegrityError("cannot validate an empty raster group")
    reference = items[0]
    for asset in items[1:]:
        if asset.crs != reference.crs or asset.shape != reference.shape:
            raise DatasetIntegrityError(
                f"unaligned rasters: {asset.path} differs from {reference.path} in CRS or shape"
            )
        if any(
            abs(left - right) > transform_tolerance
            for left, right in zip(asset.transform, reference.transform)
        ):
            raise DatasetIntegrityError(
                f"unaligned rasters: {asset.path} differs from {reference.path} in transform"
            )
    return reference


def derive_target_grid(
    *, centroid_lon: float, centroid_lat: float, source_crs: str
) -> TargetGridMetadata:
    """Derive the per-sample WGS84/UTM target CRS, without making a grid instance."""

    if not -180.0 <= centroid_lon <= 180.0 or not -90.0 <= centroid_lat <= 90.0:
        raise DatasetIntegrityError("sample centroid is outside geographic bounds")
    zone = min(60, max(1, int((centroid_lon + 180.0) // 6.0) + 1))
    hemisphere = "N" if centroid_lat >= 0 else "S"
    epsg = 32600 + zone if hemisphere == "N" else 32700 + zone
    return TargetGridMetadata(
        crs=f"EPSG:{epsg}",
        resolution_m=10.0,
        zone=zone,
        hemisphere=hemisphere,
        centroid_lon=float(centroid_lon),
        centroid_lat=float(centroid_lat),
        is_native_source_resolution=False,
    )


def raster_centroid(asset: RasterAsset) -> tuple[float, float]:
    """Return a WGS84 centroid from raster bounds, using the source CRS."""

    from rasterio.transform import array_bounds

    left, bottom, right, top = array_bounds(
        asset.shape[0], asset.shape[1], Affine(*asset.transform)
    )
    x, y = warp_transform(
        CRS.from_user_input(asset.crs),
        CRS.from_epsg(4326),
        [(left + right) / 2.0],
        [(bottom + top) / 2.0],
    )
    return float(x[0]), float(y[0])
