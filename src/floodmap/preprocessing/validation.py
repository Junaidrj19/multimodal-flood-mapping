"""Raster metadata, band and validity-mask validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .errors import InputValidationError, PreprocessingDependencyError
from .grid import AnalysisGrid, require_rasterio

try:
    import rasterio
    from rasterio.enums import MaskFlags
except ImportError as exc:  # pragma: no cover
    rasterio = None  # type: ignore[assignment]
    MaskFlags = None  # type: ignore[assignment]
    _RASTERIO_ERROR = exc
else:
    _RASTERIO_ERROR = None

__all__ = [
    "RasterMetadata",
    "RasterValidationReport",
    "inspect_raster",
    "read_valid_mask",
    "require_valid_raster",
    "validate_required_bands",
]


def _require_import() -> None:
    require_rasterio()
    if _RASTERIO_ERROR is not None:
        raise PreprocessingDependencyError(
            "rasterio is required for M2 raster validation"
        ) from _RASTERIO_ERROR


@dataclass(frozen=True)
class RasterMetadata:
    path: Path
    width: int
    height: int
    count: int
    dtypes: Tuple[str, ...]
    crs: Optional[str]
    transform: Any
    resolution: Tuple[float, float]
    bounds: Tuple[float, float, float, float]
    nodata: Optional[float]
    descriptions: Tuple[Optional[str], ...]
    tags: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtypes": list(self.dtypes),
            "crs": self.crs,
            "transform": [float(value) for value in self.transform[:6]],
            "resolution": list(self.resolution),
            "bounds": list(self.bounds),
            "nodata": self.nodata,
            "descriptions": list(self.descriptions),
            "tags": dict(self.tags),
        }

    def as_grid(self) -> AnalysisGrid:
        return AnalysisGrid(
            crs=self.crs or "",
            transform=self.transform,
            width=self.width,
            height=self.height,
            resolution_m=float((abs(self.resolution[0]) + abs(self.resolution[1])) / 2),
        )


@dataclass(frozen=True)
class RasterValidationReport:
    valid: bool
    metadata: Optional[RasterMetadata]
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def inspect_raster(path: Path | str) -> RasterMetadata:
    """Read geospatial metadata without modifying the source product."""
    _require_import()
    source = Path(path)
    if not source.is_file():
        raise InputValidationError(f"raster source does not exist: {source}")
    try:
        with rasterio.open(source) as dataset:
            if dataset.width < 1 or dataset.height < 1 or dataset.count < 1:
                raise InputValidationError(f"raster has invalid dimensions/band count: {source}")
            if dataset.crs is None:
                raise InputValidationError(f"raster has no CRS: {source}")
            if dataset.transform.is_identity:
                raise InputValidationError(f"raster has an identity transform: {source}")
            return RasterMetadata(
                path=source,
                width=int(dataset.width),
                height=int(dataset.height),
                count=int(dataset.count),
                dtypes=tuple(str(value) for value in dataset.dtypes),
                crs=dataset.crs.to_string(),
                transform=dataset.transform,
                resolution=(float(dataset.res[0]), float(dataset.res[1])),
                bounds=tuple(float(value) for value in dataset.bounds),
                nodata=float(dataset.nodata) if dataset.nodata is not None else None,
                descriptions=tuple(dataset.descriptions),
                tags={str(key): str(value) for key, value in dataset.tags().items()},
            )
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"could not read raster metadata from {source}: {exc}") from exc


def validate_required_bands(metadata: RasterMetadata, required_bands: Sequence[str]) -> None:
    """Require named band descriptions; band order is never guessed."""
    required = tuple(str(name) for name in required_bands)
    if not required:
        raise InputValidationError("required band names are unresolved; no band order is guessed")
    available = {description.strip() for description in metadata.descriptions if description}
    missing = [name for name in required if name not in available]
    if missing:
        raise InputValidationError(
            f"{metadata.path} is missing required named bands: {', '.join(missing)}; "
            f"available descriptions: {sorted(available)}"
        )


def validate_raster(
    path: Path | str,
    *,
    expected_crs: Optional[str] = None,
    min_bands: int = 1,
    required_bands: Sequence[str] = (),
) -> RasterValidationReport:
    try:
        metadata = inspect_raster(path)
    except InputValidationError as exc:
        return RasterValidationReport(valid=False, metadata=None, errors=(str(exc),))
    errors: list[str] = []
    if metadata.count < min_bands:
        errors.append(f"expected at least {min_bands} bands, found {metadata.count}")
    if expected_crs is not None:
        _require_import()
        expected = rasterio.crs.CRS.from_user_input(expected_crs).to_string()
        if metadata.crs != expected:
            errors.append(f"CRS {metadata.crs!r} does not match expected {expected!r}")
    if required_bands:
        try:
            validate_required_bands(metadata, required_bands)
        except InputValidationError as exc:
            errors.append(str(exc))
    return RasterValidationReport(valid=not errors, metadata=metadata, errors=tuple(errors))


def require_valid_raster(
    path: Path | str,
    *,
    expected_crs: Optional[str] = None,
    min_bands: int = 1,
    required_bands: Sequence[str] = (),
) -> RasterMetadata:
    """Validate and return metadata, closing all source handles."""
    metadata = inspect_raster(path)
    if metadata.count < min_bands:
        raise InputValidationError(
            f"{metadata.path} has {metadata.count} bands; minimum is {min_bands}"
        )
    if expected_crs is not None:
        _require_import()
        expected = rasterio.crs.CRS.from_user_input(expected_crs).to_string()
        if metadata.crs != expected:
            raise InputValidationError(
                f"{metadata.path} CRS {metadata.crs!r} does not match expected {expected!r}"
            )
    if required_bands:
        validate_required_bands(metadata, required_bands)
    return metadata


def read_valid_mask(
    path: Path | str, *, band_indexes: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Return a pixel-valid mask from nodata, finite values and GDAL masks."""
    _require_import()
    source = Path(path)
    with rasterio.open(source) as dataset:
        indexes = tuple(band_indexes or range(1, dataset.count + 1))
        data = dataset.read(indexes, masked=False)
        valid = np.ones((dataset.height, dataset.width), dtype=bool)
        for band_index, values in zip(indexes, data):
            valid &= np.isfinite(values)
            nodata = dataset.nodatavals[band_index - 1]
            if nodata is not None:
                valid &= values != nodata
        valid &= np.all(dataset.read_masks(indexes) > 0, axis=0)
        return valid
