"""Shared spatial reprojection to the canonical analysis grid.

ONE implementation, used by both M2 production preprocessing and M5 training.
That is the point of this module: if training resampled differently from
inference, every metric measured in training would describe a grid the served
model never sees, and the discrepancy would be invisible because both paths
would look individually reasonable.

Three decisions are enforced here rather than left to each caller.

**SAR is resampled in LINEAR POWER, before any dB conversion** (C3,
``configs/preprocessing.yaml -> target_grid.resampling.sar_domain``). Averaging
decibels averages logarithms, which is not the mean of the underlying power and
is biased low. :func:`reproject_continuous` therefore refuses a ``domain="db"``
array, and :func:`linear_to_db_after_resampling` exists so the correct ordering
is the easy one to write.

**Labels and validity masks use nearest neighbour** (C3). Bilinear on a class
label invents classes that were never in the source: a 0/2 boundary
interpolates to 1, which is "Permanent Waters".

**Invalid source pixels cannot contaminate valid output pixels.** This is the
subtle one. Kuro stores invalid SAR as ``0.0``, which is indistinguishable from
genuine near-zero backscatter once it enters a bilinear kernel — and near-zero
backscatter is precisely the open-water signal. Simply masking the output
afterwards is not enough: it cleans pixels that are *themselves* invalid while
leaving valid pixels that absorbed an invalid neighbour. See
:class:`NodataPolicy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import PreprocessingConfigurationError
from .grid import AnalysisGrid, compare_grids, require_rasterio, resampling_from_name

try:
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject, transform as warp_transform, transform_bounds
except ImportError as exc:  # pragma: no cover - environments without the geo extra
    Affine = None  # type: ignore[assignment]
    CRS = None  # type: ignore[assignment]
    array_bounds = None  # type: ignore[assignment]
    reproject = None  # type: ignore[assignment]
    warp_transform = None  # type: ignore[assignment]
    transform_bounds = None  # type: ignore[assignment]
    _RASTERIO_ERROR = exc
else:
    _RASTERIO_ERROR = None

__all__ = [
    "CANONICAL_RESOLUTION_M",
    "NodataPolicy",
    "ReprojectionResult",
    "linear_to_db_after_resampling",
    "reproject_categorical",
    "reproject_continuous",
    "target_grid_for_bounds",
    "target_grid_for_source",
    "utm_epsg_for_lonlat",
]

#: Canonical target resolution: 10.0 m TRUE GROUND in a UTM CRS (C2).
#:
#: This is the native ground resolution of no dataset in the project. It is the
#: production inference source's resolution (Sentinel-1 IW GRDH), so the path
#: actually served needs no resampling at all; every measured Kuro spacing
#: (8.736-9.798 m) is finer, so resampling to 10 m is a mild coarsening that
#: never manufactures detail the source did not contain.
CANONICAL_RESOLUTION_M = 10.0

#: Tolerance on the interpolated validity weight. A target pixel is accepted
#: only if its bilinear kernel drew essentially entirely on valid source
#: pixels; this absorbs floating-point error in the warp, nothing more.
_WEIGHT_TOLERANCE = 1e-6


def _require_import() -> None:
    require_rasterio()
    if _RASTERIO_ERROR is not None:  # pragma: no cover
        raise PreprocessingConfigurationError(
            "rasterio is required for spatial reprojection"
        ) from _RASTERIO_ERROR


# ---------------------------------------------------------------------------
# Nodata policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodataPolicy:
    """How invalid source pixels are kept out of interpolated output.

    ``strict`` (the default and the only one used by the baseline)
        A target pixel is valid only if **every** source pixel contributing to
        its interpolation kernel was valid. Implemented by warping the validity
        mask as a float coverage field with the *same* method as the data and
        requiring the resulting weight to be 1.0 within tolerance.

        This is conservative: it discards a thin fringe of target pixels around
        every nodata region. That is the correct direction to err. The
        alternative keeps pixels whose value is a blend of a real measurement
        and a sentinel, and nothing downstream could detect that such a pixel
        is part fiction.

    ``majority``
        A target pixel is valid if more than half its kernel weight came from
        valid source pixels. Provided because coarsening by a large factor
        under ``strict`` can erode a usable amount of data, and a future stage
        may need that trade-off. **Not used by the M5 baseline**, and selecting
        it is a recorded decision rather than a default.

    In both modes invalid source samples are replaced with NaN and declared as
    the warp's source nodata *before* interpolation, so they are excluded from
    the kernel rather than averaged in as a value. This is why Kuro's ``0.0``
    invalid sentinel cannot be read as genuine zero backscatter.
    """

    mode: str = "strict"
    #: Minimum fraction of kernel weight that must come from valid pixels.
    #: Derived from ``mode``; stated explicitly so provenance records a number.
    minimum_valid_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"strict", "majority"}:
            raise PreprocessingConfigurationError(
                f"unknown nodata policy mode {self.mode!r}; choose 'strict' or 'majority'"
            )
        expected = 1.0 if self.mode == "strict" else 0.5
        if abs(self.minimum_valid_weight - expected) > 1e-12:
            object.__setattr__(self, "minimum_valid_weight", expected)

    @property
    def threshold(self) -> float:
        if self.mode == "strict":
            return 1.0 - _WEIGHT_TOLERANCE
        return 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "minimum_valid_weight": self.minimum_valid_weight,
            "threshold_applied": self.threshold,
            "invalid_source_excluded_from_kernel": True,
            "invalid_source_treated_as_zero_backscatter": False,
        }


@dataclass(frozen=True)
class ReprojectionResult:
    """Reprojected data, its validity mask, and reproducible provenance."""

    data: np.ndarray
    valid: np.ndarray
    grid: AnalysisGrid
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.provenance)


# ---------------------------------------------------------------------------
# Target grid construction
# ---------------------------------------------------------------------------


def utm_epsg_for_lonlat(longitude: float, latitude: float) -> int:
    """WGS84/UTM EPSG code for a geographic coordinate.

    Zone boundaries are handled by construction rather than by rounding: zone
    width is exactly 6 degrees from -180, so ``floor((lon + 180) / 6) + 1``
    assigns a point exactly on a boundary to the zone it *starts*, which is the
    standard convention. The result is clamped to [1, 60] so ``lon == 180``
    lands in zone 60 instead of a non-existent 61.

    Hemisphere is taken from the sign of latitude, with the equator assigned
    north (EPSG:326xx), matching ``datasets/base.derive_target_grid``.
    """

    if not np.isfinite(longitude) or not np.isfinite(latitude):
        raise PreprocessingConfigurationError(
            f"cannot derive a UTM zone from non-finite coordinates ({longitude}, {latitude})"
        )
    if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
        raise PreprocessingConfigurationError(
            f"centroid ({longitude}, {latitude}) is outside geographic bounds"
        )
    zone = int(np.floor((longitude + 180.0) / 6.0)) + 1
    zone = min(60, max(1, zone))
    return (32600 if latitude >= 0.0 else 32700) + zone


def target_grid_for_bounds(
    *,
    bounds: Tuple[float, float, float, float],
    source_crs: str,
    resolution_m: float = CANONICAL_RESOLUTION_M,
    tile_size: Optional[int] = None,
) -> AnalysisGrid:
    """Build a concrete canonical grid covering a source footprint.

    Returns a real :class:`AnalysisGrid` — CRS, affine transform, width, height
    and resolution. Target-grid *metadata* alone is not reprojection; this is
    the grid the data is actually warped onto.

    The CRS is chosen per sample from the footprint centroid, because the Kuro
    corpus spans roughly -18 to +60 degrees latitude across all longitudes and
    no single projected CRS holds metre fidelity over that extent (C1).

    Parameters
    ----------
    bounds
        ``(left, bottom, right, top)`` in ``source_crs``.
    source_crs
        CRS the bounds are expressed in.
    resolution_m
        Target pixel size in metres. Defaults to the canonical 10.0 m.
    tile_size
        When given, the grid is exactly ``tile_size`` x ``tile_size`` pixels
        centred on the footprint centroid, instead of being sized to cover the
        footprint.

        This exists for a concrete reason: resampling a 224 px Kuro tile from
        its measured 8.736-9.798 m spacing to 10 m yields a footprint-sized
        output of roughly 196-220 px that **varies per activation**, and a
        dataset returning variable shapes cannot be collated into a batch.
        Fixing the pixel count makes the output batchable and identical in
        shape for training and inference; area outside the source footprint is
        marked invalid rather than filled.
    """

    _require_import()
    if resolution_m <= 0:
        raise PreprocessingConfigurationError("target resolution must be positive")
    left, bottom, right, top = (float(value) for value in bounds)
    if not (right > left and top > bottom):
        raise PreprocessingConfigurationError(f"degenerate source bounds: {bounds}")

    centre_x = (left + right) / 2.0
    centre_y = (bottom + top) / 2.0
    lon, lat = warp_transform(
        CRS.from_user_input(source_crs), CRS.from_epsg(4326), [centre_x], [centre_y]
    )
    epsg = utm_epsg_for_lonlat(float(lon[0]), float(lat[0]))
    target_crs = f"EPSG:{epsg}"

    target_bounds = transform_bounds(
        CRS.from_user_input(source_crs),
        CRS.from_epsg(epsg),
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )
    t_left, t_bottom, t_right, t_top = (float(value) for value in target_bounds)

    if tile_size is not None:
        size = int(tile_size)
        if size < 1:
            raise PreprocessingConfigurationError("tile_size must be positive")
        mid_x = (t_left + t_right) / 2.0
        mid_y = (t_bottom + t_top) / 2.0
        # Snap the centre to the global resolution lattice so the grid is a
        # deterministic function of position, not of float noise in bounds.
        origin_x = np.floor(mid_x / resolution_m) * resolution_m - (size / 2.0) * resolution_m
        origin_y = np.ceil(mid_y / resolution_m) * resolution_m + (size / 2.0) * resolution_m
        width = height = size
    else:
        origin_x = np.floor(t_left / resolution_m) * resolution_m
        origin_y = np.ceil(t_top / resolution_m) * resolution_m
        width = int(np.ceil((t_right - origin_x) / resolution_m))
        height = int(np.ceil((origin_y - t_bottom) / resolution_m))
        width = max(1, width)
        height = max(1, height)

    transform = Affine(resolution_m, 0.0, float(origin_x), 0.0, -resolution_m, float(origin_y))
    return AnalysisGrid(
        crs=target_crs,
        transform=transform,
        width=int(width),
        height=int(height),
        resolution_m=float(resolution_m),
    )


def target_grid_for_source(
    source: AnalysisGrid,
    *,
    resolution_m: float = CANONICAL_RESOLUTION_M,
    tile_size: Optional[int] = None,
) -> AnalysisGrid:
    """Canonical grid for an existing source grid. See :func:`target_grid_for_bounds`."""

    return target_grid_for_bounds(
        bounds=source.bounds,
        source_crs=source.crs,
        resolution_m=resolution_m,
        tile_size=tile_size,
    )


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------


def _warp_plane(
    plane: np.ndarray,
    *,
    source: AnalysisGrid,
    target: AnalysisGrid,
    method: Any,
    src_nodata: Optional[float],
    dst_nodata: float,
) -> np.ndarray:
    destination = np.full((target.height, target.width), dst_nodata, dtype="float64")
    reproject(
        source=np.ascontiguousarray(plane, dtype="float64"),
        destination=destination,
        src_transform=source.transform,
        src_crs=CRS.from_user_input(source.crs),
        src_nodata=src_nodata,
        dst_transform=target.transform,
        dst_crs=CRS.from_user_input(target.crs),
        dst_nodata=dst_nodata,
        resampling=method,
    )
    return destination


def reproject_continuous(
    data: np.ndarray,
    valid: np.ndarray,
    *,
    source: AnalysisGrid,
    target: AnalysisGrid,
    resampling_name: str = "bilinear",
    nodata: float = -9999.0,
    nodata_policy: Optional[NodataPolicy] = None,
    domain: str = "linear_power",
) -> ReprojectionResult:
    """Reproject continuous bands, excluding invalid source pixels.

    Parameters
    ----------
    data
        ``[B, H, W]`` or ``[H, W]`` continuous values.
    valid
        ``[H, W]`` boolean source validity mask. Authoritative: it decides what
        may enter an interpolation kernel.
    domain
        Representation of ``data``. Must be ``linear_power`` for SAR when the
        method averages (C3). ``"db"`` is rejected for averaging methods so the
        linear-then-convert ordering cannot be bypassed; convert afterwards with
        :func:`linear_to_db_after_resampling`.
    """

    _require_import()
    policy = nodata_policy or NodataPolicy()
    method = resampling_from_name(resampling_name, field="continuous resampling")

    averaging = resampling_name.lower() not in {"nearest", "mode"}
    if averaging and domain == "db":
        raise PreprocessingConfigurationError(
            "refusing to resample decibel data with an averaging method: averaging "
            "logarithms is not the mean of the underlying power and is biased low. "
            "Resample in linear power, then convert (C3, "
            "configs/preprocessing.yaml -> target_grid.resampling.sar_domain)"
        )
    if domain not in {"linear_power", "db", "reflectance", "elevation", "unitless"}:
        raise PreprocessingConfigurationError(f"unknown continuous domain {domain!r}")

    stack = np.asarray(data)
    squeeze = stack.ndim == 2
    if squeeze:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise PreprocessingConfigurationError(
            f"continuous data must be [B, H, W] or [H, W], got {stack.shape}"
        )
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != stack.shape[-2:]:
        raise PreprocessingConfigurationError(
            f"validity mask shape {mask.shape} does not match data {stack.shape[-2:]}"
        )

    alignment = compare_grids(source, target)

    # Interpolated coverage weight, warped with the SAME method as the data.
    # This is what makes the guarantee provable rather than a property of
    # GDAL's internal nodata handling: a target pixel whose weight is below
    # threshold had an invalid contributor, regardless of how the warp chose
    # to fill it.
    weight = _warp_plane(
        mask.astype("float64"),
        source=source,
        target=target,
        method=method,
        src_nodata=None,
        dst_nodata=0.0,
    )
    target_valid = weight >= policy.threshold

    out = np.full((stack.shape[0], target.height, target.width), float(nodata), dtype="float64")
    for index in range(stack.shape[0]):
        plane = np.array(stack[index], dtype="float64", copy=True)
        # Invalid samples become NaN and are declared as source nodata, so the
        # warp excludes them from the kernel instead of averaging a sentinel.
        plane[~mask] = np.nan
        warped = _warp_plane(
            plane,
            source=source,
            target=target,
            method=method,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        finite = np.isfinite(warped)
        out[index] = np.where(target_valid & finite, warped, float(nodata))
        # A pixel the warp could not fill is invalid even if its coverage
        # weight passed, so the mask never promises a value that is not there.
        target_valid &= finite

    out[:, ~target_valid] = float(nodata)
    if squeeze:
        out = out[0]

    provenance = {
        "operation": "reproject_continuous",
        "resampling": resampling_name,
        "domain": domain,
        "db_conversion_applied_before_resampling": False,
        "source_grid": source.to_dict(),
        "target_grid": target.to_dict(),
        "source_to_target_alignment": alignment.to_dict(),
        "reprojected": not alignment.aligned,
        "nodata": float(nodata),
        "nodata_policy": policy.to_dict(),
        "source_valid_pixels": int(mask.sum()),
        "target_valid_pixels": int(target_valid.sum()),
    }
    return ReprojectionResult(
        data=out.astype("float32", copy=False),
        valid=target_valid,
        grid=target,
        provenance=provenance,
    )


def reproject_categorical(
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    source: AnalysisGrid,
    target: AnalysisGrid,
    ignore_index: int,
    resampling_name: str = "nearest",
    semantic_values: Optional[Sequence[int]] = None,
) -> ReprojectionResult:
    """Reproject a class mask with nearest neighbour, preserving class values.

    Nearest neighbour cannot produce a value absent from the source, so it
    cannot invent a class at a boundary. An averaging method is refused rather
    than permitted-with-a-warning: bilinear across a 0/2 boundary yields 1,
    which in this scheme is "Permanent Waters" — a different, plausible class.

    Target pixels with no valid source coverage become ``ignore_index``, so an
    unobserved pixel is never presented as an observed class.
    """

    _require_import()
    if resampling_name.lower() not in {"nearest", "mode"}:
        raise PreprocessingConfigurationError(
            f"categorical data must not be resampled with {resampling_name!r}: an "
            "averaging method invents classes that were never in the source "
            "(a 0/2 boundary interpolates to 1)"
        )
    method = resampling_from_name(resampling_name, field="categorical resampling")

    source_labels = np.asarray(labels)
    mask = np.asarray(valid, dtype=bool)
    if source_labels.shape != mask.shape:
        raise PreprocessingConfigurationError(
            f"label shape {source_labels.shape} does not match validity {mask.shape}"
        )
    observed = set(int(v) for v in np.unique(source_labels[mask])) if mask.any() else set()
    if semantic_values is not None:
        unexpected = observed - set(int(v) for v in semantic_values)
        if unexpected:
            raise PreprocessingConfigurationError(
                f"valid source pixels carry non-semantic label values {sorted(unexpected)}"
            )

    alignment = compare_grids(source, target)

    # Validity travels by nearest neighbour too, so mask and labels land on
    # exactly the same target pixels.
    warped_valid = _warp_plane(
        mask.astype("float64"),
        source=source,
        target=target,
        method=resampling_from_name("nearest", field="validity resampling"),
        src_nodata=None,
        dst_nodata=0.0,
    )
    target_valid = warped_valid >= 0.5

    # Warp labels with invalid positions carrying the ignore index, so a
    # nearest pick from an invalid neighbour yields ignore, never a class.
    prepared = np.where(mask, source_labels, ignore_index).astype("float64")
    warped = _warp_plane(
        prepared,
        source=source,
        target=target,
        method=method,
        src_nodata=None,
        dst_nodata=float(ignore_index),
    )
    rounded = np.rint(warped).astype("int64")
    out = np.where(target_valid, rounded, int(ignore_index)).astype("int64")
    # Defensive: nearest + rint cannot interpolate, but if a value outside the
    # declared set ever appeared it must not reach a loss function silently.
    if semantic_values is not None:
        allowed = set(int(v) for v in semantic_values) | {int(ignore_index)}
        present = set(int(v) for v in np.unique(out))
        stray = present - allowed
        if stray:
            raise PreprocessingConfigurationError(
                f"reprojected labels contain values outside {sorted(allowed)}: {sorted(stray)}"
            )

    provenance = {
        "operation": "reproject_categorical",
        "resampling": resampling_name,
        "ignore_index": int(ignore_index),
        "source_grid": source.to_dict(),
        "target_grid": target.to_dict(),
        "source_to_target_alignment": alignment.to_dict(),
        "reprojected": not alignment.aligned,
        "source_classes_present": sorted(observed),
        "source_valid_pixels": int(mask.sum()),
        "target_valid_pixels": int(target_valid.sum()),
    }
    return ReprojectionResult(
        data=out, valid=target_valid, grid=target, provenance=provenance
    )


def linear_to_db_after_resampling(
    linear: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert resampled linear power to decibels. Call AFTER reprojection.

    Thin delegation to :func:`floodmap.features.numerics.linear_to_db`, which
    already handles the domain correctly: zero and negative backscatter are
    marked invalid rather than floored, because low backscatter is itself the
    candidate open-water signal and flooring would turn a non-observation into
    a water-like value.

    It exists in this module so the correct ordering is expressible as one
    obvious call at the right point, rather than something a caller has to
    remember not to do earlier.
    """

    from ..features.numerics import linear_to_db

    return linear_to_db(linear, valid)
