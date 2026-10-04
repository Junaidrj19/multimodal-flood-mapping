"""Explicit analysis-grid and alignment primitives.

The grid is never inferred for a production run. Tests may construct one from
small synthetic rasters, but a configuration used by a real run must provide
CRS, transform, dimensions, resolution and resampling methods explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .errors import AlignmentError, PreprocessingConfigurationError

try:  # Imported lazily by the public module so import errors are explanatory.
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds
except ImportError as exc:  # pragma: no cover - exercised in environments without geo extra
    Affine = None  # type: ignore[assignment]
    CRS = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]
    array_bounds = None  # type: ignore[assignment]
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None

__all__ = [
    "AlignmentReport",
    "AnalysisGrid",
    "compare_grids",
    "require_aligned",
    "require_rasterio",
    "resampling_from_name",
]


def require_rasterio() -> None:
    if _RASTERIO_IMPORT_ERROR is not None:
        raise PreprocessingConfigurationError(
            "M2 requires rasterio for raster metadata, masks and reprojection; "
            "install the project dependencies before running preprocessing"
        ) from _RASTERIO_IMPORT_ERROR


def _normalise_crs(value: Any) -> str:
    require_rasterio()
    if value is None:
        raise PreprocessingConfigurationError("analysis-grid CRS is unresolved")
    try:
        return CRS.from_user_input(value).to_string()
    except Exception as exc:
        raise PreprocessingConfigurationError(f"invalid analysis-grid CRS: {value!r}") from exc


def resampling_from_name(name: Optional[str], *, field: str = "resampling") -> Any:
    """Resolve an explicitly configured Rasterio resampling method."""
    require_rasterio()
    if name is None:
        raise PreprocessingConfigurationError(
            f"{field} is unresolved; no resampling default is used"
        )
    try:
        return getattr(Resampling, str(name).lower())
    except AttributeError as exc:
        allowed = ", ".join(item.name for item in Resampling)
        raise PreprocessingConfigurationError(
            f"unsupported {field}={name!r}; choose one of {allowed}"
        ) from exc


@dataclass(frozen=True)
class AnalysisGrid:
    """A complete, immutable raster grid."""

    crs: str
    transform: Any
    width: int
    height: int
    resolution_m: float

    def __post_init__(self) -> None:
        require_rasterio()
        if self.width < 1 or self.height < 1:
            raise PreprocessingConfigurationError("analysis-grid dimensions must be positive")
        if self.resolution_m <= 0:
            raise PreprocessingConfigurationError("analysis-grid resolution must be positive")
        if not isinstance(self.transform, Affine):
            raise PreprocessingConfigurationError(
                "analysis-grid transform must be an affine transform"
            )
        if (
            abs(abs(self.transform.a) - self.resolution_m) > 1e-9
            or abs(abs(self.transform.e) - self.resolution_m) > 1e-9
        ):
            raise PreprocessingConfigurationError(
                "analysis-grid resolution_m contradicts transform pixel size"
            )
        object.__setattr__(self, "crs", _normalise_crs(self.crs))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AnalysisGrid":
        require_rasterio()
        missing = [
            key
            for key in ("crs", "resolution_m", "transform", "width", "height")
            if mapping.get(key) is None
        ]
        if missing:
            raise PreprocessingConfigurationError(
                "target_grid is unresolved; required fields are null: " + ", ".join(missing)
            )
        raw_transform = mapping["transform"]
        if not isinstance(raw_transform, (list, tuple)) or len(raw_transform) != 6:
            raise PreprocessingConfigurationError(
                "target_grid.transform must contain six affine coefficients"
            )
        return cls(
            crs=str(mapping["crs"]),
            transform=Affine(*[float(value) for value in raw_transform]),
            width=int(mapping["width"]),
            height=int(mapping["height"]),
            resolution_m=float(mapping["resolution_m"]),
        )

    @classmethod
    def from_dataset(cls, dataset: Any) -> "AnalysisGrid":
        require_rasterio()
        if dataset.crs is None:
            raise PreprocessingConfigurationError("source raster has no CRS")
        resolution = float((abs(dataset.transform.a) + abs(dataset.transform.e)) / 2)
        return cls(
            crs=dataset.crs.to_string(),
            transform=dataset.transform,
            width=int(dataset.width),
            height=int(dataset.height),
            resolution_m=resolution,
        )

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        require_rasterio()
        return tuple(
            float(value) for value in array_bounds(self.height, self.width, self.transform)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crs": self.crs,
            "resolution_m": self.resolution_m,
            "transform": [float(value) for value in self.transform[:6]],
            "width": self.width,
            "height": self.height,
            "bounds": list(self.bounds),
        }


@dataclass(frozen=True)
class AlignmentReport:
    """Machine-readable result of comparing two complete grids."""

    aligned: bool
    checks: Dict[str, bool]
    failures: Tuple[str, ...]
    tolerance: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aligned": self.aligned,
            "checks": dict(self.checks),
            "failures": list(self.failures),
            "tolerance": self.tolerance,
        }


def compare_grids(
    reference: AnalysisGrid,
    candidate: AnalysisGrid,
    *,
    tolerance: Optional[float] = None,
) -> AlignmentReport:
    """Compare CRS, transform, resolution, extent, dimensions and pixel alignment."""
    tol = 0.0 if tolerance is None else float(tolerance)
    if tol < 0:
        raise ValueError("alignment tolerance cannot be negative")

    def close(left: float, right: float) -> bool:
        return abs(left - right) <= tol

    transform_values = zip(reference.transform[:6], candidate.transform[:6])
    checks = {
        "crs": reference.crs == candidate.crs,
        "dimensions": reference.width == candidate.width and reference.height == candidate.height,
        "resolution": close(reference.resolution_m, candidate.resolution_m),
        "transform": all(close(float(left), float(right)) for left, right in transform_values),
        "extent": all(
            close(left, right) for left, right in zip(reference.bounds, candidate.bounds)
        ),
    }
    checks["pixel_alignment"] = checks["transform"] and checks["dimensions"]
    failures = tuple(name for name, passed in checks.items() if not passed)
    return AlignmentReport(
        aligned=not failures, checks=checks, failures=failures, tolerance=tolerance
    )


def require_aligned(
    reference: AnalysisGrid,
    candidate: AnalysisGrid,
    *,
    tolerance: Optional[float] = None,
) -> AlignmentReport:
    report = compare_grids(reference, candidate, tolerance=tolerance)
    if not report.aligned:
        raise AlignmentError("raster grids are not aligned: " + ", ".join(report.failures))
    return report
