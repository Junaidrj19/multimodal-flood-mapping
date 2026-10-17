"""Candidate flood-region evidence from a real Sentinel-1 RTC pair.

This module is intentionally narrower than a flood classifier.  It consumes two
already assembled, georeferenced RTC rasters, verifies that they are comparable,
computes representation-aware change features, and applies an explicit
dual-polarisation darkening rule.  The result is a *change candidate* mask, not a
confirmed flood or damage map.

The RTC products used here are gamma-nought (γ⁰) linear power.  They are not
sigma-nought (σ⁰), so no sigma-specific training clip or learned-model
equivalence is applied.  The dB change features are computed as a log ratio of
the source linear power values.

Every candidate product is scoped to the *configured* area of interest.  The
delivered RTC rasters are bounding-box crops that were cut for an earlier study
extent, so the raster window and the AOI polygon are different areas and must
never share a denominator.  The AOI is therefore rasterized onto the delivered
grid by pixel centre and intersected into the validity mask, and the summary
reports the raster-window scope and the AOI scope separately.  The AOI identity
is read from the configuration and its source bytes are verified by digest, so
a stage run cannot label its outputs with an extent it did not actually use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import shapely
import yaml
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from ..acquisition.aoi import AreaOfInterest
from ..preprocessing.grid import AnalysisGrid, require_aligned
from ..preprocessing.validation import read_valid_mask, require_valid_raster
from ..utils.provenance import (
    ArtifactProvenance,
    ArtifactType,
    BeforeAfterPair,
    ProductionInput,
    SceneReference,
)
from ..features.numerics import log_ratio_db

__all__ = [
    "AOI_MASK_NODATA",
    "APPROVED_COVERAGE_DISCLOSURE",
    "CHANGE_CANDIDATE_NODATA",
    "CHANGE_FEATURE_NAMES",
    "RTC_CHANGE_VERSION",
    "AoiBinding",
    "RtcChangeCandidateResult",
    "RtcChangeError",
    "RtcPairMetadata",
    "compute_rtc_change_candidates",
    "load_configured_aoi_binding",
    "load_rtc_change_config",
    "write_rtc_change_artifacts",
]

RTC_CHANGE_VERSION = "m6.3-rtc-change-candidates-0.2.0"
CHANGE_FEATURE_NAMES: Tuple[str, str] = ("vv_change_db", "vh_change_db")
CHANGE_CANDIDATE_NODATA = 255
CHANGE_FEATURE_NODATA = -9999.0
AOI_MASK_NODATA = 0
RTC_CONVENTION = "gamma0_linear_power"
AOI_METRIC_CRS = "EPSG:32645"

#: The human-approved partial-coverage disclosure (M6.14 D6).  It is reproduced
#: verbatim on every generated artifact; paraphrasing it would weaken the claim
#: limit the decision record actually authorised.
APPROVED_COVERAGE_DISCLOSURE = (
    "The study corridor extends beyond the currently available valid satellite "
    "observation coverage. Approximately 63.4% of the AOI has delivered imagery "
    "coverage; the remaining 36.6% is unobserved in the current dataset. Results "
    "apply only to the observed, valid-data area. Unobserved areas must not be "
    "interpreted as unaffected, flood-free, or undamaged."
)


class RtcChangeError(ValueError):
    """The RTC pair or candidate-stage configuration is not usable."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AoiBinding:
    """The configured study extent, pinned by identifier *and* source bytes.

    The stage derives this from ``configs/data.yaml`` rather than accepting a
    free-text identifier.  A label is not evidence: an operator who mistypes an
    identifier, or who re-points a path at different geometry, would otherwise
    stamp an extent onto statistics that were never computed against it.  Both
    the digest and the geometry therefore come from the same verified bytes.
    """

    aoi_id: str
    source_sha256: str
    geojson_path: Path
    provenance_status: str
    geometry_4326: BaseGeometry
    name: str = ""
    coverage: Dict[str, Any] = None  # type: ignore[assignment]
    provenance_path: Optional[Path] = None
    provenance_sha256: Optional[str] = None

    @property
    def disclosure(self) -> str:
        """The coverage disclosure recorded beside the geometry, if any."""
        recorded = (self.coverage or {}).get("disclosure")
        if isinstance(recorded, str) and recorded.strip():
            return recorded.strip()
        return APPROVED_COVERAGE_DISCLOSURE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aoi_id": self.aoi_id,
            "name": self.name,
            "source_sha256": self.source_sha256,
            "geojson_path": str(self.geojson_path),
            "provenance_status": self.provenance_status,
            "provenance_path": (
                str(self.provenance_path) if self.provenance_path is not None else None
            ),
            "provenance_sha256": self.provenance_sha256,
            "binding": "configuration_acquisition_aoi_verified_by_digest",
        }


def load_configured_aoi_binding(
    data_config_path: Path | str,
    *,
    project_root: Optional[Path] = None,
) -> AoiBinding:
    """Read and verify the configured AOI that every output will be keyed to.

    The digest in ``acquisition.aoi.sha256`` is *enforced* against the geometry
    file.  When a provenance sidecar is configured it must agree with the same
    identifier and digest, so coverage figures cannot be inherited from a
    different extent.  Nothing is downloaded and no gate is consulted or opened.
    """
    path = Path(data_config_path)
    root = Path(project_root) if project_root is not None else path.resolve().parents[1]
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RtcChangeError(f"could not read data configuration {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise RtcChangeError(f"data configuration must be a mapping: {path}")
    raw = document.get("acquisition", {})
    raw = raw.get("aoi") if isinstance(raw, Mapping) else None
    if not isinstance(raw, Mapping):
        raise RtcChangeError(f"{path} has no acquisition.aoi block")

    aoi_id = raw.get("aoi_id")
    if not isinstance(aoi_id, str) or not aoi_id.strip():
        raise RtcChangeError("acquisition.aoi.aoi_id is required to key candidate outputs")
    digest = raw.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest.lower())
    ):
        raise RtcChangeError("acquisition.aoi.sha256 must be a 64-character hex digest")
    digest = digest.lower()
    geometry_path = root / str(raw["geojson_path"])

    # from_geojson_file enforces the digest and refuses MultiPolygon reduction.
    aoi = AreaOfInterest.from_geojson_file(aoi_id.strip(), geometry_path, expected_sha256=digest)
    geometry = shapely.from_wkt(aoi.to_wkt())
    if geometry.is_empty or not geometry.is_valid:
        raise RtcChangeError(f"configured AOI geometry is empty or invalid: {geometry_path}")

    coverage: Dict[str, Any] = {}
    provenance_sha256: Optional[str] = None
    # Same resolution order as the dashboard binding: an explicit configured
    # path, otherwise the sidecar sitting beside the geometry. Resolving these
    # differently would let the two surfaces disagree about coverage.
    provenance_path: Optional[Path] = (
        root / str(raw["provenance_path"])
        if raw.get("provenance_path")
        else geometry_path.with_suffix(".provenance.yaml")
    )
    if provenance_path is not None:
        if not provenance_path.is_file():
            raise RtcChangeError(f"configured AOI provenance file is absent: {provenance_path}")
        sidecar = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, Mapping):
            raise RtcChangeError(f"AOI provenance must be a mapping: {provenance_path}")
        if sidecar.get("aoi_id") != aoi_id.strip():
            raise RtcChangeError(
                f"AOI provenance identifier {sidecar.get('aoi_id')!r} does not match "
                f"acquisition.aoi.aoi_id {aoi_id!r}"
            )
        recorded = (sidecar.get("geometry") or {}).get("sha256")
        if recorded != digest:
            raise RtcChangeError(
                "AOI provenance geometry digest does not match acquisition.aoi.sha256"
            )
        sidecar_coverage = sidecar.get("coverage")
        coverage = dict(sidecar_coverage) if isinstance(sidecar_coverage, Mapping) else {}
        provenance_sha256 = _file_sha256(provenance_path)

    return AoiBinding(
        aoi_id=aoi_id.strip(),
        source_sha256=digest,
        geojson_path=geometry_path,
        provenance_status=str(raw.get("provenance_status", "unverified")),
        geometry_4326=geometry,
        name=str(raw.get("name", "")),
        coverage=coverage,
        provenance_path=provenance_path,
        provenance_sha256=provenance_sha256,
    )


@dataclass(frozen=True)
class RtcPairMetadata:
    """Operator-supplied identity for the selected, already verified pair."""

    aoi_id: str
    event_date: str
    pre_scene_id: str
    post_scene_id: str
    pre_acquired_at: str
    post_acquired_at: str
    relative_orbit: int
    orbit_direction: str
    aoi_provenance_status: str = "UNVERIFIED"
    aoi_source_sha256: Optional[str] = None
    aoi_geojson_path: Optional[str] = None
    aoi_name: str = ""

    def __post_init__(self) -> None:
        if not self.aoi_id.strip():
            raise RtcChangeError("aoi_id is required")
        if not self.pre_scene_id.strip() or not self.post_scene_id.strip():
            raise RtcChangeError("pre_scene_id and post_scene_id are required")
        if self.aoi_provenance_status.strip().upper() != "UNVERIFIED":
            raise RtcChangeError(
                "the local RTC candidate stage preserves AOI provenance as UNVERIFIED; "
                "a different status requires an explicit upstream milestone decision"
            )
        if self.aoi_source_sha256 is not None:
            digest = self.aoi_source_sha256.strip().lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise RtcChangeError(
                    "aoi_source_sha256 must be a 64-character hex digest when supplied; "
                    "an identifier alone is not evidence of the extent actually used"
                )
        for label, value in (
            ("event_date", self.event_date),
            ("pre_acquired_at", self.pre_acquired_at),
            ("post_acquired_at", self.post_acquired_at),
        ):
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise RtcChangeError(f"{label} must be ISO-8601: {value!r}") from exc
        if self.relative_orbit < 0:
            raise RtcChangeError("relative_orbit must be non-negative")

    @classmethod
    def from_binding(
        cls,
        binding: AoiBinding,
        *,
        event_date: str,
        pre_scene_id: str,
        post_scene_id: str,
        pre_acquired_at: str,
        post_acquired_at: str,
        relative_orbit: int,
        orbit_direction: str,
    ) -> "RtcPairMetadata":
        """Key the pair to a verified configured extent rather than a label."""
        return cls(
            aoi_id=binding.aoi_id,
            event_date=event_date,
            pre_scene_id=pre_scene_id,
            post_scene_id=post_scene_id,
            pre_acquired_at=pre_acquired_at,
            post_acquired_at=post_acquired_at,
            relative_orbit=relative_orbit,
            orbit_direction=orbit_direction,
            aoi_provenance_status=binding.provenance_status.strip().upper(),
            aoi_source_sha256=binding.source_sha256,
            aoi_geojson_path=str(binding.geojson_path),
            aoi_name=binding.name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aoi_id": self.aoi_id,
            "aoi_name": self.aoi_name,
            "aoi_source_sha256": self.aoi_source_sha256,
            "aoi_geojson_path": self.aoi_geojson_path,
            "event_date": self.event_date,
            "aoi_provenance_status": self.aoi_provenance_status,
            "pre": {
                "scene_id": self.pre_scene_id,
                "acquired_at": self.pre_acquired_at,
                "relative_orbit": self.relative_orbit,
                "orbit_direction": self.orbit_direction,
            },
            "post": {
                "scene_id": self.post_scene_id,
                "acquired_at": self.post_acquired_at,
                "relative_orbit": self.relative_orbit,
                "orbit_direction": self.orbit_direction,
            },
        }


@dataclass(frozen=True)
class RtcChangeCandidateResult:
    """In-memory result of the representation-aware change calculation.

    ``valid`` is already intersected with ``aoi_mask``, so ``candidates`` is a
    subset of the observed, in-extent pixels by construction.  ``aoi_mask``
    is retained separately so that "outside the study extent" and "inside the
    study extent but not observed" stay distinguishable in the written outputs.
    """

    pre_valid: np.ndarray
    post_valid: np.ndarray
    valid: np.ndarray
    vv_change_db: np.ndarray
    vh_change_db: np.ndarray
    candidates: np.ndarray
    alignment: Dict[str, Any]
    input_metadata: Dict[str, Any]
    aoi_mask: np.ndarray
    aoi_scope: Dict[str, Any]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.candidates.shape

    def summary(self, *, vv_threshold_db: float, vh_threshold_db: float) -> Dict[str, Any]:
        total = int(self.valid.size)
        valid_count = int(self.valid.sum())
        candidate_count = int(self.candidates.sum())
        grid_resolution = float(self.input_metadata["grid"]["resolution_m"])
        pixel_area_m2 = grid_resolution * grid_resolution

        def distribution(values: np.ndarray) -> Dict[str, float | None]:
            observed = values[self.valid & np.isfinite(values)]
            if observed.size == 0:
                return {str(p): None for p in (1, 5, 25, 50, 75, 95, 99)}
            return {
                str(p): float(np.percentile(observed, p))
                for p in (1, 5, 25, 50, 75, 95, 99)
            }

        return {
            "stage": RTC_CHANGE_VERSION,
            "status": "candidate_evidence_only",
            "radiometric_convention": RTC_CONVENTION,
            # The top-level counts are mechanical tallies over the *delivered
            # raster window*, which is the array these products are written on.
            # The window is a bounding-box crop, not the study extent; the
            # ``aoi_scope`` block below carries the extent-correct denominators.
            "scope": "delivered_raster_window_array_counts",
            "scope_note": (
                "total_pixels, valid_fraction and candidate_fraction_of_grid use the "
                "delivered raster window as their denominator. The window is a "
                "bounding-box crop and is NOT the AOI; see aoi_scope for AOI-relative "
                "figures and their denominators."
            ),
            "total_pixels": total,
            "valid_pixels": valid_count,
            "invalid_pixels": total - valid_count,
            "valid_fraction": float(valid_count / total) if total else 0.0,
            "candidate_pixels": candidate_count,
            "candidate_fraction_of_valid": (
                float(candidate_count / valid_count) if valid_count else 0.0
            ),
            "candidate_fraction_of_grid": float(candidate_count / total) if total else 0.0,
            "candidate_grid_area_km2": float(candidate_count * pixel_area_m2 / 1_000_000.0),
            "pixel_size_m": [grid_resolution, grid_resolution],
            "candidate_rule": {
                "name": "dual_polarisation_darkening_candidate",
                "vv_change_db_max": float(vv_threshold_db),
                "vh_change_db_max": float(vh_threshold_db),
                "expression": (
                    "valid AND vv_change_db <= vv_change_db_max AND "
                    "vh_change_db <= vh_change_db_max"
                ),
                "scope": "pixels_inside_configured_aoi_with_valid_pre_and_post_observations",
            },
            "aoi_scope": self.aoi_scope,
            "change_distribution_db": {
                "vv": distribution(self.vv_change_db),
                "vh": distribution(self.vh_change_db),
                "support": "configured_AOI_valid_pixels_only",
            },
            "alignment": self.alignment,
            "input_metadata": self.input_metadata,
            "interpretation": (
                "These are Sentinel-1 RTC gamma0 darkening candidates inside the "
                "configured study extent. They are not confirmed flood, damage, or "
                "ground-truth labels, and the unobserved part of the extent carries "
                "no finding at all."
            ),
        }


def load_rtc_change_config(config_path: Path | str) -> Dict[str, Any]:
    """Load and validate the small candidate-stage configuration."""
    path = Path(config_path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RtcChangeError(f"could not read RTC change configuration {path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise RtcChangeError(f"RTC change configuration must be a mapping: {path}")
    inputs = config.get("inputs")
    rule = config.get("candidate_rule")
    if not isinstance(inputs, Mapping) or not isinstance(rule, Mapping):
        raise RtcChangeError("RTC change configuration needs inputs and candidate_rule mappings")
    if inputs.get("radiometric_convention") != RTC_CONVENTION:
        raise RtcChangeError(
            "candidate stage requires inputs.radiometric_convention=gamma0_linear_power; "
            "sigma0 settings are not accepted"
        )
    if inputs.get("require_same_grid") is not True:
        raise RtcChangeError(
            "inputs.require_same_grid must be true; pixel-level change cannot proceed "
            "without explicit grid alignment"
        )
    for key in ("vv_change_db_max", "vh_change_db_max"):
        value = rule.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise RtcChangeError(f"candidate_rule.{key} must be a finite number")
        if float(value) >= 0.0:
            raise RtcChangeError(
                f"candidate_rule.{key} must be negative for a darkening candidate rule"
            )
    return dict(config)


def _grid_from_metadata(metadata: Any) -> AnalysisGrid:
    """Build the shared grid from a validated raster metadata record."""
    return metadata.as_grid()


def _input_record(path: Path, metadata: Any, valid: np.ndarray) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "crs": metadata.crs,
        "shape": [metadata.height, metadata.width],
        "pixel_size_m": [float(abs(metadata.resolution[0])), float(abs(metadata.resolution[1]))],
        "transform": [float(value) for value in metadata.transform[:6]],
        "bounds": list(metadata.bounds),
        "nodata": metadata.nodata,
        "band_names": [value.strip() if value else None for value in metadata.descriptions],
        "radiometric_convention": metadata.tags.get("radiometric_convention"),
        "valid_pixels_all_bands": int(valid.sum()),
        "valid_fraction_all_bands": float(valid.mean()) if valid.size else 0.0,
        "valid_fraction_denominator": "delivered_raster_window_not_aoi",
        "source_item_id": metadata.tags.get("source_item_id"),
    }


def _project(geometry: BaseGeometry, source: str, target: str) -> BaseGeometry:
    """Reproject a geometry vertex-wise; no densification is performed."""
    if source == target:
        return geometry
    transformer = Transformer.from_crs(source, target, always_xy=True)
    return shapely.transform(
        geometry,
        lambda coords: np.column_stack(transformer.transform(coords[:, 0], coords[:, 1])),
    )


def _rasterize_aoi(
    binding: AoiBinding, grid: Mapping[str, Any], shape: Tuple[int, int]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mark pixels whose *centre* lies inside the configured AOI.

    ``all_touched=False`` keeps a pixel only when its centre is inside the
    polygon, which is the same convention the AOI coverage measurement used, so
    the two pixel counts are comparable.  No resampling or geometry repair
    happens: the delivered grid and the approved geometry are both taken as-is.
    """
    raster_crs = str(grid["crs"])
    native = _project(binding.geometry_4326, "EPSG:4326", raster_crs)
    if native.is_empty or not native.is_valid:
        raise RtcChangeError("configured AOI is empty or invalid in the raster CRS")
    transform = Affine(*grid["transform"])
    mask = geometry_mask(
        [mapping(native)],
        out_shape=shape,
        transform=transform,
        all_touched=False,
        invert=True,
    )
    return mask, {"raster_crs": raster_crs, "native_geometry": native}


def _aoi_scope(
    *,
    binding: AoiBinding,
    aoi_mask: np.ndarray,
    delivered_valid: np.ndarray,
    candidates: np.ndarray,
    grid: Mapping[str, Any],
    native_aoi: BaseGeometry,
) -> Dict[str, Any]:
    """Build AOI-relative statistics with every denominator named explicitly.

    Three different areas are reported and never conflated: the *full* geometric
    AOI, the part of it the delivered raster window reaches, and the part that
    carries a jointly valid pre/post observation.  The full AOI area is never
    used as an observed-area denominator.
    """
    raster_crs = str(grid["crs"])
    pixel_area_m2 = float(grid["resolution_m"]) ** 2
    metric_aoi = _project(binding.geometry_4326, "EPSG:4326", AOI_METRIC_CRS)
    aoi_area_m2 = float(metric_aoi.area)

    window_native = shapely.box(*[float(value) for value in grid["bounds"]])
    delivered_native = native_aoi.intersection(window_native)
    delivered_native = delivered_native if delivered_native.area > 0 else shapely.Polygon()
    delivered_metric = _project(delivered_native, raster_crs, AOI_METRIC_CRS)
    delivered_area_m2 = min(aoi_area_m2, max(0.0, float(delivered_metric.area)))

    centers = int(np.count_nonzero(aoi_mask))
    valid_count = int(np.count_nonzero(delivered_valid))
    candidate_count = int(np.count_nonzero(candidates))
    outside = int(aoi_mask.size - centers)
    delivered_fraction = float(delivered_area_m2 / aoi_area_m2) if aoi_area_m2 else None

    def fraction(numerator: int, denominator: int) -> Optional[float]:
        return float(numerator / denominator) if denominator else None

    return {
        "aoi": binding.to_dict(),
        "metric_crs": AOI_METRIC_CRS,
        "raster_crs": raster_crs,
        # --- the three distinct areas -------------------------------------
        "aoi_geometric_area_km2": aoi_area_m2 / 1e6,
        "aoi_delivered_area_km2": delivered_area_m2 / 1e6,
        "aoi_unobserved_area_km2": (aoi_area_m2 - delivered_area_m2) / 1e6,
        "aoi_delivered_fraction": delivered_fraction,
        "aoi_unobserved_fraction": (
            None if delivered_fraction is None else max(0.0, 1.0 - delivered_fraction)
        ),
        # --- pixel-centre tallies inside the AOI ---------------------------
        "aoi_pixel_centers_in_window": centers,
        "aoi_pixel_center_area_km2": centers * pixel_area_m2 / 1e6,
        "aoi_valid_pixels": valid_count,
        "aoi_valid_area_km2": valid_count * pixel_area_m2 / 1e6,
        "aoi_delivered_but_invalid_pixels": centers - valid_count,
        "aoi_valid_fraction_of_delivered_pixels": fraction(valid_count, centers),
        "window_pixels_outside_aoi": outside,
        "window_pixels_outside_aoi_fraction": fraction(outside, int(aoi_mask.size)),
        # --- candidates, against named denominators ------------------------
        "candidate_pixels": candidate_count,
        "candidate_area_km2": candidate_count * pixel_area_m2 / 1e6,
        "candidate_fraction_of_aoi_valid_pixels": fraction(candidate_count, valid_count),
        "candidate_fraction_of_aoi_delivered_pixels": fraction(candidate_count, centers),
        "candidate_fraction_of_aoi_geometric_area": (
            float(candidate_count * pixel_area_m2 / aoi_area_m2) if aoi_area_m2 else None
        ),
        "denominator_definitions": {
            "aoi_geometric_area_km2": (
                "Area of the full approved AOI polygon in EPSG:32645. This is the study "
                "extent, NOT an observed area, and must never be described as observed."
            ),
            "aoi_delivered_area_km2": (
                "Geometric intersection of the AOI polygon with the delivered raster "
                "window bounding box. Delivery is where pixels exist, which is distinct "
                "from where those pixels are valid."
            ),
            "aoi_unobserved_area_km2": (
                "AOI area the delivered raster window does not reach. No observation "
                "exists there; it is unknown, not a negative finding."
            ),
            "aoi_valid_area_km2": (
                "Pixel-centre count times native pixel area, for centres inside the AOI "
                "with a jointly valid pre and post VV+VH observation. A rasterization "
                "estimate, not a geometric intersection."
            ),
            "candidate_fraction_of_aoi_valid_pixels": (
                "Primary candidate rate. Denominator is the observed, valid, in-extent "
                "pixel count, the only support the rule could actually evaluate."
            ),
            "candidate_fraction_of_aoi_geometric_area": (
                "Candidate area over the FULL AOI area. A lower bound only: 36.6% of the "
                "AOI is unobserved, so this is not a prevalence rate over observed land."
            ),
            "window_pixels_outside_aoi": (
                "Delivered window pixels whose centres fall outside the AOI. They are "
                "excluded from every AOI-scoped figure and from the candidate mask."
            ),
        },
        "methods": {
            "rasterization": "center",
            "all_touched": False,
            "resampling": "none",
            "geometry_repair": "none",
            "validity": (
                "pre AND post, VV AND VH: GDAL mask AND not_nodata AND finite AND "
                "strictly_positive_linear_power"
            ),
            "candidate_subset_of_valid": True,
            "aoi_intersected_into_validity_before_rule": True,
        },
        "coverage_reference": {
            "source": (
                str(binding.provenance_path) if binding.provenance_path is not None else None
            ),
            "recorded_delivered_fraction": (binding.coverage or {}).get("delivered_fraction"),
            "recorded_combined_valid_pixel_count": (binding.coverage or {}).get(
                "combined_valid_pixel_count"
            ),
            "note": (
                "Independently recomputed here from the same local rasters; the AOI "
                "sidecar figures are carried for cross-checking, not copied as results."
            ),
        },
        "disclosure": binding.disclosure,
        "disclosure_matches_m614_approved_text": (
            binding.disclosure.strip() == APPROVED_COVERAGE_DISCLOSURE
        ),
        "claims_scope": (
            "Observed valid-data area inside the configured AOI only. Unobserved "
            "regions are unknown, not unaffected, flood-free, or undamaged."
        ),
    }


def _read_rtc_input(path: Path | str) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read a two-band assembled RTC raster and its authoritative valid mask."""
    path = Path(path)
    metadata = require_valid_raster(path, min_bands=2, required_bands=("VV", "VH"))
    if metadata.tags.get("radiometric_convention") != RTC_CONVENTION:
        raise RtcChangeError(
            f"{path} does not declare {RTC_CONVENTION!r}; found "
            f"{metadata.tags.get('radiometric_convention')!r}. The candidate stage "
            "refuses to infer or convert the radiometric convention."
        )
    if metadata.nodata is None:
        raise RtcChangeError(f"{path} has no nodata value; valid-data accounting would be ambiguous")
    with rasterio.open(path) as dataset:
        values = dataset.read((1, 2)).astype("float64", copy=False)
    valid = read_valid_mask(path, band_indexes=(1, 2))
    valid &= np.all(np.isfinite(values), axis=0)
    return metadata, values, valid, _input_record(path, metadata, valid)


def compute_rtc_change_candidates(
    pre_path: Path | str,
    post_path: Path | str,
    *,
    aoi: AoiBinding,
    vv_threshold_db: float,
    vh_threshold_db: float,
    alignment_tolerance: float | None = 0.0,
) -> RtcChangeCandidateResult:
    """Compute γ⁰ change features and candidate pixels from one RTC pair.

    The function refuses a grid mismatch, a missing/incorrect radiometric
    convention, a missing nodata declaration, non-positive power, and any
    non-finite result. Invalid pixels remain invalid; they are never filled from
    neighbouring observations.

    ``aoi`` is mandatory: the configured study extent is intersected into the
    validity mask *before* the candidate rule is applied, so a candidate is
    always an observed, in-extent pixel.  A pair that does not intersect the
    extent at all is refused rather than returning an empty product that could
    be mistaken for "no change detected".
    """
    if not np.isfinite(vv_threshold_db) or not np.isfinite(vh_threshold_db):
        raise RtcChangeError("candidate thresholds must be finite")
    pre_metadata, pre_values, pre_valid, pre_record = _read_rtc_input(pre_path)
    post_metadata, post_values, post_valid, post_record = _read_rtc_input(post_path)

    alignment = require_aligned(
        _grid_from_metadata(pre_metadata),
        _grid_from_metadata(post_metadata),
        tolerance=alignment_tolerance,
    ).to_dict()
    if pre_values.shape != post_values.shape:
        # require_aligned already catches this, but the explicit guard makes the
        # array contract obvious if the reader changes later.
        raise RtcChangeError("pre and post RTC arrays do not have the same shape")

    grid = _grid_from_metadata(pre_metadata).to_dict()
    aoi_mask, aoi_geometry = _rasterize_aoi(aoi, grid, pre_valid.shape)
    if not aoi_mask.any():
        raise RtcChangeError(
            f"configured AOI {aoi.aoi_id} has no pixel centre inside the delivered raster "
            "window; this stage will not emit an all-nodata product that could be read "
            "as an absence of change"
        )

    vv_pre, vh_pre = pre_values
    vv_post, vh_post = post_values
    vv_pre_valid = pre_valid & np.isfinite(vv_pre) & (vv_pre > 0.0)
    vv_post_valid = post_valid & np.isfinite(vv_post) & (vv_post > 0.0)
    vh_pre_valid = pre_valid & np.isfinite(vh_pre) & (vh_pre > 0.0)
    vh_post_valid = post_valid & np.isfinite(vh_post) & (vh_post > 0.0)

    vv_change, vv_change_valid = log_ratio_db(
        vv_post, vv_pre, vv_post_valid, vv_pre_valid
    )
    vh_change, vh_change_valid = log_ratio_db(
        vh_post, vh_pre, vh_post_valid, vh_pre_valid
    )
    # Scope validity to the study extent before the rule runs. Pixels outside the
    # extent are observations the project is not analysing, not observations that
    # failed; either way the rule must not be evaluated on them.
    valid = vv_change_valid & vh_change_valid & aoi_mask
    candidates = valid & (vv_change <= float(vv_threshold_db)) & (
        vh_change <= float(vh_threshold_db)
    )

    # Keep invalid values out of downstream arithmetic and make nodata visible
    # in the written feature raster. The boolean validity mask remains separate.
    vv_change = np.where(valid, vv_change, CHANGE_FEATURE_NODATA).astype("float32")
    vh_change = np.where(valid, vh_change, CHANGE_FEATURE_NODATA).astype("float32")
    return RtcChangeCandidateResult(
        pre_valid=pre_valid,
        post_valid=post_valid,
        valid=valid,
        vv_change_db=vv_change,
        vh_change_db=vh_change,
        candidates=candidates,
        alignment=alignment,
        input_metadata={"pre": pre_record, "post": post_record, "grid": grid},
        aoi_mask=aoi_mask,
        aoi_scope=_aoi_scope(
            binding=aoi,
            aoi_mask=aoi_mask,
            delivered_valid=valid,
            candidates=candidates,
            grid=grid,
            native_aoi=aoi_geometry["native_geometry"],
        ),
    )


def _scene_pair(metadata: RtcPairMetadata) -> BeforeAfterPair:
    return BeforeAfterPair(
        before=SceneReference(
            scene_id=metadata.pre_scene_id,
            acquired_at=metadata.pre_acquired_at,
            relative_orbit=metadata.relative_orbit,
            orbit_direction=metadata.orbit_direction,
            processing_level="sentinel-1-rtc",
            platform="sentinel-1d",
            product_type="sentinel-1-rtc",
            source=ProductionInput.SENTINEL1,
        ),
        after=SceneReference(
            scene_id=metadata.post_scene_id,
            acquired_at=metadata.post_acquired_at,
            relative_orbit=metadata.relative_orbit,
            orbit_direction=metadata.orbit_direction,
            processing_level="sentinel-1-rtc",
            platform="sentinel-1d",
            product_type="sentinel-1-rtc",
            source=ProductionInput.SENTINEL1,
        ),
        same_relative_orbit=True,
    )


def _provenance(
    *,
    artifact_type: ArtifactType,
    artifact_id: str,
    metadata: RtcPairMetadata,
    result: RtcChangeCandidateResult,
    summary: Mapping[str, Any],
    config_version: str,
    operations: list[str],
) -> ArtifactProvenance:
    scope = result.aoi_scope
    unobserved = scope.get("aoi_unobserved_area_km2")
    unobserved_fraction = scope.get("aoi_unobserved_fraction")
    return ArtifactProvenance(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        aoi=metadata.aoi_id,
        aoi_crs="EPSG:4326",
        event_date=metadata.event_date,
        sentinel1=_scene_pair(metadata),
        production_inputs=[ProductionInput.SENTINEL1],
        source_product_ids=[metadata.pre_scene_id, metadata.post_scene_id],
        preprocessing_operations=operations,
        quality=dict(summary),
        preprocessing_version=RTC_CHANGE_VERSION,
        config_version=config_version,
        limitations=[
            "Candidate regions are rule-based Sentinel-1 RTC gamma0 change evidence, "
            "not confirmed flood or damage.",
            "The RTC inputs are gamma0 linear power, not sigma0; no sigma0-specific "
            "threshold or learned-model equivalence is claimed.",
            "AOI provenance status is UNVERIFIED and is preserved as such; the extent is "
            f"the human-approved analytical corridor {metadata.aoi_id} pinned by SHA-256 "
            f"{metadata.aoi_source_sha256}, not a judge-selected or endorsed boundary.",
            (
                "Partial observation: the delivered Sentinel-1 RTC crop reaches only "
                f"{scope.get('aoi_delivered_area_km2')} km2 of the "
                f"{scope.get('aoi_geometric_area_km2')} km2 study extent. "
                f"{unobserved} km2 "
                f"({'unknown' if unobserved_fraction is None else f'{100 * unobserved_fraction:.4f}%'})"
                " carries no observation and therefore no finding."
            ),
            scope.get("disclosure", APPROVED_COVERAGE_DISCLOSURE),
            "Statistics are reported in two separate scopes. The delivered raster window "
            "is a bounding-box crop cut for an earlier extent and is not the AOI; window "
            "and AOI denominators must never be interchanged.",
            "No independent validation, ground truth, DEM-derived slope mask, or "
            "segmentation checkpoint was used in this stage.",
            "Surface changes other than inundation can produce radar darkening, and "
            "flooded vegetation or built-up areas may be missed.",
        ],
        notes=(
            "This artifact advances the real-pixel workflow only through a candidate "
            "change region inside the configured study extent. Infrastructure exposure "
            "and connectivity consume a validated class mask and are intentionally not "
            "run here. Candidate pixels are a strict subset of the AOI-scoped valid mask."
        ),
    )


def write_rtc_change_artifacts(
    result: RtcChangeCandidateResult,
    output_dir: Path | str,
    *,
    pair: RtcPairMetadata,
    vv_threshold_db: float,
    vh_threshold_db: float,
    config_version: str,
) -> Dict[str, Path]:
    """Write georeferenced change features, candidate mask, QA and provenance.

    The candidate and validity masks are AOI-scoped, and ``rtc_aoi_mask.tif``
    is written alongside them so that a reader can still separate "outside the
    study extent" from "inside the extent but not observed" — the candidate
    mask's single not-observed class cannot express that difference on its own.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = result.input_metadata["grid"]
    transform = Affine(*grid["transform"])
    crs = grid["crs"]
    height, width = result.shape
    summary = result.summary(
        vv_threshold_db=vv_threshold_db,
        vh_threshold_db=vh_threshold_db,
    )
    summary["pair"] = pair.to_dict()
    summary["aoi_provenance_status"] = pair.aoi_provenance_status
    # Surfaced at summary top level as well, because the dashboard AOI binding
    # check treats an identifier without its source digest as insufficient.
    summary["aoi_id"] = pair.aoi_id
    summary["aoi_source_sha256"] = pair.aoi_source_sha256
    summary["coverage_disclosure"] = result.aoi_scope.get(
        "disclosure", APPROVED_COVERAGE_DISCLOSURE
    )

    # The rule can only have been evaluated where the AOI-scoped validity mask is
    # set, so this must hold by construction. Assert it on the way out rather than
    # trusting it: a silent leak here would put candidates outside the extent.
    leaked = int(np.count_nonzero(result.candidates & ~result.valid))
    if leaked:
        raise RtcChangeError(
            f"{leaked} candidate pixels fall outside the valid mask; refusing to write "
            "a product whose candidate and validity layers describe different supports"
        )
    outside = int(np.count_nonzero(result.candidates & ~result.aoi_mask))
    if outside:
        raise RtcChangeError(
            f"{outside} candidate pixels fall outside the configured AOI; refusing to "
            "write a product that would attribute out-of-extent pixels to the study area"
        )

    features_path = output_dir / "rtc_change_features.tif"
    mask_path = output_dir / "rtc_change_candidates.tif"
    valid_path = output_dir / "rtc_valid_mask.tif"
    aoi_mask_path = output_dir / "rtc_aoi_mask.tif"
    summary_path = output_dir / "rtc_change_summary.json"
    features_provenance_path = output_dir / "rtc_change_features_provenance.yaml"
    mask_provenance_path = output_dir / "rtc_change_candidates_provenance.yaml"

    aoi_tags = {
        "aoi_id": pair.aoi_id,
        "aoi_source_sha256": str(pair.aoi_source_sha256),
        "aoi_geojson_path": str(pair.aoi_geojson_path),
    }
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(
        features_path,
        "w",
        count=2,
        dtype="float32",
        nodata=CHANGE_FEATURE_NODATA,
        predictor=3,
        **profile,
    ) as destination:
        destination.write(np.stack([result.vv_change_db, result.vh_change_db]), masked=False)
        for index, name in enumerate(CHANGE_FEATURE_NAMES, start=1):
            destination.set_band_description(index, name)
        destination.update_tags(
            artifact_stage=RTC_CHANGE_VERSION,
            radiometric_convention=RTC_CONVENTION,
            source_pair=json.dumps(pair.to_dict(), sort_keys=True),
            candidate_rule=json.dumps(summary["candidate_rule"], sort_keys=True),
            analysis_scope="configured_aoi_valid_pixels_only",
            **aoi_tags,
        )

    mask = np.where(result.valid, result.candidates.astype("uint8"), CHANGE_CANDIDATE_NODATA)
    with rasterio.open(
        mask_path,
        "w",
        count=1,
        dtype="uint8",
        nodata=CHANGE_CANDIDATE_NODATA,
        **profile,
    ) as destination:
        destination.write(mask, 1)
        destination.set_band_description(1, "change_candidate")
        destination.update_tags(
            artifact_stage=RTC_CHANGE_VERSION,
            radiometric_convention=RTC_CONVENTION,
            semantic_values="0=not_candidate,1=darkening_candidate,255=not_observed",
            not_observed_meaning=(
                "255 covers both outside-the-configured-AOI and inside-AOI-without-a-"
                "valid-observation; use rtc_aoi_mask.tif to separate the two"
            ),
            analysis_scope="configured_aoi_valid_pixels_only",
            **aoi_tags,
        )

    with rasterio.open(
        valid_path,
        "w",
        count=1,
        dtype="uint8",
        nodata=0,
        **profile,
    ) as destination:
        destination.write(result.valid.astype("uint8"), 1)
        destination.set_band_description(1, "valid_change_observation")
        destination.update_tags(
            mask_semantics=(
                "1=inside configured AOI with valid pre/post VV+VH change,"
                "0=outside AOI or not observed"
            ),
            artifact_stage=RTC_CHANGE_VERSION,
            analysis_scope="configured_aoi_valid_pixels_only",
            **aoi_tags,
        )

    with rasterio.open(
        aoi_mask_path,
        "w",
        count=1,
        dtype="uint8",
        nodata=AOI_MASK_NODATA,
        **profile,
    ) as destination:
        destination.write(result.aoi_mask.astype("uint8"), 1)
        destination.set_band_description(1, "inside_configured_aoi")
        destination.update_tags(
            artifact_stage=RTC_CHANGE_VERSION,
            mask_semantics="1=pixel centre inside configured AOI,0=outside",
            rasterization="center",
            all_touched="false",
            resampling="none",
            note=(
                "Delivery footprint of the study extent on this grid. Not a valid-pixel "
                "mask and not the whole AOI: the AOI extends beyond this raster window"
            ),
            **aoi_tags,
        )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    operations = [
        "verify_rtc_gamma0_linear_power_convention",
        "verify_pre_post_grid_alignment",
        "mask_nodata_nonfinite_and_nonpositive_power",
        "bind_configured_aoi_by_identifier_and_source_sha256",
        "rasterize_configured_aoi_by_pixel_center_without_resampling",
        "intersect_configured_aoi_into_validity_mask",
        "compute_vv_vh_log_ratio_db_from_linear_power",
        "apply_dual_polarisation_darkening_candidate_rule",
        "report_raster_window_and_aoi_scopes_with_named_denominators",
    ]
    feature_provenance = _provenance(
        artifact_type=ArtifactType.FEATURE_STACK,
        artifact_id=f"rtc-change-features-{pair.aoi_id}-{pair.event_date}",
        metadata=pair,
        result=result,
        summary=summary,
        config_version=config_version,
        operations=operations,
    )
    mask_provenance = _provenance(
        artifact_type=ArtifactType.CHANGE_CANDIDATE_MASK,
        artifact_id=f"rtc-change-candidates-{pair.aoi_id}-{pair.event_date}",
        metadata=pair,
        result=result,
        summary=summary,
        config_version=config_version,
        operations=operations,
    )
    features_provenance_path.write_text(feature_provenance.to_yaml(), encoding="utf-8")
    mask_provenance_path.write_text(mask_provenance.to_yaml(), encoding="utf-8")

    return {
        "features": features_path,
        "candidate_mask": mask_path,
        "valid_mask": valid_path,
        "aoi_mask": aoi_mask_path,
        "summary": summary_path,
        "features_provenance": features_provenance_path,
        "candidate_provenance": mask_provenance_path,
    }
