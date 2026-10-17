"""Synthetic M6.3-shaped fixtures for dashboard tests.

Everything produced here is explicitly synthetic. These fixtures exist to
exercise reader/validator/renderer behaviour; their outputs are never real
Trishuli results and the constants below say so on every artifact they write.

The layout and metadata mirror what ``scripts/rtc_change_candidates.py`` writes,
so a validation failure here corresponds to a real defect rather than to a
fixture that never matched production in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floodmap.utils.provenance import (
    ArtifactProvenance,
    ArtifactType,
    ProductionInput,
)

#: Explicit marker embedded in every fixture artifact. ASCII-only so it survives
#: YAML serialization (``yaml.safe_dump`` escapes non-ASCII by default).
SYNTHETIC_MARKER = "SYNTHETIC FIXTURE - not a real Trishuli measurement."

FIXTURE_CRS = "EPSG:32645"
FIXTURE_RESOLUTION_M = 10.0
FIXTURE_ORIGIN = (500_000.0, 3_000_000.0)
FIXTURE_SHAPE = (24, 24)

M63_STAGE = "m6.3-rtc-change-candidates-0.2.0"
RTC_CONVENTION = "gamma0_linear_power"

FEATURE_NODATA = -9999.0
CANDIDATE_NODATA = 255
AOI_NODATA = 0


def _transform():
    return from_origin(FIXTURE_ORIGIN[0], FIXTURE_ORIGIN[1], FIXTURE_RESOLUTION_M, FIXTURE_RESOLUTION_M)


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    band_names: Tuple[str, ...],
    nodata: Optional[float],
    tags: Optional[Dict[str, str]] = None,
    crs: str = FIXTURE_CRS,
) -> None:
    profile = {
        "driver": "GTiff",
        "width": int(data.shape[-1]),
        "height": int(data.shape[-2]),
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": _transform(),
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(band_names, start=1):
            dst.set_band_description(index, name)
        dst.update_tags(**{"synthetic_fixture": SYNTHETIC_MARKER, **(tags or {})})


def build_arrays(
    *,
    n_candidate: int = 20,
    n_not_observed: int = 30,
    n_invalid_valid_mask: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(candidate_mask, valid_mask, aoi_mask, features)`` for a small grid.

    The synthetic AOI excludes the trailing not-observed block, so the fixture
    exercises the real contract: ``candidate`` is a subset of ``valid``, which is
    a subset of ``aoi``, and some window pixels lie outside the extent.
    """
    height, width = FIXTURE_SHAPE
    total = height * width
    flat_candidate = np.zeros(total, dtype="uint8")
    flat_valid = np.ones(total, dtype="uint8")
    flat_aoi = np.ones(total, dtype="uint8")

    if n_not_observed:
        flat_candidate[-n_not_observed:] = CANDIDATE_NODATA
        flat_valid[-n_not_observed:] = 0
        # Outside the synthetic study extent, not merely unobserved inside it.
        flat_aoi[-n_not_observed:] = 0
    flat_candidate[:n_candidate] = 1
    if n_invalid_valid_mask:
        flat_valid[:n_invalid_valid_mask] = 0

    candidate = flat_candidate.reshape(height, width)
    valid = flat_valid.reshape(height, width)
    aoi = flat_aoi.reshape(height, width)

    features = np.full((2, height, width), FEATURE_NODATA, dtype="float32")
    observed = valid == 1
    features[:, observed] = -4.0
    features[0, candidate == 1] = -6.0
    features[1, candidate == 1] = -7.0
    return candidate, valid, aoi, features


def summarize(
    candidate: np.ndarray,
    valid: np.ndarray,
    aoi: np.ndarray,
    *,
    vv_threshold_db: float = -3.0,
    vh_threshold_db: float = -3.0,
) -> Dict[str, Any]:
    """Build a summary JSON consistent with the arrays (unless overridden)."""
    total = int(candidate.size)
    n_valid = int(np.count_nonzero(valid == 1))
    n_candidate = int(np.count_nonzero(candidate == 1))
    n_in_aoi = int(np.count_nonzero(aoi == 1))
    pixel_area = FIXTURE_RESOLUTION_M * FIXTURE_RESOLUTION_M
    # A deliberately partial synthetic extent: the fixture's "full" AOI is larger
    # than its delivered footprint, mirroring the real partial-coverage case.
    aoi_full_area_km2 = (n_in_aoi * pixel_area / 1_000_000.0) * 1.5
    aoi_delivered_area_km2 = n_in_aoi * pixel_area / 1_000_000.0
    return {
        "stage": M63_STAGE,
        "status": "candidate_evidence_only",
        "radiometric_convention": RTC_CONVENTION,
        "scope": "delivered_raster_window_array_counts",
        "total_pixels": total,
        "valid_pixels": n_valid,
        "invalid_pixels": total - n_valid,
        "valid_fraction": float(n_valid / total),
        "candidate_pixels": n_candidate,
        "candidate_fraction_of_valid": float(n_candidate / n_valid) if n_valid else 0.0,
        "candidate_fraction_of_grid": float(n_candidate / total),
        "candidate_grid_area_km2": float(n_candidate * pixel_area / 1_000_000.0),
        "pixel_size_m": [FIXTURE_RESOLUTION_M, FIXTURE_RESOLUTION_M],
        "candidate_rule": {
            "name": "dual_polarisation_darkening_candidate",
            "vv_change_db_max": vv_threshold_db,
            "vh_change_db_max": vh_threshold_db,
            "expression": (
                "valid AND vv_change_db <= vv_change_db_max AND vh_change_db <= vh_change_db_max"
            ),
        },
        "aoi_scope": {
            "aoi_geometric_area_km2": aoi_full_area_km2,
            "aoi_delivered_area_km2": aoi_delivered_area_km2,
            "aoi_unobserved_area_km2": aoi_full_area_km2 - aoi_delivered_area_km2,
            "aoi_delivered_fraction": (
                aoi_delivered_area_km2 / aoi_full_area_km2 if aoi_full_area_km2 else 0.0
            ),
            "aoi_unobserved_fraction": (
                1.0 - aoi_delivered_area_km2 / aoi_full_area_km2 if aoi_full_area_km2 else 0.0
            ),
            "aoi_pixel_centers_in_window": n_in_aoi,
            "aoi_valid_pixels": n_valid,
            "aoi_valid_area_km2": float(n_valid * pixel_area / 1_000_000.0),
            "candidate_pixels": n_candidate,
            "candidate_area_km2": float(n_candidate * pixel_area / 1_000_000.0),
            "candidate_fraction_of_aoi_valid_pixels": (
                float(n_candidate / n_valid) if n_valid else 0.0
            ),
            "window_pixels_outside_aoi": total - n_in_aoi,
            "denominator_definitions": {"synthetic": SYNTHETIC_MARKER},
            "disclosure": SYNTHETIC_MARKER,
        },
        "pair": {
            "aoi_id": "fixture-aoi-SYNTHETIC",
            "event_date": "2026-08-26",
            "aoi_provenance_status": "UNVERIFIED",
            "pre": {
                "scene_id": "FIXTURE_PRE_SYNTHETIC",
                "acquired_at": "2026-08-16T12:21:53.557589Z",
                "relative_orbit": 85,
                "orbit_direction": "ASCENDING",
            },
            "post": {
                "scene_id": "FIXTURE_POST_SYNTHETIC",
                "acquired_at": "2026-08-28T12:21:54.261291Z",
                "relative_orbit": 85,
                "orbit_direction": "ASCENDING",
            },
        },
        "aoi_provenance_status": "UNVERIFIED",
        "interpretation": SYNTHETIC_MARKER,
    }


def _provenance(artifact_type: ArtifactType, artifact_id: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        aoi="fixture-aoi-SYNTHETIC",
        aoi_crs="EPSG:4326",
        event_date="2026-08-26",
        production_inputs=[ProductionInput.SENTINEL1],
        limitations=[SYNTHETIC_MARKER],
        notes=SYNTHETIC_MARKER,
    )


def write_m63_fixture(
    directory: Path,
    *,
    summary_overrides: Optional[Dict[str, Any]] = None,
    feature_grid: Optional[Dict[str, Any]] = None,
    feature_band_names: Tuple[str, ...] = ("vv_change_db", "vh_change_db"),
    n_candidate: int = 20,
    n_not_observed: int = 30,
    n_invalid_valid_mask: int = 0,
    write_provenance: bool = True,
) -> Path:
    """Write a complete, contract-shaped M6.3 artifact directory.

    ``summary_overrides`` injects deliberate summary inconsistencies, and
    ``feature_grid`` (e.g. ``{"crs": "EPSG:4326"}``) injects a grid mismatch, so
    the validator's negative paths can be tested.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    candidate, valid, aoi, features = build_arrays(
        n_candidate=n_candidate,
        n_not_observed=n_not_observed,
        n_invalid_valid_mask=n_invalid_valid_mask,
    )

    _write_raster(
        directory / "rtc_change_candidates.tif",
        candidate[np.newaxis, ...],
        band_names=("change_candidate",),
        nodata=CANDIDATE_NODATA,
        tags={
            "artifact_stage": M63_STAGE,
            "radiometric_convention": RTC_CONVENTION,
            "semantic_values": "0=not_candidate,1=darkening_candidate,255=not_observed",
        },
    )
    _write_raster(
        directory / "rtc_valid_mask.tif",
        valid[np.newaxis, ...],
        band_names=("valid_change_observation",),
        nodata=0,
        tags={"artifact_stage": M63_STAGE},
    )
    _write_raster(
        directory / "rtc_aoi_mask.tif",
        aoi[np.newaxis, ...],
        band_names=("inside_configured_aoi",),
        nodata=AOI_NODATA,
        tags={
            "artifact_stage": M63_STAGE,
            "mask_semantics": "1=pixel centre inside configured AOI,0=outside",
        },
    )
    feature_kwargs = dict(feature_grid or {})
    _write_raster(
        directory / "rtc_change_features.tif",
        features,
        band_names=feature_band_names,
        nodata=FEATURE_NODATA,
        **feature_kwargs,
    )

    summary = summarize(candidate, valid, aoi)
    if summary_overrides:
        summary.update(summary_overrides)
    (directory / "rtc_change_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if write_provenance:
        (directory / "rtc_change_candidates_provenance.yaml").write_text(
            _provenance(ArtifactType.CHANGE_CANDIDATE_MASK, "fixture-candidates").to_yaml(),
            encoding="utf-8",
        )
        (directory / "rtc_change_features_provenance.yaml").write_text(
            _provenance(ArtifactType.FEATURE_STACK, "fixture-features").to_yaml(),
            encoding="utf-8",
        )
    return directory


def write_blocker(directory: Path, filename: str, payload: Dict[str, Any]) -> Path:
    """Write a stage blocker report, or malformed content when asked."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def valid_blocker(stage: str, *, reason: str = "prerequisite missing (SYNTHETIC)") -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "blocked",
        "reason": reason,
        "missing_layers": ["roads", "buildings", "settlements"],
        "searched_directory": "artifacts/impact/fixture-aoi-SYNTHETIC",
        "retrieval_performed": False,
        "authorisation": {
            "aoi_provenance_status": "UNVERIFIED",
            "authorises_download": False,
            "gate": "fixture gate text",
        },
        "next_step": ["Do not bypass the authorisation gate."],
        "not_done": ["No infrastructure layer was synthesised."],
    }
