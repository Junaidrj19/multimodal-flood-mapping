"""Validation gates for acquisition selection: coverage, polarisation, pairing.

These tests exist because the previous implementation could report a *usable*
pre/post pair while verifying almost nothing about it: the AOI footprint check
was a constant ``True``, Sentinel-1 polarisations were never inspected, and
tile/processing-level agreement was recorded only as prose. Each test below
pins one of the gates that replaced that behaviour, and — just as importantly —
pins the *reason* recorded when a gate rejects a candidate, because a selection
that discards candidates silently is indistinguishable from one that found
nothing.

No test reaches the network. Selection is a pure function of the candidate list,
the temporal plan and the AOI.
"""

from __future__ import annotations

import importlib.util
import sys
import typing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from floodmap.acquisition.aoi import AreaOfInterest
from floodmap.acquisition.config import load_acquisition_config
from floodmap.acquisition.coverage import (
    FootprintStatus,
    evaluate_footprint_coverage,
)
from floodmap.acquisition.errors import AcquisitionError, NoSameTrackPairError
from floodmap.acquisition.manifest import AcquisitionManifest
from floodmap.acquisition.outcomes import (
    DiscoveryStatus,
    DownloadStatus,
    DownloadSummary,
    ManifestStatus,
    RejectionReason,
    SelectionStrategy,
    summarise_downloads,
)
from floodmap.acquisition.providers.base import (
    DiscoveryResult,
    DownloadResult,
    SearchRequest,
)
from floodmap.acquisition.scenes import (
    SceneFootprint,
    Sentinel1Scene,
    Sentinel2Scene,
    SensorKind,
)
from floodmap.acquisition.selection import (
    NoSameTrackPair,
    SelectionOutcome,
    select_sentinel1_pair,
    select_sentinel2_pair,
)
from floodmap.acquisition.temporal import EventSpec, SearchWindow
from floodmap.utils.provenance import Unknown

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

AOI = AreaOfInterest.from_bbox("synthetic-validation", [0.0, 0.0, 0.1, 0.1], is_synthetic=True)
EVENT = EventSpec(
    event_date=date(2024, 1, 10),
    event_time_utc=datetime(2024, 1, 10, 12, tzinfo=timezone.utc),
)
PLAN = EVENT.plan(SearchWindow(before_days=5, after_days=5))

BEFORE = datetime(2024, 1, 9, tzinfo=timezone.utc)
AFTER = datetime(2024, 1, 11, tzinfo=timezone.utc)

#: Exactly the AOI (0,0)-(0.1,0.1).
FULL = [[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]]]
#: The eastern half of the AOI: 50% coverage.
HALF = [[[0.05, 0.0], [0.1, 0.0], [0.1, 0.1], [0.05, 0.1], [0.05, 0.0]]]
#: A sliver of the AOI: about 5% coverage.
SLIVER = [[[0.005, 0.0], [0.1, 0.0], [0.1, 0.1], [0.005, 0.1], [0.005, 0.0]]]
#: Nowhere near the AOI.
ELSEWHERE = [[[10.0, 10.0], [10.1, 10.0], [10.1, 10.1], [10.0, 10.1], [10.0, 10.0]]]


def footprint(coordinates=FULL, *, kind="Polygon", crs="EPSG:4326") -> SceneFootprint:
    return SceneFootprint(type=kind, coordinates=coordinates, crs=crs)


def s2(scene_id, acquired_at, cloud, *, tile="45RUL", level="L2A", fp=None):
    return Sentinel2Scene(
        scene_id=scene_id,
        product_id=scene_id,
        acquired_at=acquired_at,
        product_type="S2MSI2A" if level == "L2A" else "S2MSI1C",
        platform_short_name="SENTINEL-2",
        platform_unit="S2A",
        provider="cdse-odata",
        scene_cloud_percent=cloud,
        processing_level=level,
        tile_id=tile,
        footprint=footprint() if fp is None else fp,
    )


def s1(scene_id, acquired_at, *, orbit=85, direction="ASCENDING", polarisations=("VV", "VH"), fp=None):
    return Sentinel1Scene(
        scene_id=scene_id,
        product_id=scene_id,
        acquired_at=acquired_at,
        product_type="IW_GRDH_1S",
        platform_short_name="SENTINEL-1",
        platform_unit="S1D",
        provider="cdse-odata",
        relative_orbit=orbit,
        orbit_direction=direction,
        polarisations=polarisations,
        footprint=footprint() if fp is None else fp,
    )


def _load_acquire_module():
    """Import ``scripts/acquire.py`` as a module without installing it."""
    path = REPOSITORY_ROOT / "scripts" / "acquire.py"
    spec = importlib.util.spec_from_file_location("acquire_cli", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["acquire_cli"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Coverage geometry: the measure itself
# ---------------------------------------------------------------------------


def test_full_coverage_is_measured_and_named():
    verdict = evaluate_footprint_coverage(footprint(), AOI, minimum_coverage=0.9)
    assert verdict.status is FootprintStatus.FULL
    assert verdict.accepted is True
    assert verdict.coverage_fraction == pytest.approx(1.0, abs=1e-9)
    assert verdict.reason is None


def test_partial_coverage_is_distinguished_from_full():
    verdict = evaluate_footprint_coverage(footprint(HALF), AOI, minimum_coverage=0.4)
    assert verdict.status is FootprintStatus.PARTIAL
    assert verdict.coverage_fraction == pytest.approx(0.5, abs=1e-6)
    assert verdict.accepted is True


def test_partial_coverage_below_the_threshold_is_rejected_with_the_measured_fraction():
    verdict = evaluate_footprint_coverage(footprint(HALF), AOI, minimum_coverage=0.9)
    assert verdict.status is FootprintStatus.PARTIAL
    assert verdict.accepted is False
    assert verdict.coverage_fraction == pytest.approx(0.5, abs=1e-6)
    assert verdict.reason is RejectionReason.FOOTPRINT_COVERAGE_BELOW_MINIMUM


def test_non_intersecting_footprint_is_rejected():
    verdict = evaluate_footprint_coverage(footprint(ELSEWHERE), AOI, minimum_coverage=None)
    assert verdict.status is FootprintStatus.NONE
    assert verdict.accepted is False
    assert verdict.coverage_fraction == 0.0
    assert verdict.reason is RejectionReason.FOOTPRINT_DOES_NOT_INTERSECT_AOI


def test_no_threshold_requires_intersection_without_inventing_a_number():
    """A sliver passes a threshold-free gate; the fraction is still recorded."""
    verdict = evaluate_footprint_coverage(footprint(SLIVER), AOI, minimum_coverage=None)
    assert verdict.accepted is True
    assert verdict.status is FootprintStatus.PARTIAL
    assert verdict.coverage_fraction == pytest.approx(0.95, abs=0.01)
    assert verdict.minimum_fraction is None


@pytest.mark.parametrize(
    ("label", "parts", "expected", "expected_fraction"),
    [
        (
            "parts_touching_along_an_edge",
            [
                [[0.0, 0.0], [0.05, 0.0], [0.05, 0.1], [0.0, 0.1], [0.0, 0.0]],
                [[0.05, 0.0], [0.1, 0.0], [0.1, 0.1], [0.05, 0.1], [0.05, 0.0]],
            ],
            FootprintStatus.FULL,
            1.0,
        ),
        (
            "parts_disjoint",
            [
                [[0.0, 0.0], [0.04, 0.0], [0.04, 0.1], [0.0, 0.1], [0.0, 0.0]],
                [[0.06, 0.0], [0.1, 0.0], [0.1, 0.1], [0.06, 0.1], [0.06, 0.0]],
            ],
            FootprintStatus.PARTIAL,
            0.8,
        ),
    ],
)
def test_multipolygon_footprint_is_supported(label, parts, expected, expected_fraction):
    """A multi-part footprint is measured as the union of its parts.

    The touching case matters: GEOS calls a MultiPolygon whose parts meet along
    an edge invalid, although its meaning is unambiguous, and catalogues emit
    footprints split that way. Each part is still validity-checked separately.
    """
    multi = [[part] for part in parts]  # one polygon per part
    verdict = evaluate_footprint_coverage(footprint(multi, kind="MultiPolygon"), AOI)
    assert verdict.accepted is True, label
    assert verdict.status is expected, label
    assert verdict.coverage_fraction == pytest.approx(expected_fraction, abs=0.01), label


def test_multipolygon_with_one_invalid_part_fails_closed():
    parts = [
        [[0.0, 0.0], [0.05, 0.0], [0.05, 0.1], [0.0, 0.1], [0.0, 0.0]],
        [[0.05, 0.0], [0.1, 0.1], [0.1, 0.0], [0.05, 0.1], [0.05, 0.0]],  # bowtie
    ]
    verdict = evaluate_footprint_coverage(
        footprint([[part] for part in parts], kind="MultiPolygon"), AOI
    )
    assert verdict.accepted is False
    assert verdict.status is FootprintStatus.UNVERIFIABLE


@pytest.mark.parametrize(
    ("label", "bad_footprint"),
    [
        ("missing", None),
        ("unknown", Unknown(reason="not_available_from_source")),
        ("empty_coordinates", SceneFootprint(type="Polygon", coordinates=[])),
        ("null_coordinates", SceneFootprint(type="Polygon", coordinates=None)),
        (
            "malformed_nesting",
            SceneFootprint(type="Polygon", coordinates=[[[0.0, 0.0], [0.1, 0.1]]]),
        ),
        (
            "zero_area",
            SceneFootprint(
                type="Polygon",
                coordinates=[[[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [0.0, 0.0]]],
            ),
        ),
        (
            "self_intersecting",
            SceneFootprint(
                type="Polygon",
                coordinates=[[[0.0, 0.0], [0.1, 0.1], [0.1, 0.0], [0.0, 0.1], [0.0, 0.0]]],
            ),
        ),
        ("unsupported_crs", footprint(FULL, crs="EPSG:32645")),
    ],
)
def test_unusable_geometry_fails_closed(label, bad_footprint):
    """Every unverifiable footprint is rejected, never assumed to cover the AOI.

    This is the direction that matters: the previous implementation returned
    ``True`` for all of these.
    """
    verdict = evaluate_footprint_coverage(bad_footprint, AOI, minimum_coverage=None)
    assert verdict.accepted is False, label
    assert verdict.status is FootprintStatus.UNVERIFIABLE, label
    assert verdict.reason is RejectionReason.FOOTPRINT_UNVERIFIABLE, label
    assert verdict.coverage_fraction is None, label
    assert verdict.detail, label


def test_degenerate_aoi_raises_instead_of_rejecting_every_candidate():
    """A zero-area AOI is a caller error, not a data-availability finding."""
    collinear = AreaOfInterest.from_geojson_geometry(
        "collinear",
        {
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [0.0, 0.0]]],
        },
    )
    with pytest.raises(ValueError, match="zero area"):
        evaluate_footprint_coverage(footprint(), collinear)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_impossible_coverage_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError, match="minimum_coverage"):
        evaluate_footprint_coverage(footprint(), AOI, minimum_coverage=threshold)


def test_coverage_is_latitude_corrected():
    """The ratio must not be a ratio of degree-squared areas.

    Two AOIs of identical degree extent but different latitude: a half-covering
    footprint of the same degree shape must measure 0.5 in both, while the raw
    degree-area ratio would drift. That drift is the reason the measure is taken
    in a locally scaled plane.
    """
    low = AreaOfInterest.from_bbox("low", [0.0, 0.0, 0.1, 0.1])
    high = AreaOfInterest.from_bbox("high", [0.0, 60.0, 0.1, 60.1])
    half_low = footprint([[[0.05, 0.0], [0.1, 0.0], [0.1, 0.1], [0.05, 0.1], [0.05, 0.0]]])
    half_high = footprint([[[0.05, 60.0], [0.1, 60.0], [0.1, 60.1], [0.05, 60.1], [0.05, 60.0]]])
    assert evaluate_footprint_coverage(
        half_low, low, minimum_coverage=None
    ).coverage_fraction == pytest.approx(0.5, abs=1e-6)
    assert evaluate_footprint_coverage(
        half_high, high, minimum_coverage=None
    ).coverage_fraction == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Sentinel-2: per-candidate and pair-level gates
# ---------------------------------------------------------------------------


def test_s2_non_covering_candidate_is_rejected_and_recorded():
    result = select_sentinel2_pair(
        [
            s2("pre-far", BEFORE, 1.0, fp=footprint(ELSEWHERE)),
            s2("pre-ok", BEFORE, 30.0),
            s2("post", AFTER, 20.0),
        ],
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
        minimum_aoi_coverage=0.9,
    )
    assert result.selected_before.scene_id == "pre-ok"
    rejection = next(r for r in result.rejected_candidates if r.scene_id == "pre-far")
    assert rejection.reason is RejectionReason.FOOTPRINT_DOES_NOT_INTERSECT_AOI
    assert rejection.coverage_fraction == 0.0


def test_s2_coverage_below_threshold_is_rejected():
    result = select_sentinel2_pair(
        [s2("pre", BEFORE, 5.0, fp=footprint(HALF)), s2("post", AFTER, 5.0)],
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
        minimum_aoi_coverage=0.9,
    )
    assert result.status == "no_usable_scene"
    assert result.is_usable is False
    assert RejectionReason.FOOTPRINT_COVERAGE_BELOW_MINIMUM in {
        r.reason for r in result.rejected_candidates
    }


def test_s2_processing_level_filter_now_records_a_reason():
    """Disabling fallback must not look like an empty catalogue."""
    result = select_sentinel2_pair(
        [s2("pre-l1c", BEFORE, 5.0, level="L1C"), s2("post-l1c", AFTER, 5.0, level="L1C")],
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
        processing_level_preference="L2A",
        allow_processing_level_fallback=False,
    )
    assert result.status == "no_usable_scene"
    assert RejectionReason.PROCESSING_LEVEL_EXCLUDED in {
        r.reason for r in result.rejected_candidates
    }


def test_s2_pair_ranking_beats_picking_the_best_scene_per_side():
    """A pair is limited by its worse side, so the pair is ranked as a pair."""
    result = select_sentinel2_pair(
        [
            s2("pre-clear", BEFORE, 1.0),
            s2("pre-cloudy", BEFORE - timedelta(days=1), 90.0),
            s2("post-cloudy", AFTER, 95.0),
            s2("post-clear", AFTER + timedelta(days=1), 2.0),
        ],
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
    )
    # Best individual scenes would be pre-clear + post-clear; that IS the best
    # pair here, so the assertion is that the pairing does not invert it.
    assert (result.selected_before.scene_id, result.selected_after.scene_id) == (
        "pre-clear",
        "post-clear",
    )
    # And the runner-ups are recorded rather than dropped.
    assert {"pre-cloudy", "post-cloudy"} <= {r.scene_id for r in result.rejected_candidates}


def test_s2_mixed_processing_levels_are_disclosed_not_silent():
    scenes = [
        s2("pre-l2a", BEFORE, 5.0, level="L2A"),
        s2("post-l1c", AFTER, 5.0, level="L1C"),
    ]
    mixed = select_sentinel2_pair(
        scenes, PLAN, aoi=AOI, strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST
    )
    assert mixed.is_usable is True
    assert mixed.pair_validation["mixed_processing_levels"] is True
    assert "MIXED" in mixed.rationale

    refused = select_sentinel2_pair(
        scenes,
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
        allow_mixed_processing_levels=False,
    )
    # Both sides had eligible candidates, so this is a pair-compatibility
    # failure rather than "no usable scene" — a distinction an operator needs.
    assert refused.status == "no_compatible_pair"
    assert refused.is_usable is False
    assert RejectionReason.MIXED_PROCESSING_LEVELS_NOT_ALLOWED in {
        r.reason for r in refused.rejected_candidates
    }


def test_s2_unverifiable_tile_pairing_is_rejected():
    result = select_sentinel2_pair(
        [
            s2("pre", BEFORE, 5.0),
            s2("post", AFTER, 5.0, tile=Unknown(reason="not_available_from_source")),
        ],
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
    )
    assert result.is_usable is False
    assert RejectionReason.TILE_UNVERIFIABLE in {r.reason for r in result.rejected_candidates}


def test_s2_records_the_selection_policy_for_reproduction():
    result = select_sentinel2_pair(
        [s2("pre", BEFORE, 5.0), s2("post", AFTER, 6.0)],
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
        minimum_aoi_coverage=0.9,
        require_same_tile=True,
    )
    policy = result.selection_policy
    assert policy is not None
    assert policy["aoi_id"] == AOI.aoi_id
    assert policy["aoi_crs"] == "EPSG:4326"
    assert policy["minimum_aoi_coverage"] == 0.9
    assert policy["require_same_tile"] is True
    # The window that admitted the pair is part of the record, not a repo value.
    assert policy["window_before_days"] == 5
    assert policy["window_after_days"] == 5
    assert policy["event_date"] == "2024-01-10"
    assert result.coverage["pre"]["coverage_fraction"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Sentinel-1: coverage and polarisation
# ---------------------------------------------------------------------------


def test_s1_non_covering_pair_is_rejected():
    result = select_sentinel1_pair(
        [s1("pre", BEFORE, fp=footprint(ELSEWHERE)), s1("post", AFTER)],
        PLAN,
        aoi=AOI,
        minimum_aoi_coverage=0.9,
    )
    assert result.is_usable is False
    assert RejectionReason.FOOTPRINT_DOES_NOT_INTERSECT_AOI in {
        r.reason for r in result.rejected_candidates
    }


@pytest.mark.parametrize(
    ("polarisations", "expected"),
    [
        (("VV", "VH"), None),
        (("vv", "vh"), None),
        (("VV",), RejectionReason.POLARISATION_MISMATCH),
        ((), RejectionReason.POLARISATION_UNVERIFIABLE),
        (Unknown(reason="not_available_from_source"), RejectionReason.POLARISATION_UNVERIFIABLE),
    ],
)
def test_s1_required_polarisations_are_enforced(polarisations, expected):
    result = select_sentinel1_pair(
        [
            s1("pre", BEFORE, polarisations=polarisations),
            s1("post", AFTER, polarisations=polarisations),
        ],
        PLAN,
        aoi=AOI,
        required_polarisations=["VV", "VH"],
    )
    if expected is None:
        assert result.is_usable is True
        assert result.pair_validation["gates"]["required_polarisations"] == "passed"
        return
    assert result.is_usable is False
    assert expected in {r.reason for r in result.rejected_candidates}


def test_s1_pair_requires_a_covering_footprint_on_both_sides():
    result = select_sentinel1_pair(
        [s1("pre", BEFORE), s1("post", AFTER, fp=footprint(HALF))],
        PLAN,
        aoi=AOI,
        minimum_aoi_coverage=0.9,
    )
    assert result.is_usable is False
    rejection = next(r for r in result.rejected_candidates if r.scene_id == "post")
    assert rejection.reason is RejectionReason.FOOTPRINT_COVERAGE_BELOW_MINIMUM
    assert rejection.coverage_fraction == pytest.approx(0.5, abs=1e-6)


def test_s1_same_track_failure_is_still_explicit():
    result = select_sentinel1_pair(
        [s1("pre", BEFORE, orbit=85), s1("post", AFTER, orbit=41)],
        PLAN,
        aoi=AOI,
    )
    assert isinstance(result, NoSameTrackPair)
    assert result.is_usable is False


# ---------------------------------------------------------------------------
# is_usable consistency
# ---------------------------------------------------------------------------


def test_is_usable_cannot_be_true_without_a_passed_validation_record():
    """The flag reads the recorded validation, not the mere presence of scenes."""
    pre, post = s1("pre", BEFORE), s1("post", AFTER)
    forged = SelectionOutcome(
        sensor=SensorKind.SENTINEL1,
        status="selected",
        selected_before=pre,
        selected_after=post,
    )
    assert forged.pair is None
    assert forged.selected_pair is None
    assert forged.is_usable is False

    stamped = SelectionOutcome(
        sensor=SensorKind.SENTINEL1,
        status="selected",
        selected_before=pre,
        selected_after=post,
        pair_validation={"passed": True},
    )
    assert stamped.is_usable is True


def test_rejection_records_name_the_rejected_candidate_and_side():
    result = select_sentinel2_pair(
        [
            s2("pre-far", BEFORE, 1.0, fp=footprint(ELSEWHERE)),
            s2("pre-ok", BEFORE, 30.0),
            s2("post", AFTER, 20.0),
        ],
        PLAN,
        aoi=AOI,
        strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST,
    )
    for rejection in result.rejected_candidates:
        assert rejection.scene_id
        assert rejection.side in {"before", "after"}
        assert rejection.rationale
        assert rejection.sensor is SensorKind.SENTINEL2


def test_selection_outcome_type_hints_resolve():
    """``Any`` must be importable where it is annotated (M6.1 defect fix)."""
    hints = typing.get_type_hints(SelectionOutcome)
    assert "tile_consistency" in hints
    assert "coverage" in hints
    assert "pair_validation" in hints


# ---------------------------------------------------------------------------
# CLI policy enforcement and manifest status
# ---------------------------------------------------------------------------


def test_on_no_same_track_pair_fail_actually_fails():
    """The configured policy was validated but never read; now it is enforced."""
    acquire = _load_acquire_module()
    config, _ = load_acquisition_config()
    assert config.sentinel1.on_no_same_track_pair == "fail"
    outcome = NoSameTrackPair(sensor=SensorKind.SENTINEL1, rationale="no same-track pair")
    with pytest.raises(NoSameTrackPairError):
        acquire._enforce_no_pair_policy("sentinel-1", outcome, config)


def test_on_no_same_track_pair_record_policy_does_not_raise():
    acquire = _load_acquire_module()
    config, _ = load_acquisition_config()
    recorded = config.model_copy(
        update={
            "sentinel1": config.sentinel1.model_copy(
                update={"on_no_same_track_pair": "record"}
            )
        }
    )
    outcome = NoSameTrackPair(sensor=SensorKind.SENTINEL1, rationale="no same-track pair")
    assert acquire._enforce_no_pair_policy("sentinel-1", outcome, recorded) is None


def _download(status, scene_id="s"):
    if status is DownloadStatus.SUCCESS:
        return DownloadResult(status=status, scene_id=scene_id, destination=Path("/tmp/x.zip"))
    if status is DownloadStatus.FAILED:
        return DownloadResult(status=status, scene_id=scene_id, error="synthetic failure")
    return DownloadResult(
        status=status, scene_id=scene_id, reason_not_attempted="downloads not enabled"
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((), DownloadSummary.NOT_ATTEMPTED),
        ((DownloadStatus.NOT_ATTEMPTED,), DownloadSummary.NOT_ATTEMPTED),
        ((DownloadStatus.SUCCESS, DownloadStatus.SUCCESS), DownloadSummary.COMPLETE),
        ((DownloadStatus.FAILED, DownloadStatus.FAILED), DownloadSummary.FAILED),
        ((DownloadStatus.SUCCESS, DownloadStatus.FAILED), DownloadSummary.PARTIAL),
        ((DownloadStatus.FAILED, DownloadStatus.NOT_ATTEMPTED), DownloadSummary.PARTIAL),
    ],
)
def test_download_summary_is_truthful(statuses, expected):
    assert summarise_downloads([_download(s) for s in statuses]) is expected


def _manifest_with(downloads):
    request = SearchRequest(aoi=AOI, sensor=SensorKind.SENTINEL1, interval=PLAN.full_extent)
    discovery = DiscoveryResult(
        status=DiscoveryStatus.SUCCESS,
        request=request,
        scenes=(s1("pre", BEFORE), s1("post", AFTER)),
        provider="cdse-odata",
    )
    selection = select_sentinel1_pair(
        [s1("pre", BEFORE), s1("post", AFTER)], PLAN, aoi=AOI
    )
    return AcquisitionManifest.from_run(
        aoi=AOI,
        event=EVENT,
        temporal_plans={"sentinel-1": PLAN},
        requests={"sentinel-1": request},
        discoveries={"sentinel-1": discovery},
        selections={"sentinel-1": selection},
        downloads=downloads,
        provider="cdse-odata",
    )


def test_failed_download_does_not_make_the_manifest_claim_completeness():
    manifest = _manifest_with([_download(DownloadStatus.FAILED)])
    # Selection succeeded, so the selection status is COMPLETE ...
    assert manifest.status is ManifestStatus.COMPLETE
    # ... but the download status must say so independently.
    assert manifest.download_status is DownloadSummary.FAILED
    serialised = manifest.to_dict()
    assert serialised["status"] == "complete"
    assert serialised["download_status"] == "download_failed"
    assert "download_status" in manifest.to_artifact_provenance().notes


def test_pre_download_manifest_is_traceable_and_not_reported_as_complete():
    pending = AcquisitionManifest.from_run(
        aoi=AOI,
        event=EVENT,
        temporal_plans={"sentinel-1": PLAN},
        requests={},
        discoveries={},
        selections={},
        downloads=[],
        download_pending=True,
    )
    assert pending.status is ManifestStatus.SELECTION_COMPLETE_DOWNLOAD_PENDING
    assert pending.status is not ManifestStatus.COMPLETE
    assert pending.download_status is DownloadSummary.NOT_ATTEMPTED


def test_cli_dry_run_writes_a_manifest_without_network_io(tmp_path):
    """The CLI surface is exercised end to end, with no provider call."""
    acquire = _load_acquire_module()
    target = tmp_path / "manifest.json"
    exit_code = acquire.main(
        [
            "--dry-run",
            "--aoi-id",
            "synthetic-validation",
            "--aoi-bbox",
            "0.0",
            "0.0",
            "0.1",
            "0.1",
            "--before-days",
            "15",
            "--after-days",
            "15",
            "--manifest",
            str(target),
        ]
    )
    assert exit_code == 0
    manifest = AcquisitionManifest.from_json(target.read_text(encoding="utf-8"))
    assert manifest.status is ManifestStatus.DRY_RUN
    assert manifest.download_status is DownloadSummary.NOT_ATTEMPTED
    # The search window actually used is recorded, so the run is reproducible:
    # the config ships these as null, so without the manifest value the run that
    # admitted this pair could not be reconstructed.
    assert manifest.temporal_plans["sentinel-1"]["window"]["before_days"] == 15
    assert manifest.temporal_plans["sentinel-1"]["window"]["after_days"] == 15
    # No --event-time-utc was supplied, so the whole event day is excluded from
    # both windows rather than guessed into one of them (AGENTS.md §4).
    assert manifest.temporal_plans["sentinel-1"]["event_day_excluded"] is True
    # The AOI actually queried is recorded with its CRS and WKT, not just an id.
    assert manifest.requests["sentinel-1"]["aoi_crs"] == "EPSG:4326"
    assert "POLYGON" in manifest.requests["sentinel-1"]["aoi_wkt"]
    assert manifest.aoi["is_synthetic"] is True


# ---------------------------------------------------------------------------
# AOI GeoJSON container handling
#
# The only candidate corridor artifact this project has seen is a
# single-feature ``FeatureCollection`` -- the shape ``ogr2ogr`` emits by
# default (docs/dataset-registry.md §1.6). The CLI previously passed such a
# document straight through to ``from_geojson_geometry``, which rejected it as
# "only GeoJSON Polygon is supported, got 'FeatureCollection'". The AOI could
# therefore not be supplied in the one format it actually existed in, so these
# tests pin the container handling and, equally, pin what is still refused.
# ---------------------------------------------------------------------------


def _polygon_geometry():
    """A small closed square; coordinates are synthetic, not a study area."""
    return {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2], [0.0, 0.0]]],
    }


def test_single_feature_featurecollection_geometry_is_unwrapped():
    """The ogr2ogr default container is accepted."""
    acquire = _load_acquire_module()
    document = {
        "type": "FeatureCollection",
        "name": "synthetic-collection",
        "features": [{"type": "Feature", "properties": {}, "geometry": _polygon_geometry()}],
    }
    assert acquire._geometry_from_geojson(document) == _polygon_geometry()


def test_bare_geometry_and_feature_containers_still_work():
    """The previously supported containers are unchanged."""
    acquire = _load_acquire_module()
    assert acquire._geometry_from_geojson(_polygon_geometry()) == _polygon_geometry()
    feature = {"type": "Feature", "properties": {}, "geometry": _polygon_geometry()}
    assert acquire._geometry_from_geojson(feature) == _polygon_geometry()


def test_multi_feature_featurecollection_is_refused_not_reduced():
    """Picking one of several features would invent an analysis extent.

    The first feature is *not* adopted and the features are *not* unioned:
    either would silently choose a study area nobody specified.
    """
    acquire = _load_acquire_module()
    document = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": _polygon_geometry()},
            {"type": "Feature", "properties": {}, "geometry": _polygon_geometry()},
        ],
    }
    with pytest.raises(ValueError, match="contains 2 features"):
        acquire._geometry_from_geojson(document)


def test_empty_featurecollection_is_refused():
    acquire = _load_acquire_module()
    with pytest.raises(ValueError, match="empty"):
        acquire._geometry_from_geojson({"type": "FeatureCollection", "features": []})


def test_featurecollection_without_features_array_is_refused():
    acquire = _load_acquire_module()
    with pytest.raises(ValueError, match="no 'features' array"):
        acquire._geometry_from_geojson({"type": "FeatureCollection"})


def test_feature_without_geometry_is_refused():
    acquire = _load_acquire_module()
    with pytest.raises(ValueError, match="no usable 'geometry' member"):
        acquire._geometry_from_geojson({"type": "Feature", "properties": {}})


def test_non_object_geojson_is_refused():
    acquire = _load_acquire_module()
    with pytest.raises(ValueError, match="must be a JSON object"):
        acquire._geometry_from_geojson([_polygon_geometry()])


def test_multipolygon_is_still_refused_through_the_cli_path():
    """Unwrapping the container must not weaken the MultiPolygon rule."""
    acquire = _load_acquire_module()
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [_polygon_geometry()["coordinates"]],
                },
            }
        ],
    }
    # The container is unwrapped, and the geometry type gate then refuses it.
    geometry = acquire._geometry_from_geojson(document)
    assert geometry["type"] == "MultiPolygon"
    with pytest.raises(ValueError, match="only GeoJSON Polygon is supported"):
        AreaOfInterest.from_geojson_geometry("synthetic-multipolygon", geometry)


def test_cli_dry_run_accepts_a_featurecollection_aoi_file(tmp_path):
    """End to end: a single-feature FeatureCollection reaches the manifest.

    The geometry is synthetic. This asserts the *container* is ingestible and
    that polygon vertices survive into the AOI provenance record; it asserts
    nothing about any real study area.
    """
    acquire = _load_acquire_module()
    import json as _json

    aoi_file = tmp_path / "aoi.geojson"
    aoi_file.write_text(
        _json.dumps(
            {
                "type": "FeatureCollection",
                "name": "synthetic-collection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": _polygon_geometry()}
                ],
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "manifest.json"
    exit_code = acquire.main(
        [
            "--dry-run",
            "--aoi-id",
            "synthetic-featurecollection",
            "--aoi-geojson",
            str(aoi_file),
            "--before-days",
            "12",
            "--after-days",
            "12",
            "--manifest",
            str(target),
        ]
    )
    assert exit_code == 0
    manifest = AcquisitionManifest.from_json(target.read_text(encoding="utf-8"))
    assert manifest.status is ManifestStatus.DRY_RUN
    assert manifest.download_status is DownloadSummary.NOT_ATTEMPTED
    # The polygon ring survived into the recorded AOI, rather than being
    # flattened to its bounding box.
    assert len(manifest.aoi["geometry"]["exterior"]) == 5
    assert manifest.aoi["is_synthetic"] is True


# ---------------------------------------------------------------------------
# The configured human-approved analytical extent (geometry approval only)
#
# M6.15 adopts the explicitly approved M6.14 HydroRIVERS corridor. It remains
# **UNVERIFIED** for acquisition; geometry approval is not download permission.
# These tests pin the properties that keep that safe: the bytes
# are hash-pinned (a swapped extent cannot load silently) and an unverified
# extent cannot authorise a product download.
# ---------------------------------------------------------------------------


def test_configured_aoi_bytes_are_pinned_and_a_swap_refuses_to_load(tmp_path):
    """A different file under the configured path must not load silently.

    The study extent is the one input every downstream number inherits, so a
    hash mismatch has to be an error rather than a quietly different analysis.
    """
    config, _ = load_acquisition_config()
    assert config.aoi is not None
    # The pin matched, so the loaded geometry carries the digest actually read.
    assert config.aoi.source_sha256 == (
        "2ae4183f0d44f80200a889ed401ba25525c52134ba1f31d65e9f2ebc3d8965e3"
    )
    from floodmap.acquisition.aoi import AreaOfInterest as _AoI

    swapped = tmp_path / "other.geojson"
    swapped.write_text(
        '{"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="was pinned"):
        _AoI.from_geojson_file(
            "swapped", swapped, expected_sha256=config.aoi.source_sha256
        )


def test_configured_corridor_geometry_is_valid_and_matches_the_recorded_artifact():
    """The pinned artifact is checked as geometry, not only as bytes.

    A digest proves the file is the one recorded; it says nothing about whether
the geometry is usable. These are the properties `docs/dataset-registry.md`
§1.9 states for the approved corridor, re-derived here rather than restated, so
the record and the artifact cannot drift apart unnoticed.
    """
    import numpy as np
    import shapely
    from pyproj import Transformer
    from shapely.geometry import Polygon

    config, _ = load_acquisition_config()
    aoi = config.aoi
    assert aoi is not None
    ring = list(aoi.geometry.exterior)
    polygon = Polygon(ring, holes=[list(h) for h in aoi.geometry.holes])
    assert polygon.is_valid and not polygon.is_empty and polygon.is_simple
    assert ring[0] == ring[-1], "the ring must be closed"
    # The exact reach-end bounded construction records 2,164 coordinate pairs.
    assert len(ring) == 2_164, f"2,164 coordinate pairs, got {len(ring)}"
    assert not polygon.exterior.is_ccw

    # RFC 7946 lon/lat delivery; buffering occurred in EPSG:32645.
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    assert (min_lon, min_lat, max_lon, max_lat) == pytest.approx(
        (84.7812496, 27.7888937, 85.3872770, 28.2837609), abs=1e-7
    )

    # M6.15 measures 206.1506509 km² in EPSG:32645. The
    # tolerance is relative and generous: this detects a swapped or truncated
    # geometry, it is not a survey.
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

    def _to_utm(coordinates):
        lon, lat = transformer.transform(coordinates[:, 0], coordinates[:, 1])
        return np.column_stack([lon, lat])

    area_km2 = shapely.transform(polygon, _to_utm).area / 1e6
    assert area_km2 == pytest.approx(206.1506509, rel=1e-8)


def test_envelope_aoi_admits_a_product_that_leaves_the_corridor_partly_outside():
    """The bounding box is not a substitute for the corridor, demonstrated.

    Pins the worst case found by the construction recorded in
    `docs/dataset-registry.md` §1.6.2: a 1.0° square rotated 20° about
    `84.958°E, 27.744°N` — the Sentinel-2 tile scale, in a realistic
    orientation, since the corridor runs diagonally while tiles are not
    north-aligned. Scored at the configured `minimum_aoi_coverage: 0.9` it is
    **accepted against the envelope and rejected against the corridor**, which
    is the failure mode that makes the envelope unacceptable rather than merely
    imprecise: ~23% of the real corridor would fall outside the image.

    This does not call the network and does not assert anything about any real
    product; it asserts that the two AOIs are not interchangeable under the
    project's own coverage measure.
    """
    from shapely import affinity
    from shapely.geometry import Polygon

    # This is the historical, numerically pinned M6.2 counterexample, not a
    # claim about the adopted M6.15 geometry. Keep its source geometry explicit.
    from floodmap.utils.config import project_root
    corridor = AreaOfInterest.from_geojson_file(
        "historical-envelope-counterexample",
        project_root() / "configs/aoi/trishuli-corridor-UNVERIFIED.geojson",
        expected_sha256="9035eeee025f311be7f16af1d23737a46caee20eb0b2a8dfa144d111b9362610",
    )
    envelope = AreaOfInterest.from_bbox(
        "envelope-demo", [85.05814, 27.84397, 85.38789, 28.28743], is_synthetic=True
    )
    square = Polygon(
        [
            (84.958 - 0.5, 27.744 - 0.5),
            (84.958 + 0.5, 27.744 - 0.5),
            (84.958 + 0.5, 27.744 + 0.5),
            (84.958 - 0.5, 27.744 + 0.5),
        ]
    )
    footprint = SceneFootprint(
        type="Polygon",
        coordinates=[list(affinity.rotate(square, 20, origin="center").exterior.coords)],
    )

    against_envelope = evaluate_footprint_coverage(footprint, envelope, minimum_coverage=0.9)
    against_corridor = evaluate_footprint_coverage(footprint, corridor, minimum_coverage=0.9)

    assert against_envelope.accepted is True
    assert against_envelope.coverage_fraction == pytest.approx(0.9144, abs=5e-4)
    assert against_corridor.accepted is False
    assert against_corridor.coverage_fraction == pytest.approx(0.7712, abs=5e-4)
    assert against_corridor.reason is RejectionReason.FOOTPRINT_COVERAGE_BELOW_MINIMUM
    # The dangerous direction: a stricter AOI would have REFUSED this product.
    assert against_corridor.coverage_fraction < against_envelope.coverage_fraction


def test_query_envelope_is_a_strict_superset_and_is_labelled_as_a_prefilter():
    """The pre-filter may only ever widen the candidate set.

    Detailed corridor polygons can exceed catalogue URL limits (measured for
    the historical M6.2 geometry), so the envelope can be sent instead. That
    is only safe if it cannot exclude a candidate the AOI would have matched, and
    if nothing can mistake it for the study extent.
    """
    from shapely.geometry import Polygon

    config, _ = load_acquisition_config()
    aoi = config.aoi
    assert aoi is not None
    envelope = aoi.query_envelope()

    assert envelope.aoi_id == f"{aoi.aoi_id}-query-envelope-prefilter"
    assert envelope.aoi_id != aoi.aoi_id
    assert envelope.bounds().as_list() == aoi.bounds().as_list()
    assert len(envelope.geometry.to_ring()) == 5  # a closed rectangle, not the corridor
    # The superset property itself, not just the label.
    polygon = Polygon(
        list(aoi.geometry.exterior), holes=[list(h) for h in aoi.geometry.holes]
    )
    assert Polygon(envelope.geometry.to_ring()).contains(polygon)
    # It is a query geometry, so it never authorises a download on its own.
    assert envelope.authorises_download is False
    assert envelope.provenance_gaps


def test_cli_envelope_prefilter_records_the_substitution_and_keeps_the_aoi(tmp_path):
    """End to end: query geometry and coverage geometry are both recorded."""
    acquire = _load_acquire_module()

    target = tmp_path / "prefilter.json"
    exit_code = acquire.main(
        [
            "--dry-run",
            "--query-envelope-prefilter",
            "--before-days",
            "12",
            "--after-days",
            "12",
            "--manifest",
            str(target),
        ]
    )
    assert exit_code == 0
    manifest = AcquisitionManifest.from_json(target.read_text(encoding="utf-8"))

    # The query geometry is the 5-vertex envelope...
    query_wkt = manifest.requests["sentinel-1"]["aoi_wkt"]
    assert query_wkt.count(",") == 4  # five coordinates
    assert manifest.requests["sentinel-1"]["aoi_id"].endswith("-query-envelope-prefilter")
    # ...while the manifest's AOI is still the full corridor polygon.
    assert len(manifest.aoi["geometry"]["exterior"]) == 2_164
    assert manifest.aoi["aoi_id"] == "trishuli-corridor-hydrorivers-v10-m615-2ae4183f"

    prefilter = manifest.query_prefilter
    assert prefilter is not None
    assert prefilter["superset_of_aoi"] is True
    assert prefilter["coverage_measured_against"] == manifest.aoi["aoi_id"]
    assert "URL" in prefilter["reason"]
    assert any("bounding envelope as a spatial pre-filter" in line for line in manifest.limitations)


def test_cli_without_the_flag_queries_the_aoi_polygon_itself(tmp_path):
    """The default is unchanged: no substitution unless it is asked for."""
    acquire = _load_acquire_module()
    target = tmp_path / "no-prefilter.json"
    assert (
        acquire.main(
            [
                "--dry-run",
                "--before-days",
                "12",
                "--after-days",
                "12",
                "--manifest",
                str(target),
            ]
        )
        == 0
    )
    manifest = AcquisitionManifest.from_json(target.read_text(encoding="utf-8"))
    assert manifest.query_prefilter is None
    assert manifest.requests["sentinel-1"]["aoi_id"] == manifest.aoi["aoi_id"]
    assert not any("pre-filter" in line for line in manifest.limitations)


def test_unverified_configured_aoi_is_disclosed_and_cannot_authorise_a_download(tmp_path):
    """The manifest states the AOI caveat, and a fetch is refused in code.

    Metadata discovery is permitted against an unverified extent; fetching the
    rasters that every downstream flood claim is computed from is not.
    """
    acquire = _load_acquire_module()
    import argparse
    from floodmap.acquisition.aoi import AoiProvenanceStatus

    target = tmp_path / "manifest.json"
    # No --aoi-* flag: this resolves the AOI from the configuration, which is what
    # makes the provisional extent part of a real run rather than a test fixture.
    exit_code = acquire.main(
        [
            "--dry-run",
            "--before-days",
            "12",
            "--after-days",
            "12",
            "--manifest",
            str(target),
        ]
    )
    assert exit_code == 0
    manifest = AcquisitionManifest.from_json(target.read_text(encoding="utf-8"))
    assert manifest.aoi["provenance_status"] == AoiProvenanceStatus.UNVERIFIED.value
    assert manifest.aoi["source_sha256"] == (
        "2ae4183f0d44f80200a889ed401ba25525c52134ba1f31d65e9f2ebc3d8965e3"
    )
    assert len(manifest.aoi["provenance_gaps"]) >= 4
    # The caveat travels with the artifact rather than living only in the docs.
    assert any("UNVERIFIED provenance" in line for line in manifest.limitations)
    assert any("sha256" in line for line in manifest.limitations)
    assert manifest.status is ManifestStatus.DRY_RUN
    assert manifest.download_status is DownloadSummary.NOT_ATTEMPTED

    # Both download gates are exercised directly, because the configuration's
    # `download.enabled: false` would otherwise mask the provenance gate.
    config, _ = load_acquisition_config()
    enabled = config.model_copy(
        update={"download": config.download.model_copy(update={"enabled": True})}
    )
    args = argparse.Namespace(download=True, dry_run=False)
    with pytest.raises(AcquisitionError, match="is not authorised"):
        acquire._authorise_download(args, config.aoi, enabled)


# ---------------------------------------------------------------------------
# Exploratory download authorisation (M6.2)
#
# Downloads are gated on AOI provenance. An operator may grant a BOUNDED
# exception to see real pixels before the provenance work is done. These tests
# pin that the exception is narrow: it must name exact products, pin the AOI
# bytes, carry a justification, and never silently become a blanket permission
# or a substitute for verifying the AOI.
# ---------------------------------------------------------------------------

_AUTH_SHA = "2ae4183f0d44f80200a889ed401ba25525c52134ba1f31d65e9f2ebc3d8965e3"
_AUTH_PRODUCT = "844b9a32-9aa2-4611-83f6-ae6501c01955"


def _authorisation(**overrides):
    from floodmap.acquisition.config import ExploratoryDownloadAuthorisation

    kwargs = dict(
        enabled=True,
        product_ids=(_AUTH_PRODUCT,),
        aoi_sha256=_AUTH_SHA,
        authorised_by="operator",
        reason="exploratory real-pixel validation",
    )
    kwargs.update(overrides)
    return ExploratoryDownloadAuthorisation(**kwargs)


def test_unverified_aoi_does_not_authorise_download_by_default():
    """The default posture is refusal, with no exception configured."""
    from floodmap.acquisition.aoi import AoiProvenanceStatus

    aoi = AreaOfInterest.from_bbox("plain", [0.0, 0.0, 0.1, 0.1])
    assert aoi.provenance_status is AoiProvenanceStatus.UNVERIFIED
    assert aoi.authorises_download is False


def test_enabled_authorisation_must_name_products_and_pin_the_aoi():
    """An exception with no scope would be a blanket download permission."""
    for field in ("product_ids", "aoi_sha256", "authorised_by", "reason"):
        blank = {"product_ids": (), "aoi_sha256": None, "authorised_by": "", "reason": ""}[field]
        with pytest.raises(Exception):
            _authorisation(**{field: blank})


def test_authorisation_covers_only_listed_products_and_pinned_aoi():
    auth = _authorisation()
    # The authorised product against the pinned AOI is permitted.
    assert auth.covers(aoi_sha256=_AUTH_SHA, product_id=_AUTH_PRODUCT) is True
    # A product that was not named is refused even inside an authorised run.
    assert auth.covers(aoi_sha256=_AUTH_SHA, product_id="not-listed") is False
    # The exception was granted for one study extent; a different AOI voids it.
    assert auth.covers(aoi_sha256="0" * 64, product_id=_AUTH_PRODUCT) is False
    assert auth.covers(aoi_sha256=None, product_id=_AUTH_PRODUCT) is False
    # A scene with no product id cannot be matched against an allowlist.
    assert auth.covers(aoi_sha256=_AUTH_SHA, product_id=None) is False


def test_disabled_authorisation_permits_nothing():
    auth = _authorisation(enabled=False)
    assert auth.covers(aoi_sha256=_AUTH_SHA, product_id=_AUTH_PRODUCT) is False


def test_authorisation_record_discloses_scope_and_keeps_aoi_unverified():
    """The manifest record must read as an exception, not as a verification."""
    record = _authorisation().to_record()
    assert record["enabled"] is True
    assert record["product_ids"] == [_AUTH_PRODUCT]
    assert record["aoi_sha256"] == _AUTH_SHA
    assert record["authorised_by"] == "operator"
    # It must say in its own text that nothing was verified.
    assert "UNCHANGED" in record["scope_note"]
    assert "unverified" in record["scope_note"]
    # No credential material is serialised.
    import json as _json

    serialised = _json.dumps(record).lower()
    for forbidden in ("password", "passwd", "secret", "token", "cdse_password"):
        assert forbidden not in serialised


def test_manifest_discloses_an_exploratory_download_exception(tmp_path):
    """An exception must surface as a limitation of the artifact."""
    aoi = AreaOfInterest.from_bbox("unverified-aoi", [0.0, 0.0, 0.1, 0.1])
    manifest = AcquisitionManifest.from_run(
        aoi=aoi,
        event=EventSpec(event_date=date(2026, 8, 26)),
        temporal_plans={},
        requests={},
        discoveries={},
        selections={},
        download_authorisation=_authorisation().to_record(),
    )
    blob = "\n".join(manifest.limitations)
    assert "narrow operator exception" in blob
    assert "UNVERIFIED" in blob
    assert "authorises no flood-evidence claim" in blob
    # The AOI's own provenance disclosure is still present and still unverified.
    assert any("UNVERIFIED provenance" in line for line in manifest.limitations)


def test_cli_refuses_download_against_unverified_aoi_without_an_exception(tmp_path):
    """End to end: the CLI exits non-zero and writes no manifest."""
    acquire = _load_acquire_module()
    target = tmp_path / "manifest.json"
    with pytest.raises(SystemExit) as excinfo:
        acquire.main(
            [
                "--download",
                "--aoi-id",
                "unverified-cli-aoi",
                "--aoi-bbox",
                "0.0",
                "0.0",
                "0.1",
                "0.1",
                "--before-days",
                "12",
                "--after-days",
                "12",
                "--manifest",
                str(target),
            ]
        )
    assert excinfo.value.code != 0
    assert not target.exists()
