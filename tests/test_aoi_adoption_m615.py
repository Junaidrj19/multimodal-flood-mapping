"""Approved AOI identity, evidence, reproducibility and closed authorization gates."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import shapely
import yaml
from pyproj import Geod

from floodmap.acquisition.config import load_acquisition_config
from floodmap.hydrology.coverage import APPROVED_COVERAGE_DISCLOSURE
from floodmap.hydrology.hydrorivers import reproject_geometry
from floodmap.utils.config import load_config, project_root

ROOT = project_root()
CONFIG = "aoi/trishuli-corridor-hydrorivers-v10-m615"
SHA = "2ae4183f0d44f80200a889ed401ba25525c52134ba1f31d65e9f2ebc3d8965e3"
ID = "trishuli-corridor-hydrorivers-v10-m615-2ae4183f"
OLD_SHA = "9035eeee025f311be7f16af1d23737a46caee20eb0b2a8dfa144d111b9362610"
HISTORICAL_SHA = "941a4079d3f317e260a91560eaebdd019ab5cf59c87f82e8e6165c7b0564b586"


@pytest.fixture(scope="module")
def evidence():
    config = load_config(CONFIG)
    payload = (ROOT / config["geojson_path"]).read_bytes()
    document = json.loads(payload)
    record = yaml.safe_load((ROOT / config["provenance_path"]).read_text())
    return config, payload, document, record


def _builder():
    spec = importlib.util.spec_from_file_location("m615_builder", ROOT / "scripts/build_corridor_aoi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_human_approval_and_claim_scope_are_recorded(evidence):
    config, _, _, record = evidence
    assert config["adoption"] == "human_approved_geometry_only"
    assert config["approval"]["partial_coverage_accepted"] is True
    assert record["approval"] == config["approval"]
    assert config["approval"]["disclosure"] == APPROVED_COVERAGE_DISCLOSURE
    decision = (ROOT / config["approval"]["record"]).read_text()
    assert "## 0. Human approval" in decision
    assert "40801674" in decision and "250 m" in decision
    assert APPROVED_COVERAGE_DISCLOSURE in decision
    assert record["coverage"]["disclosure"] == APPROVED_COVERAGE_DISCLOSURE
    assert "Observed valid-data area only" in record["coverage"]["claims_scope"]


def test_configuration_identity_is_consistent_in_both_actual_blocks(evidence):
    config, payload, _, record = evidence
    data = load_config("data")
    top, driving = data["aoi"], data["acquisition"]["aoi"]
    assert "aoi_id" not in top  # the loader does not consume one here
    for key in ("name", "geojson_path", "sha256", "provenance_status", "provenance_gaps"):
        assert top[key] == driving[key]
    assert top["geojson_path"] == config["geojson_path"]
    assert hashlib.sha256(payload).hexdigest() == top["sha256"] == SHA
    assert record["geometry"]["sha256"] == SHA
    assert driving["aoi_id"] == record["aoi_id"] == ID
    assert record["geometry"]["bytes"] == len(payload)
    assert len(driving["provenance_gaps"]) == 5
    gaps = " ".join(driving["provenance_gaps"])
    for obsolete in ("Producer unknown", "Source centreline unidentified", "NPL-FL", "~66 km"):
        assert obsolete not in gaps
    assert "human-ratified" in gaps and "unmeasured" in gaps and "63.4%" in gaps


def test_historical_artifacts_and_replacement_invariants(evidence):
    config, _, _, record = evidence
    assert config["replaces"] == "configs/aoi/trishuli-corridor-UNVERIFIED.geojson"
    assert hashlib.sha256((ROOT / config["replaces"]).read_bytes()).hexdigest() == OLD_SHA
    assert record["comparison_to_replaced_artifact"]["sha256"] == OLD_SHA
    assert hashlib.sha256((ROOT / config["historical_candidate"]).read_bytes()).hexdigest() == HISTORICAL_SHA
    assert record["historical_candidate"]["sha256"] == HISTORICAL_SHA
    assert SHA not in {OLD_SHA, HISTORICAL_SHA}


def test_adopted_topology_crs_area_bounds_and_network_endpoints(evidence):
    config, _, document, record = evidence
    assert document["type"] == "FeatureCollection" and len(document["features"]) == 1
    assert "crs" not in document  # RFC 7946 geographic delivery
    polygon = shapely.geometry.shape(document["features"][0]["geometry"])
    assert polygon.geom_type == "Polygon" and polygon.is_valid and not polygon.interiors
    assert len(polygon.exterior.coords) == record["geometry"]["vertex_count"] == 2164
    assert polygon.bounds == pytest.approx((84.7812496, 27.7888937, 85.3872770, 28.2837609), abs=1e-7)
    metric_area = reproject_geometry(polygon).area / 1e6
    assert metric_area == pytest.approx(206.1506509, abs=1e-7)
    geodesic_area = abs(Geod(ellps="WGS84").geometry_area_perimeter(polygon)[0]) / 1e6
    assert geodesic_area == pytest.approx(metric_area, rel=0.001)
    selection = record["geography"]["reach_selection"]
    reaches = selection["selected_reaches"]
    assert reaches["reach_count"] == 50
    assert reaches["reach_ids"][0] == 40771762
    assert reaches["reach_ids"][49] == selection["terminus_reach_id"] == 40801674
    assert selection["terminus_index"] == 49
    assert selection["downstream_stop_reason"] == "caller_stop_reach"
    assert selection["coverage_fraction"] is None  # selection is not observation
    assert selection["centreline_length_m"] == pytest.approx(105857.91, abs=0.01)
    assert config["construction"]["inset_role"] == "acceptance_only_not_terminus_definition"
    assert 40801919 not in reaches["reach_ids"]


def test_independent_scene_inset_and_complete_buffer_sensitivity(evidence):
    _, _, _, record = evidence
    gate = record["coverage"]["scene_acceptance"]
    assert gate["accepted"] and gate["fully_covered"]
    assert gate["clearance_m"] == pytest.approx(6763.0714, abs=0.01)
    assert gate["inset_m"] == 250
    for scene in gate["footprints"]:
        assert scene["vertex_count"] == 59
        assert scene["relative_orbit"] == 85 and scene["orbit_direction"] == "ASCENDING"
        assert scene["path"].endswith(".SAFE.zip") and "sentinel1_rtc" not in scene["path"]
        for key in ("annotation_sha256", "archive_sha256", "footprint_wkb_sha256"):
            assert len(scene[key]) == 64
    sensitivity = record["geography"]["buffer"]["sensitivity"]
    assert [row["buffer_m"] for row in sensitivity] == [500, 1000, 2000]
    for row in sensitivity:
        assert row["status"] == "constructed" and row["inset_accepted"]
        assert row["terminus_reach_id"] == 40801674
        assert row["clearance_m"] > 250
        assert row["valid_fraction_of_delivered_pixels"] == 1
        assert "unmeasured" in row["exposure_and_connectivity_sensitivity"]


def test_observed_valid_area_is_not_the_aoi_area(evidence):
    _, _, _, record = evidence
    coverage = record["coverage"]
    assert coverage["aoi_area_km2"] == pytest.approx(206.1506509, abs=1e-7)
    assert coverage["delivered_area_km2"] == pytest.approx(130.6416323, abs=1e-7)
    assert coverage["undelivered_area_km2"] == pytest.approx(75.5090186, abs=1e-7)
    assert coverage["delivered_fraction"] == pytest.approx(0.63371923247, abs=1e-10)
    assert coverage["unobserved_fraction"] == pytest.approx(1 - coverage["delivered_fraction"])
    assert coverage["valid_fraction_of_delivered_pixels"] == 1
    assert coverage["combined_valid_pixel_count"] == coverage["delivered_pixel_count"] == 1306418
    assert coverage["valid_area_km2"] == pytest.approx(130.6418)
    assert coverage["valid_area_km2"] < coverage["aoi_area_km2"]
    assert 0.63 < coverage["valid_fraction_of_aoi"] < 0.64
    assert coverage["methods"]["all_touched"] is False
    assert coverage["grid"]["resampling"] == "none"
    for source in coverage["sources"]:
        for band in source["bands"]:
            assert band["valid_pixel_count"] == 1306418 and band["invalid_pixel_count"] == 0
            assert band["nodata"] == -32768
    assert "not_valid_mask" in coverage["observation_footprint"]["definition"]
    unknown = coverage["unobserved_footprint_4326"]
    assert unknown["crs"] == "EPSG:4326"
    unknown_area = reproject_geometry(shapely.geometry.shape(unknown["geometry"])).area / 1e6
    assert unknown_area == pytest.approx(coverage["undelivered_area_km2"], abs=1e-7)
    assert "unknown_not_negative" in unknown["definition"]


def test_closed_gates_through_actual_loader(evidence):
    _, _, _, record = evidence
    loaded, data = load_acquisition_config()
    assert loaded.aoi.aoi_id == ID and loaded.aoi.source_sha256 == SHA
    assert loaded.aoi.authorises_download is False
    assert loaded.download.enabled is False
    assert "exploratory_authorisation" not in data["acquisition"]["download"]
    assert loaded.download.exploratory_authorisation.enabled is False
    assert loaded.download.exploratory_authorisation.product_ids == ()
    assert data["production"]["osm"]["allow_post_event_edits"] is False
    for key in ("authorises_download", "training_started", "data_downloaded", "falkordb_populated"):
        assert record["authorisation"][key] is False


def test_builder_implementation_and_configuration_are_content_pinned(evidence):
    config, _, _, record = evidence
    for path, digest in record["implementation_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert record["configuration"] == config
    assert record["configuration_sha256"] == hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    assert all(record["software"][name] for name in ("python", "shapely", "pyproj", "pyogrio", "rasterio"))


def test_real_dashboard_binds_reconciled_results_and_keeps_approved_disclosure(evidence):
    """M6.16 reconciled the real M6.3 outputs, so the binding now succeeds.

    M6.15 left these artifacts stale and the dashboard correctly withheld them.
    The guarantee under test is the binding itself, not the staleness: the
    negative path is covered separately below so that neither direction is
    asserted on faith.
    """
    from floodmap.dashboard.realdata import ValidationState, discover_m63_artifacts

    if not (ROOT / "artifacts/m6_3/rtc_change_candidates/rtc_change_summary.json").is_file():
        pytest.skip("real M6.3 artifacts absent (git-ignored)")
    discovered = discover_m63_artifacts(ROOT)
    assert discovered.state is ValidationState.VALID, discovered.problems_text()
    assert discovered.is_usable
    assert discovered.configured_aoi.aoi_id == ID
    assert discovered.configured_aoi.coverage["disclosure"] == APPROVED_COVERAGE_DISCLOSURE
    assert not discovered.configured_aoi.problems

    # The extent is bound by digest, and partial coverage survives into the result.
    assert discovered.summary["aoi_id"] == ID
    assert discovered.summary["aoi_source_sha256"] == SHA
    assert discovered.summary["coverage_disclosure"] == APPROVED_COVERAGE_DISCLOSURE
    scope = discovered.summary["aoi_scope"]
    assert scope["aoi_delivered_area_km2"] < scope["aoi_geometric_area_km2"]
    assert scope["aoi_valid_area_km2"] < scope["aoi_geometric_area_km2"]
    assert scope["aoi_unobserved_area_km2"] > 0
    assert scope["window_pixels_outside_aoi"] > 0
    # Candidates are a subset of the AOI-scoped valid support, recomputed from
    # the rasters by the validator rather than taken from the summary.
    assert discovered.computed["candidate_pixels"] <= discovered.computed["aoi_valid_pixels"]
    assert discovered.computed["aoi_valid_pixels"] == discovered.computed["valid_pixels"]


def test_real_dashboard_still_refuses_results_bound_to_a_different_extent(evidence, tmp_path):
    """Re-keying an output to the configured extent must not make it current."""
    import shutil

    from floodmap.dashboard.realdata import (
        M63_RELATIVE_DIR,
        ValidationState,
        discover_m63_artifacts,
    )

    source = ROOT / M63_RELATIVE_DIR
    if not (source / "rtc_change_summary.json").is_file():
        pytest.skip("real M6.3 artifacts absent (git-ignored)")

    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    shutil.copytree(ROOT / "configs/aoi", project / "configs/aoi")
    shutil.copy(ROOT / "configs/data.yaml", project / "configs/data.yaml")
    (project / M63_RELATIVE_DIR).mkdir(parents=True)
    shutil.copytree(source, project / M63_RELATIVE_DIR, dirs_exist_ok=True)

    summary_path = project / M63_RELATIVE_DIR / "rtc_change_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["aoi_source_sha256"] = OLD_SHA
    summary["pair"]["aoi_source_sha256"] = OLD_SHA
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    discovered = discover_m63_artifacts(project)
    assert discovered.state is ValidationState.STALE and not discovered.is_usable
    assert discovered.computed == {} and discovered.rasters == {}
    assert any("source hash does not match" in problem for problem in discovered.problems)


def test_build_and_check_reproduce_geometry_and_measured_provenance(evidence, monkeypatch):
    config, payload, _, record = evidence
    paths = [config["construction"]["source"]["gdb_path"]]
    paths += config["construction"]["scene_footprints"] + config["construction"]["observation_rasters"]
    if not all((ROOT / p).exists() for p in paths):
        pytest.skip("local HydroRIVERS/SAFE/RTC inputs absent (git-ignored)")
    builder = _builder()
    document, rebuilt = builder.build(config)
    assert (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode() == payload
    assert builder.check_outputs(document, config, rebuilt)
    # A forged provenance measurement is not accepted just because geometry matches.
    changed = copy.deepcopy(rebuilt)
    changed["coverage"]["delivered_fraction"] = 1.0
    assert not builder.check_outputs(document, config, changed)
    assert rebuilt["source"]["local_sha256"] == record["source"]["local_sha256"]
