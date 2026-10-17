"""Current-AOI binding and approved partial-coverage presentation regressions.

All geometry, rasters and coverage numbers below are synthetic software fixtures.
They exercise the human-approved disclosure, not a new coverage measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from floodmap.dashboard import build_real_dashboard
from floodmap.dashboard.realdata import (
    M63_RELATIVE_DIR,
    PROJECT_ROOT_ENV,
    ValidationState,
    discover_m63_artifacts,
)
from tests.conftest import REPO_ROOT
from tests.m63_fixtures import write_m63_fixture

APPROVED_DISCLOSURE = (
    "The study corridor extends beyond the currently available valid satellite observation coverage. "
    "Approximately 63.4% of the AOI has delivered imagery coverage; the remaining 36.6% is "
    "unobserved in the current dataset. Results apply only to the observed, valid-data area. "
    "Unobserved areas must not be interpreted as unaffected, flood-free, or undamaged."
)
CONFIGURED_BOUNDS = [84.8, 27.2, 85.6, 28.8]


@pytest.fixture()
def adoption_project(tmp_path: Path) -> Path:
    """Represent adoption without changing any workspace config or artifact."""
    geometry_path = tmp_path / "configs/aoi/trishuli-corridor-hydrorivers-v10-m615.geojson"
    geometry_path.parent.mkdir(parents=True)
    west, south, east, north = CONFIGURED_BOUNDS
    geometry_path.write_text(
        json.dumps({
            "type": "Polygon",
            "coordinates": [
                [[west, south], [east, south], [east, north], [west, north], [west, south]]
            ],
        }),
        encoding="utf-8",
    )
    digest = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    aoi = {
        "aoi_id": f"fixture-hydrorivers-v10-m615-{digest[:8]}",
        "geojson_path": str(geometry_path.relative_to(tmp_path)),
        "sha256": digest,
        "provenance_status": "unverified",
    }
    (tmp_path / "configs/data.yaml").write_text(
        yaml.safe_dump({"aoi": aoi, "acquisition": {"aoi": aoi}}), encoding="utf-8"
    )
    geometry_path.with_suffix(".provenance.yaml").write_text(
        yaml.safe_dump({
            "aoi_id": aoi["aoi_id"],
            "geometry": {"path": aoi["geojson_path"], "sha256": digest},
            "authorisation": {"provenance_status": "unverified", "authorises_download": False},
            "coverage": {
                "aoi_area_km2": 200.0,
                "delivered_area_km2": 126.8,
                "delivered_fraction": 0.634,
                "valid_area_km2": 126.8,
                "valid_fraction_of_delivered_pixels": 1.0,
                "valid_fraction_of_aoi": 0.634,
                "unobserved_fraction": 0.366,
                "disclosure": APPROVED_DISCLOSURE,
            },
        }),
        encoding="utf-8",
    )
    write_m63_fixture(tmp_path / M63_RELATIVE_DIR)
    return tmp_path


def _config(project: Path) -> dict:
    return yaml.safe_load((project / "configs/data.yaml").read_text())


def _set_artifact_identity(project: Path, *, source_hash: str | None) -> None:
    """Simulate outputs for a different fixture extent, never relabel real files."""
    aoi = _config(project)["acquisition"]["aoi"]
    directory = project / M63_RELATIVE_DIR
    summary_path = directory / "rtc_change_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["pair"]["aoi_id"] = aoi["aoi_id"]
    if source_hash is not None:
        summary["pair"]["aoi_source_sha256"] = source_hash
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for path in directory.glob("*_provenance.yaml"):
        provenance = yaml.safe_load(path.read_text())
        provenance["aoi"] = aoi["aoi_id"]
        path.write_text(yaml.safe_dump(provenance), encoding="utf-8")


def test_stale_aoi_cannot_build_current_dashboard(adoption_project: Path):
    directory = adoption_project / M63_RELATIVE_DIR
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    result = discover_m63_artifacts(adoption_project)
    assert result.state is ValidationState.STALE
    assert not result.is_usable
    assert "artifact AOI" in result.problems_text()
    assert "configured AOI" in result.problems_text()
    assert result.computed == result.recorded == {}
    with pytest.raises(ValueError, match="stale"):
        build_real_dashboard(result)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


@pytest.mark.parametrize("source_hash", [None, "0" * 64])
def test_matching_id_requires_matching_source_hash(adoption_project: Path, source_hash):
    _set_artifact_identity(adoption_project, source_hash=source_hash)
    result = discover_m63_artifacts(adoption_project)
    assert result.state is ValidationState.STALE
    assert not result.check("configured_aoi.binding").ok
    assert "source hash" in result.problems_text()
    assert result.computed == result.recorded == {}


def test_matching_id_and_source_hash_can_bind_unverified_aoi(adoption_project: Path):
    aoi = _config(adoption_project)["acquisition"]["aoi"]
    _set_artifact_identity(adoption_project, source_hash=aoi["sha256"])
    result = discover_m63_artifacts(adoption_project)
    assert result.is_usable, result.problems
    assert result.check("configured_aoi.binding").ok
    assert result.configured_aoi.provenance_status == "unverified"
    assert result.configured_aoi.coverage["disclosure"] == APPROVED_DISCLOSURE
    # Delivered-grid validity is distinct from the recorded AOI coverage.
    assert result.computed["valid_fraction"] != result.configured_aoi.coverage["valid_fraction_of_aoi"]
    assert build_real_dashboard(result).manifest.aoi_id == aoi["aoi_id"]


def test_configured_geometry_hash_is_checked(adoption_project: Path):
    aoi = _config(adoption_project)["acquisition"]["aoi"]
    _set_artifact_identity(adoption_project, source_hash=aoi["sha256"])
    path = adoption_project / aoi["geojson_path"]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    result = discover_m63_artifacts(adoption_project)
    assert not result.is_usable
    assert result.configured_aoi.bounds_4326 is None
    assert "sha256" in result.problems_text().lower()


def test_summary_identity_cannot_override_stale_layer_provenance(adoption_project: Path):
    aoi = _config(adoption_project)["acquisition"]["aoi"]
    _set_artifact_identity(adoption_project, source_hash=aoi["sha256"])
    path = adoption_project / M63_RELATIVE_DIR / "rtc_change_candidates_provenance.yaml"
    provenance = yaml.safe_load(path.read_text())
    provenance["aoi"] = "older-fixture-extent"
    path.write_text(yaml.safe_dump(provenance), encoding="utf-8")
    result = discover_m63_artifacts(adoption_project)
    assert not result.is_usable
    assert not result.check("configured_aoi.candidate_provenance").ok
    with pytest.raises(ValueError):
        build_real_dashboard(result)


def test_missing_configured_hash_does_not_disable_binding(adoption_project: Path):
    data = _config(adoption_project)
    del data["acquisition"]["aoi"]["sha256"]
    (adoption_project / "configs/data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    result = discover_m63_artifacts(adoption_project)
    assert result.state is ValidationState.STALE
    assert "acquisition.aoi.sha256 is missing" in result.problems_text()


def test_stale_coverage_provenance_is_not_disclosed_as_current(adoption_project: Path):
    aoi = _config(adoption_project)["acquisition"]["aoi"]
    path = (adoption_project / aoi["geojson_path"]).with_suffix(".provenance.yaml")
    provenance = yaml.safe_load(path.read_text())
    provenance["geometry"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(provenance), encoding="utf-8")
    result = discover_m63_artifacts(adoption_project)
    assert not result.is_usable
    assert result.configured_aoi.coverage == {}
    assert "AOI provenance ID/hash does not match" in result.problems_text()


def test_explicit_artifact_directory_cannot_bypass_workspace_binding(adoption_project: Path):
    custom = write_m63_fixture(adoption_project / "standalone-old-artifacts")
    result = discover_m63_artifacts(adoption_project, directory=custom)
    assert result.state is ValidationState.STALE


def test_acquisition_aoi_is_authoritative(adoption_project: Path):
    data = _config(adoption_project)
    data["aoi"] = {"aoi_id": "old-top-level-extent"}
    (adoption_project / "configs/data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    aoi = data["acquisition"]["aoi"]
    _set_artifact_identity(adoption_project, source_hash=aoi["sha256"])
    assert discover_m63_artifacts(adoption_project).is_usable


def test_synthetic_workspace_preserves_standalone_discovery(adoption_project: Path):
    data = _config(adoption_project)
    data["acquisition"]["aoi"]["is_synthetic"] = True
    (adoption_project / "configs/data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    result = discover_m63_artifacts(adoption_project)
    assert result.is_usable, result.problems
    assert result.configured_aoi is None


def test_standalone_discovery_without_data_config(tmp_path: Path):
    write_m63_fixture(tmp_path / M63_RELATIVE_DIR)
    assert discover_m63_artifacts(tmp_path).is_usable


@pytest.mark.parametrize("result_state", ["stale", "missing", "current"])
def test_ui_discloses_partial_coverage_and_frames_configured_aoi(
    adoption_project: Path, monkeypatch, result_state: str
):
    pytest.importorskip("streamlit")
    import folium
    import streamlit_folium
    from streamlit.testing.v1 import AppTest

    if result_state == "missing":
        (adoption_project / M63_RELATIVE_DIR / "rtc_change_candidates.tif").unlink()
    elif result_state == "current":
        aoi = _config(adoption_project)["acquisition"]["aoi"]
        _set_artifact_identity(adoption_project, source_hash=aoi["sha256"])
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(adoption_project))
    maps = []
    monkeypatch.setattr(streamlit_folium, "st_folium", lambda overview, **_: maps.append(overview))
    at = AppTest.from_file(str(REPO_ROOT / "dashboard/app.py")).run(timeout=60)
    assert not at.exception, at.exception
    assert APPROVED_DISCLOSURE in [item.value for item in at.warning]
    assert any("provenance `unverified`" in item.value for item in at.caption)
    assert maps
    overview = maps[-1]
    assert overview.location == pytest.approx([28.0, 85.2])
    fit = next(
        child for child in overview._children.values() if isinstance(child, folium.map.FitBounds)
    )
    assert fit.bounds == [[27.2, 84.8], [28.8, 85.6]]
    has_result_overlay = any(
        isinstance(child, folium.raster_layers.ImageOverlay) for child in overview._children.values()
    )
    if result_state == "current":
        assert has_result_overlay
        assert at.table
    else:
        assert f"state: `{result_state}`" in " ".join(item.value for item in at.error)
        assert not has_result_overlay
        assert not at.table and not at.metric and not at.sidebar.checkbox
        assert "Recorded by M6.3" not in " ".join(item.value for item in at.caption)
        assert "Completed on real data" not in " ".join(item.value for item in at.markdown)


def test_synthetic_demo_ui_is_unaffected_by_adoption(adoption_project: Path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import floodmap.dashboard

    def refuse_generation(*args, **kwargs):
        pytest.fail("Adoption must not regenerate the existing synthetic demo")

    monkeypatch.setattr(floodmap.dashboard, "build_demo_bundle", refuse_generation)
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(adoption_project))
    demo_dir = REPO_ROOT / "artifacts/dashboard_demo"
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in demo_dir.rglob("*")
        if path.is_file()
    }
    at = AppTest.from_file(str(REPO_ROOT / "dashboard/app.py")).run(timeout=60)
    at.sidebar.radio[0].set_value("Synthetic demo bundle").run(timeout=60)
    assert not at.exception, at.exception
    assert any("SYNTHETIC PLACEHOLDER" in item.value for item in at.warning)
    assert APPROVED_DISCLOSURE not in [item.value for item in at.warning]
    assert at.sidebar.checkbox
    assert before == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in demo_dir.rglob("*")
        if path.is_file()
    }
