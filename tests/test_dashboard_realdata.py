"""M6.3 real-artifact discovery + validation tests.

Every fixture is built by :mod:`tests.m63_fixtures` and is explicitly synthetic.
These tests check *software behaviour*: that valid artifacts are accepted, and
that each realistic failure mode (absent, unreadable, malformed, inconsistent) is
reported rather than papered over. None of them produce a real Trishuli result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from floodmap.dashboard.realdata import (
    M63_RELATIVE_DIR,
    PROJECT_ROOT_ENV,
    ValidationState,
    discover_m63_artifacts,
    resolve_project_root,
)
from tests.m63_fixtures import (
    FIXTURE_CRS,
    M63_STAGE,
    SYNTHETIC_MARKER,
    write_m63_fixture,
)
from tests.m63_fixtures import _write_raster  # noqa: PLC2701 - fixture internals


def _m63_dir(project_root: Path) -> Path:
    return project_root / M63_RELATIVE_DIR


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    write_m63_fixture(_m63_dir(tmp_path))
    return tmp_path


class TestValidArtifacts:
    def test_valid_fixture_is_discovered(self, project: Path):
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.VALID, result.problems
        assert result.is_usable is True
        assert result.problems == []
        assert all(check.ok for check in result.checks)

    def test_directory_is_absolute(self, project: Path):
        result = discover_m63_artifacts(project)
        assert result.directory.is_absolute()
        assert result.directory == _m63_dir(project)

    def test_computed_counts_match_the_arrays(self, project: Path):
        result = discover_m63_artifacts(project)
        computed = result.computed
        assert computed["total_pixels"] == 24 * 24
        assert computed["candidate_pixels"] == 20
        assert computed["not_observed_pixels"] == 30
        assert computed["valid_pixels"] == computed["total_pixels"] - 30
        assert computed["candidate_fraction_of_valid"] == pytest.approx(20 / 546)
        assert computed["candidate_grid_area_km2"] == pytest.approx(20 * 100 / 1e6)
        assert computed["grid_area_km2"] == pytest.approx(576 * 100 / 1e6)

    def test_recorded_summary_is_cross_checked(self, project: Path):
        result = discover_m63_artifacts(project)
        assert result.recorded["candidate_pixels"] == 20
        assert result.recorded["valid_pixels"] == 546
        # Every comparable summary figure must be present and agree.
        for key in (
            "total_pixels",
            "valid_pixels",
            "candidate_pixels",
            "candidate_fraction_of_valid",
            "candidate_grid_area_km2",
        ):
            check = result.check(f"summary.{key}")
            assert check is not None and check.ok, key

    def test_grid_metadata_is_read_back(self, project: Path):
        result = discover_m63_artifacts(project)
        grid = result.rasters["candidate_mask"]
        assert grid.crs == FIXTURE_CRS
        assert grid.shape == (24, 24)
        assert grid.resolution == (10.0, 10.0)
        assert grid.band_names == ("change_candidate",)
        features = result.rasters["features"]
        assert features.band_names == ("vv_change_db", "vh_change_db")
        assert features.count == 2

    def test_provenance_records_are_loaded_and_typed(self, project: Path):
        result = discover_m63_artifacts(project)
        assert set(result.provenance) == {"candidate_provenance", "features_provenance"}
        assert result.provenance["candidate_provenance"].artifact_type.value == (
            "change_candidate_mask"
        )
        assert result.provenance["features_provenance"].artifact_type.value == "feature_stack"
        assert result.provenance["candidate_provenance"].notes == SYNTHETIC_MARKER


class TestMissingArtifacts:
    def test_empty_directory_reports_missing(self, tmp_path: Path):
        result = discover_m63_artifacts(tmp_path)
        assert result.state is ValidationState.MISSING
        assert result.is_usable is False
        assert any("missing required M6.3 files" in p for p in result.problems)

    @pytest.mark.parametrize(
        "filename",
        [
            "rtc_change_candidates.tif",
            "rtc_valid_mask.tif",
            "rtc_change_features.tif",
            "rtc_change_summary.json",
            "rtc_change_candidates_provenance.yaml",
        ],
    )
    def test_each_required_file_is_enforced(self, project: Path, filename: str):
        (_m63_dir(project) / filename).unlink()
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.MISSING
        assert result.is_usable is False
        assert any(filename in p for p in result.problems)


class TestMalformedArtifacts:
    def test_invalid_summary_json(self, project: Path):
        (_m63_dir(project) / "rtc_change_summary.json").write_text("{not json", encoding="utf-8")
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.MALFORMED
        assert any("malformed" in p for p in result.problems)

    def test_summary_that_is_not_an_object(self, project: Path):
        (_m63_dir(project) / "rtc_change_summary.json").write_text("[1, 2]", encoding="utf-8")
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.MALFORMED

    def test_unreadable_raster(self, project: Path):
        (_m63_dir(project) / "rtc_change_candidates.tif").write_bytes(b"not a geotiff")
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.MALFORMED
        assert any("unreadable" in p for p in result.problems)

    def test_unreadable_provenance(self, project: Path):
        (_m63_dir(project) / "rtc_change_candidates_provenance.yaml").write_text(
            ": not: valid: yaml::", encoding="utf-8"
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert any("unreadable" in p for p in result.problems)


class TestGridAndBandInconsistency:
    def test_crs_mismatch_between_rasters(self, project: Path):
        _write_raster(
            _m63_dir(project) / "rtc_change_features.tif",
            np.zeros((2, 24, 24), dtype="float32"),
            band_names=("vv_change_db", "vh_change_db"),
            nodata=-9999.0,
            crs="EPSG:4326",
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        check = result.check("grid.candidate_mask~features")
        assert check is not None and check.ok is False
        assert any("grids do not match" in p for p in result.problems)

    def test_shape_mismatch_between_rasters(self, project: Path):
        _write_raster(
            _m63_dir(project) / "rtc_change_features.tif",
            np.zeros((2, 12, 12), dtype="float32"),
            band_names=("vv_change_db", "vh_change_db"),
            nodata=-9999.0,
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert any("grids do not match" in p for p in result.problems)

    def test_wrong_band_names(self, project: Path):
        write_m63_fixture(
            _m63_dir(project),
            feature_band_names=("VV", "VH"),
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        check = result.check("bands.features")
        assert check is not None and check.ok is False
        assert any("band names" in p for p in result.problems)


class TestSummaryConsistency:
    def test_candidate_count_mismatch_is_detected(self, project: Path):
        write_m63_fixture(_m63_dir(project), summary_overrides={"candidate_pixels": 999})
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        check = result.check("summary.candidate_pixels")
        assert check is not None and check.ok is False
        assert any("disagrees with the rasters" in p for p in result.problems)

    def test_candidate_percentage_mismatch_is_detected(self, project: Path):
        write_m63_fixture(
            _m63_dir(project), summary_overrides={"candidate_fraction_of_valid": 0.5}
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        check = result.check("summary.candidate_fraction_of_valid")
        assert check is not None and check.ok is False

    def test_missing_summary_field_is_detected(self, project: Path):
        write_m63_fixture(_m63_dir(project), summary_overrides={"candidate_pixels": None})
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert any("absent from summary" in c.detail for c in result.failed_checks())

    def test_candidate_outside_valid_mask_is_detected(self, project: Path):
        write_m63_fixture(_m63_dir(project), n_invalid_valid_mask=5)
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert any("fall outside the valid-data mask" in p for p in result.problems)

    def test_unexpected_candidate_class_value_is_detected(self, project: Path):
        candidate = np.zeros((1, 24, 24), dtype="uint8")
        candidate[0, 0, 0] = 3
        _write_raster(
            _m63_dir(project) / "rtc_change_candidates.tif",
            candidate,
            band_names=("change_candidate",),
            nodata=255,
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert any("unexpected class values" in p for p in result.problems)

    def test_wrong_stage_tag_is_detected(self, project: Path):
        write_m63_fixture(_m63_dir(project), summary_overrides={"stage": "some-other-stage"})
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert result.check("summary.stage").ok is False

    def test_wrong_radiometric_convention_is_detected(self, project: Path):
        write_m63_fixture(
            _m63_dir(project), summary_overrides={"radiometric_convention": "sigma0"}
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert result.check("summary.radiometric_convention").ok is False

    def test_mismatched_provenance_artifact_type_is_detected(self, project: Path):
        # Overwrite the candidate provenance with the feature-stack type.
        source = _m63_dir(project) / "rtc_change_features_provenance.yaml"
        (_m63_dir(project) / "rtc_change_candidates_provenance.yaml").write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.INCONSISTENT
        assert any("declares artifact_type" in p for p in result.problems)


class TestPathResolution:
    def test_project_root_env_override(self, project: Path, monkeypatch):
        monkeypatch.setenv(PROJECT_ROOT_ENV, str(project))
        assert resolve_project_root() == project.resolve()

    def test_explicit_root_beats_environment(self, project: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(PROJECT_ROOT_ENV, str(project))
        assert resolve_project_root(tmp_path) == tmp_path.resolve()

    def test_discovery_ignores_working_directory(self, project: Path, tmp_path: Path, monkeypatch):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        result = discover_m63_artifacts(project)
        assert result.state is ValidationState.VALID
        assert result.directory == _m63_dir(project)

    def test_explicit_directory_wins(self, project: Path, tmp_path: Path):
        custom = tmp_path / "somewhere-else"
        write_m63_fixture(custom)
        result = discover_m63_artifacts(project, directory=custom)
        assert result.state is ValidationState.VALID
        assert result.directory == custom

    def test_summary_stage_matches_declared_constant(self, project: Path):
        result = discover_m63_artifacts(project)
        assert json.loads((result.directory / "rtc_change_summary.json").read_text())["stage"] == (
            M63_STAGE
        )
