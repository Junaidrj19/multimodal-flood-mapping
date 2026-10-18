"""Workflow stage-status and blocker-report tests.

The point of these tests is the distinction the brief insists on: a blocked
downstream stage is not the same as a missing output, and neither is the same as
"zero impact". All fixtures are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from floodmap.dashboard.realdata import discover_m63_artifacts
from floodmap.dashboard.workflow import (
    M64_BLOCKER,
    M64_DIR,
    M64_OUTPUT,
    M65_BLOCKER,
    M65_DIR,
    M65_OUTPUT,
    BlockerReport,
    BlockerState,
    StageState,
    blocker_prerequisites_present,
    build_workflow,
    load_blocker_report,
)
from tests.m63_fixtures import valid_blocker, write_blocker, write_m63_fixture


def _prepare(tmp_path: Path) -> None:
    write_m63_fixture(tmp_path / "artifacts" / "m6_3" / "rtc_change_candidates")


class TestBlockerLoading:
    def test_loaded_blocker(self, tmp_path: Path):
        path = write_blocker(tmp_path / "out", "b.json", valid_blocker("m6.4"))
        report = load_blocker_report(path, stage="m6.4")
        assert report.state is BlockerState.LOADED
        assert report.is_blocked is True
        assert report.reason
        assert report.missing == ["roads", "buildings", "settlements"]
        assert report.authorisation["authorises_download"] is False
        assert report.next_step

    def test_missing_blocker(self, tmp_path: Path):
        report = load_blocker_report(tmp_path / "absent.json", stage="m6.5")
        assert report.state is BlockerState.MISSING
        assert report.is_blocked is False

    def test_malformed_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        report = load_blocker_report(path, stage="m6.4")
        assert report.state is BlockerState.MALFORMED
        assert any("not valid JSON" in p for p in report.problems)

    def test_non_object_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        report = load_blocker_report(path, stage="m6.4")
        assert report.state is BlockerState.MALFORMED

    def test_missing_required_keys(self, tmp_path: Path):
        path = write_blocker(tmp_path / "out", "b.json", {"stage": "m6.4", "status": "blocked"})
        report = load_blocker_report(path, stage="m6.4")
        assert report.state is BlockerState.MALFORMED
        assert any("reason" in p for p in report.problems)

    def test_next_step_wrong_type(self, tmp_path: Path):
        payload = valid_blocker("m6.4")
        payload["next_step"] = [1, 2]
        path = write_blocker(tmp_path / "out", "b.json", payload)
        report = load_blocker_report(path, stage="m6.4")
        # A non-string list is coerced to strings rather than crashing the UI.
        assert report.state is BlockerState.LOADED
        assert report.next_step == ["1", "2"]

    def test_string_next_step_is_wrapped(self, tmp_path: Path):
        payload = valid_blocker("m6.4")
        payload["next_step"] = "do not bypass the gate"
        path = write_blocker(tmp_path / "out", "b.json", payload)
        report = load_blocker_report(path, stage="m6.4")
        assert report.next_step == ["do not bypass the gate"]


class TestWorkflowStates:
    def test_blocked_stages_are_reported_with_reasons(self, tmp_path: Path):
        _prepare(tmp_path)
        write_blocker(
            tmp_path / M64_DIR, M64_BLOCKER, valid_blocker("m6.4", reason="no OSM layers")
        )
        write_blocker(tmp_path / M65_DIR, M65_BLOCKER, valid_blocker("m6.5", reason="no M6.4"))
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)

        exposure = workflow.stage("m6_4_exposure")
        assert exposure is not None
        assert exposure.state is StageState.BLOCKED
        assert "no OSM layers" in exposure.detail
        assert exposure.prerequisites
        assert exposure.blocker is not None and exposure.blocker.is_blocked

        disruption = workflow.stage("m6_5_disruption")
        assert disruption is not None
        assert disruption.state is StageState.BLOCKED
        assert workflow.any_blocked() is True

    def test_completed_upstream_stages(self, tmp_path: Path):
        _prepare(tmp_path)
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        for stage_id in ("rtc_acquisition", "preprocessing", "m6_3_change_candidates"):
            status = workflow.stage(stage_id)
            assert status is not None and status.state is StageState.REAL_DATA_COMPLETE, stage_id

    def test_blocked_is_not_confused_with_missing_output(self, tmp_path: Path):
        _prepare(tmp_path)
        write_blocker(tmp_path / M64_DIR, M64_BLOCKER, valid_blocker("m6.4"))
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        exposure = workflow.stage("m6_4_exposure")
        assert exposure is not None
        assert exposure.state is not StageState.MISSING_OUTPUT
        assert exposure.state is not StageState.REAL_DATA_COMPLETE

    def test_no_output_and_no_blocker_is_not_executed(self, tmp_path: Path):
        _prepare(tmp_path)
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        for stage_id in ("m6_4_exposure", "m6_5_disruption"):
            status = workflow.stage(stage_id)
            assert status is not None
            assert status.state is StageState.IMPLEMENTED_NOT_EXECUTED

    def test_real_output_marks_stage_complete(self, tmp_path: Path):
        _prepare(tmp_path)
        output = tmp_path / M64_DIR / M64_OUTPUT
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"stage": "m6.4-fixture"}), encoding="utf-8")
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        exposure = workflow.stage("m6_4_exposure")
        assert exposure is not None
        assert exposure.state is StageState.REAL_DATA_COMPLETE
        assert str(output) in exposure.evidence

    def test_malformed_blocker_is_invalid_not_blocked(self, tmp_path: Path):
        _prepare(tmp_path)
        path = tmp_path / M64_DIR / M64_BLOCKER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        exposure = workflow.stage("m6_4_exposure")
        assert exposure is not None
        assert exposure.state is StageState.INVALID_OUTPUT
        assert "malformed" in exposure.detail

    def test_unexpected_blocker_status_is_unknown(self, tmp_path: Path):
        _prepare(tmp_path)
        payload = valid_blocker("m6.4")
        payload["status"] = "resolved"
        write_blocker(tmp_path / M64_DIR, M64_BLOCKER, payload)
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        exposure = workflow.stage("m6_4_exposure")
        assert exposure is not None
        assert exposure.state is StageState.UNKNOWN

    def test_missing_m63_marks_candidate_stage_missing(self, tmp_path: Path):
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        stage = workflow.stage("m6_3_change_candidates")
        assert stage is not None
        assert stage.state is StageState.MISSING_OUTPUT

    def test_aoi_authorisation_facts_are_surfaced(self, tmp_path: Path):
        _prepare(tmp_path)
        write_blocker(tmp_path / M64_DIR, M64_BLOCKER, valid_blocker("m6.4"))
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        assert workflow.aoi_provenance_status == "UNVERIFIED"
        assert workflow.authorises_download is False
        assert workflow.unblock_requirements

    def test_unblock_requirements_are_not_bypassable_wording(self, tmp_path: Path):
        _prepare(tmp_path)
        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        joined = " ".join(workflow.unblock_requirements).lower()
        assert "authoris" in joined
        assert "osm" in joined or "openstreetmap" in joined
        assert len(workflow.unblock_requirements) >= 4


def _m65_blocker_payload(missing=("candidate_exposure_roads.geojson",)):
    return {
        "stage": "m6.5-candidate-network-disruption-0.1.0",
        "status": "blocked",
        "reason": "required M6.4 candidate-exposure artifacts are not present on disk: "
        + ", ".join(missing),
        "missing_inputs": list(missing),
        "searched_directory": str(M64_DIR),
        "retrieval_performed": False,
        "authorisation": {"aoi_provenance_status": "UNVERIFIED", "authorises_download": False},
        "next_step": ["Run M6.5 network disruption on the resulting M6.4 exposure artifacts."],
        "not_done": ["No road network, settlement or hub was synthesised."],
    }


def _write_m64_inputs(tmp_path: Path, names) -> None:
    directory = tmp_path / M64_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(
            '{"type": "FeatureCollection", "features": []}', encoding="utf-8"
        )


class TestStaleBlocker:
    """A blocker describes the moment it was written, not the current one.

    M6.4 has since run, which made the on-disk M6.5 blocker claim something that
    is no longer true: that the M6.4 artifacts it looked for are absent. The
    dashboard must stop repeating that sentence without pretending M6.5 ran.
    """

    def test_present_inputs_make_a_stale_blocker_not_blocked(self, tmp_path: Path):
        _prepare(tmp_path)
        names = ("candidate_exposure_roads.geojson", "candidate_exposure_settlements.geojson")
        write_blocker(tmp_path / M65_DIR, M65_BLOCKER, _m65_blocker_payload(names))
        _write_m64_inputs(tmp_path, names)

        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        disruption = workflow.stage("m6_5_disruption")

        assert disruption is not None
        assert disruption.state is StageState.IMPLEMENTED_NOT_EXECUTED
        assert "stale" in disruption.detail
        # It must not claim the stage ran, and must not claim it is blocked.
        assert disruption.state is not StageState.REAL_DATA_COMPLETE
        assert workflow.any_blocked() is False

    def test_one_still_missing_input_keeps_the_blocker_current(self, tmp_path: Path):
        _prepare(tmp_path)
        names = ("candidate_exposure_roads.geojson", "candidate_exposure_settlements.geojson")
        write_blocker(tmp_path / M65_DIR, M65_BLOCKER, _m65_blocker_payload(names))
        _write_m64_inputs(tmp_path, names[:1])

        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        disruption = workflow.stage("m6_5_disruption")

        assert disruption is not None
        assert disruption.state is StageState.BLOCKED
        assert "not present on disk" in disruption.detail

    def test_a_blocker_without_recorded_inputs_stays_blocked(self, tmp_path: Path):
        """Staleness cannot be proven from an empty list, so nothing is assumed."""
        _prepare(tmp_path)
        payload = _m65_blocker_payload(())
        write_blocker(tmp_path / M65_DIR, M65_BLOCKER, payload)

        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        disruption = workflow.stage("m6_5_disruption")

        assert disruption is not None
        assert disruption.state is StageState.BLOCKED

    def test_the_stale_reason_is_kept_as_history(self, tmp_path: Path):
        _prepare(tmp_path)
        names = ("candidate_exposure_roads.geojson",)
        write_blocker(tmp_path / M65_DIR, M65_BLOCKER, _m65_blocker_payload(names))
        _write_m64_inputs(tmp_path, names)

        workflow = build_workflow(discover_m63_artifacts(tmp_path), project_root=tmp_path)
        disruption = workflow.stage("m6_5_disruption")

        assert disruption is not None
        assert disruption.blocker is not None
        # The original sentence is still available, attached to the report itself.
        assert "not present on disk" in (disruption.blocker.reason or "")
        # ... but it is not what the stage's current detail says.
        assert "not present on disk" not in disruption.detail


class TestBlockerPrerequisitesPresent:
    def _blocker(self, searched, missing):
        return BlockerReport(
            stage="fixture",
            path=Path("fixture_blocker.json"),
            state=BlockerState.LOADED,
            status="blocked",
            missing=list(missing),
            searched=searched,
        )

    def test_relative_searched_directory_resolves_against_the_project_root(self, tmp_path: Path):
        directory = tmp_path / "artifacts" / "stage"
        directory.mkdir(parents=True)
        (directory / "a.geojson").write_text("{}", encoding="utf-8")
        blocker = self._blocker("artifacts/stage", ["a.geojson"])

        assert blocker_prerequisites_present(blocker, project_root=tmp_path) is True

    def test_an_absolute_searched_directory_is_used_as_given(self, tmp_path: Path):
        directory = tmp_path / "elsewhere"
        directory.mkdir(parents=True)
        (directory / "a.geojson").write_text("{}", encoding="utf-8")
        blocker = self._blocker(str(directory), ["a.geojson"])

        assert blocker_prerequisites_present(blocker, project_root=tmp_path / "unused") is True

    def test_missing_absolute_path_entry_is_checked_directly(self, tmp_path: Path):
        target = tmp_path / "a.geojson"
        target.write_text("{}", encoding="utf-8")
        blocker = self._blocker(None, [str(target)])

        assert blocker_prerequisites_present(blocker, project_root=tmp_path) is True

    def test_no_recorded_inputs_is_not_treated_as_present(self, tmp_path: Path):
        assert (
            blocker_prerequisites_present(self._blocker(None, []), project_root=tmp_path) is False
        )

    def test_blank_names_are_ignored_rather_than_resolving_to_the_directory(self, tmp_path: Path):
        assert (
            blocker_prerequisites_present(self._blocker(None, ["   "]), project_root=tmp_path)
            is False
        )
