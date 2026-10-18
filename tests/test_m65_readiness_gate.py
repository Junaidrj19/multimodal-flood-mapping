"""M6.5 readiness-gate tests.

These cover the gate's *decisions*, not the science behind them: status
reconciliation against a stale blocker, the three independent verdicts, each
declared limitation, and the §8 claim screen. Fixtures are hand-written JSON, so
every assertion is checkable by reading the fixture.

What these tests deliberately do **not** establish: that M6.5's numbers are
correct, or that its artifacts are independently reproducible. A passing suite
here is evidence about the gate, not about the flood analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from floodmap.dashboard.readiness import (
    AOI_COVERAGE,
    GEOLOCATION_ACCEPTANCE,
    M65_SCENARIO_STATUS,
    M65_STAGE_PREFIX,
    M65_REQUIRED_OUTPUTS,
    REQUIRED_PRESENTATION_LABELS,
    AoiCoverage,
    ExecutionVerdict,
    GateDecision,
    GeolocationAcceptance,
    LimitationState,
    ReadinessGateError,
    UseClass,
    authorise_use,
    build_readiness_report,
    declared_limitations,
    reconcile_execution,
    screen_claim,
)
from floodmap.dashboard.workflow import M65_BLOCKER, M65_OUTPUT

SUMMARY_AT = "2026-10-18T07:15:13.281034+00:00"
BLOCKER_AT = "2026-10-17T09:24:48.123371+00:00"


def _summary(
    *,
    stage: str = f"{M65_STAGE_PREFIX}-0.1.0",
    status: str = M65_SCENARIO_STATUS,
    generated_at: str = SUMMARY_AT,
    hubs_assessed: bool = True,
    aoi_status: str = "UNVERIFIED",
) -> dict:
    return {
        "stage": stage,
        "status": status,
        "generated_at": generated_at,
        "hub_reachability": {"assessed": hubs_assessed},
        "inputs": {"aoi_provenance_status": aoi_status},
    }


def _blocker(*, generated_at: str = BLOCKER_AT) -> dict:
    return {
        "stage": f"{M65_STAGE_PREFIX}-0.1.0",
        "status": "blocked",
        "generated_at": generated_at,
        "reason": "required M6.4 candidate-exposure artifacts are not present on disk",
        "missing_inputs": ["candidate_exposure_roads.geojson"],
        "searched_directory": "artifacts/m6_4/candidate_infrastructure_exposure",
        "authorisation": {
            "aoi_provenance_status": "UNVERIFIED",
            "authorises_download": False,
        },
        "next_step": ["Obtain explicit human authorisation for pre-event OSM retrieval."],
    }


def _write(directory: Path, *, summary: dict | None, blocker: dict | None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / M65_OUTPUT
    blocker_path = directory / M65_BLOCKER
    if summary is not None:
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        # A summary is execution evidence only when the successful-run output
        # set and its provenance sidecar are present and agree with it.
        for name in M65_REQUIRED_OUTPUTS:
            path = directory / name
            if name == M65_OUTPUT:
                continue
            if name.endswith(".yaml"):
                path.write_text(
                    "\n".join(
                        [
                            "artifact_type: settlement_connectivity",
                            "aoi_provenance_status: UNVERIFIED",
                            "quality:",
                            f"  stage: {summary['stage']}",
                            f"  status: {summary['status']}",
                            f"  generated_at: {summary['generated_at']}",
                            "source_pair:",
                            "  aoi_provenance_status: UNVERIFIED",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                path.write_text(
                    '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
                )
    if blocker is not None:
        blocker_path.write_text(json.dumps(blocker), encoding="utf-8")
    return summary_path, blocker_path


# ---------------------------------------------------------------------------
# Status reconciliation
# ---------------------------------------------------------------------------


class TestExecutionReconciliation:
    def test_summary_newer_than_blocker_marks_blocker_stale(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.EXECUTED_BLOCKER_STALE
        assert record.blocker_is_stale is True
        assert record.executed is True

    def test_stale_blocker_is_not_rewritten(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        before = blocker.read_bytes()
        reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert blocker.read_bytes() == before, "the blocker must be preserved byte-for-byte"

    def test_stale_blocker_reason_is_kept_as_history(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.blocker is not None
        assert "not present on disk" in (record.blocker.reason or "")
        assert str(blocker) in record.evidence

    def test_blocker_newer_than_summary_is_current_not_stale(self, tmp_path: Path):
        summary, blocker = _write(
            tmp_path,
            summary=_summary(generated_at="2026-10-16T00:00:00+00:00"),
            blocker=_blocker(generated_at="2026-10-17T00:00:00+00:00"),
        )
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.BLOCKED_CURRENT
        assert record.blocker_is_stale is False
        assert record.executed is False

    def test_summary_without_blocker_is_executed(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=None)
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.EXECUTED_NO_BLOCKER
        assert record.executed is True

    def test_blocker_without_summary_is_blocked(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=None, blocker=_blocker())
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.BLOCKED_CURRENT
        assert record.executed is False

    def test_nothing_on_disk_is_not_executed(self, tmp_path: Path):
        record = reconcile_execution(
            summary_path=tmp_path / M65_OUTPUT, blocker_path=tmp_path / M65_BLOCKER
        )
        assert record.verdict is ExecutionVerdict.NOT_EXECUTED
        assert record.executed is False

    def test_hub_assessment_flag_is_carried(self, tmp_path: Path):
        summary, blocker = _write(
            tmp_path, summary=_summary(hubs_assessed=False), blocker=None
        )
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.hub_reachability_assessed is False


class TestFileExistenceIsNotExecutionEvidence:
    """The specific defect this gate corrects: presence != execution."""

    def test_wrong_stage_tag_is_indeterminate(self, tmp_path: Path):
        summary, blocker = _write(
            tmp_path, summary=_summary(stage="m6.4-candidate-infrastructure-exposure-0.1.0"), blocker=None
        )
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.INDETERMINATE
        assert record.executed is False

    def test_unrecognised_status_is_indeterminate(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(status="complete"), blocker=None)
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.INDETERMINATE
        assert any("status" in problem for problem in record.problems)

    def test_empty_but_present_summary_is_indeterminate(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        summary = tmp_path / M65_OUTPUT
        summary.write_text("", encoding="utf-8")
        record = reconcile_execution(summary_path=summary, blocker_path=tmp_path / M65_BLOCKER)
        assert record.summary_present is True
        assert record.verdict is ExecutionVerdict.INDETERMINATE

    def test_non_object_summary_is_indeterminate(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        summary = tmp_path / M65_OUTPUT
        summary.write_text("[1, 2, 3]", encoding="utf-8")
        record = reconcile_execution(summary_path=summary, blocker_path=tmp_path / M65_BLOCKER)
        assert record.verdict is ExecutionVerdict.INDETERMINATE

    def test_unparseable_timestamps_do_not_silently_pick_a_winner(self, tmp_path: Path):
        summary, blocker = _write(
            tmp_path,
            summary=_summary(generated_at="not-a-timestamp"),
            blocker=_blocker(),
        )
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.INDETERMINATE
        assert record.blocker_is_stale is None

    def test_indeterminate_is_never_treated_as_permission(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(status="complete"), blocker=None)
        report = build_readiness_report(summary_path=summary, blocker_path=blocker)
        for use in UseClass:
            outcome = report.outcome(use)
            assert outcome is not None
            assert outcome.permitted is False


class TestExecutionEvidenceIntegrity:
    """A valid-looking summary cannot stand in for the complete run record."""

    def test_missing_success_outputs_is_indeterminate(self, tmp_path: Path):
        summary = tmp_path / M65_OUTPUT
        summary.write_text(json.dumps(_summary()), encoding="utf-8")
        record = reconcile_execution(
            summary_path=summary, blocker_path=tmp_path / M65_BLOCKER
        )
        assert record.verdict is ExecutionVerdict.INDETERMINATE
        assert record.evidence_integrity is not None
        assert set(record.evidence_integrity.missing_outputs) == {
            name for name in M65_REQUIRED_OUTPUTS if name != M65_OUTPUT
        }

    def test_provenance_disagreement_invalidates_execution_evidence(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=None)
        provenance = tmp_path / "candidate_disruption_provenance.yaml"
        provenance.write_text(
            provenance.read_text(encoding="utf-8").replace(
                "status: hypothetical_closure_scenario",
                "status: not_a_scenario",
            ),
            encoding="utf-8",
        )
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.INDETERMINATE
        assert record.evidence_integrity is not None
        assert "status values disagree" in " ".join(record.evidence_integrity.problems)

    def test_nonstandard_missing_name_is_warning_not_a_number(self, tmp_path: Path):
        summary_payload = _summary()
        summary_payload["scenario_unreachable"] = [{"name": float("nan")}]
        summary, blocker = _write(tmp_path, summary=summary_payload, blocker=None)
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        assert record.verdict is ExecutionVerdict.EXECUTED_NO_BLOCKER
        assert record.evidence_integrity is not None
        assert any("NaN" in warning for warning in record.evidence_integrity.warnings)


# ---------------------------------------------------------------------------
# The three verdicts
# ---------------------------------------------------------------------------


class TestThreeIndependentVerdicts:
    @pytest.fixture()
    def report(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        return build_readiness_report(summary_path=summary, blocker_path=blocker)

    def test_all_three_use_classes_are_reported(self, report):
        assert {item.use for item in report.outcomes} == set(UseClass)

    def test_verdicts_are_not_collapsed_into_one_value(self, report):
        decisions = {item.use: item.decision for item in report.outcomes}
        assert decisions[UseClass.STAGE_COMPLETION] is GateDecision.COMPLETE_FOR_STATED_SCOPE
        assert decisions[UseClass.RESEARCH_SCENARIO] is GateDecision.PERMITTED_WITH_LABELS
        assert decisions[UseClass.OPERATIONAL] is GateDecision.BLOCKED
        assert len(set(decisions.values())) == 3, "three questions, three distinct answers"

    def test_stage_completion_is_scoped_not_a_fitness_claim(self, report):
        outcome = report.outcome(UseClass.STAGE_COMPLETION)
        assert any("stated scope" in reason for reason in outcome.reasons)
        assert any("not a statement" in reason for reason in outcome.reasons)

    def test_research_use_is_allowed(self, report):
        outcome = authorise_use(report, UseClass.RESEARCH_SCENARIO)
        assert outcome.decision is GateDecision.PERMITTED_WITH_LABELS
        assert outcome.required_labels == list(REQUIRED_PRESENTATION_LABELS)

    def test_operational_use_is_blocked_and_raises(self, report):
        with pytest.raises(ReadinessGateError, match="operational_use"):
            authorise_use(report, UseClass.OPERATIONAL)

    def test_operational_blockers_name_all_three_open_blockers(self, report):
        blockers = " ".join(report.open_blockers(UseClass.OPERATIONAL)).lower()
        assert "geolocation" in blockers
        assert "coverage" in blockers
        assert "provenance" in blockers

    def test_operationally_blocked_property(self, report):
        assert report.operationally_blocked is True

    def test_completion_does_not_unblock_operational_use(self, report):
        assert (
            report.outcome(UseClass.STAGE_COMPLETION).decision
            is GateDecision.COMPLETE_FOR_STATED_SCOPE
        )
        assert report.outcome(UseClass.OPERATIONAL).permitted is False


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------


class TestDeclaredLimitations:
    @pytest.fixture()
    def report(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        return build_readiness_report(summary_path=summary, blocker_path=blocker)

    def test_all_five_limitations_are_reported(self, report):
        assert {item.limitation_id for item in report.limitations} == {
            "absolute_geolocation",
            "aoi_coverage",
            "aoi_provenance",
            "evidence_semantics",
            "reproducibility",
        }

    def test_geolocation_is_unresolved_with_measured_failure(self, report):
        limitation = report.limitation("absolute_geolocation")
        assert limitation.state is LimitationState.UNRESOLVED
        assert "4.472" in limitation.detail
        assert "0.237917" in limitation.detail

    def test_geolocation_acceptance_fails_both_criteria(self):
        assert GEOLOCATION_ACCEPTANCE.shift_passes is False
        assert GEOLOCATION_ACCEPTANCE.correlation_passes is False
        assert GEOLOCATION_ACCEPTANCE.passes is False

    def test_datum_investigation_does_not_establish_geolocation(self, report):
        detail = report.limitation("absolute_geolocation").detail
        assert "did not establish validated absolute geolocation" in detail

    def test_passing_geolocation_would_still_not_authorise_operations(self, tmp_path: Path):
        """Even a hypothetical pass leaves coverage and provenance blocking."""
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        report = build_readiness_report(
            summary_path=summary,
            blocker_path=blocker,
            geolocation=GeolocationAcceptance(
                max_shift_px=2.0,
                measured_shift_px=1.0,
                min_correlation=0.5,
                measured_correlation=0.9,
                evidence=("hypothetical",),
            ),
        )
        assert report.outcome(UseClass.OPERATIONAL).permitted is False

    def test_coverage_is_partial_with_exact_fractions(self, report):
        limitation = report.limitation("aoi_coverage")
        assert limitation.state is LimitationState.PARTIAL
        assert "63.3719" in limitation.detail
        assert "36.6281" in limitation.detail

    def test_unobserved_fraction_is_never_hidden(self, report):
        assert any("36.6281" in label for label in report.required_labels)

    def test_coverage_forbids_calling_unobserved_area_safe(self, report):
        detail = report.limitation("aoi_coverage").detail.lower()
        for word in ("dry", "safe", "unaffected", "flood-free", "connected"):
            assert word in detail
        assert "must never be classified" in detail

    def test_coverage_fractions_sum_to_the_extent(self):
        total = AOI_COVERAGE.delivered_fraction + AOI_COVERAGE.unobserved_fraction
        assert total == pytest.approx(1.0, abs=1e-9)
        assert AOI_COVERAGE.is_complete is False

    def test_aoi_provenance_is_unverified(self, report):
        limitation = report.limitation("aoi_provenance")
        assert limitation.state is LimitationState.NOT_ESTABLISHED
        assert "UNVERIFIED" in limitation.detail
        assert "not a verified official competition boundary" in limitation.detail

    def test_evidence_semantics_separates_candidate_from_confirmed(self, report):
        detail = report.limitation("evidence_semantics").detail
        assert "not confirmed flood extent" in detail
        assert "hypothetical scenarios" in detail
        assert "not validated real-world responder routes" in detail

    def test_reproducibility_distinguishes_prior_run_from_clean_run(self, report):
        limitation = report.limitation("reproducibility")
        assert limitation.state is LimitationState.NOT_INDEPENDENTLY_VERIFIED
        assert "passing test suite" in limitation.detail
        assert "has not been performed" in limitation.detail

    def test_clean_run_reproducibility_is_never_inferred(self, tmp_path: Path):
        """Successful execution must not flip the reproducibility claim."""
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=None)
        report = build_readiness_report(summary_path=summary, blocker_path=blocker)
        assert report.execution.executed is True
        assert (
            report.limitation("reproducibility").state
            is LimitationState.NOT_INDEPENDENTLY_VERIFIED
        )

    def test_limitations_survive_a_successful_execution(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=None)
        report = build_readiness_report(summary_path=summary, blocker_path=blocker)
        assert len(report.limitations) == 5

    def test_every_limitation_blocks_operational_use(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=None)
        record = reconcile_execution(summary_path=summary, blocker_path=blocker)
        for limitation in declared_limitations(execution=record):
            assert UseClass.OPERATIONAL in limitation.blocks

    def test_every_limitation_carries_inspectable_evidence(self, report):
        for limitation in report.limitations:
            assert limitation.evidence, f"{limitation.limitation_id} has no evidence"
            assert limitation.detail.strip()

    def test_evidence_paths_are_not_duplicated(self, report):
        for limitation in report.limitations:
            assert len(limitation.evidence) == len(set(limitation.evidence))


# ---------------------------------------------------------------------------
# Claim screening
# ---------------------------------------------------------------------------


class TestClaimScreen:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The road was destroyed.", "destroyed"),
            ("Confirmed flood extent is shown.", "confirmed flood"),
            ("The village is cut off.", "cut off"),
            ("This area is unaffected.", "unaffected"),
            ("The road is impassable.", "impassable"),
            ("The settlement is isolated.", "isolated"),
            ("These buildings are undamaged.", "undamaged"),
            ("The hamlet is inaccessible.", "inaccessible"),
            ("This region is flood-free.", "flood-free"),
        ],
    )
    def test_forbidden_claims_are_caught(self, text, expected):
        assert expected in [item.term for item in screen_claim(text)]

    @pytest.mark.parametrize(
        "text",
        [
            "The settlement is potentially cut off.",
            "Roads are potentially inaccessible.",
            "The road is potentially disrupted.",
            "SCENARIO_UNREACHABLE under the stated closure rule.",
            "UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED",
            "This is not confirmed damage.",
            "No road is impassable.",
            "Unobserved area is not dry, safe, unaffected or connected.",
            "Candidate exposure, never confirmed damage.",
            "",
        ],
    )
    def test_permitted_phrasing_is_not_flagged(self, text):
        assert screen_claim(text) == []

    def test_negation_does_not_leak_across_a_sentence_boundary(self):
        text = "No data exists. The road was destroyed."
        assert [item.term for item in screen_claim(text)] == ["destroyed"]

    def test_violation_names_the_permitted_alternative(self):
        violation = screen_claim("The road was destroyed.")[0]
        assert violation.permitted_instead == "candidate-exposed"
        assert violation.position >= 0

    def test_multiple_violations_are_reported_in_position_order(self):
        violations = screen_claim("The road was destroyed. The village is cut off.")
        assert [item.term for item in violations] == ["destroyed", "cut off"]

    def test_screen_is_case_insensitive(self):
        assert screen_claim("The road was DESTROYED.")

    def test_substring_matches_do_not_false_positive(self):
        # "isolated" must not fire on "isolation", nor "cut off" on "cutoff".
        assert screen_claim("Thermal isolation of the sensor. A cutoff frequency.") == []

    def test_the_gates_own_labels_pass_its_own_screen(self):
        for label in REQUIRED_PRESENTATION_LABELS:
            assert screen_claim(label) == [], f"gate label violates its own rule: {label}"

    def test_the_gates_own_limitation_text_passes_its_own_screen(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        report = build_readiness_report(summary_path=summary, blocker_path=blocker)
        for limitation in report.limitations:
            assert screen_claim(limitation.detail) == [], limitation.limitation_id
        for outcome in report.outcomes:
            for text in list(outcome.reasons) + list(outcome.blockers):
                assert screen_claim(text) == [], text


# ---------------------------------------------------------------------------
# Presentation labels and serialisation
# ---------------------------------------------------------------------------


class TestPresentationLabels:
    @pytest.fixture()
    def report(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        return build_readiness_report(summary_path=summary, blocker_path=blocker)

    def test_labels_cover_every_required_distinction(self, report):
        joined = " ".join(report.required_labels).lower()
        assert "candidate" in joined
        assert "hypothetical" in joined
        assert "unobserved" in joined
        assert "unverified" in joined
        assert "geolocation" in joined

    def test_labels_refuse_the_confirmed_damage_reading(self, report):
        joined = " ".join(report.required_labels)
        assert "not confirmed flood extent or infrastructure damage" in joined

    def test_labels_refuse_the_confirmed_accessibility_reading(self, report):
        joined = " ".join(report.required_labels)
        assert "not validated responder routes" in joined

    def test_research_outcome_carries_the_labels(self, report):
        outcome = report.outcome(UseClass.RESEARCH_SCENARIO)
        assert outcome.required_labels == list(REQUIRED_PRESENTATION_LABELS)

    def test_blocked_research_use_does_not_hand_out_labels(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=None, blocker=_blocker())
        report = build_readiness_report(summary_path=summary, blocker_path=blocker)
        assert report.outcome(UseClass.RESEARCH_SCENARIO).required_labels == []


class TestSerialisation:
    @pytest.fixture()
    def payload(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        return build_readiness_report(summary_path=summary, blocker_path=blocker).to_dict()

    def test_record_is_json_serialisable(self, payload):
        json.dumps(payload)

    def test_record_keeps_the_three_verdicts(self, payload):
        assert {item["use"] for item in payload["outcomes"]} == {
            use.value for use in UseClass
        }

    def test_record_states_geolocation_failure(self, payload):
        assert payload["geolocation_acceptance"]["passes"] is False

    def test_record_states_partial_coverage(self, payload):
        assert payload["aoi_coverage"]["is_complete"] is False
        assert payload["aoi_coverage"]["unobserved_fraction"] > 0.36

    def test_record_names_the_stale_blocker_without_replacing_it(self, payload):
        assert payload["execution"]["blocker_is_stale"] is True
        assert payload["execution"]["blocker_path"].endswith(M65_BLOCKER)

    def test_record_carries_all_labels(self, payload):
        assert payload["required_presentation_labels"] == list(REQUIRED_PRESENTATION_LABELS)


# ---------------------------------------------------------------------------
# The gate cannot be talked into authorising operations
# ---------------------------------------------------------------------------


class TestOperationalUseCannotBePromoted:
    def test_no_combination_of_evidence_authorises_operations(self, tmp_path: Path):
        """Clean geolocation, full coverage and a claimed clean run still block.

        AOI provenance and evidence semantics are unconditional: a scenario
        output does not become an operational recommendation because the other
        numbers improved.
        """
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=None)
        report = build_readiness_report(
            summary_path=summary,
            blocker_path=blocker,
            geolocation=GeolocationAcceptance(
                max_shift_px=2.0,
                measured_shift_px=0.1,
                min_correlation=0.5,
                measured_correlation=0.99,
                evidence=("hypothetical",),
            ),
            coverage=AoiCoverage(
                delivered_fraction=1.0,
                unobserved_fraction=0.0,
                delivered_area_km2=206.15,
                unobserved_area_km2=0.0,
                evidence=("hypothetical",),
            ),
            clean_run_reproduced=True,
        )
        assert report.outcome(UseClass.OPERATIONAL).permitted is False
        with pytest.raises(ReadinessGateError):
            authorise_use(report, UseClass.OPERATIONAL)

    def test_blockers_are_never_suppressed_to_pass(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        report = build_readiness_report(summary_path=summary, blocker_path=blocker)
        assert len(report.open_blockers(UseClass.OPERATIONAL)) == 5

    def test_authorise_use_rejects_an_unknown_verdict(self, tmp_path: Path):
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        report = build_readiness_report(summary_path=summary, blocker_path=blocker)
        report.outcomes = [
            item for item in report.outcomes if item.use is not UseClass.OPERATIONAL
        ]
        with pytest.raises(ReadinessGateError, match="no verdict"):
            authorise_use(report, UseClass.OPERATIONAL)

    def test_no_invented_fields_in_the_record(self, tmp_path: Path):
        """The gate must not fabricate people, damage labels or recommendations.

        Only the record payload is screened, not the ``tmp_path`` strings inside
        it: pytest derives its temp directory from the test name, so checking
        the whole JSON would match this test's own path.
        """
        summary, blocker = _write(tmp_path, summary=_summary(), blocker=_blocker())
        payload = build_readiness_report(
            summary_path=summary, blocker_path=blocker
        ).to_dict()
        payload["execution"].pop("summary_path", None)
        payload["execution"].pop("blocker_path", None)
        payload["execution"].pop("evidence", None)
        for item in payload["limitations"]:
            item.pop("evidence", None)
        text = json.dumps(payload).lower()
        for invented in (
            "people_needing_help",
            "population",
            "damage_label",
            "responder_recommendation",
            "casualt",
        ):
            assert invented not in text
