"""M6.5 readiness gate: what is complete, what may be used, and for what.

Why this module exists
----------------------
``workflow.py`` answers one question — *did a stage run?* — and it answers it
from the presence of an output file (``_output_stage_status``). That is the right
question for a pipeline view, but it is not sufficient to authorise anything. A
stage can have run, produced internally consistent artifacts and passed its
tests, and still be unfit for operational use, because fitness depends on
evidence that lives *outside* the stage: whether absolute geolocation was ever
validated, how much of the AOI was observed at all, and whether the AOI boundary
itself has established provenance.

Collapsing those into one green status is the specific failure mode
``AGENTS.md`` §8, §16 and §17 exist to prevent. So this module reports three
separate verdicts and refuses to reduce them to a Boolean:

======================  ====================================================
:attr:`UseClass`        Question
======================  ====================================================
``STAGE_COMPLETION``    Is M6.5 complete *for its own stated scope*?
``RESEARCH_SCENARIO``   May its outputs be used for labelled scenario work?
``OPERATIONAL``         Is downstream operational use authorised?
======================  ====================================================

For the current evidence the answers are, respectively: yes; yes with mandatory
labels; **no**. The third stays blocked while any of the three operational
blockers is open, and :func:`authorise_use` raises rather than returning a value
a caller could ignore.

Relationship to the stale blocker
---------------------------------
``candidate_disruption_blocker.json`` records a real refusal from
2026-10-17. The stage has since run. This module does not rewrite that file —
``AGENTS.md`` §22 wants the audit trail — it *reconciles* it, by comparing
``generated_at`` timestamps and reporting the blocker as superseded history
rather than current state (:func:`reconcile_execution`). It also does not accept
mere file existence as execution evidence: the summary must parse, carry the
expected stage tag, and declare a recognised status.

This module runs no science. It reads artifacts and declared evidence and
decides what may be claimed about them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .realdata import resolve_project_root
from .workflow import (
    M65_BLOCKER,
    M65_DIR,
    M65_OUTPUT,
    BlockerReport,
    BlockerState,
    load_blocker_report,
)

__all__ = [
    "ALLOWED_PHRASES",
    "AOI_COVERAGE",
    "FORBIDDEN_CLAIMS",
    "GEOLOCATION_ACCEPTANCE",
    "M65_STAGE_PREFIX",
    "M65_SCENARIO_STATUS",
    "M65_REQUIRED_OUTPUTS",
    "NEGATION_CUES",
    "NEGATION_WINDOW_WORDS",
    "REQUIRED_PRESENTATION_LABELS",
    "AoiCoverage",
    "ClaimViolation",
    "ExecutionReconciliation",
    "ExecutionEvidenceIntegrity",
    "ExecutionVerdict",
    "GateDecision",
    "GateOutcome",
    "GeolocationAcceptance",
    "Limitation",
    "LimitationState",
    "ReadinessGateError",
    "ReadinessReport",
    "UseClass",
    "authorise_use",
    "build_readiness_report",
    "declared_limitations",
    "reconcile_execution",
    "screen_claim",
    "validate_execution_evidence",
]

#: The stage tag M6.5 artifacts must carry. Checked as a prefix so a patch
#: release still matches, but a different stage never does.
M65_STAGE_PREFIX = "m6.5-candidate-network-disruption"

#: The only ``status`` an executed M6.5 summary may declare. The stage writes
#: this string precisely because its outputs are a stated what-if; a summary
#: claiming anything else is not a document this gate knows how to authorise.
M65_SCENARIO_STATUS = "hypothetical_closure_scenario"

# These are the five files a successful M6.5 run writes.  The blocker and the
# readiness execution record are deliberately not in this set: they are audit
# records, not successful-run outputs, and the historical blocker must remain
# untouched when a later run succeeds.
M65_REQUIRED_OUTPUTS: Tuple[str, ...] = (
    "candidate_disruption_pieces.geojson",
    "candidate_disruption_settlements.geojson",
    "candidate_disruption_routes.geojson",
    "candidate_disruption_summary.json",
    "candidate_disruption_provenance.yaml",
)


class ReadinessGateError(RuntimeError):
    """Raised when a caller requests a use the gate does not authorise."""


class UseClass(str, Enum):
    """The three independent questions this gate answers."""

    #: Does M6.5 have implementation, test, provenance and execution evidence
    #: sufficient to be reported as completed for its stated scope?
    STAGE_COMPLETION = "stage_completion"
    #: May the outputs be used for explicitly labelled candidate/scenario work?
    RESEARCH_SCENARIO = "research_scenario_use"
    #: Is downstream operational / responder-facing use authorised?
    OPERATIONAL = "operational_use"


class GateDecision(str, Enum):
    """The verdict for one :class:`UseClass`."""

    #: Complete for the stage's own stated scope. Not a fitness claim.
    COMPLETE_FOR_STATED_SCOPE = "complete_for_stated_scope"
    #: Permitted, but only carrying :data:`REQUIRED_PRESENTATION_LABELS`.
    PERMITTED_WITH_LABELS = "permitted_with_labels"
    #: Refused, with named blockers.
    BLOCKED = "blocked"
    #: Evidence was missing or unreadable, so no verdict was reached. Never
    #: treated as permission.
    INDETERMINATE = "indeterminate"


class LimitationState(str, Enum):
    """How a declared limitation currently stands."""

    #: Measured and failing a preregistered acceptance criterion.
    UNRESOLVED = "unresolved"
    #: Quantified and incomplete, e.g. partial observation coverage.
    PARTIAL = "partial"
    #: Nobody has established it; absence of evidence, not evidence of absence.
    NOT_ESTABLISHED = "not_established"
    #: Working as designed, but the output is a stated assumption, not a
    #: measurement, and must be labelled as one.
    STATED_SCENARIO = "stated_scenario"
    #: Prior execution and passing tests exist; an independent clean-run
    #: reproduction does not.
    NOT_INDEPENDENTLY_VERIFIED = "not_independently_verified"


class ExecutionVerdict(str, Enum):
    """What the artifacts say about whether M6.5 actually ran."""

    #: Valid summary present; blocker present but older, so superseded.
    EXECUTED_BLOCKER_STALE = "executed_blocker_stale"
    #: Valid summary present, no blocker on disk.
    EXECUTED_NO_BLOCKER = "executed_no_blocker"
    #: Blocker is current — newer than any summary, or no summary exists.
    BLOCKED_CURRENT = "blocked_current"
    #: Neither a valid summary nor a blocker.
    NOT_EXECUTED = "not_executed"
    #: A summary exists but could not be validated. Not execution evidence.
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class GeolocationAcceptance:
    """The preregistered M6.20 absolute-geolocation criteria and what was measured.

    Declared here as data rather than read from ``configs/rtc_reprocessing.yaml``
    so that the gate cannot be silently unblocked by editing a config, and so
    that it keeps working in a checkout where that config is absent. The
    ``evidence`` paths are where the numbers come from and must be checkable by
    hand.
    """

    max_shift_px: float
    measured_shift_px: float
    min_correlation: float
    measured_correlation: float
    evidence: Tuple[str, ...]

    @property
    def shift_passes(self) -> bool:
        return self.measured_shift_px <= self.max_shift_px

    @property
    def correlation_passes(self) -> bool:
        return self.measured_correlation >= self.min_correlation

    @property
    def passes(self) -> bool:
        """Both criteria. Absolute geolocation is validated only if both hold."""
        return self.shift_passes and self.correlation_passes

    def failure_text(self) -> str:
        return (
            f"geolocation shift {self.measured_shift_px:g} px against a maximum of "
            f"{self.max_shift_px:g} px; correlation {self.measured_correlation:g} "
            f"against a minimum of {self.min_correlation:g}"
        )


#: M6.20's two criteria, both failed, unchanged by the M6.21 investigation. The
#: investigation established a *cause* (vertical-datum convention) — it did not
#: establish validated absolute geolocation, and does not re-score these.
GEOLOCATION_ACCEPTANCE = GeolocationAcceptance(
    max_shift_px=2.0,
    measured_shift_px=4.472,
    min_correlation=0.5,
    measured_correlation=0.237917,
    evidence=(
        "docs/m6.20-grd-rtc-readiness.md",
        "docs/m6.21-geolocation-investigation.md",
        "configs/rtc_reprocessing.yaml",
    ),
)


@dataclass(frozen=True)
class AoiCoverage:
    """How much of the adopted AOI the delivered RTC window actually covers."""

    delivered_fraction: float
    unobserved_fraction: float
    delivered_area_km2: float
    unobserved_area_km2: float
    evidence: Tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return self.unobserved_fraction <= 0.0

    def coverage_text(self) -> str:
        return (
            f"The delivered RTC window covers {100 * self.delivered_fraction:.4f}% of the "
            f"adopted AOI; {100 * self.unobserved_fraction:.4f}% is unobserved"
        )


#: Measured by ``floodmap.hydrology.coverage`` and recorded in the M6 release
#: notes. ``delivered + unobserved`` equals the extent exactly.
AOI_COVERAGE = AoiCoverage(
    delivered_fraction=0.633719232,
    unobserved_fraction=0.366280768,
    delivered_area_km2=130.641632,
    unobserved_area_km2=75.509019,
    evidence=(
        "docs/m6-release-notes.md",
        "docs/data-contract.md",
        "src/floodmap/hydrology/coverage.py",
    ),
)

#: Labels any dashboard, report or downstream consumer must carry when it
#: presents M6.5 output. Stated once so a UI cannot paraphrase them away.
REQUIRED_PRESENTATION_LABELS: Tuple[str, ...] = (
    "SAR-change candidate exposure — not confirmed flood extent or infrastructure damage.",
    "Road closures and connectivity changes are hypothetical scenarios under a stated "
    "closure rule — not observed impassability and not validated responder routes.",
    f"Partial observation coverage: {100 * AOI_COVERAGE.unobserved_fraction:.4f}% of the AOI "
    "is unobserved. Unobserved area is not dry, safe, unaffected or connected.",
    "AOI provenance is UNVERIFIED: a project-defined analytical corridor, not a verified "
    "official competition boundary.",
    "Absolute geolocation is not validated: the preregistered acceptance criteria failed.",
)

#: Claims that must never be made about M6.4/M6.5 output, mapped to the
#: phrasing ``AGENTS.md`` §8 permits instead. Matched case-insensitively on word
#: boundaries, after the permitted phrasings below have been masked out.
FORBIDDEN_CLAIMS: Mapping[str, str] = {
    "destroyed": "candidate-exposed",
    "structurally failed": "candidate-exposed",
    "confirmed flood": "SAR-change candidate",
    "confirmed damage": "candidate exposure",
    "observed flooding": "SAR-change candidate",
    "impassable": "closed under the stated closure scenario",
    "cut off": "SCENARIO_UNREACHABLE under the stated closure rule",
    "isolated": "potentially cut off",
    "inaccessible": "potentially inaccessible",
    "flood-free": "not observed",
    "unaffected": "not observed",
    "undamaged": "not observed",
}

#: Phrasings that are permitted even though they contain a forbidden substring.
#: Masked out before matching, so the hedged §8 vocabulary and the M6.5 status
#: names do not register as violations.
ALLOWED_PHRASES: Tuple[str, ...] = (
    "potentially inaccessible",
    "potentially disrupted",
    "potentially cut off",
    "potentially unaffected",
    "may be cut off",
    "SCENARIO_UNREACHABLE",
    "UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED",
)

#: Words that, appearing shortly before a forbidden term, make it a *denial* of
#: the claim rather than the claim itself. "not confirmed flood extent" and
#: "must never be classified as unaffected" are the phrasings ``AGENTS.md`` §8
#: actually wants, so flagging them would make this screen unusable and invite
#: callers to bypass it.
#:
#: This is a lexical heuristic over a short window, not semantic analysis. It
#: can be defeated deliberately, and a double negative reads as permitted. It is
#: a guard against careless phrasing, not a proof of correct phrasing.
NEGATION_CUES: Tuple[str, ...] = (
    "not",
    "never",
    "no",
    "nor",
    "non",
    "cannot",
    "isn't",
    "aren't",
    "wasn't",
    "without",
    "rather",
    "instead",
    "neither",
)

#: How many preceding words :func:`screen_claim` scans for a negation cue. Set
#: wide enough to cover a list-form denial — "must never be classified as dry,
#: safe, unaffected, flood-free or connected" puts the cue six words back — but
#: still bounded, so a negation early in a long sentence does not license an
#: unrelated bare claim at the end of it. A compound sentence that denies one
#: claim and asserts another can still slip through; that is a known limit.
NEGATION_WINDOW_WORDS = 12


@dataclass(frozen=True)
class ClaimViolation:
    """One forbidden claim found in presentation text."""

    term: str
    permitted_instead: str
    position: int

    def __str__(self) -> str:  # pragma: no cover - diagnostic convenience
        return (
            f"{self.term!r} at offset {self.position}: say {self.permitted_instead!r} instead"
        )


@dataclass
class Limitation:
    """One declared limitation, its state, and what it forbids."""

    limitation_id: str
    title: str
    state: LimitationState
    detail: str
    evidence: List[str] = field(default_factory=list)
    #: Which use classes this limitation blocks outright.
    blocks: List[UseClass] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        """Every declared state here is an open limitation, not a resolved one."""
        return True


@dataclass
class ExecutionEvidenceIntegrity:
    """Structural checks on the evidence that a successful M6.5 run exists.

    A summary with a plausible stage tag is not enough on its own: a truncated
    copy, a summary from another stage, or a summary whose provenance disagrees
    with it must not become execution evidence.  This record intentionally does
    not claim numerical validation of the network result; it only binds the
    successful-run output set and its provenance metadata together.
    """

    valid: bool
    required_outputs: Tuple[str, ...] = M65_REQUIRED_OUTPUTS
    missing_outputs: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExecutionReconciliation:
    """Whether M6.5 ran, reconciled against a possibly stale blocker report."""

    verdict: ExecutionVerdict
    detail: str
    summary_path: Path
    blocker_path: Path
    summary_present: bool = False
    summary_status: Optional[str] = None
    summary_stage: Optional[str] = None
    summary_generated_at: Optional[str] = None
    blocker_state: BlockerState = BlockerState.MISSING
    blocker_status: Optional[str] = None
    blocker_generated_at: Optional[str] = None
    blocker_is_stale: Optional[bool] = None
    hub_reachability_assessed: Optional[bool] = None
    evidence_integrity: Optional[ExecutionEvidenceIntegrity] = None
    problems: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    blocker: Optional[BlockerReport] = None

    @property
    def executed(self) -> bool:
        return self.verdict in (
            ExecutionVerdict.EXECUTED_BLOCKER_STALE,
            ExecutionVerdict.EXECUTED_NO_BLOCKER,
        )


@dataclass
class GateOutcome:
    """The verdict for one use class, with the reasons behind it."""

    use: UseClass
    decision: GateDecision
    reasons: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    required_labels: List[str] = field(default_factory=list)

    @property
    def permitted(self) -> bool:
        return self.decision in (
            GateDecision.COMPLETE_FOR_STATED_SCOPE,
            GateDecision.PERMITTED_WITH_LABELS,
        )


@dataclass
class ReadinessReport:
    """The full gate: execution reconciliation, limitations, and three verdicts."""

    stage: str
    execution: ExecutionReconciliation
    limitations: List[Limitation] = field(default_factory=list)
    outcomes: List[GateOutcome] = field(default_factory=list)
    required_labels: List[str] = field(
        default_factory=lambda: list(REQUIRED_PRESENTATION_LABELS)
    )

    def outcome(self, use: UseClass) -> Optional[GateOutcome]:
        for item in self.outcomes:
            if item.use is use:
                return item
        return None

    def limitation(self, limitation_id: str) -> Optional[Limitation]:
        for item in self.limitations:
            if item.limitation_id == limitation_id:
                return item
        return None

    def open_blockers(self, use: UseClass) -> List[str]:
        found = self.outcome(use)
        return list(found.blockers) if found else []

    @property
    def operationally_blocked(self) -> bool:
        found = self.outcome(UseClass.OPERATIONAL)
        return found is None or found.decision is not GateDecision.PERMITTED_WITH_LABELS

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable record, for the execution-record artifact."""

        def _plain(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: _plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [_plain(item) for item in value]
            return value

        execution = {
            key: _plain(value)
            for key, value in asdict(self.execution).items()
            if key != "blocker"
        }
        return {
            "stage": self.stage,
            "gate": "m6.5-readiness-gate",
            "execution": execution,
            "limitations": [_plain(asdict(item)) for item in self.limitations],
            "outcomes": [_plain(asdict(item)) for item in self.outcomes],
            "required_presentation_labels": list(self.required_labels),
            "geolocation_acceptance": {
                **_plain(asdict(GEOLOCATION_ACCEPTANCE)),
                "passes": GEOLOCATION_ACCEPTANCE.passes,
            },
            "aoi_coverage": {
                **_plain(asdict(AOI_COVERAGE)),
                "is_complete": AOI_COVERAGE.is_complete,
            },
        }


# ---------------------------------------------------------------------------
# Execution reconciliation
# ---------------------------------------------------------------------------


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 stamp, tolerating a trailing ``Z``."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed


def _load_json_object(path: Path) -> Tuple[Any, List[str]]:
    """Load JSON while making non-standard constants visible to the caller.

    Python's JSON decoder accepts ``NaN`` and infinities by default even though
    they are not JSON values.  Existing generated summaries contain a few
    missing OSM names represented this way.  They are retained as ``None`` in
    the in-memory evidence view and surfaced as a warning; they are never
    silently treated as a measured number.
    """
    constants: List[str] = []

    def _constant(value: str) -> None:
        constants.append(value)
        return None

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_constant), constants


def validate_execution_evidence(
    *,
    summary_path: Path,
    summary: Mapping[str, Any],
    output_dir: Optional[Path] = None,
) -> ExecutionEvidenceIntegrity:
    """Validate the successful-run companion files without running the stage.

    This is deliberately a structural gate.  It checks that all five outputs
    exist and that the YAML provenance record agrees with the summary's stage,
    status, timestamp and AOI provenance declaration.  It does not validate
    the scientific correctness of the network numbers and it does not hash
    large payloads; :mod:`scripts.validate_m65_publication` is the separate,
    read-only checksum/manifest check.
    """
    summary_path = Path(summary_path)
    directory = Path(output_dir) if output_dir is not None else summary_path.parent
    missing = [name for name in M65_REQUIRED_OUTPUTS if not (directory / name).is_file()]
    problems: List[str] = []
    warnings: List[str] = []
    if missing:
        problems.append("successful-run output set is incomplete")

    provenance_path = directory / "candidate_disruption_provenance.yaml"
    provenance: Optional[Dict[str, Any]] = None
    if not missing or provenance_path.is_file():
        try:
            loaded = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"provenance is not valid YAML: {exc}")
        else:
            if isinstance(loaded, dict):
                provenance = loaded
            else:
                problems.append("provenance is not a YAML mapping")

    if provenance is not None:
        quality = provenance.get("quality")
        if not isinstance(quality, dict):
            problems.append("provenance is missing its quality mapping")
        else:
            if quality.get("stage") != summary.get("stage"):
                problems.append("summary and provenance stage tags disagree")
            if quality.get("status") != summary.get("status"):
                problems.append("summary and provenance status values disagree")
            if _parse_timestamp(quality.get("generated_at")) is None:
                problems.append("provenance quality.generated_at is not ISO-8601")

        provenance_inputs = provenance.get("inputs")
        if not isinstance(provenance_inputs, dict):
            provenance_inputs = quality.get("inputs") if isinstance(quality, dict) else None
        if not isinstance(provenance_inputs, dict):
            provenance_inputs = {}
        source_pair = provenance.get("source_pair") or provenance_inputs.get("source_pair")
        summary_inputs = summary.get("inputs")
        if not isinstance(source_pair, dict):
            problems.append("provenance is missing source_pair")
        elif isinstance(summary_inputs, dict):
            summary_aoi = summary_inputs.get("aoi_provenance_status")
            provenance_aoi = source_pair.get("aoi_provenance_status")
            if summary_aoi != provenance_aoi:
                problems.append("summary and provenance AOI provenance statuses disagree")

        provenance_aoi = provenance_inputs.get("aoi_provenance_status")
        if provenance_aoi is None:
            provenance_aoi = provenance.get("aoi_provenance_status")
        if provenance_aoi != (
            source_pair.get("aoi_provenance_status") if isinstance(source_pair, dict) else None
        ):
            problems.append("provenance and source-pair AOI statuses disagree")

        if provenance.get("artifact_type") != "settlement_connectivity":
            problems.append("provenance artifact_type is not settlement_connectivity")

    # This is a format warning, not a scientific value.  The separate
    # publication validator reports it so a strict downstream JSON consumer can
    # refuse the raw generated summary rather than silently accepting NaN.
    try:
        _, constants = _load_json_object(summary_path)
    except Exception:
        constants = []
    if constants:
        warnings.append(
            "summary uses non-standard JSON constants: " + ", ".join(sorted(set(constants)))
        )

    return ExecutionEvidenceIntegrity(
        valid=not problems,
        missing_outputs=missing,
        problems=problems,
        warnings=warnings,
    )


def reconcile_execution(
    *,
    summary_path: Path,
    blocker_path: Path,
    stage_prefix: str = M65_STAGE_PREFIX,
) -> ExecutionReconciliation:
    """Decide whether M6.5 ran, from artifact content rather than file existence.

    A summary file is accepted as execution evidence only when it parses as a
    JSON object, carries ``stage_prefix``, and declares
    :data:`M65_SCENARIO_STATUS`. Anything else is ``INDETERMINATE`` — which is
    not permission.

    When both a valid summary and a blocker exist, the newer ``generated_at``
    wins. A blocker that predates the summary is reported stale and kept; a
    blocker that postdates it is current, and the stage is blocked again.
    """
    summary_path = Path(summary_path)
    blocker_path = Path(blocker_path)
    blocker = load_blocker_report(blocker_path, stage="m6.5")

    record = ExecutionReconciliation(
        verdict=ExecutionVerdict.NOT_EXECUTED,
        detail="",
        summary_path=summary_path,
        blocker_path=blocker_path,
        blocker_state=blocker.state,
        blocker_status=blocker.status,
        blocker=blocker,
    )
    if blocker.state is BlockerState.LOADED:
        record.blocker_generated_at = (
            str((blocker.raw or {}).get("generated_at"))
            if (blocker.raw or {}).get("generated_at") is not None
            else None
        )
        record.evidence.append(str(blocker_path))

    raw: Optional[Dict[str, Any]] = None
    if summary_path.is_file():
        record.summary_present = True
        try:
            loaded, constants = _load_json_object(summary_path)
        except Exception as exc:
            record.problems.append(f"summary is not valid JSON: {exc}")
        else:
            if isinstance(loaded, dict):
                raw = loaded
                if constants:
                    record.evidence_integrity = ExecutionEvidenceIntegrity(
                        valid=True,
                        warnings=[
                            "summary uses non-standard JSON constants: "
                            + ", ".join(sorted(set(constants)))
                        ],
                    )
            else:
                record.problems.append("summary is not a JSON object")

    if raw is not None:
        record.summary_stage = str(raw.get("stage")) if raw.get("stage") is not None else None
        record.summary_status = str(raw.get("status")) if raw.get("status") is not None else None
        record.summary_generated_at = (
            str(raw.get("generated_at")) if raw.get("generated_at") is not None else None
        )
        hub = raw.get("hub_reachability")
        if isinstance(hub, dict) and "assessed" in hub:
            record.hub_reachability_assessed = bool(hub["assessed"])
        if not (record.summary_stage or "").startswith(stage_prefix):
            record.problems.append(
                f"summary stage {record.summary_stage!r} is not a {stage_prefix!r} artifact"
            )
        if record.summary_status != M65_SCENARIO_STATUS:
            record.problems.append(
                f"summary status {record.summary_status!r} is not {M65_SCENARIO_STATUS!r}"
            )

    if raw is not None and not record.problems:
        record.evidence_integrity = validate_execution_evidence(
            summary_path=summary_path,
            summary=raw,
        )
        record.problems.extend(record.evidence_integrity.problems)

    valid_summary = raw is not None and not record.problems
    if valid_summary:
        record.evidence.insert(0, str(summary_path))

    # --- classify -----------------------------------------------------------
    if not record.summary_present and blocker.is_blocked:
        record.verdict = ExecutionVerdict.BLOCKED_CURRENT
        record.blocker_is_stale = False
        record.detail = (
            "no summary artifact exists and the blocker report is the current state: "
            f"{blocker.reason or 'blocked on missing prerequisites'}"
        )
        return record

    if not record.summary_present:
        record.verdict = ExecutionVerdict.NOT_EXECUTED
        record.detail = "no summary artifact and no current blocker report"
        return record

    if not valid_summary:
        record.verdict = ExecutionVerdict.INDETERMINATE
        record.detail = (
            "a summary artifact exists but could not be validated as M6.5 execution "
            "evidence: " + "; ".join(record.problems)
        )
        return record

    if not blocker.is_blocked:
        record.verdict = ExecutionVerdict.EXECUTED_NO_BLOCKER
        record.detail = (
            f"M6.5 executed: {summary_path.name} declares status "
            f"{record.summary_status!r} at {record.summary_generated_at}, and no "
            "current blocker report contradicts it"
        )
        return record

    summary_at = _parse_timestamp(record.summary_generated_at)
    blocker_at = _parse_timestamp(record.blocker_generated_at)
    if summary_at is None or blocker_at is None:
        record.verdict = ExecutionVerdict.INDETERMINATE
        record.blocker_is_stale = None
        record.detail = (
            "a valid summary and a blocker both exist, but their timestamps cannot be "
            "compared, so which one is current was not established"
        )
        return record

    if blocker_at > summary_at:
        record.verdict = ExecutionVerdict.BLOCKED_CURRENT
        record.blocker_is_stale = False
        record.detail = (
            f"the blocker report ({record.blocker_generated_at}) postdates the summary "
            f"({record.summary_generated_at}), so the stage was blocked after it last ran"
        )
        return record

    record.verdict = ExecutionVerdict.EXECUTED_BLOCKER_STALE
    record.blocker_is_stale = True
    record.detail = (
        f"M6.5 executed at {record.summary_generated_at} with status "
        f"{record.summary_status!r}. The blocker report at {record.blocker_generated_at} "
        "predates it and is superseded history, not current state; it is kept as audit "
        "evidence and deliberately not rewritten"
    )
    return record


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------


def _dedupe(values: Sequence[str]) -> List[str]:
    """Preserve order, drop repeats. Evidence lists should not repeat a path."""
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def declared_limitations(
    *,
    execution: ExecutionReconciliation,
    geolocation: GeolocationAcceptance = GEOLOCATION_ACCEPTANCE,
    coverage: AoiCoverage = AOI_COVERAGE,
    aoi_provenance_status: Optional[str] = None,
    clean_run_reproduced: bool = False,
) -> List[Limitation]:
    """The five limitations that constrain what M6.5 output may be used for.

    Every one of these is reported whatever the execution state: a stage that
    ran successfully does not retire any of them.
    """
    status_text = (aoi_provenance_status or "UNVERIFIED").upper()
    limitations = [
        Limitation(
            limitation_id="absolute_geolocation",
            title="Absolute geolocation accuracy",
            state=LimitationState.UNRESOLVED,
            detail=(
                "Unresolved. The GRD-to-RTC probe failed both preregistered acceptance "
                f"criteria: {geolocation.failure_text()}. The M6.21 datum-convention "
                "investigation established a measured cause for the discrepancy; it did "
                "not establish validated absolute geolocation, and did not re-score "
                "these criteria."
            ),
            evidence=list(geolocation.evidence),
            blocks=[UseClass.OPERATIONAL],
        ),
        Limitation(
            limitation_id="aoi_coverage",
            title="AOI observation coverage",
            state=LimitationState.PARTIAL,
            detail=(
                f"Partial. {coverage.coverage_text()} "
                f"({coverage.delivered_area_km2:g} km² delivered, "
                f"{coverage.unobserved_area_km2:g} km² unobserved). Unobserved area and "
                "the infrastructure within it must never be classified as dry, safe, "
                "unaffected, flood-free or connected; absence of observation is not a "
                "negative finding."
            ),
            evidence=list(coverage.evidence),
            blocks=[UseClass.OPERATIONAL],
        ),
        Limitation(
            limitation_id="aoi_provenance",
            title="AOI provenance",
            state=LimitationState.NOT_ESTABLISHED,
            detail=(
                f"{status_text}. The HydroRIVERS-derived corridor is a project-defined "
                "analytical AOI, not a verified official competition boundary. Adopting "
                "the geometry did not establish its origin and does not authorise "
                "retrieval, so every count is relative to an extent whose provenance is "
                "not established."
            ),
            evidence=[
                "src/floodmap/acquisition/aoi.py",
                "docs/m6.11-aoi-adoption-review.md",
                "docs/m6.15-aoi-adoption-implementation.md",
            ],
            blocks=[UseClass.OPERATIONAL],
        ),
        Limitation(
            limitation_id="evidence_semantics",
            title="Evidence semantics",
            state=LimitationState.STATED_SCENARIO,
            detail=(
                "M6.3 SAR-change candidates are candidate exposure — rule-based "
                "Sentinel-1 RTC dual-polarisation darkening evidence — not confirmed "
                "flood extent and not infrastructure damage. M6.5 road closures and "
                "connectivity changes are hypothetical scenarios under a stated closure "
                "rule: not observed road impassability, and not validated real-world "
                "responder routes. Presenting either as confirmed is prohibited "
                "(AGENTS.md §8)."
            ),
            evidence=[
                "docs/m6.5-candidate-network-disruption.md",
                "artifacts/m6_5/candidate_network_disruption/candidate_disruption_summary.json",
            ],
            blocks=[UseClass.OPERATIONAL],
        ),
        Limitation(
            limitation_id="reproducibility",
            title="Reproducibility",
            state=(
                LimitationState.NOT_INDEPENDENTLY_VERIFIED
                if not clean_run_reproduced
                else LimitationState.PARTIAL
            ),
            detail=(
                "Prior successful execution and a passing test suite are recorded. "
                "Neither is evidence of independent clean-run artifact "
                "reproducibility, which has not been performed: no from-scratch "
                "re-execution was run and no per-output-file checksum manifest exists "
                "for the M6.5 artifacts to compare against."
                if not clean_run_reproduced
                else
                "A clean-run reproduction was recorded by the caller. Scope and method "
                "of that reproduction must be stated wherever it is relied on."
            ),
            evidence=_dedupe([str(execution.summary_path), *execution.evidence]),
            blocks=[UseClass.OPERATIONAL],
        ),
    ]
    return limitations


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _completion_outcome(execution: ExecutionReconciliation) -> GateOutcome:
    """Whether M6.5 may be reported complete *for its own stated scope*."""
    if execution.executed:
        reasons = [
            execution.detail,
            "Implementation, tests and an ArtifactProvenance record exist for the stage.",
            "Completion is scoped to the stage's own stated scope — a hypothetical "
            "closure scenario — and is not a statement that its outputs are "
            "operationally fit.",
        ]
        if execution.hub_reachability_assessed:
            reasons.append("Hub reachability was assessed, not skipped.")
        return GateOutcome(
            use=UseClass.STAGE_COMPLETION,
            decision=GateDecision.COMPLETE_FOR_STATED_SCOPE,
            reasons=reasons,
        )
    if execution.verdict is ExecutionVerdict.BLOCKED_CURRENT:
        return GateOutcome(
            use=UseClass.STAGE_COMPLETION,
            decision=GateDecision.BLOCKED,
            reasons=[execution.detail],
            blockers=["M6.5 has a current blocker report and no valid execution evidence."],
        )
    if execution.verdict is ExecutionVerdict.INDETERMINATE:
        return GateOutcome(
            use=UseClass.STAGE_COMPLETION,
            decision=GateDecision.INDETERMINATE,
            reasons=[execution.detail],
            blockers=list(execution.problems)
            or ["M6.5 execution evidence could not be validated."],
        )
    return GateOutcome(
        use=UseClass.STAGE_COMPLETION,
        decision=GateDecision.BLOCKED,
        reasons=[execution.detail or "no execution evidence"],
        blockers=["M6.5 has not been executed."],
    )


def _research_outcome(execution: ExecutionReconciliation) -> GateOutcome:
    """Labelled candidate/scenario analysis, which the contract does permit."""
    if not execution.executed:
        return GateOutcome(
            use=UseClass.RESEARCH_SCENARIO,
            decision=(
                GateDecision.INDETERMINATE
                if execution.verdict is ExecutionVerdict.INDETERMINATE
                else GateDecision.BLOCKED
            ),
            reasons=[execution.detail or "no execution evidence"],
            blockers=["No validated M6.5 outputs exist to analyse."],
        )
    return GateOutcome(
        use=UseClass.RESEARCH_SCENARIO,
        decision=GateDecision.PERMITTED_WITH_LABELS,
        reasons=[
            "The stage's own contract permits explicitly labelled candidate-exposure and "
            "scenario-based connectivity analysis.",
            "Threshold sensitivity and coverage uncertainty are reported alongside every "
            "number, so a reader can see what the result depends on.",
        ],
        required_labels=list(REQUIRED_PRESENTATION_LABELS),
    )


def _operational_outcome(
    execution: ExecutionReconciliation, limitations: Sequence[Limitation]
) -> GateOutcome:
    """Operational use, which stays blocked while any limitation blocks it."""
    blockers = [
        f"{item.title}: {item.detail}"
        for item in limitations
        if UseClass.OPERATIONAL in item.blocks
    ]
    reasons = [
        "Operational readiness is not implied by stage completion or by a passing test "
        "suite. It requires evidence this project does not have.",
    ]
    if not execution.executed:
        blockers.insert(0, "M6.5 has no validated execution evidence.")
    if not blockers:
        # Defensive: the declared limitations always block operational use. If a
        # caller supplies a set that does not, say so rather than inventing
        # authorisation.
        return GateOutcome(
            use=UseClass.OPERATIONAL,
            decision=GateDecision.INDETERMINATE,
            reasons=reasons
            + ["No limitation blocked operational use, which the declared set always does."],
            blockers=["Operational authorisation was not established by any evidence."],
        )
    return GateOutcome(
        use=UseClass.OPERATIONAL,
        decision=GateDecision.BLOCKED,
        reasons=reasons,
        blockers=blockers,
    )


def build_readiness_report(
    *,
    project_root: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    blocker_path: Optional[Path] = None,
    geolocation: GeolocationAcceptance = GEOLOCATION_ACCEPTANCE,
    coverage: AoiCoverage = AOI_COVERAGE,
    clean_run_reproduced: bool = False,
) -> ReadinessReport:
    """Build the three-verdict readiness gate from artifacts on disk.

    ``clean_run_reproduced`` defaults to ``False`` and must stay false unless a
    from-scratch reproduction was actually performed. It is a parameter rather
    than a detected fact precisely so that nothing can infer it.
    """
    root = resolve_project_root(project_root)
    m65_dir = root / M65_DIR
    summary = Path(summary_path) if summary_path is not None else m65_dir / M65_OUTPUT
    blocker = Path(blocker_path) if blocker_path is not None else m65_dir / M65_BLOCKER

    execution = reconcile_execution(summary_path=summary, blocker_path=blocker)

    aoi_status: Optional[str] = None
    if execution.blocker is not None and execution.blocker.authorisation:
        aoi_status = execution.blocker.authorisation.get("aoi_provenance_status")
    if aoi_status is None and summary.is_file():
        try:
            loaded = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            inputs = loaded.get("inputs")
            if isinstance(inputs, dict):
                aoi_status = inputs.get("aoi_provenance_status")

    limitations = declared_limitations(
        execution=execution,
        geolocation=geolocation,
        coverage=coverage,
        aoi_provenance_status=aoi_status,
        clean_run_reproduced=clean_run_reproduced,
    )
    outcomes = [
        _completion_outcome(execution),
        _research_outcome(execution),
        _operational_outcome(execution, limitations),
    ]
    return ReadinessReport(
        stage=execution.summary_stage or M65_STAGE_PREFIX,
        execution=execution,
        limitations=limitations,
        outcomes=outcomes,
    )


def authorise_use(report: ReadinessReport, use: UseClass) -> GateOutcome:
    """Return the outcome for ``use``, or raise if the gate does not permit it.

    Raising is deliberate. A downstream consumer that asks for operational
    authorisation should not be able to proceed by ignoring a return value.
    """
    outcome = report.outcome(use)
    if outcome is None:
        raise ReadinessGateError(f"the readiness gate reached no verdict for {use.value!r}")
    if not outcome.permitted:
        detail = "; ".join(outcome.blockers) or outcome.decision.value
        raise ReadinessGateError(
            f"{use.value!r} is not authorised ({outcome.decision.value}): {detail}"
        )
    return outcome


def _is_negated(haystack: str, start: int) -> bool:
    """Whether a forbidden term at ``start`` is preceded by a negation cue.

    Scans the last :data:`NEGATION_WINDOW_WORDS` words before the match. Stops
    at a sentence boundary so a negation in the previous sentence does not
    license a bare claim in this one.
    """
    before = haystack[:start]
    for boundary in (".", ";", ":", "!", "?", "\n"):
        cut = before.rfind(boundary)
        if cut >= 0:
            before = before[cut + 1 :]
    words = re.findall(r"[\w'-]+", before)
    for word in words[-NEGATION_WINDOW_WORDS:]:
        if word in NEGATION_CUES or word.rstrip("-") in NEGATION_CUES:
            return True
    return False


def screen_claim(text: str) -> List[ClaimViolation]:
    """Find forbidden claims in presentation text.

    Guards the §8 vocabulary boundary: a dashboard, report or downstream
    consumer can run its own strings through this and refuse to render a
    sentence that promotes candidate exposure to confirmed damage, a scenario to
    confirmed accessibility, or missing coverage to a negative finding.

    Two kinds of legitimate phrasing are not flagged:

    * :data:`ALLOWED_PHRASES` — hedged §8 vocabulary and the ``SCENARIO_*``
      status names, masked out before matching;
    * negated claims — a term preceded by a :data:`NEGATION_CUES` word within
      :data:`NEGATION_WINDOW_WORDS`, because "not confirmed flood extent" is the
      phrasing the project requires.

    Returns an empty list for compliant text. This is a lexical guard and not a
    correctness proof; passing it does not make a claim true.
    """
    if not text:
        return []

    masked = list(text)
    lowered = text.lower()
    for phrase in ALLOWED_PHRASES:
        start = 0
        needle = phrase.lower()
        while True:
            found = lowered.find(needle, start)
            if found < 0:
                break
            for offset in range(found, found + len(needle)):
                masked[offset] = "\x00"
            start = found + len(needle)
    haystack = "".join(masked).lower()

    violations: List[ClaimViolation] = []
    for term, permitted in FORBIDDEN_CLAIMS.items():
        for match in re.finditer(rf"(?<!\w){re.escape(term.lower())}(?!\w)", haystack):
            if _is_negated(haystack, match.start()):
                continue
            violations.append(
                ClaimViolation(
                    term=term, permitted_instead=permitted, position=match.start()
                )
            )
    violations.sort(key=lambda item: (item.position, item.term))
    return violations
