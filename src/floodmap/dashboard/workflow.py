"""Workflow stage status: what actually ran, what is blocked, and on what.

The dashboard's job (``AGENTS.md`` §16) is to distinguish, honestly, an observed
result from a prediction from an inference from something that simply has not
happened. For the M6 vertical slice that means showing, stage by stage:

``RTC acquisition -> preprocessing -> M6.3 change candidates -> M6.4 exposure
-> M6.5 disruption/connectivity``

and saying for each whether real-data execution completed, whether it is blocked
on a missing prerequisite, or whether no output exists at all. The distinction
matters: a missing output is *not* "zero impact" (the M6.6 brief says so
explicitly), and a blocked stage must name the prerequisite that blocks it.

The M6.4/M6.5 blocker reports on disk are read verbatim and shown. If a blocker
file is absent or malformed this module reports that state as unknown/unavailable
and never synthesises replacement content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .realdata import M63Artifacts, resolve_project_root

__all__ = [
    "M64_DIR",
    "M65_DIR",
    "BlockerReport",
    "BlockerState",
    "StageState",
    "StageStatus",
    "WorkflowReport",
    "build_workflow",
    "load_blocker_report",
]

#: Artifact directories, relative to the project root.
M64_DIR = Path("artifacts") / "m6_4" / "candidate_infrastructure_exposure"
M65_DIR = Path("artifacts") / "m6_5" / "candidate_network_disruption"

#: Real-data *output* files that prove a stage actually ran (as opposed to the
#: blocker report, which proves it could not).
M64_OUTPUT = "candidate_exposure_summary.json"
M65_OUTPUT = "candidate_disruption_summary.json"
M64_BLOCKER = "candidate_exposure_blocker.json"
M65_BLOCKER = "candidate_disruption_blocker.json"

#: Keys a well-formed blocker report must carry.
BLOCKER_REQUIRED_KEYS = ("stage", "status", "reason", "next_step")

#: The course of action every blocker defers to. Stated in one place so the UI
#: cannot paraphrase the safeguard into something weaker.
UNBLOCK_REQUIREMENTS = (
    "Verify the AOI and authorise data retrieval through the process that owns "
    "that decision (the AOI is currently UNVERIFIED and does not authorise a download).",
    "Fetch permitted pre-event OpenStreetMap layers with the existing adapter.",
    "Run M6.4 candidate exposure on those real inputs.",
    "Run M6.5 network disruption on the resulting M6.4 exposure artifacts.",
)


class StageState(str, Enum):
    """The distinct states a pipeline stage can be in for this dashboard.

    Kept deliberately granular: collapsing "blocked" into "no output" would let a
    blocked downstream analysis read as "nothing to report", which the brief and
    ``architecture.md`` §17 forbid.
    """

    #: Implementation exists and real-data execution produced validated output.
    REAL_DATA_COMPLETE = "real_data_complete"
    #: Implementation exists but no real execution has happened.
    IMPLEMENTED_NOT_EXECUTED = "implemented_not_executed"
    #: Execution was attempted and stopped on a missing prerequisite, recorded.
    BLOCKED = "blocked"
    #: Output is expected but absent, with no blocker explaining why.
    MISSING_OUTPUT = "missing_output"
    #: Output exists but is malformed or internally inconsistent.
    INVALID_OUTPUT = "invalid_output"
    #: State could not be determined from the available artifacts.
    UNKNOWN = "unknown"


class BlockerState(str, Enum):
    """Whether a stage's blocker report could be read."""

    LOADED = "loaded"
    MISSING = "missing"
    MALFORMED = "malformed"


@dataclass
class BlockerReport:
    """A parsed M6.4/M6.5 blocker report, or the reason none could be parsed."""

    stage: str
    path: Path
    state: BlockerState
    reason: Optional[str] = None
    status: Optional[str] = None
    missing: List[str] = field(default_factory=list)
    searched: Optional[str] = None
    authorisation: Dict[str, Any] = field(default_factory=dict)
    next_step: List[str] = field(default_factory=list)
    not_done: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None
    problems: List[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.state is BlockerState.LOADED and self.status == "blocked"


@dataclass
class StageStatus:
    """One stage's status, ready to render."""

    stage_id: str
    title: str
    state: StageState
    detail: str
    evidence: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    blocker: Optional[BlockerReport] = None


@dataclass
class WorkflowReport:
    """The full stage list plus the AOI-authorisation facts shown alongside it."""

    stages: List[StageStatus]
    aoi_provenance_status: Optional[str] = None
    authorises_download: Optional[bool] = None
    unblock_requirements: List[str] = field(default_factory=lambda: list(UNBLOCK_REQUIREMENTS))

    def stage(self, stage_id: str) -> Optional[StageStatus]:
        for item in self.stages:
            if item.stage_id == stage_id:
                return item
        return None

    def any_blocked(self) -> bool:
        return any(s.state is StageState.BLOCKED for s in self.stages)


def load_blocker_report(path: Path, *, stage: str) -> BlockerReport:
    """Read one blocker report, classifying absence and malformation explicitly."""
    path = Path(path)
    if not path.is_file():
        return BlockerReport(stage=stage, path=path, state=BlockerState.MISSING)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return BlockerReport(
            stage=stage,
            path=path,
            state=BlockerState.MALFORMED,
            problems=[f"not valid JSON: {exc}"],
        )
    if not isinstance(raw, dict):
        return BlockerReport(
            stage=stage,
            path=path,
            state=BlockerState.MALFORMED,
            problems=["blocker report is not a JSON object"],
        )
    problems = [f"missing key {key!r}" for key in BLOCKER_REQUIRED_KEYS if key not in raw]
    next_step = raw.get("next_step") or []
    if isinstance(next_step, str):
        next_step = [next_step]
    if not isinstance(next_step, list):
        problems.append("next_step must be a list of strings")
        next_step = []
    missing = raw.get("missing_layers") or raw.get("missing_inputs") or []
    if not isinstance(missing, list):
        missing = []
    authorisation = raw.get("authorisation")
    if not isinstance(authorisation, dict):
        authorisation = {}
    if problems:
        return BlockerReport(
            stage=stage,
            path=path,
            state=BlockerState.MALFORMED,
            raw=raw,
            problems=problems,
        )
    return BlockerReport(
        stage=stage,
        path=path,
        state=BlockerState.LOADED,
        reason=str(raw.get("reason")) if raw.get("reason") is not None else None,
        status=str(raw.get("status")) if raw.get("status") is not None else None,
        missing=[str(item) for item in missing],
        searched=raw.get("searched_directory"),
        authorisation=authorisation,
        next_step=[str(step) for step in next_step],
        not_done=[str(item) for item in (raw.get("not_done") or [])],
        raw=raw,
    )


def _output_stage_status(
    stage_id: str,
    title: str,
    *,
    output_path: Path,
    blocker: BlockerReport,
    prerequisites: List[str],
) -> StageStatus:
    """Classify a downstream stage from its output file and blocker report."""
    if output_path.is_file():
        return StageStatus(
            stage_id=stage_id,
            title=title,
            state=StageState.REAL_DATA_COMPLETE,
            detail=f"real-data output present: {output_path}",
            evidence=[str(output_path)],
        )
    if blocker.state is BlockerState.LOADED:
        if blocker.is_blocked:
            return StageStatus(
                stage_id=stage_id,
                title=title,
                state=StageState.BLOCKED,
                detail=blocker.reason or "blocked on missing prerequisites",
                evidence=[str(blocker.path)],
                prerequisites=prerequisites,
                blocker=blocker,
            )
        return StageStatus(
            stage_id=stage_id,
            title=title,
            state=StageState.UNKNOWN,
            detail=f"blocker report has unexpected status {blocker.status!r}",
            evidence=[str(blocker.path)],
            blocker=blocker,
        )
    if blocker.state is BlockerState.MALFORMED:
        return StageStatus(
            stage_id=stage_id,
            title=title,
            state=StageState.INVALID_OUTPUT,
            detail="blocker report is malformed: " + "; ".join(blocker.problems),
            evidence=[str(blocker.path)],
            blocker=blocker,
        )
    return StageStatus(
        stage_id=stage_id,
        title=title,
        state=StageState.IMPLEMENTED_NOT_EXECUTED,
        detail="no output artifact and no blocker report found",
    )


def build_workflow(
    m63: M63Artifacts,
    *,
    project_root: Optional[Path] = None,
) -> WorkflowReport:
    """Assemble the stage-status view from the real artifacts on disk."""
    root = resolve_project_root(project_root)
    m64_dir = root / M64_DIR
    m65_dir = root / M65_DIR

    m64_blocker = load_blocker_report(m64_dir / M64_BLOCKER, stage="m6.4")
    m65_blocker = load_blocker_report(m65_dir / M65_BLOCKER, stage="m6.5")

    aoi_status: Optional[str] = None
    authorises: Optional[bool] = None
    for report in (m64_blocker, m65_blocker):
        auth = report.authorisation
        if auth:
            aoi_status = aoi_status or auth.get("aoi_provenance_status")
            if authorises is None and "authorises_download" in auth:
                authorises = bool(auth["authorises_download"])

    # --- acquisition / preprocessing / M6.3 from the validated candidate set ---
    if m63.is_usable:
        m63_state = StageState.REAL_DATA_COMPLETE
        m63_detail = "validated M6.3 candidate artifacts"
        m63_evidence = [str(m63.files["candidate_mask"].path), str(m63.files["summary"].path)]
    elif m63.state.value == "missing":
        m63_state = StageState.MISSING_OUTPUT
        m63_detail = m63.problems_text()
        m63_evidence = []
    else:
        m63_state = StageState.INVALID_OUTPUT
        m63_detail = m63.problems_text()
        m63_evidence = []

    pair = (m63.summary or {}).get("pair") or {}
    pre = pair.get("pre") or {}
    post = pair.get("post") or {}

    grid_info = m63.rasters.get("candidate_mask")
    if grid_info is not None and pre and post:
        upstream_state = StageState.REAL_DATA_COMPLETE
    else:
        upstream_state = StageState.UNKNOWN

    acquisition_evidence = [
        f"pre: {pre.get('scene_id')} @ {pre.get('acquired_at')}",
        f"post: {post.get('scene_id')} @ {post.get('acquired_at')}",
        f"relative orbit {pre.get('relative_orbit')} ({pre.get('orbit_direction')})",
    ] if pre and post else []

    stages = [
        StageStatus(
            stage_id="rtc_acquisition",
            title="Sentinel-1 RTC acquisition",
            state=upstream_state,
            detail=(
                "real pre/post RTC scenes recorded in the M6.3 summary"
                if upstream_state is StageState.REAL_DATA_COMPLETE
                else "no acquisition evidence readable from M6.3"
            ),
            evidence=acquisition_evidence,
        ),
        StageStatus(
            stage_id="preprocessing",
            title="Preprocessing (RTC assembly, gamma0 linear power)",
            state=upstream_state,
            detail=(
                "pre/post assembled to a shared EPSG:32645 10 m grid"
                if upstream_state is StageState.REAL_DATA_COMPLETE
                else "no preprocessing evidence readable from M6.3"
            ),
            evidence=(
                [
                    f"convention: {(m63.summary or {}).get('radiometric_convention')}",
                    f"grid: {grid_info.crs} {grid_info.resolution[0]:g} m {grid_info.shape}",
                ]
                if m63.summary and grid_info is not None
                else []
            ),
        ),
        StageStatus(
            stage_id="m6_3_change_candidates",
            title="M6.3 change candidates",
            state=m63_state,
            detail=m63_detail,
            evidence=m63_evidence,
        ),
        _output_stage_status(
            "m6_4_exposure",
            "M6.4 infrastructure exposure",
            output_path=m64_dir / M64_OUTPUT,
            blocker=m64_blocker,
            prerequisites=[
                "Verified/authorised AOI",
                "Permitted pre-event OSM layers",
            ],
        ),
        _output_stage_status(
            "m6_5_disruption",
            "M6.5 network disruption / connectivity",
            output_path=m65_dir / M65_OUTPUT,
            blocker=m65_blocker,
            prerequisites=[
                "M6.4 candidate exposure artifacts",
                "M6.4 OSM hub layer",
            ],
        ),
    ]

    return WorkflowReport(
        stages=stages,
        aoi_provenance_status=aoi_status,
        authorises_download=authorises,
    )
