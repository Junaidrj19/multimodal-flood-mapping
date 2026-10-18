"""Dashboard artifact contract: the boundary between science and presentation.

Why this package exists
-----------------------
``architecture.md`` §15 makes the dashboard a visualization and decision-support
layer that *consumes generated artifacts* and runs no science of its own.
``AGENTS.md`` §16 requires it to distinguish, on screen, four different kinds of
claim:

* observed / directly derived;
* model prediction;
* network-level inference;
* validation reference.

Both rules are only enforceable if the dashboard reads a *declared contract*
rather than guessing at file locations, and if every displayed layer can be
traced to an :class:`~floodmap.utils.provenance.ArtifactProvenance` record. This
package defines that contract and its loader. It contains no acquisition,
preprocessing, segmentation, network or hydrology logic, so a frontend that
imports it could still not compute a result of its own (the guard in
``tests/test_data_boundary.py`` covers the production-input boundary, and
``tests/test_dashboard_contract.py`` covers this one).
"""

from .contract import (
    MANIFEST_SCHEMA,
    DashboardContractError,
    DashboardLayer,
    DashboardManifest,
    EvidenceTier,
    LayerFormat,
    LoadedDashboard,
    load_manifest,
)
from .demobundle import build_demo_bundle
from .realbundle import build_real_dashboard
from .realdata import (
    M63_STAGE,
    M63Artifacts,
    ValidationState,
    discover_m63_artifacts,
    resolve_project_root,
)
from .readiness import (
    AOI_COVERAGE,
    GEOLOCATION_ACCEPTANCE,
    REQUIRED_PRESENTATION_LABELS,
    ClaimViolation,
    ExecutionReconciliation,
    ExecutionEvidenceIntegrity,
    ExecutionVerdict,
    GateDecision,
    GateOutcome,
    Limitation,
    LimitationState,
    ReadinessGateError,
    ReadinessReport,
    UseClass,
    authorise_use,
    build_readiness_report,
    reconcile_execution,
    screen_claim,
    validate_execution_evidence,
)
from .workflow import (
    BlockerReport,
    BlockerState,
    StageState,
    StageStatus,
    WorkflowReport,
    build_workflow,
    load_blocker_report,
)

__all__ = [
    "AOI_COVERAGE",
    "GEOLOCATION_ACCEPTANCE",
    "MANIFEST_SCHEMA",
    "REQUIRED_PRESENTATION_LABELS",
    "M63_STAGE",
    "M63Artifacts",
    "BlockerReport",
    "BlockerState",
    "ClaimViolation",
    "DashboardContractError",
    "DashboardLayer",
    "DashboardManifest",
    "EvidenceTier",
    "ExecutionReconciliation",
    "ExecutionEvidenceIntegrity",
    "ExecutionVerdict",
    "GateDecision",
    "GateOutcome",
    "LayerFormat",
    "Limitation",
    "LimitationState",
    "LoadedDashboard",
    "ReadinessGateError",
    "ReadinessReport",
    "StageState",
    "StageStatus",
    "UseClass",
    "ValidationState",
    "WorkflowReport",
    "authorise_use",
    "build_demo_bundle",
    "build_readiness_report",
    "build_real_dashboard",
    "build_workflow",
    "discover_m63_artifacts",
    "load_blocker_report",
    "load_manifest",
    "reconcile_execution",
    "resolve_project_root",
    "screen_claim",
    "validate_execution_evidence",
]
