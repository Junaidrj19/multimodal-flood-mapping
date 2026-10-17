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
    "MANIFEST_SCHEMA",
    "M63_STAGE",
    "M63Artifacts",
    "BlockerReport",
    "BlockerState",
    "DashboardContractError",
    "DashboardLayer",
    "DashboardManifest",
    "EvidenceTier",
    "LayerFormat",
    "LoadedDashboard",
    "StageState",
    "StageStatus",
    "ValidationState",
    "WorkflowReport",
    "build_demo_bundle",
    "build_real_dashboard",
    "build_workflow",
    "discover_m63_artifacts",
    "load_blocker_report",
    "load_manifest",
    "resolve_project_root",
]
