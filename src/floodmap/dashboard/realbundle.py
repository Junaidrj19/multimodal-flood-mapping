"""Assemble an in-memory dashboard bundle from validated real M6.3 artifacts.

The M6.3 stage writes rasters, a summary and provenance records, but no
``dashboard_manifest.json`` (that descriptor belongs to the presentation layer,
not the science pipeline, ``AGENTS.md`` §16). This module builds the descriptor
*in memory* from what was validated, so the existing contract viewer renders real
data through exactly the same code path as the synthetic demo — no bespoke
rendering branch, and no manifest file written to disk.

Separating this from ``dashboard/app.py`` keeps it unit-testable without the
``streamlit`` extra: the tests can assert that a real bundle is
``is_synthetic=False``, that its layers resolve to the real files, and that the
blocked downstream stages are declared unavailable *with a reason*.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..utils.provenance import REQUIRED_ATTRIBUTIONS, ArtifactType
from .contract import (
    MANIFEST_SCHEMA,
    DashboardLayer,
    DashboardManifest,
    EvidenceTier,
    LayerFormat,
    LoadedDashboard,
)
from .realdata import M63Artifacts
from .workflow import StageState, WorkflowReport

__all__ = ["REAL_LAYER_TITLES", "build_real_dashboard"]

REAL_LAYER_TITLES: Dict[str, str] = {
    "change_candidate_mask": "M6.3 change candidates (rule-based, gamma-0)",
    "change_vv": "VV log-ratio change (dB)",
    "change_vh": "VH log-ratio change (dB)",
}

#: Downstream stages presented as explicitly-absent layers with a real reason.
_DOWNSTREAM_LAYERS = (
    ("m6_4_exposure", "M6.4 infrastructure exposure",
     ArtifactType.INFRASTRUCTURE_EXPOSURE, EvidenceTier.SPATIAL_INFERENCE),
    ("m6_5_disruption", "M6.5 network disruption / connectivity",
     ArtifactType.SETTLEMENT_CONNECTIVITY, EvidenceTier.NETWORK_INFERENCE),
)


def build_real_dashboard(
    artifacts: M63Artifacts,
    workflow: Optional[WorkflowReport] = None,
    *,
    event_name: str = (
        "Trishuli Valley flood — real M6.3 Sentinel-1 RTC change candidates"
    ),
) -> LoadedDashboard:
    """Build a contract-valid bundle backed by the real on-disk M6.3 artifacts.

    Requires ``artifacts.is_usable``: a partially-valid set is not dressed up as a
    renderable bundle. Callers must handle the not-usable case explicitly.
    """
    if not artifacts.is_usable:
        raise ValueError(
            f"cannot build a real dashboard bundle from a {artifacts.state.value} "
            "artifact set; report the problem instead of rendering a partial result"
        )

    files = artifacts.files
    summary = artifacts.summary or {}
    pair = summary.get("pair") or {}
    grid = artifacts.rasters.get("candidate_mask")
    bbox = list(grid.bounds) if grid is not None else None
    crs = grid.crs if grid is not None else None

    candidate_prov = artifacts.provenance["candidate_provenance"]
    features_prov = artifacts.provenance["features_provenance"]

    def _rel(role: str) -> str:
        return files[role].path.relative_to(artifacts.directory).as_posix()

    layers: List[DashboardLayer] = [
        DashboardLayer(
            layer_id="change_candidate_mask",
            title=REAL_LAYER_TITLES["change_candidate_mask"],
            artifact_type=candidate_prov.artifact_type,
            evidence_tier=EvidenceTier.OBSERVED_DERIVED,
            format=LayerFormat.GEOTIFF,
            path=_rel("candidate_mask"),
            crs=crs,
            bbox=bbox,
            provenance_path=_rel("candidate_provenance"),
        ),
        DashboardLayer(
            layer_id="change_vv",
            title=REAL_LAYER_TITLES["change_vv"],
            artifact_type=features_prov.artifact_type,
            evidence_tier=EvidenceTier.OBSERVED_DERIVED,
            format=LayerFormat.GEOTIFF,
            path=_rel("features"),
            crs=crs,
            bbox=bbox,
            provenance_path=_rel("features_provenance"),
        ),
        DashboardLayer(
            layer_id="change_vh",
            title=REAL_LAYER_TITLES["change_vh"],
            artifact_type=features_prov.artifact_type,
            evidence_tier=EvidenceTier.OBSERVED_DERIVED,
            format=LayerFormat.GEOTIFF,
            path=_rel("features"),
            crs=crs,
            bbox=bbox,
            provenance_path=_rel("features_provenance"),
        ),
    ]

    if workflow is not None:
        for stage_id, title, artifact_type, tier in _DOWNSTREAM_LAYERS:
            status = workflow.stage(stage_id)
            if status is None or status.state is StageState.REAL_DATA_COMPLETE:
                continue
            layers.append(
                DashboardLayer(
                    layer_id=stage_id,
                    title=title,
                    artifact_type=artifact_type,
                    evidence_tier=tier,
                    format=LayerFormat.GEOJSON,
                    available=False,
                    unavailable_reason=status.detail,
                    path=None,
                    provenance_path=None,
                )
            )

    manifest = DashboardManifest(
        schema=MANIFEST_SCHEMA,
        aoi_id=str(pair.get("aoi_id") or candidate_prov.aoi),
        event_name=event_name,
        event_date=pair.get("event_date"),
        is_synthetic=False,
        attribution=list(REQUIRED_ATTRIBUTIONS),
        layers=layers,
    )

    provenance_by_id: Dict[str, Optional[object]] = {
        "change_candidate_mask": candidate_prov,
        "change_vv": features_prov,
        "change_vh": features_prov,
    }
    for layer in layers:
        if not layer.available:
            provenance_by_id[layer.layer_id] = None

    return LoadedDashboard(
        manifest=manifest, provenance=provenance_by_id, base_dir=artifacts.directory
    )
