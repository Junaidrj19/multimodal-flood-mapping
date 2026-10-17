"""Flood impact dashboard (Multimodal AI Hackathon 2026, Track B).

This is a visualization and decision-support layer only. ``AGENTS.md`` §16 and
``architecture.md`` §15: it consumes generated artifacts and runs NO acquisition,
preprocessing, segmentation, network or hydrology logic. Every number shown is
read from an artifact or recomputed from an artifact's pixels for cross-checking;
nothing is estimated, interpolated or invented here.

Two data sources exist and are kept strictly apart (M6.6 §E):

* **Real M6.3 artifacts** — the on-disk Sentinel-1 RTC change-candidate output.
  ``src/floodmap/dashboard/realdata.py`` locates and validates it. If it is
  missing, malformed or internally inconsistent the app says so and shows the
  problem; it never substitutes synthetic values.
* **Synthetic demo bundle** — a deterministic placeholder bundle used before real
  artifacts existed. It is always bannered and its values are never mixed into the
  real metrics.

Evidence tiers are rendered as separate, explicitly labelled toggle groups
(``AGENTS.md`` §16, ``docs/scientific-assumptions.md`` §10.6). A validation
reference can never be presented as an input.

Run with:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.utilities import image_to_url
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

from floodmap.dashboard import (
    BlockerState,
    DashboardContractError,
    EvidenceTier,
    LayerFormat,
    StageState,
    build_demo_bundle,
    build_real_dashboard,
    build_workflow,
    discover_m63_artifacts,
    load_manifest,
)
from floodmap.dashboard.realdata import ConfiguredAOI, M63Artifacts, resolve_project_root
from floodmap.dashboard.render import candidate_mask_rgba, change_rgba, read_display
from floodmap.dashboard.workflow import BlockerReport, WorkflowReport
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "dashboard_demo" / "dashboard_manifest.json"

SOURCE_REAL = "Real M6.3 artifacts"
SOURCE_DEMO = "Synthetic demo bundle"

TIER_LABELS: dict[EvidenceTier, str] = {
    EvidenceTier.OBSERVED_DERIVED: "Observed / derived",
    EvidenceTier.MODEL_PREDICTION: "Model prediction",
    EvidenceTier.SPATIAL_INFERENCE: "Spatially inferred",
    EvidenceTier.NETWORK_INFERENCE: "Network inference",
    EvidenceTier.VALIDATION_REFERENCE: "Validation reference",
}

#: Human-readable rendering of each workflow stage state, with a traffic marker.
STAGE_STYLE: dict[StageState, tuple[str, str]] = {
    StageState.REAL_DATA_COMPLETE: ("🟢", "Completed on real data"),
    StageState.IMPLEMENTED_NOT_EXECUTED: ("⚪", "Implemented — not executed"),
    StageState.BLOCKED: ("🔴", "Blocked — missing prerequisite"),
    StageState.MISSING_OUTPUT: ("🟠", "Missing output"),
    StageState.INVALID_OUTPUT: ("🟠", "Invalid or inconsistent output"),
    StageState.UNKNOWN: ("⚫", "Unknown"),
}

st.set_page_config(page_title="Flood impact dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Raster -> RGBA (presentation only). Real layer ids use the nodata-aware
# helpers in floodmap.dashboard.render so invalid pixels stay transparent.
# ---------------------------------------------------------------------------


def _normalize(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile-normalize a band to 0..1 for display (presentation only)."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype="uint8")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(band, dtype="uint8")
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out.astype("float32")


def _flood_mask_rgba(data: np.ndarray) -> np.ndarray:
    """Flood water mask -> translucent blue overlay."""
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")
    flooded = data > 0
    rgba[flooded] = np.array([30, 144, 255, 170], dtype="uint8")
    return rgba


def _confidence_rgba(data: np.ndarray) -> np.ndarray:
    """Confidence 0..1 -> orange-red heat overlay, more opaque when confident.

    Synthetic demo only: the per-pixel Python loop is fine on the 128x128 demo
    grid but is never used for a real scene.
    """
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")
    conf = np.clip(data.astype("float32"), 0.0, 1.0)
    low = np.array([59, 76, 192], dtype="float32")
    mid = np.array([221, 130, 58], dtype="float32")
    high_c = np.array([240, 59, 32], dtype="float32")
    for y in range(height):
        for x in range(width):
            c = conf[y, x]
            if c <= 0.0:
                continue
            if c < 0.5:
                colour = low + (mid - low) * (c / 0.5)
            else:
                colour = mid + (high_c - mid) * ((c - 0.5) / 0.5)
            rgba[y, x, :3] = colour.astype("uint8")
            rgba[y, x, 3] = int(40 + 200 * c)
    return rgba


def _gray_rgba(band: np.ndarray) -> np.ndarray:
    gray = (np.clip(band, 0.0, 1.0) * 255.0).astype("uint8")
    return np.dstack([gray, gray, gray, np.full_like(gray, 255)])


def _rgb_rgba(bands: np.ndarray) -> np.ndarray:
    height, width = bands.shape[1:]
    stacked = [(_normalize(bands[i]) * 255.0).astype("uint8") for i in range(bands.shape[0])]
    alpha = np.full((height, width), 255, dtype="uint8")
    return np.dstack([*stacked, alpha])


def _layer_rgba(path: Path, layer_id: str) -> np.ndarray:
    """Read a geotiff and turn it into an RGBA display array (presentation only).

    Real M6.3 layer ids are decimated to a bounded display resolution and keep
    nodata transparent. Demo layer ids keep their original rendering.
    """
    if layer_id == "change_candidate_mask":
        data, nodata = read_display(path, band=1)
        return candidate_mask_rgba(data, nodata=nodata)
    if layer_id == "change_vv":
        data, nodata = read_display(path, band=1)
        return change_rgba(data, nodata=nodata)
    if layer_id == "change_vh":
        data, nodata = read_display(path, band=2)
        return change_rgba(data, nodata=nodata)
    with rasterio.open(path) as src:
        count = src.count
        if count == 1:
            if layer_id == "flood_water_extent":
                return _flood_mask_rgba(src.read(1))
            if layer_id == "flood_water_confidence":
                return _confidence_rgba(src.read(1))
            return _gray_rgba(_normalize(src.read(1)))
        if count == 3:
            return _rgb_rgba(src.read())
    return _gray_rgba(_normalize(src.read(1)))


def _wgs84_bounds(path: Path):
    with rasterio.open(path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return south, west, north, east


def _center_bounds(bundle):
    """Center the map on the first available geotiff layer's footprint."""
    for layer in bundle.manifest.layers:
        if layer.available and layer.format is LayerFormat.GEOTIFF:
            south, west, north, east = _wgs84_bounds(bundle.resolve_path(layer))
            return [(south + north) / 2.0, (west + east) / 2.0]
    return [28.0, 85.1]


def _colour_for_feature(layer_id: str, props: dict) -> str:
    """Traffic-light styling for vector overlays. Presentational only."""
    if layer_id == "buildings_exposed" and props.get("intersects_flood_extent"):
        return "red"
    if layer_id == "roads":
        return "red" if props.get("potentially_disrupted") else "gray"
    if layer_id == "settlements":
        return "red" if props.get("potentially_cut_off") else "green"
    return "blue"


def _add_layers(folium_map, bundle, enabled: set[str], errors: list[str] | None = None) -> None:
    for layer in bundle.manifest.layers:
        if not layer.available or layer.layer_id not in enabled:
            continue
        path = bundle.resolve_path(layer)
        label = f"{layer.title} ({TIER_LABELS[layer.evidence_tier]})"
        if layer.format is LayerFormat.GEOTIFF:
            try:
                south, west, north, east = _wgs84_bounds(path)
                rgba = _layer_rgba(path, layer.layer_id)
                url = image_to_url(rgba)
                overlay = folium.raster_layers.ImageOverlay(
                    image=url, bounds=[[south, west], [north, east]], name=label, opacity=1.0
                )
            except Exception as exc:  # surface, never hide, a broken layer
                if errors is not None:
                    errors.append(f"{layer.layer_id}: {exc}")
                continue
        elif layer.format is LayerFormat.GEOJSON:
            overlay = folium.GeoJson(
                str(path),
                name=label,
                style_function=lambda feature, l_id=layer.layer_id: {
                    "color": _colour_for_feature(l_id, feature.get("properties", {}))
                },
            )
        else:
            continue
        overlay.add_to(folium_map)
    folium.LayerControl(collapsed=True).add_to(folium_map)


def _render_sidebar(bundle) -> dict:
    st.sidebar.header("Layers")
    st.sidebar.caption(
        "Grouped by evidence strength (strongest first). Each layer's provenance "
        "and limitations are shown under the map."
    )
    selected: dict[str, bool] = {}
    for tier in (
        EvidenceTier.OBSERVED_DERIVED,
        EvidenceTier.MODEL_PREDICTION,
        EvidenceTier.SPATIAL_INFERENCE,
        EvidenceTier.NETWORK_INFERENCE,
        EvidenceTier.VALIDATION_REFERENCE,
    ):
        tiers = bundle.manifest.layers_by_tier()[tier]
        available = [layer for layer in tiers if layer.available]
        if not available:
            continue
        st.sidebar.markdown(f"**{TIER_LABELS[tier]}**")
        for layer in available:
            default = layer.layer_id in {
                "s1_pre",
                "s1_post",
                "flood_water_extent",
                "flood_water_confidence",
                "buildings_exposed",
                "roads",
                "settlements",
                "change_candidate_mask",
                "change_vv",
            }
            selected[layer.layer_id] = st.sidebar.checkbox(
                layer.title, key=f"layer-{layer.layer_id}", value=default
            )
    st.sidebar.markdown("---")
    # Explicit absence (architecture.md §17): an unavailable layer is shown,
    # not omitted.
    unavailable = [layer for layer in bundle.manifest.layers if not layer.available]
    if unavailable:
        with st.sidebar.expander("Unavailable layers"):
            for layer in unavailable:
                reason = layer.unavailable_reason
                st.caption(f"**{layer.title}** unavailable.")
                st.caption(f"Reason: {getattr(reason, 'note', reason)}")
    st.sidebar.markdown("---")
    st.sidebar.caption("Attribution")
    for line in bundle.manifest.attribution:
        st.sidebar.caption(line)
    return selected


def _render_provenance(bundle, enabled: dict[str, bool]) -> None:
    st.markdown("## Evidence & provenance")
    st.caption(
        "Every displayed layer is traceable to its source chain (AGENTS.md §22): "
        "source data → processing → model → metric/rule → derived result → uncertainty."
    )
    shown = 0
    for layer in bundle.manifest.layers:
        if not layer.available or not enabled[layer.layer_id]:
            continue
        shown += 1
        provenance = bundle.provenance[layer.layer_id]
        tier = TIER_LABELS[layer.evidence_tier]
        with st.expander(f"{layer.title} — {tier}", expanded=False):
            st.markdown(
                f"**Production inputs:** {', '.join(p for p in provenance.production_inputs)}"
            )
            st.markdown("**Known limitations:**")
            for limitation in provenance.limitations:
                st.markdown(f"- {limitation}")
            if provenance.notes:
                st.markdown(f"**Notes:** {provenance.notes}")
            if provenance.model_version or provenance.preprocessing_version:
                st.markdown(
                    f"**Versions:** model={provenance.model_version or '—'}, "
                    f"preprocessing={provenance.preprocessing_version or '—'}"
                )
            st.markdown("**Attribution:**")
            for line in provenance.attribution:
                st.markdown(f"- {line}")
    st.markdown("---")
    if shown == 0:
        st.caption("No layers selected. Use the sidebar to enable layers.")


# ---------------------------------------------------------------------------
# Real M6.3 dashboard construction
# ---------------------------------------------------------------------------

def _render_aoi_scope(summary: dict) -> None:
    """Show the three distinct areas and the denominator each rate uses.

    The delivered raster window is a bounding-box crop that reaches far outside
    the study extent, so a single "area" figure would be ambiguous at best. The
    full extent, the delivered part of it and the validly observed part of it are
    therefore always shown together, and the full extent is labelled as the
    study extent rather than as something that was observed.
    """
    scope = summary.get("aoi_scope")
    if not isinstance(scope, dict):
        st.error(
            "This M6.3 result carries no `aoi_scope` block, so AOI-relative figures "
            "and their denominators are unavailable. Nothing below may be read as a "
            "statement about the study extent."
        )
        return

    st.markdown("### Observed support within the study extent")
    disclosure = scope.get("disclosure")
    if isinstance(disclosure, str) and disclosure.strip():
        st.warning(disclosure)

    full = scope.get("aoi_geometric_area_km2")
    delivered = scope.get("aoi_delivered_area_km2")
    unobserved = scope.get("aoi_unobserved_area_km2")
    valid_area = scope.get("aoi_valid_area_km2")
    candidate_area = scope.get("candidate_area_km2")

    def _km2(value: object) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "unknown"

    def _pct(value: object) -> str:
        return f"{value:.4%}" if isinstance(value, (int, float)) else "unknown"

    st.table(
        {
            "area": [
                "Study extent (AOI polygon)",
                "Delivered (AOI ∩ raster window)",
                "Unobserved (no delivered pixels)",
                "Validly observed (pre/post VV+VH, pixel-centre estimate)",
                "Darkening candidates",
            ],
            "km²": [
                _km2(full),
                _km2(delivered),
                _km2(unobserved),
                _km2(valid_area),
                _km2(candidate_area),
            ],
            "share of study extent": [
                "100% (extent, NOT observed)",
                _pct(scope.get("aoi_delivered_fraction")),
                _pct(scope.get("aoi_unobserved_fraction")),
                _pct(
                    valid_area / full
                    if isinstance(valid_area, (int, float)) and isinstance(full, (int, float)) and full
                    else None
                ),
                _pct(scope.get("candidate_fraction_of_aoi_geometric_area")),
            ],
            "status": [
                "Study extent — context, not evidence",
                "Observed / derived",
                "Unknown — not unaffected, flood-free or undamaged",
                "Observed / derived",
                "Change candidate — NOT confirmed flood or damage",
            ],
        }
    )
    st.caption(
        "Primary candidate rate: "
        f"{_pct(scope.get('candidate_fraction_of_aoi_valid_pixels'))} of the "
        f"{scope.get('aoi_valid_pixels'):,} validly observed in-extent pixels. "
        "The share-of-extent column is a lower bound only, because "
        f"{_pct(scope.get('aoi_unobserved_fraction'))} of the extent carries no observation."
    )
    st.caption(
        "Measurement types differ by row and are not interchangeable: the delivered "
        "area is a geometric polygon intersection, while the validly observed area is "
        "a pixel-centre estimate (valid pixel centres × 100 m²), so it can differ "
        "marginally from the delivered area at the boundary. Every rate above names "
        "its own denominator; none of them is a whole-extent prevalence rate."
    )
    st.caption(
        f"{scope.get('window_pixels_outside_aoi'):,} delivered raster-window pixels lie "
        "outside the study extent and are excluded from every figure above. Window-wide "
        "counts in the next section are mechanical array tallies, not extent-relative rates."
    )
    with st.expander("Denominator definitions (as recorded by the M6.3 stage)"):
        for name, text in (scope.get("denominator_definitions") or {}).items():
            st.markdown(f"- **`{name}`** — {text}")


def _render_real_metrics(artifacts: M63Artifacts) -> None:
    """Show the M6.3 figures, distinguishing recorded metadata from recomputation."""
    if not artifacts.is_usable:
        raise ValueError("Cannot display metrics from unusable or stale M6.3 results")
    summary = artifacts.summary or {}
    pair = summary.get("pair") or {}
    pre = pair.get("pre") or {}
    post = pair.get("post") or {}
    grid = artifacts.rasters.get("candidate_mask")
    computed = artifacts.computed
    recorded = artifacts.recorded

    st.markdown("## Real M6.3 artifacts")
    st.caption(f"Artifact directory: `{artifacts.directory}`")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Acquisition (Sentinel-1 RTC pair)**")
        st.table(
            {
                "role": ["pre-event", "post-event"],
                "scene id": [pre.get("scene_id"), post.get("scene_id")],
                "acquired at": [pre.get("acquired_at"), post.get("acquired_at")],
                "relative orbit": [pre.get("relative_orbit"), post.get("relative_orbit")],
                "orbit direction": [pre.get("orbit_direction"), post.get("orbit_direction")],
            }
        )
        st.caption(
            f"AOI: `{pair.get('aoi_id')}` · event date {pair.get('event_date')} · "
            f"AOI provenance {summary.get('aoi_provenance_status')}"
        )
        prov_pair = artifacts.provenance["candidate_provenance"].sentinel1
        if prov_pair is not None and prov_pair.before is not None:
            before = prov_pair.before
            st.caption(
                "Product metadata (from the M6.3 provenance record): "
                f"platform `{before.platform}` · processing level `{before.processing_level}` · "
                f"product type `{before.product_type}` · "
                f"same relative orbit: {prov_pair.same_relative_orbit}"
            )
    with col_b:
        st.markdown("**Raster / radiometry**")
        st.table(
            {
                "property": [
                    "radiometric convention",
                    "input bands",
                    "CRS",
                    "resolution (m)",
                    "dimensions (h × w)",
                    "candidate-mask nodata",
                    "feature nodata",
                ],
                "value": [
                    f"{summary.get('radiometric_convention')}",
                    ", ".join(
                        str(name)
                        for name in ((summary.get("input_metadata") or {})
                                     .get("pre", {})
                                     .get("band_names") or [])
                    )
                    or "unknown",
                    f"{grid.crs if grid else 'unknown'}",
                    f"{grid.resolution if grid else 'unknown'}",
                    f"{grid.shape if grid else 'unknown'}",
                    "255 (not observed)",
                    "-9999.0",
                ],
            }
        )
        st.caption(
            "Input RTC power is gamma-0 linear power, NOT sigma-0. The M5 "
            "segmentation baseline was trained on sigma-0 sigma-nought inputs, so "
            "no equivalence between them is claimed."
        )

    _render_aoi_scope(summary)

    st.markdown("### Candidates")
    recorded_pct = recorded.get("candidate_fraction_of_valid")
    comp_pct = computed.get("candidate_fraction_of_valid")
    st.caption(
        "Recorded by M6.3: "
        f"valid pixels {recorded.get('valid_pixels'):,} · "
        f"candidate pixels {recorded.get('candidate_pixels'):,} · "
        f"candidate % of valid {recorded_pct:.4%} · "
        f"candidate grid area {recorded.get('candidate_grid_area_km2'):.4f} km²"
        if all(
            isinstance(recorded.get(key), (int, float))
            for key in (
                "valid_pixels",
                "candidate_pixels",
                "candidate_fraction_of_valid",
                "candidate_grid_area_km2",
            )
        )
        else "Recorded summary figures are incomplete; see the checks below."
    )
    st.caption(
        "Recomputed from the rasters: "
        f"valid pixels {computed.get('valid_pixels', 0):,} · "
        f"candidate pixels {computed.get('candidate_pixels', 0):,} · "
        f"candidate % of valid {(comp_pct or 0.0):.4%} · "
        f"candidate grid area {computed.get('candidate_grid_area_km2', 0.0):.4f} km²"
    )
    _metrics = (
        "valid pixels",
        "candidate pixels",
        "candidate % of valid",
        "candidate grid area (km²)",
    )
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**As recorded by the M6.3 stage** (`rtc_change_summary.json`)")
        st.table(
            {
                "metric": list(_metrics),
                "recorded": [
                    f"{recorded.get('valid_pixels'):,}",
                    f"{recorded.get('candidate_pixels'):,}",
                    f"{recorded_pct:.4%}",
                    f"{recorded.get('candidate_grid_area_km2'):.4f}",
                ],
            }
        )
    with col_d:
        st.markdown("**Recomputed from the rasters** (independent cross-check)")
        st.table(
            {
                "metric": list(_metrics),
                "recomputed": [
                    f"{computed.get('valid_pixels', 0):,}",
                    f"{computed.get('candidate_pixels', 0):,}",
                    f"{(comp_pct or 0.0):.4%}",
                    f"{computed.get('candidate_grid_area_km2', 0.0):.4f}",
                ],
            }
        )

    if artifacts.recorded:
        failed = artifacts.failed_checks()
        if failed:
            st.error(
                "Recorded summary figures disagree with the rasters: "
                + "; ".join(f"{c.name} ({c.detail})" for c in failed)
            )
        else:
            st.success("Recorded summary figures agree with the rasters.")

    st.warning(
        "**Scope of the figures in this section.** These are mechanical tallies over "
        f"the assembled raster window ({computed.get('grid_area_km2', 0.0):.1f} km² at "
        "10 m), which is a bounding-box crop and NOT the study extent — "
        f"{computed.get('window_pixels_outside_aoi', 0):,} of its pixels fall outside "
        "the AOI. Extent-relative areas and rates are in the section above. Candidate "
        "pixels are Sentinel-1 RTC gamma-0 dual-polarisation darkening evidence — they "
        "are NOT confirmed flood, NOT damage and NOT ground truth."
    )

    with st.expander("Consistency checks (raster grid, bands, summary cross-check)"):
        for check in artifacts.checks:
            mark = "✅" if check.ok else "❌"
            st.markdown(f"{mark} `{check.name}` — {check.detail}")

    with st.expander("Artifact files"):
        for key, record in artifacts.files.items():
            st.markdown(f"- `{record.path}`")


# ---------------------------------------------------------------------------
# Workflow / blocker presentation
# ---------------------------------------------------------------------------


def _render_blocker(blocker: BlockerReport) -> None:
    if blocker.state is BlockerState.LOADED:
        st.markdown(f"**Reason:** {blocker.reason}")
        st.markdown(
            f"**Searched:** `{blocker.searched}` · "
            f"retrieval performed: {blocker.raw.get('retrieval_performed') if blocker.raw else None}"
        )
        if blocker.missing:
            st.markdown("**Missing prerequisites:** " + ", ".join(f"`{m}`" for m in blocker.missing))
        if blocker.authorisation:
            st.markdown(
                "**Authorisation gate:** AOI provenance "
                f"`{blocker.authorisation.get('aoi_provenance_status')}`, "
                f"`authorises_download={blocker.authorisation.get('authorises_download')}`"
            )
        if blocker.next_step:
            st.markdown("**To unblock (must not be bypassed):**")
            for index, step in enumerate(blocker.next_step, start=1):
                st.markdown(f"{index}. {step}")
        if blocker.not_done:
            st.markdown("**Explicitly not done:**")
            for item in blocker.not_done:
                st.markdown(f"- {item}")
        st.caption(f"Blocker report: `{blocker.path}`")
    elif blocker.state is BlockerState.MISSING:
        st.info(
            "No blocker report found; the reason this stage has no output is unknown "
            f"(expected `{blocker.path}`)."
        )
    else:
        st.error(
            "Blocker report is malformed and its contents cannot be trusted: "
            + "; ".join(blocker.problems)
        )


def _render_workflow(workflow: WorkflowReport) -> None:
    st.markdown("## Workflow dependency panel")
    st.caption(
        "Stage-by-stage state. A missing output is NOT zero impact — it means the "
        "analysis has not happened."
    )
    for status in workflow.stages:
        mark, label = STAGE_STYLE[status.state]
        st.markdown(f"{mark} **{status.title}** — {label}")
        st.caption(status.detail)
        for item in status.evidence:
            st.caption(f"   • {item}")
        if status.prerequisites:
            st.caption("   Requires: " + ", ".join(status.prerequisites))
        if status.blocker is not None:
            with st.expander(f"{status.title} — blocker detail"):
                _render_blocker(status.blocker)

    st.info(
        "**Real M6.4 / M6.5 execution requires, in order:**\n\n"
        + "\n".join(f"{i}. {step}" for i, step in enumerate(workflow.unblock_requirements, start=1))
    )
    if workflow.aoi_provenance_status is not None:
        st.caption(
            f"AOI provenance status `{workflow.aoi_provenance_status}` · "
            f"authorises_download `{workflow.authorises_download}`. This milestone "
            "does not change that status or retrieve data."
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _render_configured_aoi(aoi: ConfiguredAOI | None) -> None:
    if aoi is None:
        return
    st.caption(
        f"Configured study AOI: `{aoi.aoi_id}` · source SHA-256 `{aoi.source_sha256}` · "
        f"provenance `{aoi.provenance_status}`"
    )
    disclosure = aoi.coverage.get("disclosure")
    if isinstance(disclosure, str) and disclosure.strip():
        st.warning(disclosure)
        st.caption(f"Coverage disclosure read from `{aoi.provenance_path}`.")
    else:
        st.info(
            "Configured-AOI coverage disclosure unavailable. Observation coverage "
            "must not be assumed from the artifact grid or its valid-pixel fraction."
        )
    for problem in aoi.problems:
        st.error(problem)


def _configured_aoi_map(aoi: ConfiguredAOI | None):
    """Frame the pinned configured extent, even when result layers are withheld."""
    if aoi is None or aoi.bounds_4326 is None or aoi.geojson_path is None:
        return None
    west, south, east, north = aoi.bounds_4326
    overview = folium.Map(
        location=[(south + north) / 2.0, (west + east) / 2.0], tiles="OpenStreetMap"
    )
    folium.GeoJson(
        str(aoi.geojson_path),
        name="Configured study extent — context only, not observation coverage",
        style_function=lambda _: {"color": "#555555", "weight": 2, "fillOpacity": 0},
    ).add_to(overview)
    unobserved = aoi.coverage.get("unobserved_footprint_4326", {})
    if unobserved.get("crs") == "EPSG:4326" and unobserved.get("geometry"):
        folium.GeoJson(
            unobserved["geometry"],
            name="Unobserved AOI — unknown, not unaffected",
            style_function=lambda _: {
                "color": "#666666", "weight": 1, "fillColor": "#999999", "fillOpacity": 0.3,
            },
            tooltip="No delivered observations here. Unknown — not flood-free or undamaged.",
        ).add_to(overview)
    folium.LayerControl().add_to(overview)
    overview.fit_bounds([[south, west], [north, east]])
    return overview


def _render_real() -> None:
    root = resolve_project_root()
    artifacts = discover_m63_artifacts(root)

    st.title("Flood impact dashboard — real M6.3 artifacts")
    st.info(
        "**Data source: REAL M6.3 artifacts** "
        f"(`{artifacts.directory}`). Synthetic demo data is not used on this path."
    )
    _render_configured_aoi(artifacts.configured_aoi)

    if not artifacts.is_usable:
        st.error(
            f"Real M6.3 artifacts are not usable (state: `{artifacts.state.value}`). "
            "No real result is shown, and the synthetic demo is NOT substituted."
        )
        for problem in artifacts.problems:
            st.markdown(f"- {problem}")
        with st.expander("Consistency checks"):
            for check in artifacts.checks:
                mark = "✅" if check.ok else "❌"
                st.markdown(f"{mark} `{check.name}` — {check.detail}")
        if artifacts.configured_aoi is not None:
            st.markdown("## Workflow dependency panel")
            st.info(
                "Current-AOI workflow results unavailable: M6.3 change candidates, "
                "M6.4 infrastructure exposure and M6.5 network disruption / connectivity "
                "cannot be presented as results for the configured study extent. "
                "Existing artifacts have not been relabelled."
            )
            overview = _configured_aoi_map(artifacts.configured_aoi)
            if overview is not None:
                st.caption("Configured study extent only; no analytical result layers are shown.")
                st_folium(overview, width="100%", height=560, key="real-map")
        else:
            _render_workflow(build_workflow(artifacts, project_root=root))
        st.caption("Attribution (full): " + " ".join(REQUIRED_ATTRIBUTIONS))
        return

    workflow = build_workflow(artifacts, project_root=root)
    _render_workflow(workflow)
    _render_real_metrics(artifacts)

    bundle = build_real_dashboard(artifacts, workflow)
    enabled = _render_sidebar(bundle)
    overview = _configured_aoi_map(artifacts.configured_aoi)
    if overview is None:
        overview = folium.Map(location=_center_bounds(bundle), zoom_start=11, tiles="OpenStreetMap")
    errors: list[str] = []
    _add_layers(overview, bundle, {k for k, v in enabled.items() if v}, errors=errors)
    st_folium(overview, width="100%", height=560, key="real-map")
    if errors:
        st.warning("Some layers could not be rendered: " + "; ".join(errors))

    _render_provenance(bundle, enabled)
    st.caption("Attribution (full): " + " ".join(bundle.manifest.attribution))


def _load_demo_bundle():
    """Return the synthetic bundle, materializing it if needed."""
    if not DEFAULT_MANIFEST.exists():
        build_demo_bundle(DEFAULT_MANIFEST.parent)
    try:
        return load_manifest(DEFAULT_MANIFEST)
    except DashboardContractError as exc:
        st.error(f"Manifest contract failure: {exc}")
        st.stop()


def _render_demo() -> None:
    bundle = _load_demo_bundle()
    st.title("Flood impact dashboard — SYNTHETIC DEMO")
    st.warning(
        "DEMO — this dashboard is showing SYNTHETIC PLACEHOLDER data, not a "
        "real measurement or result. It exists to exercise the viewer and the "
        "artifact contract, and must never be presented as a flood map."
    )
    st.caption(
        "Layer groups (strongest → weakest): Observed/derived · Model prediction · "
        "Spatially inferred · Network inference · Validation reference."
    )

    enabled = _render_sidebar(bundle)
    centre = _center_bounds(bundle)
    overview = folium.Map(location=centre, zoom_start=13, tiles="OpenStreetMap")
    _add_layers(overview, bundle, {k for k, v in enabled.items() if v})
    st_folium(overview, width="100%", height=560, key="demo-map")

    _render_provenance(bundle, enabled)
    st.caption("Attribution (full): " + " ".join(bundle.manifest.attribution))


def main() -> None:
    st.sidebar.header("Data source")
    source = st.sidebar.radio(
        "Data source",
        [SOURCE_REAL, SOURCE_DEMO],
        index=0,
        key="data-source",
        help=(
            "Real: on-disk M6.3 artifacts. Demo: synthetic placeholder bundle. "
            "The app never falls back from one to the other."
        ),
    )
    if source == SOURCE_REAL:
        _render_real()
    else:
        _render_demo()


if __name__ == "__main__":
    main()
