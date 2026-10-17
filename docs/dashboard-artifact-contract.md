# Dashboard artifact contract

Status: **implemented** (`src/floodmap/dashboard/`), exercised end-to-end by the
synthetic demo bundle (`src/floodmap/dashboard/demobundle.py`) and the Streamlit
app (`dashboard/app.py`).

## Purpose

The dashboard is a visualization and decision-support layer only
(`architecture.md` §15). It must never run acquisition, preprocessing,
segmentation, network or hydrology logic, and it must keep the evidence
strength of every claim visible (`AGENTS.md` §16, `docs/scientific-assumptions.md` §10.6).

Both rules are enforceable only if the frontend reads a **declared, tested
contract** rather than guessing file locations or computing values. This
document describes that contract. The authoritative implementation is
`src/floodmap/dashboard/contract.py`; this document mirrors it and cannot drift
silently, because `tests/test_dashboard_contract.py` pins the behaviour.

## Bundle layout

One AOI's generated dashboard inputs live in one directory:

```text
<aoi_dir>/
    dashboard_manifest.json     # top-level descriptor (this contract)
    layers/
        <stem>.tif              # raster artifact (UTM CRS, band names preserved)
        <stem>.geojson          # vector artifact (WGS84)
        <stem>_provenance.yaml  # ArtifactProvenance for that layer
```

`schema` is `floodmap.dashboard_manifest/1`. The loader rejects any other value
so a future breaking change is loud, not silent.

## Manifest fields

| Field | Meaning |
|---|---|
| `schema` | Contract version. Must equal `floodmap.dashboard_manifest/1`. |
| `aoi_id` | AOI identifier, carried from the acquisition chain. |
| `event_name`, `event_date` | Human-readable event identity. |
| `is_synthetic` | `true` only for placeholder/demo bundles. The UI shows an unmissable banner when set. |
| `attribution` | Must contain all three required strings (`AGENTS.md` §18); the manifest author cannot forget them. |
| `layers` | One `DashboardLayer` per displayable artifact, including unavailable ones. |

## Layer fields

| Field | Meaning |
|---|---|
| `layer_id` | Stable unique key within the manifest. |
| `title` | Plain-language label. No over-claiming (no "destroyed", no "isolated"). |
| `artifact_type` | Reuses `floodmap.utils.provenance.ArtifactType`. |
| `evidence_tier` | Closed enum below. This is the scientific honesty control. |
| `format` | `geotiff`, `geojson`, or `table`. |
| `path`, `crs`, `bbox` | Artifact location (relative to the manifest) and georeference. |
| `provenance_path` | Relative path to the linked `ArtifactProvenance` YAML. Mandatory for available layers. |
| `available` | Whether the layer could be produced. |
| `unavailable_reason` | Mandatory when `available` is `false`. Uses the same `Unknown`-style vocabulary as provenance. |

## Evidence tiers (strongest → weakest)

```
observed_derived   Observed or directly derived from permitted inputs
        ↓
model_prediction   A model's output; carries model error ("Predicted flood water extent")
        ↓
spatial_inference  Intersection of a prediction with reference data
                   (e.g. buildings/roads exposed). Depends on model AND reference.
        ↓
network_inference  Graph-based, e.g. "potentially cut off" settlements
        ↓
validation_reference  Comparison-only. NEVER a production input.
```

This refines the four presentation buckets of `AGENTS.md` §16 with the
`spatial_inference` split demanded by `PRD.md` §7 and
`docs/scientific-assumptions.md` §10.6. The UI renders each tier as a separate,
labelled toggle group.

## Rules enforced here rather than documented

1. **An available layer must resolve.** `path` and `provenance_path` are
   mandatory for available layers, and `load_manifest` re-checks both files
   exist and `provenance.artifact_type` agrees with the manifest entry.
2. **Absence is explicit.** An unavailable layer is *listed* with a reason
   (`architecture.md` §17). Omitting it would let "not produced" read as
   "nothing to report". If it is marked unavailable it must not keep a path.
3. **Validation references stay separated.** `validation_reference` layers are
   grouped apart in the UI, and a provenance record naming a validation-only
   source is rejected at parse time (`AGENTS.md` §3), so a comparison product
   can never appear as a production layer.
4. **The presentation layer does not run science.** A guard test
   (`TestDashboardIsPresentationOnly`) fails if `src/floodmap/dashboard/`
   imports any acquisition/preprocessing/features/segmentation/network/
   hydrology/evaluation logic.

## Synthetic demo bundle

`src/floodmap/dashboard/demobundle.py` generates a small, deterministic
(`seed=2026`) bundle under `artifacts/dashboard_demo/` so the viewer and its
tests exercise the real contract before pipeline artifacts exist. It is
unambiguously labelled: `is_synthetic=true`, every provenance note contains
`SYNTHETIC PLACEHOLDER`, and the app banners it. It includes one deliberately
unavailable layer (`flood_path_trace`) so the "explicit absence" rule is always
visible in the demo. No binaries are committed (`*.tif` is globally gitignored).

## Real M6.3 source

Since M6.6 the app has two sources, selected in the sidebar and never mixed:

* **Real M6.3 artifacts** (default) — `src/floodmap/dashboard/realdata.py`
  locates and validates `artifacts/m6_3/rtc_change_candidates/`;
  `src/floodmap/dashboard/realbundle.py` builds the manifest **in memory** and
  reuses the layers/provenance rendering above. Nothing is written to disk, and
  no manifest file is required from the science stage.
* **Synthetic demo bundle** — unchanged, `is_synthetic=true`, always bannered.

The real path may import `rasterio` (to read grids and count classes) but no
science module, and no `streamlit`/`folium`, so it stays unit-testable without the
dashboard extra. `src/floodmap/dashboard/render.py` provides the nodata-aware RGBA helpers:
not-observed (`255`), feature nodata (`-9999`), non-finite values and unexpected
class values all render fully transparent, because an invalid pixel must never be
painted as an observation.

Validation states are `valid`, `missing`, `malformed`, `inconsistent` and `stale`, and
only `valid` is rendered as a result. See `docs/m6.6-dashboard-integration.md`
for the full check list, the workflow stage/blocker model and the scope
reconciliation.

### M6.15 — current AOI binding and partial-observation disclosure

For a real workspace with `configs/data.yaml`, discovery requires both the
M6.3 summary's AOI ID and explicitly recorded source SHA-256 to match
`acquisition.aoi`. A matching ID without a hash is insufficient. The linked
artifact provenance must also name that AOI. Standalone fixtures without a
workspace configuration and the explicitly synthetic demo remain separate.

The real UI reads the hash-matching configured AOI sidecar, displays its coverage
disclosure verbatim, and frames the configured study polygon rather than the
historical raster crop. A stale set has **no analytical metrics or result
layers**; the outline is context only, not observed support. Missing results are
never zero exposure or zero disruption. No M6.2–M6.6 artifacts are relabelled.

> The study corridor extends beyond the currently available valid satellite observation coverage. Approximately 63.4% of the AOI has delivered imagery coverage; the remaining 36.6% is unobserved in the current dataset. Results apply only to the observed, valid-data area. Unobserved areas must not be interpreted as unaffected, flood-free, or undamaged.

### M6.16 — the 0.2.0 artifact contract the real path now requires

`discover_m63_artifacts()` validates `m6.3-rtc-change-candidates-0.2.0`, not
`0.1.0`. The differences that change what the dashboard will accept:

* **Seven required files**, not six: `rtc_change_features.tif`,
  `rtc_change_candidates.tif`, `rtc_valid_mask.tif`, **`rtc_aoi_mask.tif`**,
  `rtc_change_summary.json`, and the two provenance YAMLs. A `0.1.0` set, or a
  set missing the AOI mask, is reported as a problem and no result is rendered.
* **Band and grid agreement** across all four rasters, including the AOI mask
  (`inside_configured_aoi`), checked on CRS, shape and transform.
* **AOI-scoped invariants read from the rasters**: `candidate ⊆ valid`,
  `valid ⊆ AOI`, `candidate ⊆ AOI`, and a refusal of any result whose delivered
  or valid area exceeds the full extent area.
* **A mandatory `aoi_scope` block** in the summary, whose AOI-scoped counts the
  dashboard recomputes from the masks and cross-checks.

The vocabulary the UI must keep distinct is therefore: **study extent** (the
configured polygon — context, not evidence), **delivered observation support**
(where pixels exist), **valid observation support** (where the pre/post VV+VH
pair is jointly valid) and **candidate area** (rule-based darkening evidence).
The extent area is never presented as observed area, and the delivered raster
window is never used as an AOI denominator.

Before future real-path regeneration, carry polygon-clipped and window-wide
metrics separately, preserve valid/nodata masks, identify the unobserved part of
the full AOI, and bind AOI ID **and** hash plus acquisition-pair provenance across
stages. `floodmap.dashboard_manifest/1` and the synthetic demo are unchanged.

## Attribution

The manifest must carry the three required strings
(`floodmap.utils.provenance.REQUIRED_ATTRIBUTIONS`); the loader refuses a
manifest that omits any. Provenance records attach them automatically.
