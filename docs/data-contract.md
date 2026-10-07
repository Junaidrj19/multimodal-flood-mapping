# Data Contract

Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space.

This document defines the **expected interface** for every major data source. It
is the contract subsequent agents implement against: acquisition produces what
is described here, preprocessing consumes it, and features depend on the
guarantees stated.

## How to read this document

| Marker | Meaning |
|---|---|
| **REQUIRED** | The pipeline cannot be correct without this. |
| **TODO(verify)** | Not yet verified against an authoritative source. Must not be assumed. |
| **UNKNOWN (blocking)** | Cannot be determined from available information; blocks a downstream milestone. |
| *commonly documented — VERIFY* | Widely published mission characteristic, recorded for orientation but **not** yet confirmed for this project. Treat as a hypothesis until checked. |

Per `AGENTS.md` §11, nothing in this document is scientific validation. Values
marked *commonly documented* are starting points for verification, not
established project facts.

> ### Partially resolved by the official specification
>
> `AGENTS.md` §2 names the official Track B challenge specification as the
> primary requirement source. It has now been supplied and fixes: the event date
> (2026-08-26), the pre-event OSM snapshot (2026-07-27, ohsome API), the DEM
> source (Copernicus WorldDEM-30), the permitted training dataset list (Kuro
> Siwo, Sen1Floods11), EMSR927 as validation-only, and the three mandatory
> attributions. Those are marked **VERIFIED(spec)**.
>
> Still unresolved and therefore still TODO/UNKNOWN rather than guessed: the
> judge-selected AOI geometry, the acquisition windows around the event date,
> the target grid
> CRS/resolution, the reference hub definition, and every property that can only
> be measured from a delivered asset (delivered resolutions, CRS, vertical datum,
> band sets, orbit tracks, dataset label semantics).

## 0. Earth Observation acquisition interface (Milestone 1)

Milestone 1 implements metadata acquisition in `src/floodmap/acquisition/`.
The interface is deliberately provider-independent above
`floodmap.acquisition.providers.cdse.CdseOdataProvider`:

```text
AreaOfInterest + EventSpec + SearchWindow
        -> SearchRequest
        -> SceneProvider.search()       # metadata only
        -> SelectionOutcome              # deterministic, auditable
        -> AcquisitionManifest           # JSON or YAML
        -> ArtifactProvenance(ACQUISITION_MANIFEST)
```

### 0.1 AOI and time inputs

- `AreaOfInterest` accepts a caller-supplied EPSG:4326 bounding box or GeoJSON
  `Polygon`. There is no repository default geometry. `aoi_id`, CRS and the
  `is_synthetic` flag are retained in the manifest.
- `EventSpec` preserves an optional timezone-aware event instant. If only a
  date is known, the whole UTC event day is excluded from both half-open search
  windows rather than being guessed into the before or after side.
- `SearchWindow` requires positive, explicit before/after day counts. The
  acquisition block in `configs/data.yaml` leaves these values `null` until
  actual availability is inspected; the CLI accepts `--before-days` and
  `--after-days` overrides.

### 0.2 CDSE OData provider

`CdseOdataProvider` queries the CDSE OData Products endpoint. It emits
percent-encoded `$filter`, `$expand=Attributes`, `$top`, `$skip` and stable
`$orderby` parameters. The filter contains:

- Sentinel-1 collection `SENTINEL-1` and product type `IW_GRDH_1S`;
- Sentinel-2 collection `SENTINEL-2` and product types `S2MSI2A` or `S2MSI1C`;
- the AOI intersection and the explicit UTC interval; and
- an optional provider scene-cloud limit for Sentinel-2.

Every page is parsed into `Sentinel1Scene` or `Sentinel2Scene`. A malformed
response is a failed discovery, never a partial successful result. HTTP
timeouts, 401/403 responses and unsuccessful provider statuses remain distinct
through `ProviderTimeout`, `ProviderAuthError` and `ProviderHttpError`.
Catalogue search is metadata-only. `download()` is separate, requires the
explicit CLI `--download` flag, records byte count and SHA-256, and is disabled
by configuration by default. `CDSE_USERNAME` and `CDSE_PASSWORD` are referenced
by name in configuration and are never stored in the repository.

### 0.3 Selection contract

- Sentinel-1 selection only returns a pair whose two scenes report the same
  integer relative orbit and compatible pass direction. Missing/explicitly
  unknown orbit metadata produces `ORBIT_UNKNOWN`; cross-track scenes produce
  `ORBIT_MISMATCH`. No cross-track fallback exists. When no compatible pair is
  possible the manifest contains a `NoSameTrackPair` outcome.
- Sentinel-2 selection partitions candidates into the configured before/after
  windows and applies either `nearest_in_time` or
  `least_cloud_then_nearest`. A configured scene-cloud limit is applied before
  ranking. Missing cloud metadata is not treated as clear sky when a cloud
  filter or cloud-aware ranking needs it.
- Every discarded candidate is retained as a structured record with its scene
  ID, side, `RejectionReason` and human-readable rationale. Ties are broken by
  UTC acquisition time and scene ID.

### 0.4 Manifest shape

`AcquisitionManifest.to_json()` and `.to_yaml()` serialize the AOI, event,
resolved temporal plans, exact requests/URLs, per-sensor discovery status and
scene records, selection rationales/rejections, download outcomes, limitations
and linked provenance. A minimal dry-run excerpt is:

```yaml
status: dry_run
provider: cdse-odata
aoi:
  aoi_id: synthetic-demo
  crs: EPSG:4326
  is_synthetic: true
discoveries:
  sentinel-1:
    status: discovery_not_attempted
selections: {}
provenance:
  artifact_type: acquisition_manifest
  production_inputs: []
```

The linked provenance record always carries the mandatory challenge
attributions and uses `artifact_type: acquisition_manifest`. A successful
catalogue discovery does not imply a download.

## 0.5 Analysis-ready preprocessing interface (Milestone 2)

M2 consumes an `AcquisitionManifest` plus an explicit mapping from selected
scene IDs to local source-product paths. It does not query a catalogue or pick
replacement scenes:

```text
AcquisitionManifest
  + selected scene/product IDs
  + explicit local source paths
        -> PreprocessingInputs
        -> source validation
        -> sensor-specific masking/correction gates
        -> explicit AnalysisGrid
        -> alignment checks and resampling
        -> GeoTIFF + valid mask + QA JSON + provenance YAML
```

The real AOI and target grid remain unset in configuration. A production run
therefore fails before writing an analysis-ready artifact until the caller
supplies an AOI-bearing M1 manifest, source paths, complete target-grid
metadata, and the unresolved product-level processing metadata.

### 0.5.1 Grid and alignment

`AnalysisGrid` requires CRS, resolution, affine transform, width and height.
No CRS, origin, resolution, extent, transform tolerance or resampling method
is invented. `compare_grids` reports CRS, dimensions, resolution, transform,
extent and pixel-alignment checks; `require_aligned` raises an explicit
`AlignmentError` when a comparison cannot be justified. Reprojection to a
configured target grid records the source-grid report and method in QA and
provenance.

### 0.5.2 Sentinel-1

The local Rasterio engine handles already-readable raster assets, named
polarisation validation, invalid-pixel masking, explicit common-grid
preparation, and either preservation of a source-calibrated representation or
an explicitly requested linear-to-dB conversion. It does not claim to perform
SNAP-grade orbit-file application, radiometric calibration, or terrain
correction. Those states must be established in explicit source metadata, or
the run fails rather than guessing. Pair registration also requires explicit
`pair_geometry_verified: true` metadata.

### 0.5.3 Sentinel-2

Required named bands and any pixel-quality mask are configured explicitly.
Quality masks use configured valid values and are propagated as a separate
valid-observation mask. Scene-level catalogue cloud percentage is preserved as
metadata and is never used as AOI cloud percentage. If no pixel-level quality
mask is supplied, QA records that it was unavailable; no cloud threshold or
clear-sky conclusion is invented.

### 0.5.4 DEM and artifacts

DEM preprocessing accepts only source metadata identifying the configured
`Copernicus WorldDEM-30` product, including an exact source product ID and
version. Vertical datum is preserved when available and represented as
unknown otherwise. M2 does not derive slope, flow paths, hydrology or
settlement results.

Each successful output consists of a raster, a `1=valid observation` mask, a
machine-readable QA JSON file, and a YAML `ArtifactProvenance` record with
`artifact_type: preprocessed_raster`, the M1 acquisition manifest ID, source
product IDs, preprocessing/configuration versions, operations, and QA values.

---

## 0.6 Feature generation interface (Milestone 3)

M3 consumes M2 analysis-ready artifacts and produces model-ready features. It
does not query a catalogue, select or download scenes, resample, or classify:

```text
M2 preprocessed_raster artifacts
  + their ArtifactProvenance
  + their QA records
        -> FeatureInputs (band-name addressed)
        -> common-analysis-grid verification (no resampling)
        -> FeatureRegistry resolution from the closed catalogue
        -> per-feature computation with mask propagation
        -> feature stack + per-feature valid masks + registry + QA + provenance
```

The boundary ends at **model-ready feature artifacts**. M3 applies no
threshold, produces no class label, and makes no flood/debris decision. A
feature is evidence for the M4 segmentation model.

### 0.6.1 Input requirements

Each supplied M2 artifact **REQUIRED** to have:

| Requirement | Why |
|---|---|
| `<stem>.tif`, `<stem>_valid_mask.tif`, `<stem>_quality.json`, `<stem>_provenance.yaml` all present | A raster without its mask and provenance cannot be masked correctly or traced. |
| `artifact_type: preprocessed_raster` | M3 consumes analysis-ready products only. |
| non-empty `acquisition_manifest_id` | The chain to M1 and the source scenes must survive. |
| non-empty `production_inputs`, all within the feature set's `allowed_sources` | Enforces the `AGENTS.md` §3 boundary at load time. |
| named band descriptions | M3 addresses bands **by name, never by position**. |

Band naming is why M2's `write_raster_artifact` now takes a required
`band_names` argument: M2 validates its input bands by name, and before this
milestone those names were discarded on write. A downstream stage forced to
fall back on band *order* could feed the wrong channel into a formula, and the
resulting raster would look entirely plausible.

### 0.6.2 Band roles, not band names

Features are defined over **roles** — `vv`, `vh`, `green`, `red`, `nir`,
`swir16`, `elevation` — and `configs/features.yaml → band_roles` binds each role
to the band description M2 actually wrote.

The indirection exists because the delivered polarisation set (§1.1) and band
set (§2.3) are still **TODO(verify)**, and `configs/preprocessing.yaml` leaves
`required_polarizations` and `required_bands` null. Hard-coding `B03` as green,
or assuming VV+VH exists, would be an assumption about a product nobody has
inspected. An unbound role fails **by name**; it never resolves to a guess.

### 0.6.3 Sentinel-1 backscatter representation — REQUIRED

`configs/features.yaml → sentinel1.backscatter_representation` must be
`linear` or `decibel`, and must agree with
`configs/preprocessing.yaml → sentinel1.radiometric_calibration`. There is no
default.

This is a correctness gate, not a formatting preference. The same physical
change quantity requires different arithmetic in each representation:

| Representation | Change transform | Equivalent to |
|---|---|---|
| `decibel` | `post_dB - pre_dB` | `10·log10(post_linear / pre_linear)` |
| `linear` | `10·log10(post_linear / pre_linear)` | the decibel difference |

Applying a logarithm to decibel data yields a plausible-looking raster that is
physically meaningless, and nothing downstream would reveal it.
`tests/test_features_m3.py::TestNumericalFormulas::test_db_difference_equals_linear_log_ratio`
pins the equivalence the gate relies on.

### 0.6.4 Implemented features

The catalogue is **closed**. Every entry below carries, in
`feature_registry.json`, its family, source, inputs, exact transformation,
units, dtype, declared range, nodata policy, version, rationale, limitations
and citation where applicable.

Per-polarisation entries expand over `sentinel1.polarisation_roles`;
per-band entries expand over `sentinel2.spectral_band_roles`.

| Feature | Formula | Units | Range | Represents |
|---|---|---|---|---|
| `s1_<pol>_pre` / `_post` | M2 value, unchanged | dB or linear power | — | Absolute backscatter level on each date. Retained so a change value can be interpreted relative to its starting point. |
| `s1_<pol>_change_db` | `post − pre` (dB) **or** `10·log10(post/pre)` (linear) | dB | — | Log-ratio change. Speckle is multiplicative, so a ratio makes it additive with approximately terrain-independent statistics; the dB form is symmetric for reciprocal changes. |
| `s1_vv_vh_ratio_db_pre` / `_post` | `VV_dB − VH_dB` | dB | — | Co/cross-polarised ratio: responds to scattering mechanism, separating dark-but-rough from dark-and-smooth. Requires dual polarisation to exist. |
| `s2_<band>_pre` / `_post` | M2 value, unchanged | reflectance | `[0, 1]`\* | Spectral state on each date. |
| `s2_<band>_change` | `post − pre` | reflectance | `[-1, 1]` | Per-band reflectance change, isolating surface change from the static land-cover background. |
| `s2_ndwi_pre` / `_post` | `(green − nir) / (green + nir)` | dimensionless | `[-1, 1]`\*\* | NDWI (McFeeters 1996). Water absorbs NIR strongly while still reflecting green. |
| `s2_mndwi_pre` / `_post` | `(green − swir16) / (green + swir16)` | dimensionless | `[-1, 1]`\*\* | MNDWI (Xu 2006). SWIR suppresses the built-up false positives that affect NDWI. |
| `s2_ndvi_pre` / `_post` | `(nir − red) / (nir + red)` | dimensionless | `[-1, 1]`\*\* | NDVI (Rouse et al. 1974). Vegetation/water discrimination and SAR context, **not** a water indicator. |
| `s2_ndwi_change`, `s2_mndwi_change`, `s2_ndvi_change` | `post_index − pre_index` | dimensionless | `[-2, 2]` | Index change. This is what separates new inundation from permanent water: a river channel scores high on both dates and changes near zero. |
| `dem_elevation` | M2 value, unchanged | m | — | Terrain context constraining where water can plausibly stand. |
| `dem_slope_degrees` | `degrees(arctan(hypot(dz/dx, dz/dy)))`, Horn (1981) 3×3 kernel | degrees | `[0, 90]` | Hydrological plausibility and SAR geometry error stratification (§3.4). Radar shadow in steep terrain is a systematic false-positive mechanism for water detection. |

\* Holds for physically valid surface reflectance; atmospheric correction can
return values slightly outside it over dark surfaces.
\*\* Holds when both reflectances are non-negative.

Declared ranges are **reported, never enforced by clipping**. An excursion is
information — usually a slightly negative reflectance over a dark surface — and
clipping it would hide that while altering the recorded physical value.
`numerics.clip_to_valid_range` is `false` and enabling it requires a written
scientific justification.

Horn's kernel, written north-up as `z1 z2 z3 / z4 z5 z6 / z7 z8 z9`:

```text
dz/dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cellsize)
dz/dy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8 * cellsize)
```

**Explicitly not implemented in M3:** flow direction, flow accumulation,
watershed or settlement isolation, downstream flood routing, and road
connectivity. Those belong to later milestones (`architecture.md` §8–§10).

### 0.6.5 Numerical and mask contract

For every feature:

```text
valid_feature_pixel =
      valid(input_1) AND finite(input_1)
  AND valid(input_2) AND finite(input_2)
  AND ... AND in_mathematical_domain(transform)
```

- **REQUIRED:** no invalid input is ever replaced by a plausible scientific
  value. For backscatter this matters directly — flooring a zero at a small
  positive number would manufacture a very dark, water-like value out of a
  non-observation (§1.7).
- Log transforms require strictly positive inputs; a zero or negative input is
  invalid, with **no epsilon regularisation**.
- Normalised differences are invalid where the two bands sum to zero.
- Non-finite values (NaN, ±inf) are invalid in every transform.
- Slope: the one-pixel grid border is invalid because no 3×3 window exists, and
  an interior pixel is invalid if **any** of its nine window cells is invalid.
- Arithmetic is performed in float64 and cast once to the configured storage
  dtype, so dtype is a storage decision rather than an accuracy accident.
- Features sharing identical validity conditions receive identical masks.

### 0.6.6 Alignment — M3 never resamples

M3 requires every input to share one analysis grid, verified with M2's
`require_aligned` (§0.5.1). Incompatible CRS, transform, resolution,
dimensions, extent or pixel alignment is an explicit `FeatureGridError`.
`configs/features.yaml → grid.allow_resampling` must stay `false`: resampling
here would hide a real registration error behind an interpolation and silently
invalidate every change feature computed from the pair. Spatial normalisation
is M2's responsibility.

Terrain derivatives additionally **REQUIRE** a projected grid in metres and an
explicitly declared `terrain.elevation_unit: m`. The slope kernel divides an
elevation difference by the pixel size, so a geographic grid would produce a
gradient over degrees — not a slope, and not detectable from the output.

### 0.6.7 Normalisation — applied, never fitted

M3 emits scientifically interpretable physical quantities.
`configs/features.yaml → normalisation.method` is `null`.

When the segmentation milestone enables it, M3 **applies** parameters and never
**fits** them: the only data M3 holds is the scene being processed, and fitting
on it is the textbook form of test-time leakage. Enabling it requires an
explicit `statistics_path`, a `version`, and
`statistics_source: training_split_only` — matching
`configs/segmentation.yaml → features.normalisation.statistics_source`. Fitting
on the unseen Himalayan evaluation scenes is prohibited by `AGENTS.md` §5 and
would void the evaluation. Method, parameters, source and version are recorded
in the QA record on every run.

### 0.6.8 Training-dataset adapter boundary

`allowed_sources` deliberately excludes
`permitted-training-dataset`. Kuro Siwo and Sen1Floods11 differ from our
inference-time products in band naming, resolution, label conventions, metadata
and sensor representation (§5.4, §5.5), and adapting them *inside* the
production feature generator would make the scientific definition of a feature
depend on which corpus supplied it — a silent train/serve mismatch.

The required boundary is therefore:

```text
training corpus
      -> dataset adapter (segmentation milestone)
      -> M2-equivalent analysis-ready arrays + band names + valid masks
      -> the SAME FeatureRegistry definitions used in production
```

The adapter's obligation is to present arrays that satisfy §0.6.1 and bind the
same roles in §0.6.2. `feature_registry.json` is sufficient for this: it carries
the exact transform, inputs, units and nodata policy for every feature, so a
corpus can be mapped onto the feature contract without changing it.
`tests/test_features_m3.py::TestTrainingDataAdapterBoundary` asserts that no
dataset-specific identifier appears anywhere in the production feature package.

### 0.6.9 Output artifact

```text
<output>/features/<feature-set-id>/
  ├── features.tif                  # one band per feature; descriptions = feature names
  ├── features_valid_mask.tif       # one band per feature, same order; 1 = valid
  ├── feature_registry.json         # machine-readable feature contract
  ├── features_quality.json
  └── features_provenance.yaml      # ArtifactProvenance, artifact_type: feature_stack
```

This extends M2's `<stem>`-based convention (§1.8/§2.5/§3.5) rather than
introducing a parallel one.

The valid mask is a **band-per-feature stack, not a single plane**. Features do
not share validity: a Sentinel-1 change feature can be valid where a
Sentinel-2 index is cloud-masked. Collapsing them would either discard valid
SAR evidence or mark cloud-obscured optical pixels as observed, and
`docs/scientific-assumptions.md` §8 requires "not observed" to stay
distinguishable from "observed, not flooded". A combined-validity summary is
recorded in QA as a statistic and never substituted for the per-feature masks.

Provenance **REQUIRED** fields: `artifact_type: feature_stack`, the M1
acquisition manifest ID(s), the Sentinel-1/Sentinel-2 before/after scene pairs
carried forward from M2, DEM version, union of source product IDs, feature
pipeline and configuration versions, and the operations applied. The QA record
carries `classification_performed: false`.

### 0.6.10 Determinism

Identical inputs and configuration produce byte-identical `features.tif`,
`features_valid_mask.tif`, `feature_registry.json` and `features_quality.json`.
Feature order follows catalogue order and then configured role order, so it is
independent of mapping iteration and of the order templates are listed in
configuration. Neither the registry nor the QA record contains a timestamp;
`generated_at` in the provenance record is the only time-varying field.

### 0.6.11 Production status

**Feature generation against real Trishuli data has not been run.** The engine
is implemented and tested against synthetic M2-compatible artifacts only. A
production run fails explicitly until the operator supplies the delivered
polarisation and band bindings, the backscatter representation, the elevation
unit, the output nodata sentinel, and the M2 target grid that
`configs/preprocessing.yaml` still leaves null. No AOI coordinate, scene ID,
band name or feature distribution has been invented to fill those gaps.

---

## 1. Sentinel-1 (SAR)

**Role:** cloud-independent observation; primary pre/post change-detection
signal under monsoon conditions (`architecture.md` §3).

### 1.1 Expected acquisition metadata

Every scene **REQUIRED** to carry:

| Field | Requirement | Notes |
|---|---|---|
| `scene_id` | REQUIRED | Provider granule identifier. |
| `acquired_at` | REQUIRED | ISO 8601 **with timezone**. `AGENTS.md` §4 requires timestamps be preserved. |
| `relative_orbit` | REQUIRED | Needed to verify same-track comparison. Absence must be recorded as `Unknown`, never defaulted. |
| `orbit_direction` | REQUIRED | ASCENDING / DESCENDING. |
| `platform` | REQUIRED | Provider mission/unit metadata. CDSE mission metadata is retained and the individual unit is `Unknown` when the product name does not expose a documented prefix. |
| `product_type` | REQUIRED | TODO(verify) — GRD is the usual choice for amplitude change detection, *commonly documented — VERIFY*. |
| `polarisations` | REQUIRED | TODO(verify) — IW mode commonly provides VV+VH, *commonly documented — VERIFY*. |
| `incidence_angle` | REQUIRED | Needed to reason about geometric comparability. |

### 1.2 Before/after relationship

- Exactly one **before** and one **after** scene per change-detection pair,
  recorded together (`floodmap.utils.provenance.BeforeAfterPair`).
- `before.acquired_at` **must** precede the event date; `after.acquired_at`
  **must** follow it.
- Temporal windows must not be silently mixed (`AGENTS.md` §4). The search
  window is configuration-driven: `configs/data.yaml → event.window_before_days`
  / `window_after_days`, both currently `null`/TODO.
- **TODO(decide):** behaviour when multiple candidate scenes qualify. Options
  are nearest-in-time, least-cloud (N/A for SAR), or explicit operator choice.
  Not decided; must not default silently.

### 1.3 Orbit-track requirement

This is a hard scientific constraint, not a preference:

> Different tracks view terrain from different angles and should not be compared
> pixel-by-pixel as though they were equivalent observations.
> — `AGENTS.md` §4, `README.md` §4

- Default: `require_same_relative_orbit: true`.
- If no same-track pair exists, the pipeline **fails explicitly**
  (`configs/data.yaml → production.sentinel1.on_no_same_track_pair: fail`).
  Falling back to a cross-track pair silently would produce terrain-induced
  change that is indistinguishable from flood signal.
- A same-track claim is only recordable when **both** relative orbits are known;
  the provenance schema enforces this.

### 1.4 Spatial resolution assumptions

- **TODO(verify).** IW GRD products are *commonly documented* at ~10 m pixel
  spacing with coarser true resolution — **VERIFY** before relying on it.
- Pixel spacing and true spatial resolution are **not** the same thing, and the
  distinction matters when reporting minimum mappable flood extent and when
  intersecting with OSM features.

### 1.5 CRS and raster alignment requirements

- **REQUIRED:** CRS, transform, resolution, extent and pixel alignment are
  verified before any pixel-level comparison (`AGENTS.md` §7).
- Target analysis grid: `configs/preprocessing.yaml → target_grid` (currently
  `null`/TODO — the correct projected CRS depends on the AOI).
- Resampling method and reason **must** be documented.
- Terrain correction is **REQUIRED** in mountainous terrain, where
  radar geometry effects are severe. **TODO(verify)** the exact correction chain
  against the product level used.

### 1.6 Expected channels / features

- **RESOLVED (Milestone 3).** Per-polarisation backscatter pre and post, the
  log-ratio change, and the co/cross-polarised ratio on both dates. Exact
  formulas, units and masking behaviour are in §0.6.4.
- Which polarisations actually expand is driven by
  `configs/features.yaml → sentinel1.polarisation_roles`, still null. Dual
  polarisation is not assumed to exist.
- Backscatter representation is no longer left implicit: whether differencing
  happens in dB or as a linear log-ratio is a **required**, explicit setting
  (§0.6.3), because the two need different arithmetic for the same quantity.
  It must agree with
  `configs/preprocessing.yaml → sentinel1.radiometric_calibration`.

### 1.7 NoData handling

- **REQUIRED:** a NoData value is declared and preserved; `configs/preprocessing.yaml → sentinel1.nodata_value` (TODO).
- NoData must never be silently treated as a valid low-backscatter value — low
  backscatter is itself a candidate water signal, so conflating the two would
  manufacture flood detections.
- Border noise and invalid edge regions must be masked, not interpolated.

### 1.8 Output artifact

```text
data/interim/<aoi>/<event_date>/sentinel1/
  ├── before_<scene_id>.tif        # aligned to target grid
  ├── before_<scene_id>_valid_mask.tif
  ├── before_<scene_id>_quality.json
  ├── before_<scene_id>_provenance.yaml
  ├── after_<scene_id>.tif
  ├── after_<scene_id>_valid_mask.tif
  ├── after_<scene_id>_quality.json
  └── after_<scene_id>_provenance.yaml # ArtifactProvenance, preprocessed_raster
```

Provenance **REQUIRED** fields: both scene IDs, both acquisition timestamps,
both relative orbits, `same_relative_orbit`, preprocessing version, target grid
CRS and resolution.

---

## 2. Sentinel-2 (optical)

**Role:** optical confirmation and multispectral features when cloud permits
(`architecture.md` §3).

### 2.1 Acquisition metadata

| Field | Requirement | Notes |
|---|---|---|
| `scene_id` | REQUIRED | Provider granule identifier. |
| `acquired_at` | REQUIRED | ISO 8601 with timezone. |
| `processing_level` | REQUIRED | TODO(verify) — L2A (surface reflectance) is *commonly documented* as preferable to L1C for change work; **VERIFY** availability for the event window. |
| `tile_id` | REQUIRED | MGRS tile. |
| `scene_cloud_percent` | REQUIRED | Provider-reported scene-level cloud fraction. |
| `aoi_cloud_percent` | REQUIRED | Cloud fraction **within the AOI**, which can differ greatly from the scene value and is the figure that actually matters. |

### 2.2 Cloud and invalid-pixel handling

- **REQUIRED:** cloud, cloud shadow, cirrus and saturated/defective pixels are
  masked, not interpolated
  (`configs/preprocessing.yaml → sentinel2.invalid_pixel_policy: mask`).
- Interpolating across a cloud gap would present a reconstruction as an
  observation. If interpolation is ever enabled it requires written
  justification and must be recorded in provenance `limitations`.
- **REQUIRED:** the per-pixel valid-observation mask is propagated downstream so
  that "not observed" is distinguishable from "observed, not flooded". These are
  different findings and must not collapse into one.
- **TODO(verify):** which cloud mask product is used
  (`cloud_mask_source`), and the AOI cloud threshold above which an optical
  scene is rejected (`max_scene_cloud_percent`). Do not pick a threshold without
  inspecting actual availability.

### 2.3 Relevant bands

- **TODO(verify).** Band selection must follow the label and feature
  requirements of the permitted training datasets (Kuro Siwo, Sen1Floods11),
  whose band/label expectations are TODO(verify) until the data is inspected
  (see §5).
- *Commonly documented — VERIFY*: Sentinel-2 MSI provides visible/NIR bands at
  10 m, red-edge/SWIR at 20 m, and atmospheric bands at 60 m. Any multi-band
  stack therefore requires an explicit, documented resampling decision.
- Water-sensitive indices are now **selected and defined**: NDWI (green/NIR),
  MNDWI (green/SWIR) and NDVI (NIR/red), each pre, post and as a change
  feature. Definitions, citations and masking behaviour are in §0.6.4. They are
  computed over *roles*, so each remains unavailable until the corresponding
  band description is bound in `configs/features.yaml → band_roles` (§0.6.2).
- An index is a feature, not a classification rule. No water threshold is
  applied in M3.

### 2.4 Spatial alignment

- **REQUIRED:** resampled onto the same target grid as Sentinel-1 and the DEM.
- Mixed native resolutions (10/20/60 m) mean band-dependent resampling; method
  per band **REQUIRED** to be documented.
- **REQUIRED:** geolocation co-registration against the SAR grid is verified,
  not assumed (`AGENTS.md` §7).

### 2.5 Output artifact

```text
data/interim/<aoi>/<event_date>/sentinel2/
  ├── before_<scene_id>.tif
  ├── before_<scene_id>_valid_mask.tif # 1 = valid observation
  ├── before_<scene_id>_quality.json
  ├── before_<scene_id>_provenance.yaml
  ├── after_<scene_id>.tif
  ├── after_<scene_id>_valid_mask.tif
  ├── after_<scene_id>_quality.json
  └── after_<scene_id>_provenance.yaml
```

---

## 3. Copernicus DEM

**Role:** terrain context, slope/elevation features, optional downstream
flood-path tracing (`architecture.md` §3).

### 3.1 Elevation data

- **TODO(verify):** exact product and variant. `README.md` §12 requires
  attribution to **Copernicus WorldDEM-30**, which indicates the 30 m family,
  but the precise product identifier and version for this project are not
  stated. Do not assume a specific GLO variant without checking.
- **REQUIRED:** the vertical datum and whether elevations are orthometric or
  ellipsoidal must be recorded. Mixing these produces metre-scale errors, which
  matters directly for any terrain-based flood-path inference.
- **REQUIRED:** the DEM represents **pre-event** terrain. After an avalanche and
  debris flow the real terrain has changed, so the DEM is a *prior*, not a
  current observation. This must be stated wherever DEM-derived results appear.

### 3.2 CRS

- **TODO(verify)** native CRS; reprojected to the target grid in preprocessing.

### 3.3 Resolution

- **TODO(verify).** ~30 m is implied by "WorldDEM-30" but **must be confirmed**.
- Resolution mismatch against 10 m imagery is a real limitation: terrain
  derivatives will be smoother than the optical/SAR detail and cannot resolve
  narrow channels or individual road cuttings.

### 3.4 Terrain derivatives (only if used)

| Derivative | Status | Note |
|---|---|---|
| slope | **IMPLEMENTED (M3)** | Horn (1981) 3x3 kernel, degrees. Radar geometry reasoning and error stratification. Requires a projected metre grid and a declared elevation unit (§0.6.4, §0.6.6). |
| aspect | NOT IMPLEMENTED | Relevant to SAR shadow/layover, but no downstream stage consumes it yet. Adding it without a consumer would be an unused claim. |
| flow direction / accumulation | NOT IMPLEMENTED | Deliberately out of scope for M3; only for the optional hydrology bonus. |

`configs/preprocessing.yaml → dem.derivatives: []` — computed only when a
downstream stage actually consumes them. Void filling, if enabled, **REQUIRED**
to document method and reason.

### 3.5 Output artifact

```text
data/interim/<aoi>/dem/
  ├── elevation.tif                # aligned to target grid
  ├── elevation_valid_mask.tif     # 1 = valid observation
  ├── elevation_quality.json
  ├── elevation_provenance.yaml    # dem_version REQUIRED
  └── <derivative>.tif             # only those actually used later
```

---

## 4. OpenStreetMap (pre-event)

**Role:** pre-event buildings, roads and bridges (`architecture.md` §3).

### 4.1 Pre-event snapshot requirement

This is a **hard data rule**, not a convenience:

- The snapshot **must** predate the event date. Post-event OSM edits are
  **validation-only** (`AGENTS.md` §3).
- **Why it matters:** after a disaster, mappers add damage-related detail.
  Using a current extract would import post-event human knowledge of the damage
  into a system that claims to derive damage from satellite data — leakage that
  would invalidate the entire result.
- **REQUIRED:** `osm_snapshot_date` recorded in provenance, and asserted to be
  earlier than `event_date`.
- **TODO(verify):** which historical extract provider/mechanism is used
  (`configs/data.yaml → production.osm.snapshot_source`).

### 4.2 Buildings

| Field | Requirement |
|---|---|
| `osm_id` | REQUIRED |
| geometry | REQUIRED — polygon |
| `building` tag | REQUIRED |
| population / occupancy | **UNKNOWN** — OSM does not reliably carry this. Any exposed-population figure would be an inference requiring a separate, cited source. Do not compute one from building counts alone. |

### 4.3 Roads

| Field | Requirement |
|---|---|
| `osm_id` | REQUIRED |
| geometry | REQUIRED — LineString |
| `highway` tag | REQUIRED — road class |
| `bridge` / `tunnel` tags | REQUIRED — a flooded *tunnel* and a flooded *bridge* are different disruption cases |
| `oneway` | REQUIRED for routing correctness |
| `surface`, `width` | OPTIONAL — sparsely populated in practice |

- **TODO(decide):** which `highway` classes are in scope. Including footpaths
  changes connectivity conclusions substantially; this is a scientific choice
  requiring justification.

### 4.4 Bridges

- Derived from `bridge=yes` on ways, **not** a separate layer.
- **REQUIRED:** treated as distinct from ordinary road segments — a bridge is a
  single point of failure whose loss disconnects a route even when the
  surrounding road surface is unaffected.

### 4.5 Geometry requirements

- **REQUIRED:** valid geometries; invalid ones repaired with documented method
  or excluded with a recorded count. Silently dropping features would bias
  exposure counts downward.
- **REQUIRED:** reprojected to the target grid CRS before any intersection.
- **REQUIRED:** length and area computed in a projected CRS, never in degrees.
- Road buffer width (`configs/preprocessing.yaml → osm.road_buffer_m`) is
  **TODO** and deliberately unset: it directly determines what counts as
  "exposed" and requires a sensitivity analysis (`AGENTS.md` §9).

### 4.6 Output format

```text
data/interim/<aoi>/osm/
  ├── buildings.parquet            # or GeoPackage — TODO(decide)
  ├── roads.parquet
  ├── bridges.parquet
  └── provenance.yaml              # osm_snapshot_date REQUIRED
```

**TODO(decide):** vector interchange format. GeoParquet and GeoPackage are both
candidates; the choice affects tooling and should be fixed once the geospatial
dependency set is pinned.

---

## 5. Training datasets

> ### AUDITED — properties verified from primary sources
>
> The permitted list is **closed**: **Kuro Siwo** (required) and
> **Sen1Floods11** (optional). Training on anything outside it violates
> `AGENTS.md` §3.
>
> Both corpora have now been audited against their own repositories, LICENSE
> files and papers. The full audit with source URLs and verbatim quotes is in
> `docs/dataset-registry.md` §1.5; the machine-readable form is in
> `configs/data.yaml → production.training_datasets.datasets`. This section
> records only what the *contract* depends on.

### 5.1 What was resolved

| Question | Answer |
|---|---|
| Does either dataset label debris or sediment? | **No.** Both class lists are exhaustive and water-only. |
| Does either separate permanent water from flood water? | **Kuro Siwo yes** (3-class), **Sen1Floods11 no** (binary). |
| Do they share label semantics? | **No.** See §5.3. |
| Do they ship optical data? | **Kuro Siwo no. Sen1Floods11 yes** — L1C TOA, 13 bands. |
| Do they ship a DEM? | **Kuro Siwo yes** (SRTM 1Sec). **Sen1Floods11 no.** Neither uses Copernicus WorldDEM-30. |
| Do they agree on SAR representation? | **No. Kuro Siwo is linear σ⁰** (inferred from its SNAP graph); **Sen1Floods11 is dB** (stated). |
| Do they provide pre-event imagery? | **Kuro Siwo: 2 pre + 1 post. Sen1Floods11: post only.** |
| Official splits? | **Both event/geography based.** Must be honoured. |
| Licences? | **Neither is cleanly established.** See §5.4. |

### 5.2 Label taxonomies — VERIFIED(source)

**Kuro Siwo** — stored label mask, with validity held **separately**:

```text
mask.npy        0 = No water
                1 = Permanent Waters
                2 = Floods          (exhaustive — there is no fourth value)

valid_mask.npy  0 = invalid
                1 = valid           (a SEPARATE raster, not a label value)
```

> **Correction.** An earlier revision of this contract recorded
> `3 = Invalid pixels (ignored)` as a stored class with `ignore_index`
> semantics. That was wrong. The label mask carries only `{0, 1, 2}`; validity
> is a separate binary raster that the upstream loader reads independently and
> applies as `valid_mask == 1`. The `3: "Invalid pixels"` entry does exist in
> the upstream `CLASS_LABELS` dict, but `CLASS_LABELS[3]` is **never referenced
> anywhere in that codebase**, and no `ignore_index` key exists in any of its
> configs.
>
> The error mattered rather than being cosmetic: an adapter written against it
> would have filtered `label == 3`, matched nothing, and then trained on
> invalid pixels while appearing to handle them. **REQUIRED:** the adapter reads
> two rasters per sample and derives the loss/metric mask from `valid_mask`.

**Sen1Floods11** (hand-labelled QC layer, README):

```text
-1 = No Data / Not Valid
 0 = Not Water
 1 = Water
```

### 5.3 The two taxonomies are not interchangeable

This is the contract-relevant consequence, and it is asymmetric:

```text
Kuro Siwo  {Permanent Waters, Floods}  --lossy-->  Sen1Floods11 {Water}
Sen1Floods11 {Water}                   --IMPOSSIBLE-->  {Permanent, Flood}
```

- Mapping Kuro Siwo **down** to binary surface water is well defined but
  **discards the permanent-vs-flood distinction** — the one thing that stops a
  river being reported as flood on every run
  (`docs/scientific-assumptions.md` §7).
- Mapping Sen1Floods11 **up** to the 3-class scheme is **not possible from its
  labels**. Its `Water` class conflates both. The JRC permanent-water chips are
  a *separate subset*, not a class value, so they cannot relabel the flood
  chips.
- Therefore a naive union of the two corpora under one head is **not permitted**
  by this contract: it would either silently relabel Sen1Floods11 flood water as
  a 3-class value it does not carry, or silently collapse Kuro Siwo's most
  valuable distinction. The harmonisation rule is specified in
  `docs/evaluation-protocol.md` §2.5 and the architectural resolution in
  `docs/m4-architecture-decision.md`.

**REQUIRED:** `configs/segmentation.yaml → classes.label_mapping` must state the
mapping per source dataset, and must record which direction was used and what
was lost.

### 5.4 Licensing — REQUIRED to carry, not yet established

Per `AGENTS.md` §18 training datasets must be cited per their licence and paper.
Both must be cited; neither licence is confirmed.

| Dataset | Repository LICENSE | README | Paper | Spec claim | Conservative reading |
|---|---|---|---|---|---|
| Kuro Siwo | **MIT** | **CC BY** (no version) | MIT | MIT | **Data CC BY + code MIT; attribution required** |
| Sen1Floods11 | **ABSENT** (API `license: null`) | none | CVF boilerplate only | CC BY 4.0 (**unconfirmed**) | **No redistribution right established** |

- **REQUIRED:** citations are carried on any artifact or publication derived
  from either corpus. Both full citations are in
  `docs/dataset-registry.md` §1.5.
- **REQUIRED:** no sample from either dataset is committed to this repository
  until the licence is resolved. `redistribution_permitted` is `null` for Kuro
  Siwo and `false` for Sen1Floods11.
- Discrepancies are **preserved, not collapsed**. Asserting a single licence
  identifier would be a claim this project cannot support.

### 5.5 Required records per dataset

Unchanged in intent from the original contract; now partly satisfied by the
audit. Each dataset must have on file: source and retrieval date; licence and
citation metadata; class definitions **in the dataset's own words**; an explicit
mapping to project classes; label provenance and any stated noise; modalities
with bands, polarisations and processing level; the preprocessing it was
produced with; and its train/validation/test separation.

Still outstanding: Kuro Siwo's flood definition and stated label noise (paper
Supplemental Material), Sen1Floods11's weak-label encodings and exact test
split, and a confirmation of Kuro Siwo's linear-vs-dB representation from a
delivered raster rather than from its SNAP graph.

### 5.6 Train / validation / test separation

- **REQUIRED:** scene-level or geographic-block splits. Random pixel or random
  tile splits are **forbidden**
  (`configs/evaluation.yaml → splits.forbid_random_pixel_split: true`) because
  neighbouring pixels are strongly spatially correlated and such a split leaks
  across splits and inflates scores.
- **REQUIRED — and now actionable:** both datasets ship official event-based
  splits, and they must be honoured. Kuro Siwo's upstream **test** activations
  (`321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013`) must not
  enter our training or validation split. See
  `docs/evaluation-protocol.md` §2.4.
- **REQUIRED:** the unseen-Himalaya evaluation scenes are disjoint from training
  and validation, and are used **once**, after tuning is frozen
  (`AGENTS.md` §5).

> **Material finding.** Neither corpus is documented as containing Himalayan or
> high-mountain terrain. Kuro Siwo's only Nepal-labelled activation is tropical
> and lowland by its own metadata *and* sits in the upstream test split. This
> means there is **no labelled Himalayan test set inside the permitted data**,
> so the unseen-Himalaya requirement cannot be met with permitted labels. See
> `docs/dataset-registry.md` §5.3 and `docs/evaluation-protocol.md` §2.6.

See `docs/evaluation-protocol.md` for the full protocol and
`docs/dataset-registry.md` for the registry these entries populate.

---

## 6. Cross-source invariants

These hold across every source and are the contract's non-negotiable core:

1. **No validation-only source** (EMSR927, Copernicus EMS, UNOSAT, published
   damage maps, post-event OSM) reaches training, feature construction,
   preprocessing, inference or threshold selection (`AGENTS.md` §3).
2. **Alignment is verified, never assumed** (`AGENTS.md` §7).
3. **Acquisition timestamps are preserved** end to end (`AGENTS.md` §4).
4. **Geospatial metadata is preserved** end to end (`AGENTS.md` §7).
5. **Every artifact carries provenance** (`architecture.md` §18.9), using
   `floodmap.utils.provenance.ArtifactProvenance`.
6. **"Not observed" is distinguishable from "observed, negative."** Cloud gaps,
   SAR shadow and NoData are not evidence of absence of flooding.
7. **Insufficient data fails explicitly** rather than producing a misleading map
   (`architecture.md` §17).

## 7. Open questions blocking implementation

| # | Question | Blocks |
|---|---|---|
| 1 | Kuro Siwo label semantics: is debris/sediment distinct from water? | Debris class, product claim |
| 2 | Judge-selected AOI geometry and canonical case-study extent | Case-study acquisition run |
| 3 | Pre/post acquisition windows around 2026-08-26 | Case-study acquisition run |
| 4 | Target grid CRS and resolution | Preprocessing |
| 5 | WorldDEM-30 delivered grid and vertical datum | DEM handling, hydrology bonus |
| 6 | ohsome API query shape and snapshot verification | OSM acquisition |
| 7 | In-scope `highway` classes | Network analysis |
| 8 | Road buffer width + sensitivity range | Infrastructure exposure |
| 9 | Reference hub definition (nearest town/hospital) | Connectivity analysis |
| 10 | Vector interchange format | OSM output |

---

## Attribution

Per `AGENTS.md` §18, any submission using these sources must carry:

> Contains modified Copernicus Sentinel data 2026.

> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus
> Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European
> Union and ESA; all rights reserved.

> © OpenStreetMap contributors.
