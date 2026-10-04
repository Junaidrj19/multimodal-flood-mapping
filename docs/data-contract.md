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

- **TODO(define in features milestone).** Candidates from `architecture.md` §5:
  per-polarisation backscatter pre and post, and pre/post difference or ratio.
- Backscatter is typically handled in dB for differencing; whether this project
  does so is **TODO(verify)**
  (`configs/preprocessing.yaml → sentinel1.steps.convert_to_db`).

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
  ├── after_<scene_id>.tif
  └── provenance.yaml              # ArtifactProvenance, artifact_type=preprocessed_raster
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
- Water-sensitive indices (e.g. NDWI-type green/NIR or SWIR combinations) are
  plausible candidates but **not yet selected**
  (`configs/segmentation.yaml → features.indices: []`).

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
  ├── after_<scene_id>.tif
  ├── valid_mask_before.tif        # 1 = valid observation
  ├── valid_mask_after.tif
  └── provenance.yaml
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
| slope | TODO | Needed for radar geometry reasoning and error stratification. |
| aspect | TODO | Relevant to SAR shadow/layover. |
| flow direction / accumulation | TODO | Only for the optional hydrology bonus. |

`configs/preprocessing.yaml → dem.derivatives: []` — computed only when a
downstream stage actually consumes them. Void filling, if enabled, **REQUIRED**
to document method and reason.

### 3.5 Output artifact

```text
data/interim/<aoi>/dem/
  ├── elevation.tif                # aligned to target grid
  ├── <derivative>.tif             # only those actually used
  └── provenance.yaml              # dem_version REQUIRED
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

> ### VERIFIED(spec) — list resolved, properties not
>
> The official Track B specification enumerates the permitted training
> datasets: **Kuro Siwo** (production training input, MIT / CC BY, Bountos et
> al., 2024, NeurIPS 2024) and **Sen1Floods11** (optional training input, CC BY
> 4.0, Bonafilia et al., 2020, CVPRW 2020). The list is **closed** — training
> on anything outside it violates `AGENTS.md` §3.
>
> This resolves the previous UNKNOWN (blocking) on *which* datasets. It does
> **not** resolve their contents. Nothing below may be filled in from memory of
> these datasets' papers: every property must be read from the delivered data.
>
> Open and material: whether Kuro Siwo distinguishes debris/sediment from water.
> Track B requires both. See `docs/dataset-registry.md` §1.5.

The contract below specifies **what must be recorded for each of the two
datasets before it is used**, and applies to both equally.

### 5.1 Source

- **REQUIRED:** dataset name, version, provider, access URL, retrieval date.

### 5.2 License and citation metadata

- **REQUIRED:** license identifier and full text location.
- **REQUIRED:** the citation the license/paper demands. `AGENTS.md` §18 requires
  training datasets be cited per their respective licenses and papers.
- **REQUIRED:** confirmation that the license permits this use (a research
  prototype, publicly demonstrated).

### 5.3 Labels

- **REQUIRED:** class definitions **in the dataset's own words**, not ours.
- **REQUIRED:** an explicit mapping from dataset labels to project classes
  (`configs/segmentation.yaml → classes.label_mapping`).
- **Critical compatibility question:** does the dataset distinguish *flood
  water* from *debris/mud*? The project's stated target is "flood/debris"
  segmentation, and if the labels only cover open water then debris cannot be
  predicted and the claim must be narrowed. **UNKNOWN** until the list exists.
- **REQUIRED:** label provenance — manually annotated, semi-automatic, or
  threshold-derived — plus any stated label noise. A model cannot be more
  reliable than its labels.

### 5.4 Modalities

- **REQUIRED:** which modalities the dataset provides (SAR / optical / DEM),
  with bands, polarisations and processing level.
- **REQUIRED:** compatibility check against our inference-time inputs. Training
  on a modality combination we cannot reproduce at inference is a silent
  train/serve mismatch.

### 5.5 Expected preprocessing

- **REQUIRED:** the preprocessing the dataset was produced with, and whether it
  matches ours. A mismatch in calibration, speckle filtering or dB conversion is
  a domain shift that will not announce itself.

### 5.6 Train / validation / test separation

- **REQUIRED:** scene-level or geographic-block splits. Random pixel or random
  tile splits are **forbidden**
  (`configs/evaluation.yaml → splits.forbid_random_pixel_split: true`) because
  neighbouring pixels are strongly spatially correlated and such a split leaks
  across splits and inflates scores.
- **REQUIRED:** the unseen-Himalaya evaluation scenes are disjoint from training
  and validation, and are used **once**, after tuning is frozen
  (`AGENTS.md` §5).
- **REQUIRED:** if the dataset ships an official split, state whether we use it
  and why.

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
