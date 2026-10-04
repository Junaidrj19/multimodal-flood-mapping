# Scientific Assumptions

Multimodal AI Hackathon 2026 — Track B.

Every assumption that affects how a result may be interpreted. `AGENTS.md` §17
makes limitations part of the product, and §22 requires the system to answer
"How do you know this?" — which means being equally clear about what it does
*not* know.

Each assumption is stated as:

- **Claim the system can make** — what the evidence genuinely supports.
- **Claim the system must NOT make** — the overreach to avoid.
- **Why** — the mechanism behind the gap.
- **Consequence** — what this forces in the implementation, dashboard or report.

These are recorded **before** implementation deliberately, so they constrain the
build rather than being retrofitted as excuses afterwards.

---

## 1. Flood extent versus physical damage

**Claim the system can make:** a predicted flood/debris footprint — the area
where post-event imagery is consistent with water or debris presence.

**Claim the system must NOT make:** that the area is *damaged*.

**Why:** inundation and damage are different physical phenomena. A flooded field
may be undamaged; a debris-scoured road may be destroyed. Satellite surface
observation detects the former and only weakly constrains the latter. Nothing in
the permitted input set measures structural condition.

**Consequence:** the product maps **flood/debris extent**, not a damage
assessment. "Damage" appears only where a specific, stated inference supports
it. The dashboard must label this layer as predicted extent.

---

## 2. Spatial intersection versus infrastructure destruction

**Claim the system can make:** a building/road/bridge geometry **intersects** the
predicted footprint — i.e. it is *exposed*.

**Claim the system must NOT make:** that the feature is destroyed or
structurally failed.

**Why:** `AGENTS.md` §8 is explicit — "A flood mask intersecting a road is
evidence of spatial exposure, not automatic evidence of physical road
destruction." Intersection is a geometric relation between a predicted mask and
a pre-event vector, mediated by segmentation error, a chosen buffer width, and
positional error in both layers.

**Consequence:**
- Permitted vocabulary: *exposed*, *intersected*, *potentially disrupted*,
  *potentially inaccessible*, *potentially cut off*.
- Forbidden without stronger evidence: *destroyed*, *structurally failed*,
  *inaccessible*, *isolated*.
- Counts are reported as "features exposed", never "features destroyed".
- The buffer width is a documented, sensitivity-tested parameter, not a hidden
  constant (`configs/preprocessing.yaml → osm.road_buffer_m`, currently unset).

---

## 3. Road disruption versus confirmed road failure

**Claim the system can make:** a road segment is **potentially disrupted** under
the stated disruption rule.

**Claim the system must NOT make:** that the road is impassable.

**Why:** passability depends on water depth, flow velocity, debris depth,
surface damage and vehicle type — none of which are measured. `PRD.md` §13
records this directly: "Flood overlap does not prove a road is impassable." The
converse error also exists: a road outside the footprint may still be unusable
because its bridge failed or its approach was undercut.

**Consequence:**
- The disruption rule is explicit and configurable, with sensitivity analysis
  across thresholds (`AGENTS.md` §9).
- Reported as *potentially disrupted*.
- Bridges are modelled distinctly: a bridge is a single point of failure whose
  loss disconnects a route even when the surrounding surface is unaffected.
- A road predicted clear is **not** certified passable; absence of predicted
  flooding is weaker evidence than presence.

---

## 4. Modelled connectivity versus confirmed human isolation

**Claim the system can make:** under the disruption model, no route exists in
the **pre-event OSM road graph** from settlement *S* to reference hub *H*.

**Claim the system must NOT make:** that the settlement is cut off, or that
residents cannot leave or receive aid.

**Why:** the graph is a model of a subset of reality. It omits footpaths,
informal tracks, river crossings, helicopter access and roads missing from OSM.
`architecture.md` §18.8 states network disconnection is an inference about
modelled accessibility. Real access is a human and logistical question.

**Consequence:**
- Status vocabulary: `CONNECTED`, `POTENTIALLY DISRUPTED`,
  `POTENTIALLY CUT OFF`, `UNKNOWN / INSUFFICIENT DATA`.
- `UNKNOWN / INSUFFICIENT DATA` is a **first-class outcome**, not a fallback:
  where OSM coverage is too sparse to support a conclusion, that is the honest
  answer.
- All graph assumptions documented: road classes in scope, disruption threshold,
  settlement representation, hub definition, routing algorithm, treatment of
  disconnected components (`AGENTS.md` §9).
- **TODO(decide):** reference hub definition ("nearest town or hospital" is not
  yet operationalised).

---

## 5. Satellite acquisition time versus flood peak

**Claim the system can make:** surface conditions **at the acquisition
timestamp**.

**Claim the system must NOT make:** maximum flood extent, or conditions at any
other time.

**Why:** satellites observe on a fixed revisit cycle, uncorrelated with the
event. A post-event acquisition may precede the peak, follow substantial
recession, or fall days later. `PRD.md` §13 lists this as an explicit risk.
Flash floods and debris flows evolve over hours.

**Consequence:**
- Every flood-extent output carries its acquisition timestamp, and the
  **latency between event and acquisition** is reported prominently.
- Results are described as "extent observed at *T*", never "peak extent".
- Recession means extent can **understate** the affected area: a road flooded at
  the peak may appear clear. Debris, which persists, may be the better indicator
  of peak reach than water.
- The report must state the acquisition-to-event interval so a reader can judge
  relevance.

---

## 6. OpenStreetMap completeness

**Assumption made:** pre-event OSM is a *partial* inventory whose completeness
is unknown and spatially uneven.

**Claim the system must NOT make:** that exposure counts are complete, or that
absence from OSM means absence in reality.

**Why:** OSM is volunteer-contributed; rural mountainous Nepal is typically less
completely mapped than urban areas. `PRD.md` §13 records OSM incompleteness as a
risk. Attribute completeness (surface, width, bridge tags) is far patchier than
geometry.

**Consequence:**
- Counts are **estimates with a known downward bias** — unmapped features cannot
  be counted as exposed.
- The connectivity direction of bias is less obvious and worth stating: missing
  roads can make a settlement look *more* cut off (a real alternative route is
  invisible), while missing roads in the flooded zone can make it look *less*
  exposed. Both directions are possible.
- Where feasible, report an OSM coverage/density indicator alongside results.
- No population figures are derived from OSM: it carries no reliable population
  data, so any exposed-population number would need a separate cited source.

---

## 7. Sentinel-1 viewing geometry

**Assumption made:** SAR backscatter depends on viewing geometry, so only
comparable geometries may be differenced.

**Claim the system must NOT make:** that backscatter change between arbitrary
acquisitions is surface change.

**Why:** `AGENTS.md` §4 and the challenge both warn that different tracks view
terrain from different angles. In high relief this dominates: radar shadow,
layover and foreshortening vary with incidence angle and aspect, so a cross-track
difference contains large terrain-induced change unrelated to flooding.

**Consequence:**
- Same relative orbit is required by default; failure to find a pair is an
  explicit error, not a silent fallback
  (`configs/data.yaml → production.sentinel1.on_no_same_track_pair: fail`).
- Terrain correction is required in this terrain.
- **Low backscatter is ambiguous** — smooth open water, dry smooth surfaces
  (e.g. some bare rock, roads) and radar shadow all appear dark. Radar shadow in
  steep terrain is a systematic false-positive mechanism for water detection and
  must be in the error analysis (`configs/evaluation.yaml →
  error_analysis.inspect_failure_modes`).
- Wet soil and partially submerged vegetation are genuinely intermediate cases,
  not clean positives or negatives.
- Permanent water bodies must be distinguished from new flooding; otherwise the
  river itself is detected as flood every time.

---

## 8. Sentinel-2 cloud limitations

**Assumption made:** usable post-event optical imagery may not exist.

**Claim the system must NOT make:** that unobserved areas are unflooded.

**Why:** the case study is a monsoon-season event in a mountainous region —
among the least favourable optical conditions. Cloud, cloud shadow and terrain
shadow all remove observations. Cloud masks are themselves imperfect, and
mountain terrain makes cloud/snow/shadow discrimination harder.

**Consequence:**
- Cloud and invalid pixels are **masked and propagated**, never interpolated
  (`configs/preprocessing.yaml → sentinel2.invalid_pixel_policy: mask`).
  Interpolation would present reconstruction as observation.
- The pipeline distinguishes **"not observed"** from **"observed, not flooded"**.
  Collapsing these is a direct route to understating impact.
- SAR is the primary signal precisely because it is cloud-independent; optical
  is confirmatory where available.
- If no usable post-event optical scene exists, the system must say so and
  proceed SAR-only, with that limitation recorded in provenance and surfaced in
  the dashboard (`architecture.md` §17).
- Snow and ice complicate this further in Himalayan terrain — an ice/rock
  avalanche source region may be snow-covered, and snow is itself a
  misclassification risk.

---

## 9. Domain shift from training data to Himalayan scenes

**Assumption made:** training data probably does not represent Himalayan terrain,
so in-domain performance will overstate performance here.

**Claim the system must NOT make:** that a validation score transfers to the
Himalayan evaluation scenes.

**Why:** flood segmentation datasets are predominantly drawn from lower-relief
regions. Himalayan scenes differ in slope distribution, radar geometry effects,
snow/ice presence, land cover, river morphology and — critically — **debris
composition**: a rock-and-ice avalanche footprint may be unlike anything in the
training labels. `PRD.md` §13 lists label/domain shift as a risk.

**Consequence:**
- A dedicated **unseen-Himalaya** evaluation split, used once, after tuning is
  frozen (`AGENTS.md` §5, `docs/evaluation-protocol.md`).
- Both in-domain and unseen-Himalaya metrics are reported; the **gap between
  them is itself a headline result**, not a footnote.
- `AGENTS.md` §6: a strong score on an easy random split is not evidence of
  Himalayan generalisation.
- **Open risk:** if the permitted training datasets label only open water and not
  debris, then debris cannot be predicted and the product claim must be narrowed
  from "flood/debris" to "flood water". This is currently **UNKNOWN** because the
  dataset list is unavailable (`docs/dataset-registry.md` §1.5).

---

## 10. Cross-cutting assumptions

### 10.1 The DEM is a pre-event terrain prior

Copernicus DEM reflects terrain from its acquisition epoch (2010–2018 per the
required attribution), not current terrain. An avalanche and debris flow changes
real topography. DEM-derived flood paths therefore describe *pre-event* terrain
routing.

### 10.2 DEM flood-path tracing is not hydrodynamic simulation

`AGENTS.md` §10 — terrain-based inference only. It ignores water volume,
velocity, momentum, infiltration, channel roughness and debris rheology. It must
never be presented as a physically exact prediction.

### 10.3 Segmentation output is a prediction, not an observation

Model output carries error. It is labelled as *model prediction*, distinct from
*observed/derived* and *network inference* in both the dashboard
(`AGENTS.md` §16) and the report.

### 10.4 Reference products are comparisons, not ground truth

EMSR927 and similar carry their own method, timing and interpretation
assumptions. Disagreement is not automatically our error, and agreement is not
proof of correctness.

### 10.5 Spatial resolution bounds what can be claimed

At roughly 10 m imagery and ~30 m DEM, narrow features — single-lane roads,
small bridges, individual buildings — are near or below reliable detection size.
Per-building statements carry correspondingly low confidence.

### 10.6 The evidence-strength hierarchy is reported, not flattened

`PRD.md` §7 requires results be classified by evidence strength:

```text
Observed / directly derived
        ↓
Spatially inferred
        ↓
Network-level inferred
```

Confidence decreases down this chain. A "potentially cut-off settlement" sits at
the bottom — it depends on the segmentation being right, the buffer rule being
reasonable, OSM being complete and the graph model being valid. Every layer of
inference compounds the uncertainty above it, and the presentation must not
flatten that into a single confident number.

---

## 11. Assumptions that are NOT yet made

Recorded so a later agent does not mistake silence for a decision:

| Item | Status |
|---|---|
| Model architecture | **Not selected.** Must follow measured performance (`AGENTS.md` §5). |
| Decision threshold | **Not selected.** Chosen on validation, never on the Himalaya holdout. |
| Metric targets | **Not set.** A target before a baseline would be arbitrary. |
| Road buffer width | **Not set.** Drives exposure results; needs sensitivity analysis. |
| Road classes in scope | **Not decided.** Including footpaths materially changes connectivity. |
| Reference hub definition | **Not operationalised.** |
| Target grid CRS/resolution | **Not set.** Depends on AOI and product resolutions. |
| Permitted training datasets | **Specified:** Kuro Siwo (required), Sen1Floods11 (optional). Their label semantics are **not** yet known. |
