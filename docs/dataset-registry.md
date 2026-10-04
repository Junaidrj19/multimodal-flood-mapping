# Dataset Registry

Multimodal AI Hackathon 2026 — Track B.

Authoritative list of every dataset the project touches, and — critically —
whether each one is permitted as a **production input** or restricted to
**validation/comparison only**.

This registry exists so the `AGENTS.md` §3 data rule is auditable from one
place. `tests/test_data_boundary.py` enforces the boundary mechanically.

## Field definitions

| Field | Meaning |
|---|---|
| `dataset` | Name and version. |
| `purpose` | What it is used for in this project. |
| `allowed_as_input` | **YES** = may enter the production pipeline. **NO** = must never. |
| `validation_only` | YES = usable only for post-hoc comparison, after a production prediction exists. |
| `license` | License identifier. |
| `citation` | Required attribution/citation. |
| `temporal_requirement` | Constraint on acquisition/snapshot time. |
| `spatial_information` | CRS / resolution / extent, where known. |
| `status` | `VERIFIED` · `VERIFIED(spec)` · `TODO(verify)` · `UNKNOWN (blocking)` · `NOT ACQUIRED`. |
| `notes` | Caveats. |

### Status vocabulary

- **VERIFIED** — confirmed against an authoritative source and safe to rely on.
- **VERIFIED(spec)** — fixed by the official Track B challenge specification.
  Authoritative for *what we are required to use*. It does **not** certify any
  property that can only be measured from the delivered asset.
- **TODO(verify)** — recorded for orientation; **not** yet confirmed.
- **UNKNOWN (blocking)** — cannot be determined from available information and
  blocks a downstream milestone.
- **NOT ACQUIRED** — permitted and understood, but no data retrieved yet.

> **Partially verified as of the official Track B specification.** The
> specification has now been supplied and fixes the following, which are marked
> **VERIFIED(spec)** below: the event date (2026-08-26), the pre-event OSM
> snapshot date (2026-07-27, via the ohsome API), the DEM source (Copernicus
> WorldDEM-30), the permitted training dataset list (Kuro Siwo, Sen1Floods11),
> EMSR927's classification as validation-only, and the three mandatory
> attribution strings.
>
> **VERIFIED(spec) means "fixed by the specification", not "measured from the
> data."** No data has been acquired. Every property that must come from the
> asset itself — delivered resolution, CRS, vertical datum, band set, orbit
> track, class balance, tile geometry — remains `TODO(verify)` and must be read
> from the downloaded product, not inferred from a product name. Per
> `AGENTS.md` §11 an AI-generated statement is not scientific validation, so no
> row is promoted beyond what the specification actually states.

---

## 1. Production inputs — permitted

### 1.1 Sentinel-1

| Field | Value |
|---|---|
| **dataset** | Sentinel-1 SAR. Product type TODO(verify) — GRD expected. |
| **purpose** | Cloud-independent pre/post change detection; primary flood signal under monsoon cloud. |
| **allowed_as_input** | **YES** |
| **validation_only** | NO |
| **license** | Copernicus Sentinel Data — free, full and open. TODO(verify) exact terms version. |
| **citation** | "Contains modified Copernicus Sentinel data 2026." (`AGENTS.md` §18) |
| **temporal_requirement** | One pre-event and one post-event acquisition. **Same relative orbit strongly required** (`AGENTS.md` §4). Windows TODO. |
| **spatial_information** | TODO(verify). IW GRD ~10 m pixel spacing *commonly documented — VERIFY*. Pixel spacing ≠ true resolution. |
| **status** | **NOT ACQUIRED** / TODO(verify) product parameters |
| **notes** | Cross-track pairs must not be compared pixel-by-pixel. Terrain correction essential in Himalayan relief. Low backscatter is ambiguous: smooth water, dry smooth surfaces and radar shadow can all appear dark. |

### 1.2 Sentinel-2

| Field | Value |
|---|---|
| **dataset** | Sentinel-2 MSI. Processing level TODO(verify) — L2A expected. |
| **purpose** | Optical confirmation and multispectral features where cloud permits. |
| **allowed_as_input** | **YES** |
| **validation_only** | NO |
| **license** | Copernicus Sentinel Data — free, full and open. TODO(verify). |
| **citation** | "Contains modified Copernicus Sentinel data 2026." |
| **temporal_requirement** | Pre/post acquisitions; cloud-limited. Usable post-event optical imagery may not exist for a monsoon event. |
| **spatial_information** | TODO(verify). *Commonly documented — VERIFY*: 10 m visible/NIR, 20 m red-edge/SWIR, 60 m atmospheric. |
| **status** | **NOT ACQUIRED** / TODO(verify) |
| **notes** | Availability is a genuine risk, not a formality — an August monsoon event may have no clear post-event scene. Cloud gaps must be masked and propagated, never interpolated into apparent observations. |

### 1.3 Copernicus DEM

| Field | Value |
|---|---|
| **dataset** | **Copernicus WorldDEM-30** — VERIFIED(spec). |
| **purpose** | Terrain context, slope/elevation features, optional downstream flood-path tracing. |
| **allowed_as_input** | **YES** |
| **validation_only** | NO |
| **license** | COPERNICUS WorldDEM-30 terms via EU/ESA. TODO(verify) redistribution limits before committing any sample to Git. |
| **citation** | VERIFIED(spec), mandatory: "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved." Enforced by `provenance.REQUIRED_ATTRIBUTIONS`. |
| **temporal_requirement** | Represents **pre-event** terrain. Acquisition epoch 2010–2018 per the attribution. |
| **spatial_information** | TODO(verify) — ~30 m is implied by the product *name*, which is not a measurement. Delivered grid, CRS and vertical datum must be read from the asset. |
| **status** | **VERIFIED(spec)** source · **NOT ACQUIRED** · grid/datum TODO(verify) |
| **notes** | The DEM is a terrain *prior*, not a current observation: an avalanche and debris flow changes real terrain. 30 m cannot resolve individual road cuttings or narrow channels. Vertical datum errors are metre-scale and directly affect flood-path inference. |

### 1.4 OpenStreetMap — pre-event snapshot

| Field | Value |
|---|---|
| **dataset** | OpenStreetMap historical extract via the **ohsome API**, snapshot **2026-07-27** — VERIFIED(spec). |
| **purpose** | Pre-event buildings, roads, bridges for exposure and network analysis. |
| **allowed_as_input** | **YES — pre-event snapshot only** |
| **validation_only** | NO (but see 2.5 for post-event edits) |
| **license** | ODbL 1.0. TODO(verify) share-alike implications for derived published outputs. |
| **citation** | VERIFIED(spec), mandatory: "© OpenStreetMap contributors." Enforced by `provenance.REQUIRED_ATTRIBUTIONS`. |
| **temporal_requirement** | **Snapshot must predate the event.** Hard rule (`AGENTS.md` §3). Satisfied: 2026-07-27 < 2026-08-26, a 30-day margin. |
| **spatial_information** | Vector, EPSG:4326 native; reprojected for any length/area computation. |
| **status** | **VERIFIED(spec)** snapshot date and source · **NOT ACQUIRED** |
| **notes** | The ohsome API serves time-sliced OSM *history*, so it can return a genuine 2026-07-27 state rather than a current-day extract — this is why it satisfies §3 where a plain planet extract would not. The returned snapshot timestamp must be checked against the request, not assumed. Coverage in rural Nepal is uneven; absence of a road in OSM is not evidence the road does not exist. Omissions bias exposure counts **downward** and may hide genuinely cut-off settlements. OSM carries no reliable population data. |

### 1.5 Permitted training datasets

The permitted list is **closed and VERIFIED(spec)**: Kuro Siwo and
Sen1Floods11. Training on any dataset outside this list violates `AGENTS.md`
§3. The machine-readable list lives at
`configs/data.yaml:production.training_datasets.allowed_training_datasets`.

#### 1.5.1 Kuro Siwo

| Field | Value |
|---|---|
| **dataset** | **Kuro Siwo** — VERIFIED(spec). Release/version TODO(verify) at download. |
| **purpose** | Primary training and validation source for flood/debris segmentation. |
| **allowed_as_input** | **YES** — production training input |
| **validation_only** | NO |
| **license** | MIT / CC BY — VERIFIED(spec) as stated. TODO(verify) **which** applies to code versus data before redistributing either; the spec records both identifiers without splitting them. |
| **citation** | Bountos et al., 2024 (NeurIPS 2024). TODO(verify) full bibliographic entry from the paper itself before publication. |
| **temporal_requirement** | None — a training corpus, not an event observation. Its scenes are unrelated to the 2026-08-26 event and must never be mixed into the event's pre/post pair. |
| **spatial_information** | TODO(verify) — CRS, tile geometry, resolution and SAR product level must be read from the delivered dataset. |
| **status** | **VERIFIED(spec)** permitted · **NOT ACQUIRED** · properties TODO(verify) |
| **notes** | Permission to use is settled; suitability is not. Open and blocking for the segmentation milestone: does it label **debris/sediment separately from water**, or only water? Track B requires flood *and* debris. If it labels water only, the debris class has no training signal from this source and that limitation must be stated in the report rather than papered over. Also unknown: class balance, geographic coverage (Himalayan terrain represented or not), and whether official splits exist that must be honoured to avoid leakage. See `docs/evaluation-protocol.md`. |

#### 1.5.2 Sen1Floods11

| Field | Value |
|---|---|
| **dataset** | **Sen1Floods11** — VERIFIED(spec). Release/version TODO(verify) at download. |
| **purpose** | Optional supplementary training data for flood/water segmentation. |
| **allowed_as_input** | **YES** — optional training input |
| **validation_only** | NO |
| **license** | CC BY 4.0 — VERIFIED(spec). Attribution required on derived outputs. |
| **citation** | Bonafilia et al., 2020 (CVPRW 2020). TODO(verify) full bibliographic entry from the paper itself before publication. |
| **temporal_requirement** | None — a training corpus. Its scenes are unrelated to the 2026-08-26 event. |
| **spatial_information** | TODO(verify) — CRS, chip size, resolution and band set must be read from the delivered dataset. |
| **status** | **VERIFIED(spec)** permitted · **OPTIONAL** · **NOT ACQUIRED** · properties TODO(verify) |
| **notes** | Marked optional by the specification, so it is a deliberate choice rather than a default. It is a **surface-water** dataset; it is unlikely to carry a debris class, so it cannot resolve the debris-label gap above. Combining it with Kuro Siwo introduces a real risk of **label-definition mismatch** — two corpora can disagree on what counts as "water" at a boundary. Any combined training run must document the harmonisation rule and must not silently union incompatible label schemes. Its geographic distribution is global and may under-represent steep terrain. |

---

## 2. Validation / comparison only — never production inputs

> Everything in this section is forbidden in training, feature construction,
> preprocessing, inference and threshold selection. They may be read **only** by
> the evaluation layer, and **only after** a production prediction already
> exists (`architecture.md` §11, §18).
>
> Importing any of these into a production path is a **critical defect**
> (`AGENTS.md` §3), not a style issue.

### 2.1 Copernicus EMS — EMSR927

| Field | Value |
|---|---|
| **dataset** | Copernicus Emergency Management Service activation EMSR927 — VERIFIED(spec) as the activation for this event. |
| **purpose** | Post-hoc comparison for the Trishuli case study (`PRD.md` FR-13). |
| **allowed_as_input** | **NO** — VERIFIED(spec): explicitly prohibited as a production input. |
| **validation_only** | **YES** — VERIFIED(spec). |
| **license** | TODO(verify) Copernicus EMS terms. |
| **citation** | TODO(verify) required EMS attribution form. |
| **temporal_requirement** | Post-event. Read only after our prediction exists. |
| **spatial_information** | TODO — not acquired. |
| **status** | **VERIFIED(spec)** classification · **NOT ACQUIRED** |
| **notes** | The specification settles the *rule* (validation only), which is now enforced by `ValidationOnlySource.EMSR927` and by `tests/test_data_boundary.py`. It does not make EMSR927 ground truth: agreement is a **comparison**, since it carries its own method, timing and interpretation assumptions, so disagreement is not automatically our error. The dashboard must not visually imply EMSR927 is an input layer (`AGENTS.md` §16). |

### 2.2 UNOSAT damage maps

| Field | Value |
|---|---|
| **dataset** | UNOSAT / UNITAR damage assessments. |
| **purpose** | Optional post-hoc comparison. |
| **allowed_as_input** | **NO** |
| **validation_only** | **YES** |
| **license** | TODO(verify). |
| **citation** | TODO(verify). |
| **temporal_requirement** | Post-event. |
| **spatial_information** | TODO — not acquired. |
| **status** | **NOT ACQUIRED** |
| **notes** | Same caution as EMSR927. |

### 2.3 Other published damage maps

| Field | Value |
|---|---|
| **dataset** | Any third-party published flood/damage product. |
| **purpose** | Optional post-hoc comparison. |
| **allowed_as_input** | **NO** |
| **validation_only** | **YES** |
| **license** | Per source — TODO. |
| **citation** | Per source — TODO. |
| **temporal_requirement** | Post-event. |
| **spatial_information** | Per source. |
| **status** | **NOT ACQUIRED** |
| **notes** | Catch-all row so a newly discovered product is covered by the rule by default rather than by omission. |

### 2.4 Post-event OpenStreetMap edits

| Field | Value |
|---|---|
| **dataset** | OSM state **after** the event date. |
| **purpose** | Optional post-hoc comparison of infrastructure change. |
| **allowed_as_input** | **NO** |
| **validation_only** | **YES** |
| **license** | ODbL 1.0. |
| **citation** | "© OpenStreetMap contributors." |
| **temporal_requirement** | Post-event by definition. |
| **spatial_information** | Vector, EPSG:4326. |
| **status** | **NOT ACQUIRED** |
| **notes** | Subtle and easy to get wrong: calling a *current* OSM API returns post-event data by default. After a disaster, mappers add damage-related detail, so a current extract would import post-event human knowledge of the damage into a system claiming to derive damage from satellite data. The acquisition layer must request an explicitly dated historical snapshot. |

---

## 3. Registry summary

| dataset | allowed_as_input | validation_only | status |
|---|---|---|---|
| Sentinel-1 | YES | NO | NOT ACQUIRED |
| Sentinel-2 | YES | NO | NOT ACQUIRED |
| Copernicus WorldDEM-30 | YES | NO | VERIFIED(spec) · NOT ACQUIRED |
| OSM — pre-event snapshot (2026-07-27, ohsome) | YES | NO | VERIFIED(spec) · NOT ACQUIRED |
| Kuro Siwo | YES (required) | NO | VERIFIED(spec) · NOT ACQUIRED |
| Sen1Floods11 | YES (optional) | NO | VERIFIED(spec) · NOT ACQUIRED |
| EMSR927 | **NO** | YES | VERIFIED(spec) · NOT ACQUIRED |
| UNOSAT | **NO** | YES | NOT ACQUIRED |
| Other published damage maps | **NO** | YES | NOT ACQUIRED |
| OSM — post-event edits | **NO** | YES | NOT ACQUIRED |

## 4. Machine-readable counterparts

The boundary is represented in code, so it can be enforced rather than merely
documented:

- `floodmap.utils.provenance.ProductionInput` — closed enum of permitted sources.
- `floodmap.utils.provenance.ValidationOnlySource` — closed enum of forbidden sources.
- `ArtifactProvenance` rejects a validation-only source recorded as a production input.
- `configs/data.yaml` separates `production:` from `validation_only:`.
- `tests/test_data_boundary.py` fails if a forbidden source name appears in a production module.
- `floodmap.utils.provenance.REQUIRED_ATTRIBUTIONS` — the three mandatory
  attributions, attached automatically to every `ArtifactProvenance` and
  re-inserted if a caller omits them.

## 5. Blocking items

The training-dataset list is **no longer blocking** — the specification
enumerates Kuro Siwo and Sen1Floods11 (§1.5). Remaining items:

1. **Kuro Siwo label semantics** — does it label debris/sediment separately from
   water? Blocks the *debris* half of the segmentation task, not the flood half.
   Resolvable only by inspecting the dataset or its paper, not by assumption.
2. **Kuro Siwo license split** — the spec records "MIT / CC BY" without saying
   which covers code and which covers data. Blocks redistribution, not use.
3. **Label harmonisation rule** — required before Kuro Siwo and Sen1Floods11 are
   combined, since the two may define water boundaries differently.
4. **Official dataset splits** — if either dataset ships splits, they must be
   honoured to avoid leakage; unknown until acquired.
5. **License verification for `data/samples/`** — before committing any sample,
   confirm the license permits redistribution in a public repository.
6. **EMS/UNOSAT citation formats** — required before publishing any comparison.
7. **WorldDEM-30 attribution wording** — confirm against the official
   specification document whether "© DLR e.V." carries the copyright mark. This
   repository uses the form **with** the mark, per `AGENTS.md` §18.
