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
| `status` | `VERIFIED` · `TODO(verify)` · `UNKNOWN (blocking)` · `NOT ACQUIRED`. |
| `notes` | Caveats. |

### Status vocabulary

- **VERIFIED** — confirmed against an authoritative source and safe to rely on.
- **TODO(verify)** — recorded for orientation; **not** yet confirmed.
- **UNKNOWN (blocking)** — cannot be determined from available information and
  blocks a downstream milestone.
- **NOT ACQUIRED** — permitted and understood, but no data retrieved yet.

> **Nothing in this registry is marked VERIFIED.** The official Track B
> challenge specification named by `AGENTS.md` §2 as the primary requirement
> source is not present in this repository, and no data has been acquired. Per
> `AGENTS.md` §11, an AI-generated statement is not scientific validation, so no
> row may be promoted to VERIFIED on the strength of this document alone.

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
| **dataset** | Copernicus DEM. Exact product/variant TODO(verify); `README.md` §12 attribution indicates **WorldDEM-30**. |
| **purpose** | Terrain context, slope/elevation features, optional downstream flood-path tracing. |
| **allowed_as_input** | **YES** |
| **validation_only** | NO |
| **license** | COPERNICUS WorldDEM-30 terms via EU/ESA. TODO(verify) redistribution limits before committing any sample to Git. |
| **citation** | "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved." |
| **temporal_requirement** | Represents **pre-event** terrain. Acquisition epoch 2010–2018 per the attribution. |
| **spatial_information** | TODO(verify) — ~30 m implied by the product name; **confirm**. Vertical datum TODO(verify). |
| **status** | **NOT ACQUIRED** / TODO(verify) |
| **notes** | The DEM is a terrain *prior*, not a current observation: an avalanche and debris flow changes real terrain. 30 m cannot resolve individual road cuttings or narrow channels. Vertical datum errors are metre-scale and directly affect flood-path inference. |

### 1.4 OpenStreetMap — pre-event snapshot

| Field | Value |
|---|---|
| **dataset** | OpenStreetMap historical extract, snapshot date TODO (must be < event date). |
| **purpose** | Pre-event buildings, roads, bridges for exposure and network analysis. |
| **allowed_as_input** | **YES — pre-event snapshot only** |
| **validation_only** | NO (but see 2.5 for post-event edits) |
| **license** | ODbL 1.0. TODO(verify) share-alike implications for derived published outputs. |
| **citation** | "© OpenStreetMap contributors." |
| **temporal_requirement** | **Snapshot must predate the event.** Hard rule (`AGENTS.md` §3). |
| **spatial_information** | Vector, EPSG:4326 native; reprojected for any length/area computation. |
| **status** | **NOT ACQUIRED** / TODO(verify) extract source |
| **notes** | Coverage in rural Nepal is uneven; absence of a road in OSM is not evidence the road does not exist. Omissions bias exposure counts **downward** and may hide genuinely cut-off settlements. OSM carries no reliable population data. |

### 1.5 Permitted training datasets

| Field | Value |
|---|---|
| **dataset** | **UNKNOWN (blocking)** |
| **purpose** | Training and validating the flood/debris segmentation model. |
| **allowed_as_input** | YES — but **only** those on the official permitted list |
| **validation_only** | NO |
| **license** | UNKNOWN |
| **citation** | UNKNOWN — `AGENTS.md` §18 requires citation per each dataset's license and paper |
| **temporal_requirement** | UNKNOWN |
| **spatial_information** | UNKNOWN |
| **status** | **UNKNOWN (blocking)** |
| **notes** | `README.md` §4 and `PRD.md` §8 refer to "listed permitted training datasets" but **no document in this repository enumerates them**, and the official challenge specification is absent. This blocks the segmentation milestone. It must not be resolved by selecting a well-known flood dataset from memory: an unlisted dataset would violate `AGENTS.md` §3. Required per dataset once supplied: source, license, citation, label definitions, flood-vs-debris class coverage, modalities, expected preprocessing, official splits. See `docs/data-contract.md` §5. |

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
| **dataset** | Copernicus Emergency Management Service activation EMSR927. |
| **purpose** | Post-hoc comparison for the Trishuli case study (`PRD.md` FR-13). |
| **allowed_as_input** | **NO** |
| **validation_only** | **YES** |
| **license** | TODO(verify) Copernicus EMS terms. |
| **citation** | TODO(verify) required EMS attribution form. |
| **temporal_requirement** | Post-event. Read only after our prediction exists. |
| **spatial_information** | TODO — not acquired. |
| **status** | **NOT ACQUIRED** |
| **notes** | Agreement with EMSR927 is a **comparison, not ground truth**: it carries its own method, timing and interpretation assumptions, so disagreement is not automatically our error. The dashboard must not visually imply EMSR927 is an input layer (`AGENTS.md` §16). |

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
| Copernicus DEM (WorldDEM-30, TODO variant) | YES | NO | NOT ACQUIRED |
| OSM — pre-event snapshot | YES | NO | NOT ACQUIRED |
| Permitted training datasets | YES (list unknown) | NO | **UNKNOWN (blocking)** |
| EMSR927 | **NO** | YES | NOT ACQUIRED |
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

## 5. Blocking items

1. **Permitted training dataset list** — blocks segmentation and evaluation.
   Requires the official challenge specification.
2. **License verification for `data/samples/`** — before committing any sample,
   confirm the license permits redistribution in a public repository.
3. **EMS/UNOSAT citation formats** — required before publishing any comparison.
