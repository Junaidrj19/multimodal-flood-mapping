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

> **AUDITED against primary sources.** The properties below were read from the
> dataset's own repository, LICENSE file and published paper — not from the
> challenge specification, and not from recollection. Sources:
> [paper (arXiv HTML v2)](https://arxiv.org/html/2311.12056v2),
> [abstract](https://arxiv.org/abs/2311.12056),
> [repository](https://github.com/Orion-AI-Lab/KuroSiwo),
> [LICENSE](https://raw.githubusercontent.com/Orion-AI-Lab/KuroSiwo/main/LICENSE),
> [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/43612b0662cb6a4986edf859fd6ebafe-Abstract-Datasets_and_Benchmarks_Track.html),
> plus `configs/grd_preprocessing.xml`, `configs/slc_preprocessing.xml`,
> `configs/train/data_config.json`, `configs/config.json`,
> `catalogue/catalogue.yaml` and `training/segmentation_trainer.py` from the
> repository at `main`.
>
> The HuggingFace dataset card returned HTTP 401 and could not be read. The
> paper's Supplemental Material was not read, and it is the stated location of
> the flood definition and annotation principles.

| Field | Value |
|---|---|
| **dataset** | **Kuro Siwo** — VERIFIED(spec) as permitted. Release/version TODO(verify) at download. |
| **purpose** | **Primary training source for flood-water segmentation.** Not a debris source — see label classes below. |
| **allowed_as_input** | **YES** — production training input |
| **validation_only** | NO |
| **license** | **DISCREPANCY — UNRESOLVED.** See §1.5.5. |
| **citation** | Bountos, Sdraka, Zavras, Karavias, Karasante, Herekakis, Thanasou, Michail & Papoutsis (2024), *Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood mapping*, NeurIPS 37, 38105–38121. DOI 10.52202/079017-1204. BibTeX in the repository README. — VERIFIED(source) |
| **temporal_requirement** | None — a training corpus. Its scenes are unrelated to 2026-08-26 and must never be mixed into the event's pre/post pair. |
| **status** | **VERIFIED(source)** properties below · **NOT ACQUIRED** |

**Verified technical properties**

| Property | Value | Status |
|---|---|---|
| Sensors | Sentinel-1 SAR only, plus a bundled DEM. **No optical data.** | VERIFIED(source) |
| Bundled DEM | **SRTM 1 arc-second** — *not* Copernicus WorldDEM-30 | VERIFIED(source) |
| Polarisations | VV and VH | VERIFIED(source) |
| Product levels | Level-1 GRD and Level-1 SLC | VERIFIED(source) |
| Resolution | 10 m pixel spacing | VERIFIED(source) |
| Tile size | 224 × 224 | VERIFIED(source) |
| CRS | EPSG:3857 | VERIFIED(source) |
| Backscatter representation | **linear σ⁰** | **INFERRED** — `Calibration` sets `outputImageScaleInDb=false` and the GRD graph contains no `LinearToFromdB` node; `clamp_input: 0.15` is consistent. **No source states it in prose.** |
| Temporal structure | **Triplet: two pre-event images + one post-event image** | VERIFIED(source) |
| GRD preprocessing chain | Apply-Orbit-File (Sentinel Precise) → Subset → ThermalNoiseRemoval → Remove-GRD-Border-Noise (`borderLimit 500`) → Land-Sea-Mask (SRTM) → Calibration (`outputSigmaBand=true`) → Speckle-Filter (**Lee Sigma, 7×7, target 3×3, sigma 0.9**) → Terrain-Correction (SRTM 1Sec HGT, bilinear, 10 m, EPSG:3857) | VERIFIED(source) |
| Per-sample metadata | caption dates, climate zone, AOI id, activation id, DEM. **Orbit, incidence angle and geotransform not confirmed as per-sample fields**; incidence-angle saving is `false` in both graphs. | partly TODO(verify) |

**Label classes — VERIFIED(source)**

From `training/segmentation_trainer.py`, with the paper's own wording:

| Value | Class |
|---|---|
| 0 | `No water` |
| 1 | `Permanent Waters` |
| 2 | `Floods` |

**Validity is a separate raster, not a label value.** The HuggingFace
`Kuro-Siwo-Webdataset` card documents two arrays per sample:
`mask.npy` → *"0: no water, 1: permanent water, 2: flood"*, and
`valid_mask.npy` → *"0: invalid, 1: valid"*. `dataset/Dataset.py` loads the
validity raster independently, carries it separately through augmentation, and
applies `valid_mask == 1`.

> **Correction to an earlier revision of this registry.** This table previously
> listed `3: Invalid pixels` as a stored class with `ignore_index` semantics.
> It is not a stored value. `CLASS_LABELS` in
> `training/segmentation_trainer.py` does contain a `3` entry, but
> `CLASS_LABELS[3]` is never referenced in that repository — only `[0]`, `[1]`
> and `[2]` are — and no `ignore_index` key appears in any of its config files.
> The consequence for us is concrete: the adapter must read **two** rasters per
> sample. Pinned by
> `tests/test_configs.py::test_kuro_siwo_validity_is_a_separate_raster_not_a_label_value`.

Paper wording: *"assigning each pixel to one of three categories, i.e. Permanent
Waters, Floods and No Water."*

- **Separates permanent water from flood water: YES.** This is the single most
  valuable property of this corpus for us. `docs/scientific-assumptions.md` §7
  records that permanent water must be distinguished from new flooding or the
  river itself is detected as flood every time. Kuro Siwo supervises that
  distinction directly; Sen1Floods11 does not.
- **Debris / sediment / mud class: NONE.** The class list is exhaustive. See
  §1.5.6.
- **Label provenance:** manual photointerpretation of the preprocessed GRD
  images by five SAR experts, *initialised from Copernicus EMS shapefiles where
  those exist*. The authors state that *"errors in CEMS annotations are
  apparent"*. This ancestry is material — see §1.5.4.
- **Flood definition:** **TODO(verify).** The main text distinguishes
  *"permanent water bodies, e.g rivers and lakes"* from *"flooded areas"* and
  defers the annotation principles to the Supplemental Material, which was not
  read. Our class semantics must not be finalised without it.
- **Stated label noise for Kuro Siwo itself:** TODO(verify) — not in the main text.

**Official splits — VERIFIED(source), and binding on us**

Split is by event and geography, not random. Paper: *"The test dataset
encompasses flood events from entirely unseen locations on Earth."* From
`configs/train/data_config.json`, by activation id:

- **test:** `321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013`
- **val:** `514, 559, 279, 520, 437, 1111003, 1111008`
- **train:** the remaining 27 of 43 activations

> **Governance consequence.** These upstream test activations must not enter our
> training or validation split. Using an upstream test event as our training
> data would mean any published comparison against other Kuro Siwo results is
> measured on scenes we trained on. `docs/evaluation-protocol.md` §2.4 records
> the rule.

**The "Nepal" activation — read this before assuming Himalayan coverage**

`catalogue/catalogue.yaml` contains exactly one entry with
`act_region: Nepal`: `act_id: 1111007`, `ref_date: 20190917`,
`aoi_name: Patna`, `cl_zone: 1` (tropical). Two independent reasons it does not
give us Himalayan training or evaluation data:

1. It is in the **upstream test split**, so it is off-limits as training data.
2. Its AOI name and tropical climate-zone classification both point to lowland
   terrain rather than mountain terrain. **TODO(verify):** inspect the actual AOI
   geometry before drawing any conclusion. Treat it as **not** satisfying the
   unseen-Himalaya requirement until measured.

**No source states that Himalayan or high-mountain terrain is included.** The
domain-shift risk in `docs/scientific-assumptions.md` §9 is therefore
*unmitigated by this corpus*, not merely unquantified.

#### 1.5.2 Sen1Floods11

> **AUDITED against primary sources.** Sources:
> [repository README](https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/README.md),
> [bucket docs](https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/docs/README.md),
> [metadata GeoJSON](https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/Sen1Floods11_Metadata.geojson),
> [GitHub API](https://api.github.com/repos/cloudtostreet/Sen1Floods11),
> [CVPR Workshops paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html),
> and the [GEE S1 GRD catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
> that the README delegates its preprocessing description to.
>
> The paper PDF body could not be text-extracted, so quantified label-noise
> figures for the weak labels remain unverified. The Radiant Earth MLHub mirror
> now redirects to Source Cooperative with no dataset page.

| Field | Value |
|---|---|
| **dataset** | **Sen1Floods11** — VERIFIED(spec) as permitted, **optional**. |
| **purpose** | Optional supplementary training data for **surface-water** segmentation. |
| **allowed_as_input** | **YES** — optional training input |
| **validation_only** | NO |
| **license** | **NO LICENSE FILE UPSTREAM.** See §1.5.5. |
| **citation** | Bonafilia, Tellman, Anderson & Issenberg (2020), *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*, CVPR Workshops 2020, 210–211. BibTeX on the CVF page. — VERIFIED(source) |
| **status** | **VERIFIED(source)** properties below · **NOT ACQUIRED** |

**Verified technical properties**

| Property | Value | Status |
|---|---|---|
| Sensors | **Sentinel-1 AND Sentinel-2** | VERIFIED(source) |
| Bundled DEM | **None.** SRTM/ASTER is used inside GEE terrain correction but is not distributed as a layer. | VERIFIED(source) |
| Polarisations | VV (band 0) and VH (band 1) | VERIFIED(source) |
| Product level | GRD, IW mode | VERIFIED(source) |
| Backscatter representation | **decibels** — README states *"Unit: dB"* | **VERIFIED(source)** |
| S1 preprocessing | Per GEE: thermal noise removal → radiometric calibration → terrain correction (SRTM 30 / ASTER) → *"converted to decibels via log scaling (10*log10(x))"*. **No speckle filter.** | VERIFIED(source) |
| Optical processing level | **L1C — top-of-atmosphere reflectance, NOT L2A surface reflectance** | VERIFIED(source) |
| Optical bands | all 13 (B1–B12 incl. B8A), *"Does not contain QA mask"* | VERIFIED(source) |
| Optical scaling | TOA reflectance **scaled by 10000** | VERIFIED(source) |
| Resolution | 10 m; all S2 bands resampled to the common 10 m grid | VERIFIED(source) |
| Chip size | 512 × 512 | VERIFIED(source) |
| CRS | EPSG:4326 | VERIFIED(source) |
| Temporal structure | **Single acquisition per chip — no pre-event image** | VERIFIED(source) |
| S1/S2 date alignment | **Close but not always identical** — offsets up to 2 days observed (e.g. Ghana S1 2018-09-18 / S2 2018-09-19; Sri Lanka S1 2017-05-30 / S2 2017-05-28) | VERIFIED(source) |
| Per-sample metadata | `s1_date`, `s2_date`, `orbit` (ASC/DESC), `rel_orbit_num`, `location`, `ISO_CC`, `VH_thresh` | VERIFIED(source) |

**Label classes — VERIFIED(source)**

Hand-labelled QC layer, verbatim from the README:

| Value | Class |
|---|---|
| −1 | `No Data / Not Valid` |
| 0 | `Not Water` |
| 1 | `Water` |

- **Separates permanent water from flood water: NO.** The labels are binary
  surface water. The paper addresses the distinction by using *separate data
  subsets* with the same binary encoding, not by adding a class.
- **Debris / sediment / mud class: NONE.**
- **Encoding of the weak and JRC layers:** TODO(verify) — the README documents
  values only for the hand-labelled QC layer.

**Label portions — VERIFIED(source)**

| Portion | Chips | How produced |
|---|---|---|
| `LabelHand` | **446** | Hand-annotated ground truth |
| `S1OtsuLabelWeak` | 4,385 | Otsu thresholding of the S1 VH band |
| `S2IndexLabelWeak` | 4,385 | *"traditional Sentinel-2 Classification"* (index/threshold based) |
| `JRCPerm` | 815 | JRC permanent-water dataset (Landsat-derived) |

Total hand + weak = 4,831 chips over 120,406 km².

> **The weak labels are threshold outputs, not observations.** Training on
> `S2IndexLabelWeak` teaches a model to reproduce a spectral-index threshold
> rule; training on `S1OtsuLabelWeak` teaches it to reproduce an Otsu threshold
> on VH. That is a meaningful distillation target but it adds no information
> beyond the rule, and it would embed a decision threshold inside the learned
> weights — which is exactly the thing this project keeps explicit and
> configurable (`configs/segmentation.yaml → inference.probability_threshold`).
> Any use of the weak portions must be declared and justified, and the 446
> hand-labelled chips are the only portion that is ground truth.

**Coverage — VERIFIED(source)**

11 flood events (12 metadata entries; Colombia appears to post-date the
original 11), spanning *"all 14 biomes, 357 ecoregions, and 6 continents"*:
Bolivia, Colombia, Ghana, India, Cambodia, Nigeria, Pakistan, Paraguay,
Somalia, Spain, Sri Lanka, USA.

**No event is in Nepal.** The India event (~92.1–94.2°E, 24.8–28.3°N) reaches
the eastern-Himalayan foothills and the Pakistan event (~69.0–72.6°E,
28.0–34.4°N) reaches northern mountainous terrain, but **no source describes
either as Himalayan or high-mountain**, and neither is a substitute for
Himalayan evaluation data.

Splits are **event-based** — the metadata carries per-event `train_chip` and
`val_chip` counts, and the bucket contains `splits/` files. The exact test-split
composition is TODO(verify) (not downloaded). Colombia has `val_chip: 0`.

#### 1.5.3 Modality compatibility against our production pipeline

The comparison that actually governs M4. "Ours" is what M2/M3 produce.

| Dimension | Kuro Siwo | Sen1Floods11 | Ours (production) | Compatible? |
|---|---|---|---|---|
| S1 polarisations | VV, VH | VV, VH | TODO(verify) from product | Likely — but ours is unverified |
| S1 representation | **linear σ⁰** (inferred) | **decibel** | configurable, **must be declared** (`configs/features.yaml → sentinel1.backscatter_representation`) | **The two corpora disagree with each other.** Harmonisation mandatory. |
| S1 speckle filter | **Lee Sigma 7×7** | **none** | not implemented in M2 | **Three-way mismatch.** |
| S1 terrain correction | SRTM 1Sec | SRTM 30 / ASTER | required, source-asserted | Method differs |
| Resolution | 10 m | 10 m | TODO(verify) target grid | Both 10 m — encouraging |
| CRS | EPSG:3857 | EPSG:4326 | TODO(verify), projected metre CRS expected | All three differ; M3 requires a projected metre grid for slope |
| Tile size | 224² | 512² | N/A (full AOI) | Tiling is a training-time concern |
| Pre-event imagery | **2 pre + 1 post** | **post only** | **1 pre + 1 post** (M1 selection) | **Ours matches neither exactly.** Kuro Siwo's second pre-image has no production counterpart; Sen1Floods11 supports no change feature at all. |
| Optical | **absent** | **L1C TOA, 13 bands, no QA mask** | **L2A expected** (`configs/data.yaml` TODO) | **TOA ≠ BOA.** Index values differ systematically. |
| DEM | SRTM 1Sec, bundled | absent | **Copernicus WorldDEM-30** | Different DEM product than either corpus |
| Labels | 3-class, permanent vs flood | binary surface water | N/A | **Label taxonomies differ** — see §1.5.6 |

**The three findings that constrain M4 hardest:**

1. **Our change features have no supervised counterpart in Sen1Floods11.** It
   ships a single post-event acquisition, so `s1_<pol>_change_db` — the
   scientifically strongest SAR feature M3 produces — cannot be computed for
   that corpus at all.
2. **The two corpora disagree on SAR representation.** Training across both
   without converting to one representation would present the same physical
   backscatter on two different scales. M3's
   `backscatter_representation` gate exists precisely to make this explicit
   rather than silent, and it must be set per corpus by the adapter.
3. **Sen1Floods11's optical data is TOA, our production plan is L2A.** NDWI and
   MNDWI computed on TOA reflectance are not numerically interchangeable with
   the same indices on surface reflectance, because atmospheric path radiance
   affects the visible bands far more than the NIR/SWIR bands. An optical model
   trained on TOA and served BOA is a silent domain shift.

#### 1.5.4 Disclosure: Copernicus EMS ancestry in Kuro Siwo labels

Kuro Siwo's annotation procedure *initialises from Copernicus EMS shapefiles
where those exist*, before expert photointerpretation.

This is recorded because it is material, not because it is a violation:

- **It is not a boundary breach.** We consume the authors' finished labels. No
  EMS product is retrieved, read or referenced by this pipeline, and EMSR927
  specifically is not implicated — Kuro Siwo predates the 2026 event entirely.
- **It does weaken independence claims.** Our training labels inherit CEMS
  interpretation conventions second-hand. A later comparison against an
  EMS-family product (EMSR927) is therefore *not* a comparison between two fully
  independent methods, and must not be presented as one.
  `docs/evaluation-protocol.md` §7.3 carries the caveat.
- **Upstream label quality is explicitly imperfect.** The authors state
  *"errors in CEMS annotations are apparent"* — which is why they
  photointerpreted rather than accepting the shapefiles. A model cannot be more
  reliable than its labels.

`configs/data.yaml` records this under `upstream_label_provenance`, a field
whose only purpose is disclosure. `tests/test_data_boundary.py` grants that key
a **narrow, tested exemption** from the forbidden-marker scan and still fails if
an EMS reference appears in a path, URL or any other field.

#### 1.5.5 Licensing — discrepancies preserved

Per the instruction to preserve rather than resolve, and to take the
conservative reading.

**Kuro Siwo — DISCREPANCY(unresolved)**

| Source | States |
|---|---|
| Repository `LICENSE` file | **MIT** (Copyright (c) 2024 Orion Lab) |
| Repository README | *"The Kuro Siwo dataset is released under the CC BY license"* — **no version** |
| Paper body | *"Kuro Siwo is released under the MIT License"* |
| Challenge specification | MIT (our source also notes a repository CC BY claim) |
| arXiv paper badge | CC BY 4.0 — licenses **the paper**, not the data |
| HuggingFace card | UNVERIFIED (HTTP 401) |

**Conservative reading adopted:** treat the **code as MIT** and the **data as
CC BY** (attribution required). Rationale: CC BY is the more restrictive of the
two for data reuse, and attribution is required under *either* reading — so
requiring attribution is always safe, whereas assuming MIT for the data would
not be. Redistribution is **not** assumed: `redistribution_permitted: null`,
and no sample may be committed until resolved.

**Sen1Floods11 — UNVERIFIED(no license file upstream)**

This is worse than a discrepancy; it is an absence.

| Source | States |
|---|---|
| Repository `LICENSE` file | **DOES NOT EXIST** — root listing has no LICENSE/COPYING |
| GitHub API `license` field | **`null`** |
| Repository README | **No licence statement at all** |
| CVPR paper page | CVF boilerplate: *"rights therein are retained by authors"* — covers the paper |
| Challenge specification | CC BY 4.0 — **could not be confirmed against any primary source** |
| Radiant Earth / MLHub mirror | UNVERIFIED (redirects; no dataset page) |

**Conservative reading adopted:** **no redistribution right is established.**
Use is permitted — the authors publish the data openly and the challenge permits
it — but `redistribution_permitted: false`, the data must not be re-published
from this repository, no sample may be committed, and the citation must be
carried. The specification's "CC BY 4.0" is recorded as a *claim*, not a fact.

> Neither dataset's licence is silently asserted anywhere in this repository.
> `tests/test_configs.py::test_license_discrepancies_are_preserved_not_collapsed`
> fails if a future edit collapses either into a single confident identifier.

#### 1.5.6 Debris: the label evidence does not exist

**Neither permitted dataset contains a debris, sediment or mud class.** This is
now verified from both corpora's own class definitions rather than assumed:

- Kuro Siwo: `{0: No water, 1: Permanent Waters, 2: Floods}`, validity separate
- Sen1Floods11: `{-1: No Data, 0: Not Water, 1: Water}`

Both class lists are exhaustive. The permitted training list is **closed**
(`AGENTS.md` §3), so there is no permitted route to debris supervision.

**Consequence:** a supervised debris/sediment segmentation capability cannot be
built within the challenge rules. `PRD.md` FR-06 asks for *"flood/debris classes
supported by the training labels"* — and the training labels support flood
water only. The product claim narrows accordingly. The decision and the
permitted alternatives are recorded in `docs/m4-architecture-decision.md` §6 and
`docs/scientific-assumptions.md` §9.

### 1.6 Unverified candidate AOI artifact — `aoi.geojson`

An untracked file `aoi.geojson` appeared in the working tree. It is **not
adopted, not committed, and not referenced by any configuration.**
`configs/data.yaml → aoi` remains `null`.

It is recorded here because a plausible-looking AOI with no provenance is a
governance hazard: every acquisition, every grid and every reported number
downstream would inherit it, so adopting it silently would be the single
highest-leverage unverified assumption in the project.

**What was measured (independently verifiable, repeatable):**

| Property | Value |
|---|---|
| sha256 | `9035eeee025f311be7f16af1d23737a46caee20eb0b2a8dfa144d111b9362610` |
| size | 79,228 bytes |
| created = modified | 2026-10-05 19:46:31 (identical mtime/ctime → written once, not edited) |
| git history | **none** — untracked, never committed |
| FeatureCollection name | `NPL-FL-2026-upperstream-aoi` |
| feature name | `Upper Trishuli and Bhote Koshi flood corridor` |
| geometry | single `Polygon`, 1,515 vertices, closed ring |
| bbox (EPSG:4326) | lon 85.05814 – 85.38789, lat 27.84397 – 28.28743 |
| declared `source_crs` | `EPSG:32645` (UTM 45N) |
| declared `buffer_meters` | 1000 |
| declared `area_sq_km` | 132.93 |
| **recomputed area in EPSG:32645** | **132.93 km² — agrees to 0.001%** |
| perimeter | 133.5 km |
| implied centreline length at 2 km width | 66.5 km |
| half-perimeter (thin-corridor proxy) | 66.8 km — agrees to 0.4% |
| admin attributes | Bagmati province; Nuwakot and Rasuwa districts; 8 named municipalities |

**What this establishes.** The file is **internally consistent and machine
generated**. The recomputed area matches the stored attribute to five
significant figures, and the implied corridor length from area-over-width agrees
with half the perimeter to 0.4% — the signature of a genuine buffer around a
~66–67 km linear feature, not a hand-drawn or synthetic polygon. The declared
UTM zone is correct for the longitude range, and the named districts and
municipalities are consistent with the Bhote Koshi–Trishuli corridor.

The top-level `name` member plus a `crs` member containing
`urn:ogc:def:crs:OGC:1.3:CRS84` is the **GDAL/`ogr2ogr` fingerprint**: RFC 7946
does not define `name` and explicitly forbids `crs`, and GDAL emits both in its
non-RFC GeoJSON output. Combined with the single-space indentation and
one-coordinate-per-line layout, the file was almost certainly produced by
`ogr2ogr` converting a shapefile or GeoPackage named
`NPL-FL-2026-upperstream-aoi`.

**What this does NOT establish — the gaps that matter:**

1. **Who produced it, and from what.** No author, no tool version, no
   generation date inside the file. The river centreline it was buffered from is
   unidentified; if that centreline came from OSM, its snapshot date matters
   under `AGENTS.md` §3 and is unrecorded.
2. **Whether it is the judge-selected AOI.** Nothing connects it to the official
   Track B specification. The name `NPL-FL-2026-upperstream-aoi` is suggestive
   but self-asserted.
3. **Whether 1 km is the right buffer.** It is a scientific choice that bounds
   everything downstream, and it arrived without justification or sensitivity
   analysis.
4. **Whether "upperstream" is the intended extent.** The 132.93 km² corridor
   covers roughly 66 km of valley. Whether that is the correct study extent for
   the event is a decision, not a measurement.

**Status: `UNVERIFIED (provenance unknown)` · NOT ADOPTED.**

**Required before use:** confirm the origin and the centreline source with
whoever produced it; record the centreline's own provenance and, if OSM-derived,
its snapshot date; justify the 1 km buffer and plan its sensitivity analysis;
and only then populate `configs/data.yaml → aoi` with the file committed and
hash-pinned. Until then it is a candidate, and M1 continues to require an
explicit operator-supplied AOI.

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
| **license** | Copernicus EMS terms. TODO(verify) exact terms version. |
| **citation** | **VERIFIED(spec), mandatory when any comparison is published:** "European Union, Copernicus Emergency Management Service data". Deliberately **not** added to `provenance.REQUIRED_ATTRIBUTIONS`: that constant is attached to *every* artifact, and attaching an EMS attribution to production artifacts would imply EMS contributed to them. It belongs only on the evaluation/comparison artifact and the report section that presents the comparison. |
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
| Kuro Siwo | YES (required) | NO | VERIFIED(source) · NOT ACQUIRED · licence DISCREPANCY |
| Sen1Floods11 | YES (optional) | NO | VERIFIED(source) · NOT ACQUIRED · **licence UNVERIFIED (none upstream)** |
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

The dataset audit (§1.5) resolved most of the previous list. Resolved items are
kept with their outcome so a later reader can see what was settled and how.

### 5.1 Resolved by the audit

| # | Question | Outcome |
|---|---|---|
| 1 | Kuro Siwo label semantics — debris separate from water? | **RESOLVED — NO.** Classes are `{0: No water, 1: Permanent Waters, 2: Floods}` with validity in a separate raster. No debris class. Kuro Siwo *does* separate permanent water from flood water, which is valuable (§1.5.1). |
| 2 | Sen1Floods11 label semantics | **RESOLVED.** Binary `{-1: No Data, 0: Not Water, 1: Water}`. Does **not** separate permanent from flood water. |
| 3 | Do the two corpora share label semantics? | **RESOLVED — NO.** 3-class vs binary. Harmonisation rule required and specified in `docs/evaluation-protocol.md` §2.5. |
| 4 | Official dataset splits | **RESOLVED.** Both are event/geography based, not random. Kuro Siwo's activation IDs are recorded in `configs/data.yaml`; Sen1Floods11's are per-event in its metadata. Honouring them is now a rule (`docs/evaluation-protocol.md` §2.4). |
| 5 | Modality compatibility | **RESOLVED** and worse than assumed: the two corpora disagree on SAR representation (linear vs dB) and speckle filtering, and Sen1Floods11 has no pre-event image. See §1.5.3. |

### 5.2 Still blocking

| # | Item | Blocks | Why it cannot be closed yet |
|---|---|---|---|
| 1 | **Kuro Siwo licence discrepancy** | Redistribution, not use | `LICENSE` says MIT, README says CC BY. Conservative reading adopted (§1.5.5); resolution needs the authors or the HuggingFace card (HTTP 401). |
| 2 | **Sen1Floods11 has no licence** | Redistribution, not use | No `LICENSE` file, no README statement, GitHub API reports `null`. The spec's "CC BY 4.0" is unconfirmed. Needs the authors. |
| 3 | **Kuro Siwo flood definition and annotation principles** | Final class semantics | Stated to be in the paper's Supplemental Material, which was not read. Required before our label mapping is frozen. |
| 4 | **Kuro Siwo stated label noise** | Error analysis, metric interpretation | Not in the main text; likely in the supplement. |
| 5 | **Sen1Floods11 weak/JRC label encodings** | Any use of the weak portions | README documents values for the hand-labelled layer only. |
| 6 | **Sen1Floods11 exact test-split composition** | Leakage control | Split files live in the GCS bucket and were not downloaded. |
| 7 | **Kuro Siwo backscatter representation in prose** | Adapter correctness | Linear σ⁰ is **inferred** from the SNAP graph, not stated. A wrong inference here silently corrupts every SAR feature. Must be confirmed by reading a delivered raster's value distribution. |
| 8 | **Kuro Siwo "Nepal" AOI geometry** | Whether any near-Himalayan data exists at all | `aoi_name: Patna`, tropical climate zone, and it is in the upstream *test* split. Needs the actual geometry. |
| 9 | **Himalayan evaluation scenes** | Unseen-Himalaya evaluation | **Neither corpus is documented as containing Himalayan terrain.** This is now a confirmed gap, not an unknown — see §5.3. |
| 10 | **Licence verification for `data/samples/`** | Committing any sample | Both datasets currently forbid or fail to establish redistribution. |
| 11 | **EMS / UNOSAT citation formats** | Publishing any comparison | EMSR927 attribution string is now known (§2.1); UNOSAT's is not. |
| 12 | **WorldDEM-30 attribution wording** | Submission | Confirm whether "© DLR e.V." carries the mark; repository uses the form **with** it. |

### 5.3 The gap the audit opened

Items 1–8 are ordinary verification work. Item 9 is different, and it is the
most consequential finding in this registry.

**Neither permitted training corpus is documented as containing Himalayan or
high-mountain terrain.** Kuro Siwo's only Nepal-labelled activation is tropical
and lowland by its own metadata, and sits in the upstream test split.
Sen1Floods11's closest approaches are sub-Himalayan foothills in its India and
Pakistan events, which no source characterises as mountainous.

`PRD.md` §3 goal 3 requires evaluation on *unseen Himalayan scenes*, and
`docs/scientific-assumptions.md` §9 anticipated this as a risk. It is now a
measured fact about the permitted data, with two consequences:

1. **The domain shift is unmitigated by training data.** No permitted corpus
   teaches the model Himalayan radar geometry, snow/ice confusion, or
   high-relief shadow and layover behaviour. The in-domain-to-Himalaya
   performance gap should be expected to be large, and reporting that gap is
   itself a headline result (`AGENTS.md` §6).
2. **There is no labelled Himalayan test set within the permitted data.** So
   "unseen-Himalaya evaluation" cannot mean "metrics against permitted Himalayan
   labels". The only available Himalayan reference is EMSR927 — which is
   validation-only and must be used once, after freezing, and reported as
   *spatial agreement with an independent product*, never as accuracy against
   ground truth. The evaluation design consequence is recorded in
   `docs/evaluation-protocol.md` §2.6.
