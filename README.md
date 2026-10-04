# Multimodal AI Hackathon 2026 --- Track B

## Mapping Flood Damage from Space

A research-grade geospatial AI prototype for mapping flood/debris
impact, estimating infrastructure disruption, and identifying
settlements that may be cut off from road networks using permitted
satellite, terrain, and pre-event OpenStreetMap data.

> **Case study:** August 2026 Trishuli flood, Nepal\
> **Primary AI track:** Flood/debris segmentation with evaluation on
> unseen Himalayan scenes.

## 1. Problem

The challenge asks us to build a system that can take an area and flood
date and answer three operational questions:

1.  Where did the flood hit?
2.  What infrastructure was affected?
3.  Which settlements may have lost road connectivity to the nearest
    town or hospital?

The official challenge describes the August 2026 Trishuli disaster as
involving an avalanche of ice and rock followed by flood water, mud, and
debris along the Bhote Koshi--Trishuli corridor. The challenge
emphasizes rapid, trustworthy answers when roads, bridges, sensors, and
communications may fail.

## 2. System objective

The system will implement the following end-to-end pipeline:

```text
Area + event date
        |
        v
Satellite / terrain / OSM acquisition
        |
        v
Preprocessing + co-registration
        |
        v
Pre/post-event multimodal features
        |
        v
Flood/debris segmentation
        |
        +-------------------+
        |                   |
        v                   v
Infrastructure impact   Road-network analysis
        |                   |
        |                   v
        |              Connectivity graph
        |                   |
        +----------+--------+
                   |
                   v
          Potentially cut-off
             settlements
                   |
                   v
       Map dashboard + reports
```

A bonus hydrology component may trace a downstream flood path from an
upstream point using elevation data and list settlements along the path.

## 3. Scientific design principles

The project is designed around five principles:

- **Data legality:** only permitted inputs enter the production
  pipeline.
- **No reference-map leakage:** EMSR927 and other published damage
  maps are used only for validation/comparison.
- **Reproducibility:** acquisition, preprocessing, model inference,
  network analysis, and evaluation are versioned.
- **Independent evaluation:** the segmentation model must be evaluated
  on Himalayan scenes not used for training.
- **Honest uncertainty:** the system must distinguish what is directly
  observed from what is inferred.

The system must not claim that a flooded road is necessarily physically
destroyed. Where evidence supports only accessibility disruption, the
output should be described as potential flood-induced road disruption.

## 4. Allowed and prohibited data

### Allowed inputs

- Sentinel-1
- Sentinel-2
- Copernicus DEM / WorldDEM-30 as specified by the challenge
- OpenStreetMap using a pre-event snapshot
- Listed permitted training datasets

The challenge specifically recommends comparing Sentinel-1 acquisitions
from the same orbit track for change detection because different tracks
view terrain from different angles.

### Validation-only sources

- Copernicus EMS / EMSR927
- Other published damage maps
- Post-event OSM edits

These may be used to check results but must never become model or
production-system inputs.

## 5. AI component

The primary AI component is a flood/debris segmentation model trained
using the permitted training datasets.

The benchmark must report performance on unseen Himalayan scenes.
Planned metrics include:

- Intersection over Union (IoU)
- Dice / F1
- Precision
- Recall
- Class-specific performance where labels permit
- Error analysis by terrain, cloud/visibility conditions, and scene
  characteristics where feasible

The exact model architecture will be selected after the data contract
and baseline are established rather than being hard-coded prematurely.

## 6. Impact analysis

### Infrastructure

Pre-event OSM features will be overlaid with the predicted flood/debris
footprint.

Candidate feature classes include:

- buildings
- roads
- bridges

The result should express spatial exposure and inferred impact
separately.

### Road connectivity

The road network will be represented as a graph.

```text
roads -> graph
          |
          v
flood/disruption mask
          |
          v
remove or mark affected segments
          |
          v
recompute connectivity
          |
          v
settlements without viable
connection to reference hubs
```

The output is a **potentially cut-off settlement** assessment, not a
claim that residents are definitively inaccessible.

## 7. Dashboard

The dashboard should expose evidence rather than hide it.

Core layers:

- pre-event imagery
- post-event imagery
- predicted flood/debris extent
- confidence/uncertainty
- buildings affected/exposed
- affected road segments
- bridges
- road-network connectivity
- potentially cut-off settlements
- optional DEM/flood-path analysis

The UI should allow a judge to move from a regional overview to the
evidence supporting an individual impact.

## 8. Situation report

The system will generate a one-page situation report in:

- English
- Nepali

Every numerical claim must originate from system-generated map/analysis
outputs. The language model must not invent statistics.

## 9. Case study

The August 2026 Trishuli flood is the primary demonstration case.

EMSR927 will be used for post-hoc comparison and validation only.

The case-study workflow is:

```text
Permitted raw data
       |
       v
our pipeline
       |
       v
our prediction
       |
       +-------> EMSR927 comparison
       |
       v
quantitative + qualitative evaluation
```

## 10. Repository structure

Actual current state. Directories marked *(empty)* exist as documented
namespaces awaiting their milestone; `(planned)` entries do not exist yet.

```text
.
├── README.md
├── PRD.md
├── architecture.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── configs/                      # pipeline configuration (placeholders, many TODO)
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── segmentation.yaml
│   └── evaluation.yaml
├── docs/
│   ├── data-contract.md          # expected interface per data source
│   ├── dataset-registry.md       # permitted vs validation-only datasets
│   ├── scientific-assumptions.md # what the evidence does and does not support
│   └── evaluation-protocol.md    # splits, metrics, leakage prevention
├── data/                         # contents git-ignored (AGENTS.md §15)
│   ├── raw/                      (empty)
│   ├── interim/                  (empty)
│   ├── processed/                (empty)
│   └── samples/                  (empty; small licensed examples may be committed)
├── src/
│   └── floodmap/
│       ├── __init__.py
│       ├── acquisition/          (empty — namespace only)
│       ├── preprocessing/        (empty — namespace only)
│       ├── features/             (empty — namespace only)
│       ├── segmentation/         (empty — namespace only)
│       ├── infrastructure/       (empty — namespace only)
│       ├── network/              (empty — namespace only)
│       ├── hydrology/            (empty — namespace only)
│       ├── evaluation/           (empty — namespace only)
│       ├── reporting/            (empty — namespace only)
│       └── utils/
│           ├── provenance.py     # artifact provenance schema
│           └── config.py         # YAML configuration loader
├── tests/                        # 108 tests, all passing
│   ├── conftest.py
│   ├── test_package_structure.py
│   ├── test_configs.py
│   ├── test_provenance.py
│   └── test_data_boundary.py     # enforces the AGENTS.md §3 data rule
├── models/                       (empty; git-ignored)
├── artifacts/                    (empty; git-ignored)
├── notebooks/                    (empty)
├── scripts/                      (empty)
├── dashboard/                    (planned — not created)
└── reports/                      (planned — not created)
```

Raw and large derived datasets must remain outside Git history unless
explicitly small enough and legally appropriate.

## 11. Deliverables

**None of the following exist yet.** This list is the target scope.

- Interactive map dashboard
- One-page system-generated situation report
- GitHub repository with reproducible setup/run instructions
- Maximum 6-page technical report
- 3-minute demo video
- Required data attributions and training-dataset citations
- Trishuli case-study results compared with EMSR927

## 12. Required attribution

Every submission must include the challenge-required attribution:

> Contains modified Copernicus Sentinel data 2026.

> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010--2014 and ©
> Airbus Defence and Space GmbH 2014--2018 provided under COPERNICUS by
> the European Union and ESA; all rights reserved.

> © OpenStreetMap contributors.

Training datasets must also be cited according to their respective
requirements.

## 13. Limitations

This is an educational research prototype, not an operational
emergency-response system.

Important limitations include satellite revisit frequency,
cloud/visibility constraints for optical imagery, SAR interpretation
limitations, spatial resolution, uncertainty in damage inference, OSM
completeness, temporal mismatch between imagery and the event, and the
distinction between observed inundation and actual structural failure.

The challenge explicitly notes that Sentinel satellites cannot provide a
warning minutes before a sudden glacier collapse.

## 14. Development status

Current phase: **Phase 1 complete — scientific and engineering foundation.**

### What exists

| Component | Status |
|---|---|
| Repository scaffold, build config, data-exclusion rules | **Done** |
| `docs/data-contract.md` — per-source expected interface | **Done** (with TODO/UNKNOWN markers) |
| `docs/dataset-registry.md` — permitted vs validation-only | **Done** (no row VERIFIED; no data acquired) |
| `docs/scientific-assumptions.md` | **Done** |
| `docs/evaluation-protocol.md` | **Done** (no metric targets, no model chosen) |
| `configs/*.yaml` — machine-readable configuration | **Done** (structural placeholders) |
| `floodmap.utils.provenance` — artifact provenance schema | **Done** |
| `floodmap.utils.config` — configuration loader | **Done** |
| Test infrastructure (108 tests) incl. data-boundary guard | **Done** |

### What does NOT exist

Nothing below is implemented. The corresponding `src/floodmap/` subpackages are
documented namespaces with no logic:

- Sentinel-1 / Sentinel-2 / DEM / OSM acquisition
- Preprocessing and co-registration
- Feature construction
- Flood/debris segmentation model (**no architecture selected** — `AGENTS.md` §5)
- Infrastructure exposure analysis
- Road-network graph and disruption analysis
- Settlement connectivity analysis
- DEM flood-path tracing (bonus)
- Dashboard
- Situation report generation

**No model has been trained and no performance has been measured.** Any
performance figure anywhere in this repository would be fabricated.

### Known blocking gap

The official Track B challenge specification that `AGENTS.md` §2 names as the
primary requirement source is **not present in this repository**. As a result
the **permitted training dataset list is UNKNOWN**, which blocks the
segmentation and evaluation milestones. See `docs/dataset-registry.md` §1.5.

The AOI geometry, exact event date, target grid CRS/resolution, DEM product
variant and reference hub definition are likewise unresolved and recorded as
TODO rather than guessed.

### Progress order

1.  Repository and architecture — **done**
2.  Data provenance and legality checks — **partially done** (contract and
    registry written; licenses still require verification)
3.  Acquisition — not started
4.  Preprocessing — not started
5.  Segmentation baseline — **blocked** on the permitted dataset list
6.  Unseen-Himalaya evaluation — blocked
7.  Infrastructure impact — not started
8.  Road-network connectivity — not started
9.  Optional DEM flood-path analysis — not started
10. Trishuli case study — not started
11. Dashboard — not started
12. Situation report — not started
13. Final evaluation and limitations — not started
14. Demo and submission packaging — not started

## 15. Development setup

```bash
# Editable install with development dependencies
pip install -e ".[dev]"

# Run the test suite
python3 -m pytest

# Formatting
black src tests
```

The geospatial stack (`rasterio`, `geopandas`, `pyproj`, `rioxarray`, `osmnx`)
is declared in `pyproject.toml` under the `geo` extra but is **deliberately
unpinned and not installed**: no module imports it yet, and pinning untested
versions would be a false claim about the environment. Pin it in the milestone
that first needs it.
