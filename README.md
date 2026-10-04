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

```text
.
├── README.md
├── PRD.md
├── architecture.md
├── AGENTS.md
├── pyproject.toml
├── configs/
├── data/
├── src/
│   └── floodmap/
│       ├── acquisition/
│       ├── preprocessing/
│       ├── features/
│       ├── segmentation/
│       ├── infrastructure/
│       ├── network/
│       ├── hydrology/
│       ├── evaluation/
│       ├── reporting/
│       └── utils/
├── notebooks/
├── tests/
├── dashboard/
├── reports/
├── docs/
└── scripts/
```

Raw and large derived datasets must remain outside Git history unless
explicitly small enough and legally appropriate.

## 11. Deliverables

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

Current phase: **Phase 0 --- Scientific architecture and data contract**

The project will progress in this order:

1.  Repository and architecture
2.  Data provenance and legality checks
3.  Acquisition
4.  Preprocessing
5.  Segmentation baseline
6.  Unseen-Himalaya evaluation
7.  Infrastructure impact
8.  Road-network connectivity
9.  Optional DEM flood-path analysis
10. Trishuli case study
11. Dashboard
12. Situation report
13. Final evaluation and limitations
14. Demo and submission packaging
