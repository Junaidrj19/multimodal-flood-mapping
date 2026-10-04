# PRD --- Multimodal AI Hackathon 2026, Track B

## 1. Product definition

**Product:** Flood Damage and Accessibility Intelligence System

**Track:** Mapping Flood Damage from Space

**Primary case study:** August 2026 Trishuli flood, Nepal

**Primary AI capability:** Flood/debris segmentation evaluated on unseen
Himalayan scenes.

**Users:** Rescue teams, disaster-response analysts, and evaluators
assessing rapid satellite-derived situational awareness.

**Product type:** Research-grade educational prototype.

## 2. Problem statement

When flooding and debris destroy or obstruct roads, bridges, sensors,
and communications, response teams need rapid information about the
affected geography and potentially isolated settlements.

The product converts permitted satellite observations, terrain
information, and pre-event OpenStreetMap data into a chain of evidence:

```text
satellite observations
        ->
flood/debris extent
        ->
infrastructure exposure
        ->
road-network disruption
        ->
potentially cut-off settlements
```

## 3. Goals

### Primary goals

1.  Map flood/debris-affected areas from pre/post satellite
    observations.
2.  Provide a serious, quantitatively evaluated segmentation model.
3.  Evaluate the model on unseen Himalayan scenes.
4.  Estimate affected/exposed buildings, roads, and bridges.
5.  Determine potential loss of road connectivity from settlements to
    reference hubs.
6.  Demonstrate the complete workflow on the August 2026 Trishuli flood.
7.  Produce an interactive dashboard and system-generated situation
    report.
8.  Maintain strict separation between production inputs and validation
    references.

### Secondary goal

Implement a DEM-based downstream flood-path/settlement tracing
capability if time and scientific validation permit.

## 4. Non-goals

The system will not:

- provide minutes-ahead warnings for sudden glacier collapse;
- claim structural failure when the data only establishes spatial
  exposure;
- use EMSR927 as a model input;
- use post-event OSM edits as production inputs;
- use UNOSAT or other published damage maps as production inputs;
- replace professional emergency-response systems;
- present language-model-generated numbers without provenance.

## 5. Core user journey

A judge or user selects:

```text
Area
+
Event date
```

The system then:

1.  acquires permitted satellite/terrain/OSM data;
2.  creates pre/post analysis inputs;
3.  generates flood/debris predictions;
4.  overlays infrastructure;
5.  evaluates road connectivity;
6.  identifies potentially cut-off settlements;
7.  displays results on a map;
8.  generates a concise situation report.

The system must support an arbitrary judge-selected area/date to the
extent that suitable permitted satellite data exists.

## 6. Functional requirements

### FR-01: Area/date input

The system shall accept an area of interest and event date.

### FR-02: Satellite acquisition

The system shall acquire permitted Sentinel-1 and Sentinel-2
observations appropriate to the requested area/date.

For Sentinel-1 change detection, same-orbit-track acquisitions should be
preferred where available.

### FR-03: Terrain acquisition

The system shall obtain the permitted Copernicus DEM product needed for
terrain-aware processing and optional downstream tracing.

### FR-04: Pre-event OSM

The system shall use a pre-event OSM snapshot for buildings, roads, and
bridges.

### FR-05: Preprocessing

The system shall produce analysis-ready, spatially aligned inputs and
retain processing metadata.

### FR-06: Flood/debris segmentation

The system shall produce a spatial prediction of flood/debris classes
supported by the training labels.

### FR-07: AI evaluation

The system shall report quantitative segmentation performance on unseen
Himalayan scenes.

### FR-08: Infrastructure impact

The system shall intersect predictions with permitted pre-event OSM
features and calculate spatial exposure/impact indicators.

### FR-09: Road-network analysis

The system shall represent the road network as a graph and identify
network segments whose usability is potentially disrupted by the
predicted event footprint.

### FR-10: Cut-off settlement analysis

The system shall identify settlements whose road connectivity to defined
reference hubs is lost or potentially lost under the disruption model.

### FR-11: Dashboard

The dashboard shall expose map layers, analysis outputs, and enough
provenance/context for users to understand what each output means.

### FR-12: Situation report

The system shall generate a one-page report in English and Nepali.

Every numerical statement must be traceable to system outputs.

### FR-13: Validation

The system shall support post-hoc comparison against EMSR927 without
passing EMSR927 information into the production pipeline.

## 7. Scientific requirements

The system shall document:

- input sources;
- acquisition dates;
- preprocessing steps;
- coordinate/reference assumptions;
- model version;
- training data;
- test data;
- evaluation metrics;
- thresholding;
- uncertainty;
- known failure modes.

A result must be classified according to evidence strength. At minimum:

```text
Observed / directly derived
        |
        v
Spatially inferred
        |
        v
Network-level inferred
```

The terminology used in the dashboard and report must not imply greater
certainty than the underlying evidence supports.

## 8. AI requirements

The segmentation model must:

- train only on permitted datasets;
- avoid test-scene leakage;
- maintain a reproducible train/validation/test split;
- report performance quantitatively;
- include qualitative error analysis;
- be evaluated on unseen Himalayan imagery;
- retain model/configuration metadata.

Candidate architectures may be evaluated experimentally. Model choice
must follow measured performance, computational feasibility, data
compatibility, and reproducibility rather than popularity alone.

## 9. Road-network model

The road analysis must explicitly define:

- what counts as a road disruption;
- the spatial buffer/intersection rule;
- whether a road segment is removed or assigned a probability/cost;
- reference hubs such as nearest town/hospital;
- settlement-to-hub routing;
- treatment of disconnected components.

The output should be phrased as **potentially cut-off** where the
evidence supports accessibility inference rather than confirmed human
isolation.

## 10. Quality requirements

The prototype should prioritize:

- scientific correctness;
- reproducibility;
- traceable data provenance;
- deterministic or controlled experiments;
- test coverage for core algorithms;
- clear error handling;
- usable visualization;
- transparent limitations.

## 11. Evaluation plan

### Segmentation

Report at least:

- IoU
- Dice/F1
- precision
- recall

Where appropriate, report class-specific metrics.

### Flood/damage mapping

Assess:

- spatial agreement;
- false-positive regions;
- missed flood/debris regions;
- performance under difficult terrain/cloud/land-cover conditions.

### Infrastructure

Report counts/areas of exposed or affected features with clear
definitions.

### Connectivity

Report:

- disrupted road length/segments;
- settlements losing modeled connectivity;
- changes in route availability;
- sensitivity to disruption thresholds where feasible.

### Case study

Compare the Trishuli outputs with EMSR927 after the production analysis
has been completed.

## 12. Acceptance criteria

The project is submission-ready when:

- the complete pipeline runs from permitted raw data to final outputs;
- the pipeline does not require EMSR927 or another published damage
  map as an input;
- a judge-selected area/date can be processed to the extent allowed by
  available data;
- the AI component has quantitative unseen-Himalaya evaluation;
- infrastructure and connectivity outputs are reproducible;
- the dashboard presents the major outputs;
- the English/Nepali situation report is generated from system
  results;
- required citations/attributions are included;
- the six-page report includes limitations;
- the 3-minute demo can explain the workflow and evidence chain.

## 13. Risks

### Data availability

Suitable before/after imagery may not exist for every arbitrary
judge-selected date.

Mitigation: make acquisition transparent and fail gracefully when data
coverage is insufficient.

### SAR/optical mismatch

Different spatial, temporal, and viewing characteristics can produce
misleading change signals.

Mitigation: explicit preprocessing and co-registration; document
modality-specific assumptions.

### Label/domain shift

Training data may not represent Himalayan terrain.

Mitigation: unseen-Himalaya evaluation and explicit domain-shift
analysis.

### OSM incompleteness

Pre-event OSM may omit or misrepresent infrastructure.

Mitigation: label outputs as estimates and quantify data provenance.

### Connectivity false positives

Flood overlap does not prove a road is impassable.

Mitigation: define disruption rules and sensitivity analysis.

### Temporal mismatch

Satellite acquisition time may differ from peak flooding.

Mitigation: preserve timestamps and communicate temporal uncertainty.

## 14. Deliverables

1.  Interactive map dashboard.
2.  One-page English/Nepali situation report.
3.  GitHub repository.
4.  Maximum 6-page technical report.
5.  3-minute demonstration video.
6.  Required attribution and dataset citations.
7.  Trishuli case-study comparison with EMSR927.
