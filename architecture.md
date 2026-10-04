# Architecture --- Mapping Flood Damage from Space

## 1. Architectural objective

The architecture converts permitted remote-sensing observations into an
auditable chain of spatial and network-level evidence.

The core design is:

```text
                 +-----------------------+
                 | Area + Event Date     |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Data Acquisition      |
                 +-----------+-----------+
                             |
           +-----------------+-----------------+
           |                 |                 |
           v                 v                 v
     Sentinel-1        Sentinel-2       Copernicus DEM
           |                 |                 |
           +-----------------+-----------------+
                             |
                             v
                 +-----------------------+
                 | Preprocessing         |
                 | - alignment           |
                 | - normalization       |
                 | - QA                  |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Feature Construction  |
                 | pre/post + indices    |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Segmentation Model    |
                 +-----------+-----------+
                             |
                  flood/debris prediction
                             |
             +---------------+----------------+
             |               |                |
             v               v                v
       Infrastructure    Road Network       DEM/Hydrology
          Impact          Analysis            (bonus)
             |               |                |
             |               v                |
             |        connectivity graph      |
             |               |                |
             +---------------+----------------+
                             |
                             v
                 +-----------------------+
                 | Impact Synthesis      |
                 +-----------+-----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Interactive Map              Situation Report
```

## 2. Trust boundary

The production system has a strict input boundary.

```text
                    PRODUCTION INPUTS
        +--------------------------------------+
        | Sentinel-1                            |
        | Sentinel-2                            |
        | Copernicus DEM                       |
        | Pre-event OpenStreetMap              |
        | Permitted training datasets           |
        +------------------+-------------------+
                           |
                           v
                    PRODUCTION PIPELINE
                           |
                           v
                       OUTPUTS
                           |
                           v
                    VALIDATION ONLY
        +--------------------------------------+
        | EMSR927                              |
        | Published damage maps                |
        | Post-event OSM edits                 |
        +--------------------------------------+
```

Validation sources must be physically and logically separated from the
model inference path.

## 3. Data acquisition layer

### Sentinel-1

Purpose:

- cloud-independent radar observation;
- pre/post flood change detection;
- support for monsoon conditions.

Acquisition strategy:

- prioritize matching orbit tracks;
- preserve acquisition timestamp and orbit metadata;
- prefer comparable viewing geometry;
- record unavailable or unsuitable acquisitions.

### Sentinel-2

Purpose:

- optical confirmation and high-detail surface information when cloud
  conditions permit;
- multispectral feature construction.

Processing must account for cloud and invalid-pixel conditions.

### Copernicus DEM

Purpose:

- terrain context;
- slope/elevation features;
- optional downstream flood-path tracing.

### OpenStreetMap

Purpose:

- pre-event buildings;
- roads;
- bridges.

The project must obtain a snapshot dated before the event.

## 4. Preprocessing layer

The preprocessing layer produces standardized analysis-ready inputs.

Responsibilities include:

1.  spatial reference normalization;
2.  clipping to the analysis area;
3.  raster alignment/co-registration;
4.  missing-data handling;
5.  modality-specific normalization;
6.  quality masking;
7.  metadata/provenance recording.

No model should silently perform hidden preprocessing that is absent
from the documented pipeline.

## 5. Feature layer

The feature layer creates model inputs from permitted observations.

Potential features include:

- Sentinel-1 backscatter channels;
- pre/post SAR differences or ratios;
- Sentinel-2 spectral bands;
- spectral indices;
- pre/post optical differences;
- terrain/elevation features where justified.

Feature construction must be reproducible and configurable.

## 6. Segmentation layer

Input:

```text
analysis-ready multimodal raster
```

Output:

```text
class probability raster
+
segmentation mask
+
model metadata
```

The model must expose enough metadata to reproduce an inference.

The architecture deliberately does not lock the project to one network
architecture. The selected model must be justified experimentally.

## 7. Infrastructure impact layer

The infrastructure engine receives:

```text
segmentation mask
+
pre-event OSM
```

It produces feature-level spatial relationships.

Example:

```text
Flood/debris mask
      |
      v
spatial intersection
      |
      +--> buildings
      +--> roads
      +--> bridges
```

The output should preserve the distinction between:

- geometrically exposed;
- potentially disrupted;
- inferred inaccessible.

## 8. Road-network layer

Roads become a graph:

```text
                  road segment
             +-------------------+
             |                   |
        node A ---------------- node B
             |                   |
             |       node C      |
             +---------+---------+
                       |
                  node D
```

A disruption engine assigns each road segment a state/cost based on the
event footprint.

Conceptually:

```text
road graph
   +
predicted disruption
   |
   v
modified graph
   |
   v
shortest-path/connectivity analysis
   |
   v
settlement -> reference hub reachability
```

The architecture should support threshold sensitivity so that
conclusions are not dependent on one arbitrary overlap threshold.

## 9. Settlement analysis

Inputs:

- settlement locations;
- road graph;
- reference hubs;
- disrupted network.

Output:

```text
settlement_id
reachable: true/false
reference_hub
baseline route
post-event route
status
```

Recommended status vocabulary:

- CONNECTED
- POTENTIALLY DISRUPTED
- POTENTIALLY CUT OFF
- UNKNOWN / INSUFFICIENT DATA

## 10. DEM flood-path bonus

The optional hydrology module starts from an upstream point and uses
terrain-derived downstream direction to estimate a plausible flood path.

This is a model-based terrain inference, not a hydraulic simulation.

The implementation must not present it as a physically exact prediction
unless a validated hydrodynamic model supports that claim.

## 11. Evaluation architecture

Evaluation is separate from inference.

```text
TRAINING
permitted training datasets
        |
        v
     model

INFERENCE
unseen Himalayan scene
        |
        v
     prediction
        |
        +----------+
                   |
                   v
             evaluation
                   ^
                   |
        independent reference
```

EMSR927 enters only the final validation/comparison branch.

## 12. Provenance

Each major artifact should have provenance metadata.

Example:

```yaml
artifact:
  type: segmentation_prediction
  area: <AOI>
  event_date: <date>
  sentinel1:
    before: <scene-id>
    after: <scene-id>
  sentinel2:
    before: <scene-id>
    after: <scene-id>
  dem: <dataset-version>
  osm_snapshot: <snapshot-date>
  model: <model-version>
  preprocessing: <pipeline-version>
  generated_at: <timestamp>
```

## 13. Software architecture

Recommended source modules:

```text
src/floodmap/
├── acquisition/
├── preprocessing/
├── features/
├── segmentation/
├── infrastructure/
├── network/
├── hydrology/
├── evaluation/
├── reporting/
└── utils/
```

Module boundaries should follow scientific responsibilities rather than
individual notebooks.

## 14. Reproducibility

Experiments should be controlled by configuration files.

```text
configs/
├── data.yaml
├── preprocessing.yaml
├── segmentation.yaml
└── evaluation.yaml
```

A model result should be reproducible from:

```text
code version
+
configuration
+
input scene identifiers
+
model checkpoint
+
environment
```

## 15. Dashboard architecture

The dashboard consumes generated artifacts rather than executing core
scientific algorithms in the browser.

```text
Scientific pipeline
       |
       v
validated artifacts
       |
       v
API / artifact service
       |
       v
interactive dashboard
```

The dashboard should visualize:

- flood/debris extent;
- confidence;
- infrastructure exposure;
- road disruption;
- settlement connectivity;
- optional flood path;
- source/provenance metadata.

## 16. Reporting architecture

The reporting system consumes structured outputs.

```text
structured analysis results
        |
        +--> English report
        |
        +--> Nepali report
```

The language model, if used, is a rendering/summarization layer. It must
not become the source of numerical truth.

## 17. Failure handling

The pipeline must distinguish:

- no suitable imagery;
- cloud-contaminated optical imagery;
- insufficient temporal coverage;
- invalid geometry;
- incomplete OSM;
- model confidence too low;
- connectivity result indeterminate.

Failure should be explicit rather than silently producing a misleading
map.

## 18. Architectural invariants

These rules must not be violated:

1.  EMSR927 is validation-only.
2.  Post-event OSM is validation-only.
3.  Published damage maps are validation-only.
4.  Production inference uses only permitted inputs.
5.  Training/test leakage is prohibited.
6.  Every numerical report claim originates from structured system
    outputs.
7.  Spatial exposure is not automatically equivalent to physical damage.
8.  Network disconnection is an inference about modeled accessibility.
9.  All major outputs retain provenance.
10. Limitations are part of the product, not an afterthought.
