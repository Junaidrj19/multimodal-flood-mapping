# AGENTS.md --- Track B Engineering and Research Instructions

## 1. Mission

Build a scientifically defensible, reproducible prototype for the
Multimodal AI Hackathon 2026 --- Track B: Mapping Flood Damage from
Space.

The priority order is:

``` text
Scientific correctness
>
Data compliance
>
Reproducibility
>
Evaluation quality
>
Robust engineering
>
Usability
>
Visual polish
```

Do not sacrifice scientific validity for a visually impressive demo.

## 2. Source of truth

The official Track B challenge specification supplied to this project is
the primary requirement source.

When implementation decisions conflict with the challenge rules, the
challenge rules win.

The challenge requires the system to answer:

-   where the flood hit;
-   what infrastructure was affected;
-   who may be cut off.

It also requires end-to-end execution from raw satellite data for
judge-selected areas/dates.

## 3. Data governance

### Allowed production inputs

-   Sentinel-1
-   Sentinel-2
-   Copernicus DEM
-   pre-event OpenStreetMap
-   permitted training datasets

### Never use as production inputs

-   EMSR927;
-   Copernicus EMS damage products;
-   UNOSAT damage maps;
-   other published damage maps;
-   post-event OSM edits.

These sources can be used for validation/comparison only.

If code accidentally imports validation data into inference or feature
generation, treat this as a critical defect.

## 4. Temporal discipline

Always preserve acquisition timestamps.

For Sentinel-1 change detection, prefer the same orbit track when
possible. The challenge specifically warns that different tracks view
terrain from different angles and should not be compared pixel-by-pixel
as though they were equivalent observations.

Do not silently mix temporal windows.

## 5. AI development rules

Do not select a model because it is fashionable.

Before committing to a model, establish:

-   label compatibility;
-   input modality compatibility;
-   spatial resolution;
-   training/test split;
-   compute requirements;
-   baseline performance;
-   unseen-Himalaya performance.

Every experiment should record:

``` text
dataset
split
features
model
hyperparameters
seed
metrics
checkpoint
```

Do not tune on the final unseen-Himalaya evaluation set.

## 6. Evaluation rules

Evaluation must be independent of training.

Minimum segmentation metrics:

-   IoU
-   Dice/F1
-   precision
-   recall

Report confusion/error patterns, not only one headline score.

A strong score on an easy random split is not sufficient evidence of
Himalayan generalization.

## 7. Geospatial rules

Do not assume rasters are aligned.

Before pixel-level comparison, verify:

-   CRS;
-   transform;
-   resolution;
-   extent;
-   pixel alignment;
-   acquisition geometry where relevant.

Do not resample without documenting the method and reason.

Maintain geospatial metadata throughout the pipeline.

## 8. Infrastructure semantics

Use precise terminology.

Preferred:

-   exposed;
-   intersected;
-   potentially disrupted;
-   potentially inaccessible;
-   potentially cut off.

Avoid claiming:

-   destroyed;
-   structurally failed;
-   inaccessible;
-   isolated

unless the available evidence genuinely supports that stronger
statement.

A flood mask intersecting a road is evidence of spatial exposure, not
automatic evidence of physical road destruction.

## 9. Network analysis

Define all graph assumptions explicitly.

At minimum document:

-   road segmentation;
-   disruption threshold;
-   graph construction;
-   settlement representation;
-   reference hubs;
-   route algorithm;
-   treatment of disconnected components.

Where practical, test sensitivity to disruption thresholds.

## 10. DEM bonus

Treat DEM-derived flood-path tracing as terrain-based inference.

Do not describe it as a validated hydrodynamic simulation unless such a
model is actually implemented and validated.

## 11. LLM usage

AI coding/research agents may be used heavily for:

-   architecture;
-   implementation;
-   debugging;
-   testing;
-   documentation;
-   literature/data-source investigation;
-   code review;
-   experiment design.

However:

**An AI-generated statement is not scientific validation.**

Every important scientific claim must be checked against:

-   source data;
-   code behavior;
-   numerical results;
-   authoritative documentation;
-   reproducible experiments.

The language model must never be the source of truth for numerical
outputs.

## 12. Code organization

Keep scientific logic in `src/floodmap/`.

Do not make notebooks the only implementation.

Notebooks are for:

-   exploration;
-   visualization;
-   experiment analysis;
-   debugging;
-   presentation of results.

Reusable logic belongs in tested modules.

## 13. Testing

Every non-trivial algorithm should have tests.

High-priority tests:

-   coordinate/grid handling;
-   preprocessing;
-   feature construction;
-   segmentation output shape/classes;
-   spatial intersection;
-   road graph construction;
-   connectivity analysis;
-   metric calculation;
-   provenance generation.

Include edge cases.

## 14. Reproducibility

Use configuration files rather than hard-coded experimental parameters.

Avoid hidden state.

Record:

-   code version;
-   dataset identifiers;
-   acquisition dates;
-   configuration;
-   model version;
-   random seeds;
-   environment/dependencies.

## 15. Data storage

Do not commit large raw satellite scenes or generated datasets to Git.

Use `.gitignore` for:

``` text
data/raw/
data/interim/
data/processed/
models/
artifacts/
```

Small reproducible examples may be committed when licensing permits.

## 16. Dashboard rules

The dashboard is a visualization and decision-support layer.

Do not duplicate scientific logic in frontend code.

The dashboard must clearly distinguish:

``` text
Observed / derived
vs.
Model prediction
vs.
Network inference
vs.
Validation reference
```

Do not visually imply that EMSR927 is an input layer.

## 17. Reporting rules

The report must include limitations.

Do not hide weaknesses such as:

-   satellite revisit limitations;
-   cloud contamination;
-   SAR ambiguity;
-   spatial resolution;
-   domain shift;
-   OSM incompleteness;
-   temporal mismatch;
-   uncertain infrastructure damage;
-   uncertain connectivity inference.

A limitation that affects the result must be stated in the final report
and, where relevant, surfaced in the dashboard.

## 18. Attribution

Every submission must include the required challenge attributions:

> Contains modified Copernicus Sentinel data 2026.

> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010--2014 and ©
> Airbus Defence and Space GmbH 2014--2018 provided under COPERNICUS by
> the European Union and ESA; all rights reserved.

> © OpenStreetMap contributors.

Training datasets must be cited according to their respective licenses
and papers.

## 19. Git workflow

Use small, logically isolated commits.

Recommended commit categories:

``` text
feat:
fix:
refactor:
test:
docs:
data:
experiment:
```

Do not mix major architectural changes with unrelated formatting.

Before merging a scientific component:

1.  run tests;
2.  run the relevant experiment;
3.  inspect outputs;
4.  record metrics;
5.  update documentation if assumptions changed.

## 20. Pull-request checklist

Before accepting a PR:

-   Does it obey the data rules?
-   Does it introduce leakage?
-   Are geospatial assumptions explicit?
-   Are tests included?
-   Is the result reproducible?
-   Are metrics computed correctly?
-   Are claims stronger than the evidence?
-   Is provenance retained?
-   Does the implementation work outside the Trishuli example?
-   Does documentation match actual behavior?

## 21. Definition of done

A feature is not done because the code runs.

It is done when:

``` text
implemented
+
tested
+
scientifically checked
+
documented
+
reproducible
```

## 22. Final principle

Build the system so that a technically skeptical reviewer can ask:

> "How do you know this?"

and the system can answer with:

``` text
source data
→ processing
→ model
→ metric / rule
→ derived result
→ uncertainty
```

That chain of evidence is more important than adding another AI feature.
