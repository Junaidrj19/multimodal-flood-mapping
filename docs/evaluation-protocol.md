# Evaluation Protocol

Multimodal AI Hackathon 2026 — Track B.

Defined **before** model training, deliberately. An evaluation protocol written
after seeing results is not an evaluation protocol — it is a rationalisation.
Fixing it now is what makes the eventual numbers meaningful.

> **No metric targets are set in this document, and no model is recommended.**
> A "good" IoU chosen before a baseline exists would be arbitrary, and
> `AGENTS.md` §5 forbids selecting a model for any reason other than measured
> performance.

---

## 1. Evaluation philosophy

Four commitments, from `AGENTS.md` §6:

1. **Evaluation is independent of training.** Nothing that touches the test set
   may influence the model.
2. **Report error patterns, not a headline number.** A single IoU hides where and
   how the model fails.
3. **A strong score on an easy random split is not evidence of Himalayan
   generalisation.** The hard split is the one that matters.
4. **Honest failure beats a flattering number.** A low unseen-Himalaya score,
   reported with analysis, is a legitimate scientific result. An inflated score
   from a leaky split is not.

The headline question is not "how well does the model score?" but **"how well
does it work on Himalayan terrain it has never seen?"**

---

## 2. Data separation

### 2.1 Three splits

| Split | Purpose | May influence the model? |
|---|---|---|
| **train** | Fit model parameters. | Yes — by definition. |
| **validation** | Model selection, hyperparameters, early stopping, checkpoint selection, **decision threshold**. | Yes — this is its job. |
| **unseen_himalaya_test** | Final reported generalisation performance. | **No. Never.** |

Configured in `configs/evaluation.yaml → splits`. Dataset and scene lists are
currently empty: the permitted training datasets are **UNKNOWN (blocking)**
(`docs/dataset-registry.md` §1.5).

### 2.2 Splits must be spatial, not random

**Random pixel and random tile splits are forbidden**
(`configs/evaluation.yaml → splits.forbid_random_pixel_split: true`).

**Why this matters more than it sounds:** neighbouring pixels in a satellite
image are strongly spatially correlated. Splitting a single scene randomly puts
pixels from the same flood edge — often from the same *object* — into both train
and test. The model can then score highly by memorising local texture, which is
indistinguishable from genuine generalisation when measured on that split. This
is the single most common way a flood segmentation result is accidentally
inflated.

Permitted strategies:
- `scene_level_spatial` — whole scenes assigned to exactly one split.
- `geographic_block` — contiguous geographic blocks assigned to one split.

### 2.3 The unseen-Himalaya split

Required by `PRD.md` §3 goal 3 and `AGENTS.md` §5.

- Himalayan scenes **not used** in training or validation.
- Used **once**, after all tuning is frozen.
- `AGENTS.md` §5: *do not tune on the final unseen-Himalaya evaluation set.*
- Three separate prohibitions, all enforced in config:
  - `may_be_used_for_tuning: false`
  - `may_be_used_for_threshold_selection: false`
  - `may_be_used_for_checkpoint_selection: false`
- **If it is inspected more than once, it is no longer unseen** and must be
  reported as a validation set instead. Repeated evaluation is tuning by hand.

---

## 3. Leakage prevention

Controls in `configs/evaluation.yaml → leakage_controls`:

| Control | Rule |
|---|---|
| `assert_disjoint_scene_ids` | No scene identifier appears in more than one split. |
| `assert_spatial_disjoint` | Split regions do not overlap geographically. |
| `spatial_buffer_m` | Minimum separation between split regions. **TODO** — set when scene geometry is known. Adjacency across a shared boundary still leaks. |
| `assert_normalisation_from_train_only` | Normalisation statistics computed on training data only, reused unchanged at inference. |
| `assert_no_validation_source_in_production` | No validation-only source reaches any production path. |

### 3.1 Subtle leakage routes to guard against

Beyond the obvious split overlap:

1. **Normalisation statistics** computed over the full dataset leak test
   distribution into training. Statistics come from the training split only
   (`configs/segmentation.yaml → features.normalisation.statistics_source:
   training_split_only`).
2. **Threshold selection on test** is leakage even if weights are untouched —
   the threshold is a fitted parameter.
3. **Checkpoint selection on test** is leakage: picking the best-performing
   epoch by test score fits the test set through model selection.
4. **Repeated test evaluation** with changes in between is manual
   hyperparameter search against the test set.
5. **Overlapping tiles** spanning a split boundary put the same ground in both
   splits.
6. **Pre-trained encoder weights** may have seen related imagery. Not
   necessarily disqualifying, but **must be disclosed** —
   `configs/segmentation.yaml → model.pretrained_weights`.
7. **Validation-only reference products** (EMSR927, UNOSAT) influencing any
   threshold, feature or model decision. This is the `AGENTS.md` §3 critical
   defect; `tests/test_data_boundary.py` checks it mechanically.

---

## 4. Metrics

Required by `AGENTS.md` §6 and `PRD.md` §11. Let TP, FP, FN be pixel counts for
the class under evaluation.

### 4.1 Intersection over Union

```text
IoU = TP / (TP + FP + FN)
```

Primary segmentation metric. Penalises false positives and false negatives
without crediting the typically vast true-negative background.

### 4.2 Dice / F1

```text
Dice = F1 = 2·TP / (2·TP + FP + FN)
```

Monotonically related to IoU but less harsh on small objects. Both are reported
because both appear in the literature, which makes comparison possible.

### 4.3 Precision

```text
Precision = TP / (TP + FP)
```

Of the pixels predicted flooded, the fraction actually flooded. Low precision
means **false alarms** — which in this application means directing response
effort to unaffected areas.

### 4.4 Recall

```text
Recall = TP / (TP + FN)
```

Of the actually flooded pixels, the fraction detected. Low recall means
**missed flooding** — affected communities invisible to the system.

### 4.5 Precision and recall are not interchangeable here

They carry asymmetric operational cost, and that asymmetry is a decision for
domain experts, not a default. Reporting a single F1 hides the trade-off, so
precision and recall are always reported separately alongside the chosen
threshold and its sensitivity curve.

### 4.6 Reporting requirements

- **Per-class** metrics (`metrics.per_class: true`) — flood and debris may
  perform very differently, and an aggregate would conceal that.
- **Full confusion matrix** (`metrics.confusion_matrix: true`), per `AGENTS.md`
  §6.
- **Both macro and micro averaging.** Under heavy class imbalance — flood pixels
  are typically a small minority — these diverge sharply, and quoting only the
  favourable one would be misleading.
- **Support** (pixel counts) per class, so a metric computed over a handful of
  pixels is not mistaken for a stable estimate.
- **Accuracy is not reported as a headline.** With a small positive class, a
  model predicting "no flood" everywhere scores high accuracy while being
  useless.

---

## 5. Threshold selection

The **procedure** is fixed here; the **value** is not
(`configs/evaluation.yaml → threshold_selection`).

1. Train with tuning on the training split only.
2. Sweep the decision threshold on the **validation** split.
3. Select by a criterion **declared before the sweep**
   (`selection_criterion` — TODO, must be stated in advance so the choice is not
   made post hoc to flatter a result).
4. Freeze the threshold.
5. Apply it unchanged to the unseen-Himalaya split.
6. Record the full sensitivity curve (`record_sensitivity_curve: true`) so a
   reader can see how fragile the conclusion is.

`selection_split: validation` — never `unseen_himalaya_test`.

---

## 6. Qualitative error analysis

Required by `AGENTS.md` §6 and `PRD.md` §11; numbers alone do not explain
behaviour.

### 6.1 Stratification

Metrics recomputed within strata (`error_analysis.stratify_by`): terrain slope,
elevation, land cover, cloud condition, and per scene. A model may be adequate
on valley floors and fail on steep slopes — an aggregate hides exactly the
failure mode that matters in Himalayan terrain.

### 6.2 Failure modes to inspect explicitly

From `error_analysis.inspect_failure_modes`, each a known mechanism rather than
a generic bucket:

| Mode | Expected effect |
|---|---|
| `radar_shadow` | Dark in SAR → **false positive** water. Systematic in steep terrain. |
| `layover` | Geometric distortion → misplaced boundaries. |
| `smooth_surface_low_backscatter` | Dry smooth surfaces mimic water → false positive. |
| `permanent_water_vs_flood_water` | River detected as flood every time → inflated extent. |
| `wet_soil_vs_standing_water` | Genuinely intermediate; neither clean positive nor negative. |
| `debris_vs_bare_rock` | Rock/ice debris may be spectrally and radiometrically close to natural bare rock. |

Snow and ice are an additional Himalayan-specific confuser, relevant given an
ice/rock avalanche source region.

### 6.3 Qualitative examples

Export representative successes **and failures** per stratum
(`export_qualitative_examples: true`). Showing only successes would be
selective reporting.

---

## 7. Beyond segmentation

`PRD.md` §11 requires evaluation of the downstream stages too. Each is
**inference**, and its evaluation must not be presented as segmentation
accuracy.

### 7.1 Infrastructure exposure

- Report counts and lengths of **exposed** features with the definition and
  buffer width used.
- Report **sensitivity to buffer width** — a count quoted without it is not
  reproducible.
- Vocabulary per `AGENTS.md` §8: exposed, not destroyed.

### 7.2 Connectivity

- Disrupted road length and segment count.
- Settlements losing modelled connectivity.
- Changes in route availability (baseline vs post-event).
- **Sensitivity to the disruption threshold** (`AGENTS.md` §9) — if conclusions
  flip across a plausible threshold range, that instability is the finding.
- Compounding uncertainty is stated: connectivity depends on segmentation being
  right **and** the buffer being reasonable **and** OSM being complete **and**
  the graph model being valid.

### 7.3 Case study comparison

- EMSR927 comparison runs **after** the production analysis is complete
  (`architecture.md` §11).
- Reported as spatial agreement, **not** accuracy against ground truth.
- Reference products carry their own method and timing assumptions; disagreement
  is not automatically our error.

---

## 8. Reproducibility

Per `AGENTS.md` §5, every experiment records:

```text
dataset · split · features · model · hyperparameters · seed · metrics · checkpoint
```

Extended for provenance (`configs/evaluation.yaml → tracking.record_fields`)
with `code_version`, `config_version` and `environment`.

A result must be reproducible from code version + configuration + input scene
identifiers + model checkpoint + environment (`architecture.md` §14).

### 8.1 Determinism

- Seeds set explicitly (`configs/segmentation.yaml → training.seed` — TODO).
- `deterministic: true` requested.
- **Caveat worth stating rather than hiding:** some GPU kernels are
  non-deterministic even when seeded. Where bit-exact reproduction is not
  achievable, the limitation is recorded rather than claimed away, and
  run-to-run variance is reported instead.

### 8.2 Artifact tracking

Every evaluation result carries an `ArtifactProvenance` record
(`floodmap.utils.provenance`) with `artifact_type=evaluation_result`, written to
`artifacts/` (git-ignored).

---

## 9. What this protocol refuses to do

| Refusal | Reason |
|---|---|
| Set metric targets | A threshold before a baseline is arbitrary (`metrics.targets: null`). |
| Recommend a model | `AGENTS.md` §5 — choice follows measurement, not fashion. |
| Promise a performance level | Unseen-Himalaya performance is unknown until measured. |
| Treat EMSR927 as ground truth | It is a comparison reference with its own assumptions. |
| Report a single headline score | `AGENTS.md` §6 requires error patterns. |

---

## 10. Blocking items

1. **Permitted training dataset list** — UNKNOWN (blocking). Without it, splits
   cannot be populated and the class set cannot be defined.
2. **Himalayan evaluation scenes** — not identified; depends on item 1 and on
   AOI definition.
3. **Spatial buffer between splits** — TODO; requires scene geometry.
4. **Threshold selection criterion** — TODO; must be declared before the sweep.
5. **Class set** — depends on whether permitted labels distinguish flood water
   from debris. If they do not, the product claim narrows to flood water
   (`docs/scientific-assumptions.md` §9).
