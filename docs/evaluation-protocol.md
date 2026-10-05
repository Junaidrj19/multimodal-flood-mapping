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

Configured in `configs/evaluation.yaml → splits`. Both permitted datasets have
now been audited (`docs/dataset-registry.md` §1.5) and **both ship official
event-based splits**, which removes the main reason the scene lists were empty.
They remain empty pending acquisition, but they are no longer unconstrained:
§2.4 fixes which activations are off-limits, and §2.5 fixes the label
harmonisation rule. Inventing split membership would still be the leakage risk
this protocol exists to prevent.

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

### 2.4 Upstream splits are binding

Both corpora ship official event/geography-based splits, and honouring them is
a rule rather than a courtesy.

**Kuro Siwo** — from `configs/train/data_config.json`, by activation id:

| Upstream split | Activation IDs | Our permitted use |
|---|---|---|
| test | `321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013` | **Never train or validate on these.** |
| val | `514, 559, 279, 520, 437, 1111003, 1111008` | Validation only. |
| train | remaining 27 of 43 | Training. |

**Sen1Floods11** — splits are per-event in its metadata, with the split files in
the dataset bucket. Exact test composition is TODO(verify) until downloaded;
until then no Sen1Floods11 chip may be assigned to a split.

**Why this is a real rule and not bookkeeping.** If we train on an upstream test
activation, every published number from other work on Kuro Siwo becomes
incomparable to ours — theirs is measured on held-out scenes, ours on scenes we
fitted. The comparison would silently favour us. `assert_disjoint_scene_ids`
must therefore be checked against the **upstream** split assignment, not only
against our own.

> Recorded machine-readably at
> `configs/data.yaml → production.training_datasets.datasets[kuro_siwo].official_split_activation_ids`
> with `upstream_test_activations_are_offlimits: true`.

### 2.5 Label harmonisation rule

The two corpora do not share a label taxonomy
(`docs/data-contract.md` §5.3): Kuro Siwo is 3-class and separates permanent
water from flood water; Sen1Floods11 is binary surface water.

The mapping is **asymmetric**, and only one direction is defined:

```text
Kuro Siwo {Permanent Waters, Floods} --lossy--> {Water}        DEFINED
Sen1Floods11 {Water} --> {Permanent Waters, Floods}            NOT POSSIBLE
```

**Rules:**

1. **A naive union of the two corpora under one head is forbidden.** It would
   either relabel Sen1Floods11 water as a class it does not carry, or discard
   Kuro Siwo's permanent-vs-flood distinction without saying so.
2. If both corpora are used, they must be used under **separate heads or
   separate training stages**, with the mapping direction recorded per source.
3. Any run that collapses Kuro Siwo to binary **must report that the
   permanent-water distinction was discarded**, because that distinction is the
   defence against reporting the river as flood
   (`docs/scientific-assumptions.md` §7).
4. `configs/segmentation.yaml → classes.label_mapping` records the mapping
   **per source dataset**, never as a single global map.
5. Sen1Floods11's `-1` and Kuro Siwo's `3` are both ignore values and must be
   excluded from loss and from every metric — not mapped to a negative class.
   Treating "no data" as "not water" would make unobserved pixels count as
   correct negatives and inflate every score.

### 2.6 There is no labelled Himalayan test set in the permitted data

The most consequential finding of the dataset audit
(`docs/dataset-registry.md` §5.3).

Neither corpus is documented as containing Himalayan or high-mountain terrain.
Kuro Siwo's only Nepal-labelled activation is tropical and lowland by its own
metadata **and** sits in the upstream test split. Sen1Floods11's nearest
approaches are sub-Himalayan foothills that no source characterises as
mountainous.

`PRD.md` §3 goal 3 requires evaluation on unseen Himalayan scenes. That
requirement cannot be satisfied with permitted *labels*, so the protocol splits
it into two separately-reported things rather than pretending one exists:

| Report | What it is | What it is not |
|---|---|---|
| **Held-out generalisation** | Metrics on the most Himalaya-like held-out permitted scenes available (steepest terrain, highest relief), using permitted labels. Stratified by slope and elevation per §6.1. | Not Himalayan performance. A proxy, and labelled as a proxy. |
| **Trishuli spatial agreement** | Agreement between our frozen prediction and EMSR927 over the real Himalayan AOI. | **Not accuracy.** EMSR927 is a comparison reference, not ground truth, and it is read once after freezing (§7.3). |

**Both must be reported, and neither may be presented as the other.** Quoting
the proxy as "unseen-Himalaya IoU" would overstate what was measured; quoting
the EMSR927 agreement as accuracy would treat a reference product as truth.

The expected consequence — stated now, before measurement, so it cannot be
rationalised later — is that the gap between in-domain and Himalaya-like
performance will be **large**, because no permitted corpus teaches the model
high-relief radar geometry, snow/ice confusion, or steep-terrain shadow and
layover. Per `AGENTS.md` §6 that gap is itself a headline result.

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

### 7.3 Case study comparison — the EMSR927 freeze protocol

EMSR927 is **validation-only** (`AGENTS.md` §3, `architecture.md` §18.1). It must
never enter training, feature generation, preprocessing, threshold tuning, model
selection or production inference. The following procedure is what makes that
enforceable rather than aspirational, and the ordering is the whole point.

**Freeze gate — every item must be true and recorded BEFORE EMSR927 is fetched:**

1. Model architecture, weights and checkpoint are final and hashed.
2. The decision threshold is selected on **validation** and frozen (§5).
3. Normalisation statistics are fixed and training-split-derived.
4. The production prediction for the Trishuli AOI exists on disk, with its
   `ArtifactProvenance` written and its `generated_at` timestamp recorded.
5. `configs/evaluation.yaml → reference_comparison.enabled` is still `false`.

**Then, and only then:**

6. Flip `reference_comparison.enabled` to `true` as an explicit, reviewable act.
7. Retrieve EMSR927. Record its retrieval timestamp — it must be **later** than
   the prediction artifact's `generated_at`. That timestamp ordering is the
   audit evidence that the reference could not have influenced the prediction.
8. Compute spatial agreement. Report it as **agreement**, never as accuracy,
   precision or recall against ground truth.
9. Carry the EMS attribution on the comparison artifact and in the report
   section that presents it: *"European Union, Copernicus Emergency Management
   Service data"*.

**After the comparison, nothing may change.** If the comparison is disappointing
and the model, threshold or features are then adjusted, EMSR927 has become a
tuning signal and the result is void. A revised model requires the whole
protocol to restart from a fresh freeze, and the report must disclose that the
comparison was run more than once.

**Interpretation constraints:**

- Reported as spatial agreement, **not** accuracy against ground truth.
- Reference products carry their own method, timing and interpretation
  assumptions; disagreement is not automatically our error, and agreement is not
  proof of correctness (`docs/scientific-assumptions.md` §10.4).
- EMSR927's acquisition time almost certainly differs from ours, so part of any
  disagreement is temporal rather than methodological (§5 of
  `docs/scientific-assumptions.md`).
- The dashboard must not visually imply EMSR927 is an input layer
  (`AGENTS.md` §16).

> **Independence caveat — disclosed, not hidden.** Kuro Siwo's labels were
> *initialised from Copernicus EMS shapefiles* before expert photointerpretation
> (`docs/dataset-registry.md` §1.5.4). A model trained on Kuro Siwo therefore
> inherits CEMS interpretation conventions second-hand, and an EMS-family
> reference product shares those conventions. The comparison is consequently
> **not** between two fully independent methods, and agreement is weaker
> evidence than it appears. This does not breach the boundary — no EMS product
> is read by our pipeline, and EMSR927 postdates Kuro Siwo entirely — but it
> must be stated wherever the comparison is presented.

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

### 10.1 Resolved by the dataset audit

| # | Item | Outcome |
|---|---|---|
| 1 | Official dataset splits | **RESOLVED.** Both event-based. Kuro Siwo's activation IDs recorded; honouring them is now a rule (§2.4). |
| 2 | Class set / debris question | **RESOLVED.** Neither corpus labels debris. The segmentation claim narrows to flood water; see `docs/m4-architecture-decision.md` §6. |
| 3 | Label harmonisation rule | **RESOLVED.** Specified in §2.5; the mapping is asymmetric and a naive union is forbidden. |

### 10.2 Still blocking

1. **Himalayan evaluation data does not exist in the permitted corpora.** Not a
   gap in our knowledge any more — a measured property of the permitted data
   (§2.6, `docs/dataset-registry.md` §5.3). The protocol now reports a labelled
   proxy and an unlabelled EMSR927 agreement separately. Closing this properly
   would require a permitted Himalayan labelled source, which the closed
   training list does not provide.
2. **Spatial buffer between splits** — TODO; requires scene geometry.
3. **Threshold selection criterion** — TODO; must be declared before the sweep.
4. **Sen1Floods11 exact test-split composition** — split files not downloaded;
   until then no chip may be assigned to a split (§2.4).
5. **Kuro Siwo flood definition** — in the paper's Supplemental Material, unread.
   Required before the label mapping is frozen, because our `Floods` class
   semantics inherit theirs.
6. **Kuro Siwo stated label noise** — unread; bounds how much of any residual
   error is attributable to the model rather than the labels.
7. **UNOSAT citation format** — required before publishing any comparison
   against it.
