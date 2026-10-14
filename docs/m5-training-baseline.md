# M5 — SAR Segmentation Training Baseline

Training infrastructure for the first supervised flood segmentation baseline.
Implemented in `src/floodmap/segmentation/`, configured in
`configs/experiments/`, tested in `tests/test_segmentation_m5.py` and
`tests/test_segmentation_m5_real.py`.

**Status: infrastructure verified, baseline not trained.** The pipeline has
been shown to execute and to learn on a handful of real tiles. No full training
run has been launched, and no performance claim is made anywhere in this
milestone.

---

## 1. The frozen baseline definition

SAR only. Kuro Siwo labelled partition `01`. One pre-event epoch and one
post-event epoch. Four channels. Three semantic classes.

### 1.1 Input channels

| Index | Name | Source | Temporal role | Polarisation |
|---|---|---|---|---|
| 0 | `pre_vv` | SL1 | pre-event, `crank == 1` | VV |
| 1 | `pre_vh` | SL1 | pre-event, `crank == 1` | VH |
| 2 | `post_vv` | MS1 | post-event, `master == true` | VV |
| 3 | `post_vh` | MS1 | post-event, `master == true` | VH |

The order is frozen in `segmentation/channels.py` as `CHANNEL_ORDER` and is
recorded in both the statistics artifact and every checkpoint. Reordering it
silently re-maps a trained model's first-layer weights, so the experiment
config, the statistics artifact and the checkpoint loader each reject any other
ordering rather than accepting it.

Temporal roles are read from the M4 canonical sample (`sample.sar["pre"]` /
`sample.sar["post"]`), which the adapter resolved from `info.json`
`master`/`crank` fields. The filename prefix is never parsed here. `SL2`, the
second pre-event epoch, stays recorded as a dropped source and is not used.

### 1.2 Representation

Linear sigma0, clipped at 0.15 by the M4 adapter. **Not converted to decibel.**

`tensors.py` rejects any sample whose `sar_representation` is not
`linear_sigma0`. That check is load-bearing rather than defensive:
Sen1Floods11 is delivered in decibel, and feeding it through statistics fitted
on linear sigma0 would be a silent ~20 dB offset producing plausible, wrong
output.

### 1.3 Explicitly excluded from this baseline

Sentinel-2, Copernicus DEM, Kuro's bundled `MK0_DEM` and `MK0_SLOPE`, OSM,
roads, buildings, handcrafted indices, the `SL2` epoch, speckle filtering,
decibel conversion, radiometric augmentation, pretrained weights and class
weighting.

None of these are excluded because they are useless. A clean SAR pre/post
reference has to exist before anything can be shown to improve on it
(`AGENTS.md` §5). `configs/data.yaml -> m4_contract.baseline` already freezes
the SAR-only scope, and the experiment config validator rejects a fifth channel
or a non-null `pretrained_weights`.

---

## 2. Dataset to tensor pipeline

```
KuroSiwoAdapter  (M4 — catalogue/export check, partition-01 filter,
       |          temporal-role resolution, 3-way validity cross-check)
       v
CanonicalSample
       |
       v
KuroSiwoSarDataset.__getitem__      (lazy, one sample at a time)
       |
       +-- stack_channels      -> image      float32 [4, H, W]
       +-- labels + validity   -> target     int64   [H, W]
       +-- validity            -> valid_mask bool    [H, W]
       |
       +-- (optional) normalisation: (image - mean) / std
       +-- (optional) joint geometric augmentation
       v
collate_samples -> DataLoader
```

M5 consumes the M4 boundary and **never re-scans the Kuro directory tree**.
That matters beyond tidiness: the adapter is where the catalogue/export
consistency check, the partition-01 filter and the three redundant validity
cross-checks live. A dataset that globbed the corpus itself would skip all of
them and would ingest label-free partition-00 tiles.

### 2.1 Tensor contract

| Payload | Shape | dtype |
|---|---|---|
| `image` | `[4, H, W]` | `float32` |
| `target` | `[H, W]` | `int64` |
| `valid_mask` | `[H, W]` | `bool` |

All locally delivered Kuro tiles are 224×224.

### 2.2 Memory

Lazy and indexed. `KuroSiwoSarDataset.__init__` holds only the adapter's
lightweight sample references; pixels are read in `__getitem__`. The 7,052
locally available training tiles are never materialised together. Two tests
pin this: one counts `load_sample` calls during construction, one asserts no
decoded array is retained on the dataset object.

---

## 3. Labels and ignore handling

| Stored value | Meaning | Trainable |
|---|---|---|
| 0 | No water | yes |
| 1 | Permanent Waters | yes |
| 2 | Floods | yes |
| 3 | nodata (in-band sentinel) | **no — ignore index** |

Semantic classes are `{0, 1, 2}`. Kuro's native values already *are* the class
indices, so the identity mapping is **asserted, not applied**: a corpus with a
different encoding fails loudly instead of being silently mis-labelled.

Validity comes from the M4 canonical mask (Kuro `MK0_MNA == 1`), not from the
label raster, so Kuro's redundant mechanisms stay independently cross-checkable
instead of being collapsed into one.

```
target[valid]   in {0, 1, 2}
target[~valid]  == 3   (ignore_index)
```

Three invariants, each tested:

- nodata does **not** become a fourth trainable class — the model emits 3
  logits, and `num_classes: 4` is rejected by the config validator;
- `ignore_index` 3 is an **input-side** concern and is deliberately a different
  number from the output nodata 255 (`configs/segmentation.yaml -> output`);
- an `ignore_index` colliding with a semantic class is rejected.

### 3.1 Invalid pixels do not contribute to the loss

Handled by `CrossEntropyLoss(ignore_index=3)`, which excludes those positions
from both the loss value and the backward pass. This is verified rather than
assumed, at three levels:

1. synthetic logits — perturbing only ignored positions does not change the
   loss, and those positions receive exactly zero gradient;
2. end-to-end on the synthetic fixture, through the real mask and target;
3. **on real Kuro tiles**, where nodata covers a non-trivial fraction of every
   tile (`test_overfit_does_not_learn_from_ignored_pixels`).

---

## 4. Normalisation

**NO FITTED STATISTICS EXIST.** M5 builds the fitting mechanism and the
artifact; it does not invent values.

### 4.1 Four kinds, three of them contractual

| Kind | Source | May normalise? |
|---|---|---|
| `reference` | Kuro's published `data_mean`/`data_std` | **No** |
| `fitted` | our parameters, training split only | Yes |
| `validation` | measured on validation data | **No** — drift reporting only |
| `development` | identity transform, no data read | **No** — not part of the contract |

`reference` values are refused as fitted parameters because their pre/post
basis, split basis and aggregation basis are all unstated upstream; adopting
them would normalise our data by an unknown recipe. `StatisticsArtifact.require_fitted()`
raises for every kind except `fitted`.

### 4.2 Fitting

`fit_training_statistics()`:

- **training split only** — raises on any sample whose split is not `train`, so
  a mistaken caller cannot leak validation or test data into the parameters;
- **respects the validity mask** — accumulates only pixels the M4 canonical
  mask marks valid. Invalid SAR pixels are stored as `0.0`, so counting them
  would drag every mean toward zero. A test asserts the masked and unmasked
  means actually differ on the fixture, so the check cannot pass vacuously;
- **streaming** — count, sum and sum-of-squares per channel, so no tile stack is
  materialised and the full training split never has to fit in RAM;
- **records** the activations used, tile count, valid-pixel count, channel order,
  representation, clip state, preprocessing state, temporal-role basis and
  aggregation basis.

Aggregation is **per valid pixel, pooled across tiles** — recorded explicitly
because a per-tile-then-average basis gives a different answer on tiles with
differing valid fractions, and that ambiguity is exactly what makes the
upstream reference values unusable.

### 4.3 Artifact layout (C5)

A sidecar **pair**, not one wrapped file:

- `statistics.json` — flat, float-only, `channel_name -> {mean, std}`.
  The layout is forced rather than chosen: `floodmap.features.pipeline` coerces
  `float()` over every value of every top-level key, so an embedded provenance
  block would raise there.
- `statistics_meta.json` — provenance. All 16 required fields from
  `configs/data.yaml -> m4_contract.statistics.provenance_required_fields` are
  answered; `StatisticsArtifact` refuses to construct without them, and
  `read()` refuses a pair with a missing half.

Fit one with:

```bash
FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo \
  python3 scripts/fit_statistics.py --output artifacts/statistics/kuro_sar_fitted
```

### 4.4 Development vs fitted

The two are kept visibly distinct. A development artifact is an **identity
transform** (mean 0, std 1) that reads no data and carries
`development_only: true`, `usable_for_reported_results: false` and
`is_part_of_frozen_contract: false`. Identity is deliberate: it leaves the
linear sigma0 values exactly as the adapter produced them, so a failing test
points at the pipeline rather than at an invented scaling.

An experiment must opt in explicitly — `required_kind: development` paired with
`allow_development_statistics: true`, or the config is rejected. Every run
record and checkpoint carries the statistics kind, so a development run cannot
be mistaken for a fitted one after the fact.

The real baseline config is therefore **loadable but not runnable**:
`statistics_path` stays `null` because C5 states plainly that a declared
artifact which does not exist is worse than none, so it is *not* pre-filled with
the path the artifact will eventually occupy. `train.resolve_statistics` raises
with the remedy instead.

---

## 5. Model

A small, conventional U-Net. `segmentation/model.py`.

- input channels 4, output logits 3;
- configurable `base_channels` and `depth`;
- double 3×3 conv + BatchNorm + ReLU, 2×2 max pool down, transposed conv up,
  skip concatenation, 1×1 conv to logits;
- **no pretrained weights** — no undeclared external training corpus in the
  provenance chain, and no ImageNet 3-channel adapter hack for 4-channel SAR;
- **no transformer, no attention**;
- deterministic initialisation from an explicit seed.

Default baseline config is `base_channels: 32, depth: 4` → 7,763,395 parameters.

`forward` rejects a spatial size not divisible by `2 ** depth`, rather than
failing with a shape mismatch deep inside the decoder.

Model construction also **restores the global RNG state** when seeded.
`nn.Conv2d.__init__` runs `kaiming_uniform_` against the global stream, so
without this, dataloader shuffling would depend on whether a model happened to
be constructed first and two same-seed runs would see their batches in
different orders.

**This is a reference model, not a selected architecture.**
`configs/segmentation.yaml` still carries `model.architecture: null` because
that choice must follow measured performance (`AGENTS.md` §5). See §9.

---

## 6. Loss

**Unweighted** `CrossEntropyLoss(ignore_index=3)`.

Class weighting is *supported* by configuration and is **never applied by
default**, and no weight vector is computed or guessed. Inverse-frequency
weights look principled and are not: on this corpus the frequencies depend on
which activations happen to be delivered locally — activation 1111004 alone is
71% of the available training tiles — so a weight fitted from them would encode
a delivery accident rather than the phenomenon.

No composite loss. Dice, Tversky, focal and Lovász all change what the
optimiser considers an improvement; adding one before a plain cross-entropy
reference exists would make any later gain unattributable. The config validator
rejects any `loss.name` other than `cross_entropy`.

A supplied weight vector of the wrong length is rejected rather than broadcast.

---

## 7. Metrics

All derived from one 3×3 confusion matrix accumulated across batches, never
averaged per batch. Per-batch IoU averaged over batches is not the dataset IoU,
and the difference is largest exactly where it matters here: a batch with few
or no flood pixels yields an unstable or undefined flood IoU that then
contaminates the average. A test pins this (`3/4`, not the mean of `1/2` and `1`).

Reported:

- per-class IoU, precision, recall, F1, support
- **flood-class IoU (class 2) — the primary metric (C6)**
- mean IoU (macro), micro IoU, macro F1, pixel accuracy
- the full confusion matrix (rows truth, columns prediction)

Ignored pixels are removed **before** accumulation, on the target. Nodata is
6.48% of Kuro label pixels and the model emits a confident prediction for every
one; counting them would inflate "No water" performance with pixels that were
never observed.

A class with no support scores `nan` and is **dropped** from macro aggregates
rather than counted as zero — scoring an absent class as 0.0 would drag mean IoU
down by a third on a tile that simply contains no permanent water, which is a
property of the tile and not of the model.

---

## 8. Training, selection and reproducibility

`run_training()` in `segmentation/train.py`.

### 8.1 Selection discipline

Best checkpoint = **highest validation flood-class IoU** (C6:
`monitor: val_iou_flood_class`, `mode: max`, `selection_split: validation`).

**The test split is never loaded by this module at all** — not read, not scored,
not used for early stopping. The strongest way to avoid optimising the test
split is to have no code path that touches it, and a test asserts the string
`"test"` does not appear in `train.py`. The config validator additionally
rejects `selection_split: test` and `validation_split: test`.

A `nan` flood IoU never wins selection, so a degenerate epoch cannot be chosen.
Validation runs under `torch.no_grad` with augmentation disabled — a flipped
validation tile would score a different, unrepeatable number each epoch.

### 8.2 Reproducibility

Seeding covers Python, NumPy and Torch, plus cuDNN determinism flags.
`torch.use_deterministic_algorithms(True)` is deliberately **not** set: it raises
on several ordinary convolution and pooling backward kernels and would make the
baseline unrunnable. The weaker guarantee is stated rather than overstated.

Every run writes `run_record.json` containing the experiment config, resolved
seed, dataset identity (corpus, split, activation ids, sample count),
statistics identity, device, availability of each backend, commit, model
parameter count, loss weighting and per-epoch metrics.

Two tests pin that identical seeds give identical loss trajectories and that a
different seed changes them.

### 8.3 Checkpoint artifact

Each checkpoint stores model state, optimizer state, epoch, validation flood
IoU, full experiment config, statistics identity, dataset identity, seed,
channel order and commit. A checkpoint holding only weights would be
unreproducible — nothing in it would say which normalisation its inputs were
scaled by, and feeding a model statistics it was not fitted with produces
plausible, wrong output. `load_checkpoint` rejects a checkpoint whose channel
order differs from the current frozen order.

Checkpoints are **never committed**. `.gitignore` excludes `runs/*`, `models/*`,
`artifacts/*`, `*.pt`, `*.pth`, `*.ckpt` and `*.safetensors`.

### 8.4 Device

Priority: explicit config → CUDA → MPS → CPU. An explicitly requested device
that is unavailable **raises** rather than falling back — a run meant for a GPU
that quietly spent hours on CPU is worse than one that refused to start.
Nothing requires a GPU; unit tests run on CPU, and CUDA/MPS paths are tested
when the hardware exists.

---

## 9. Augmentation

**Disabled for baseline 1**, and configurable.

Only exact lattice operations are eligible — flips and 90° rotations permute
pixels without resampling, so they cannot manufacture a backscatter value the
sensor never measured.

Any spatial transform is drawn **once per sample** and applied identically to
all four channels, the target and the validity mask. Augmenting channels
independently would destroy the pre/post pixel correspondence the change signal
depends on; augmenting the image without the target would silently mis-align
supervision. A test brute-force searches all 16 flip/rotation combinations and
asserts one single transform reproduces all three payloads.

Not permitted:

- **radiometric** (brightness, contrast, gamma, additive noise) — sigma0 *is*
  the measurement, and scaling it trains on a physically meaningless
  distribution. The config validator rejects `radiometric: true` rather than
  merely defaulting it off;
- **synthetic speckle** — Kuro arrives already Lee Sigma filtered
  (`configs/data.yaml` C4), so injecting speckle would move training data
  further from both the corpus and the unfiltered production path.

---

## 10. Configuration

Experiment settings live in `configs/experiments/`, never defaulted in the
training script (`AGENTS.md` §14).

| File | Purpose | Runnable |
|---|---|---|
| `m5_kuro_sar_unet.yaml` | the real baseline | **No** — blocked on fitted statistics |
| `m5_kuro_sar_unet_dev.yaml` | pipeline verification for smoke/overfit tests | Yes |

Exposed: dataset, splits, channel definition, batch size, dataloader workers,
epochs, learning rate, optimizer, seed, determinism, model parameters, loss
parameters, ignore index, metric selection, checkpoint directory and criterion,
statistics artifact, augmentation and device.

Every model is `extra="forbid"`: a mistyped key is rejected rather than silently
ignored, which is how a run ends up not doing what its config appears to say.

### 10.1 Why a separate file from `configs/segmentation.yaml`

`configs/segmentation.yaml` is part of the frozen contract and deliberately
keeps `model.architecture: null`, `model.encoder: null`, `model.candidates: []`
and every `training.*` hyperparameter `null`, because the project has not
**selected** an architecture or a hyperparameter set — it has only proposed a
first baseline to measure.

The two files answer different questions. `segmentation.yaml` says what is
**frozen**; an experiment config says what one **run** did. Writing this
experiment's values into those fields would silently promote a candidate into a
project-level decision, so M5 changes **nothing** in that file. A test asserts
it still defers every one of those decisions.

### 10.2 Entrypoints

```bash
# Inspect a config without training
python3 scripts/train_segmentation.py --config m5_kuro_sar_unet --describe

# Fit training statistics (training split only)
FLOODMAP_KURO_SIWO_ROOT=... python3 scripts/fit_statistics.py \
    --output artifacts/statistics/kuro_sar_fitted

# Bounded pipeline-verification run
FLOODMAP_KURO_SIWO_ROOT=... python3 scripts/train_segmentation.py \
    --config m5_kuro_sar_unet_dev --run-dir runs/dev
```

---

## 11. Verification performed

### 11.1 Smoke test — 4 real training tiles, 1 epoch

Chain verified: adapter → canonical sample → tensor → model → loss →
`backward()` → `optimizer.step()` → validation → checkpoint, both by hand and
through `run_training`. Gradients non-zero, parameters changed, metrics
computed with nodata excluded, checkpoint saved, reloaded and reproducing
identical logits.

### 11.2 Overfit test — 4 fixed real tiles, 60 iterations

| Check | Result |
|---|---|
| gradients non-zero | yes |
| loss decreases | **1.55335 → 0.02781** (98.2% reduction) |
| parameters change | yes |
| predictions change | yes |
| train flood IoU improves | **0.0329 → 0.5885** |
| checkpoint loads | yes, reproduces trained logits |

The subset contained all three semantic classes. **This is a pipeline
correctness result, not a performance result** — a model that fits four tiles
has memorised four tiles. Its only meaning is that the pipeline is capable of
learning, which rules out a detached graph, a mask on the wrong axis, an
all-ignored target and constant inputs.

### 11.3 Bounded statistics fit on real tiles

Over 4 real training tiles, valid pixels only:

```
mean = [0.11451, 0.03792, 0.11253, 0.03775]   # pre_vv, pre_vh, post_vv, post_vh
std  = [0.03612, 0.01507, 0.03589, 0.01547]
```

Plausible for linear sigma0 clipped at 0.15, and **not** near the −10 to −15 a
decibel product would give — an independent confirmation of the representation.
These are close to the `measured_subset_2026_10_08` values already recorded in
`configs/data.yaml` (VV 0.1180, VH 0.0373). **These are not the project's fitted
statistics**: 4 tiles from one activation, and the artifact records `n_tiles` so
they cannot be mistaken for a full-split fit.

### 11.4 No full training run was launched

Only unit tests, the tiny smoke test and the tiny overfit test were executed.
No multi-epoch run over the 7,052 training tiles, no long-lived process.

---

## 12. M4 defect found and fixed

The M4 Kuro adapter **crashed on the real delivery**.
`_read_exported_catalogue` filtered only `exported == 1`, then called
`int(row["aoiid"])`. Partition-00 (unlabelled) rows carry `aoiid = NULL` with
`exported = 1` — exactly as `configs/data.yaml -> partition_contract` records —
so discovery raised `TypeError: int() argument must be ... not 'NoneType'`.
The M4 synthetic fixtures never contained such a row and the external smoke
test was skipped, so it was never exercised.

Fix: filter `aoiid == 1` alongside `exported == 1`, which is the
labelled-partition filter the partition contract already specifies. Grouping
those rows first would also have corrupted the 3-row triplet check. The original
post-group assertion is kept as a defensive invariant.

Verified across all 5 local activations: every labelled grid count now matches
its on-disk partition-01 directory count exactly, every grid has exactly 3 rows,
and NULL-`aoiid` grid ids never overlap labelled ones.

| Activation | Split | NULL-`aoiid` grids | Labelled grids | On disk |
|---|---|---|---|---|
| 1111002 | test | 2 | 32 | 32 |
| 1111003 | validation | 0 | 19 | 19 |
| 1111004 | train | 0 | 5001 | 5001 |
| 1111005 | train | 1 | 1369 | 1369 |
| 1111006 | train | 4 | 682 | 682 |

Two regression tests were added to `tests/test_dataset_adapters.py`: one proves
NULL-`aoiid` rows are filtered, one proves the labelled-partition check was not
weakened by the filter.

---

## 13. Known limitations

1. **Validation is one activation, 19 tiles.** `configs/evaluation.yaml` already
   records `min_validation_activations_for_a_reportable_threshold: 2` against
   `validation_activations_available_locally: 1`. The checkpoint-selection
   mechanism is implemented and exercised, but **any number it produces on 19
   tiles is infrastructure evidence, not a performance claim**. One activation
   cannot separate event-specific from generalisable behaviour.

2. **No fitted statistics.** The real baseline config cannot run until they are
   fitted. This is intentional.

3. **Train/serve texture mismatch remains.** Kuro arrives Lee Sigma filtered and
   cannot be un-filtered; our production path is unfiltered (`configs/data.yaml`
   C4). Disclosed, not closed, and must be quantified by ablation before any
   transfer claim.

4. **The local delivery is a subset.** 5 of 45 catalogued activations; 3 of 27
   official training activations, with 1111004 alone contributing 71% of
   available training tiles. Geographically skewed.

5. **Spatial reprojection is implemented and shared with M2** (added
   2026-10-09). Kuro is delivered on an EPSG:3857 grid whose pixels are 10
   PROJECTED units (8.736–9.798 m true ground across the five delivered
   activations), while production inference serves genuinely 10 m Sentinel-1.
   `floodmap.preprocessing.reprojection` now warps to a per-sample 10 m UTM
   grid and is called by both paths. The real baseline config enables it; see
   §14. The previously recorded train/serve spatial mismatch is **resolved in
   code**, but no run has yet been performed with it.

6. **Determinism is partial** — cuDNN flags only, not
   `use_deterministic_algorithms`. See §8.2.

7. **Two source samples were malformed; both are now resolved** (2026-10-09).
   Of 7,103 samples, 2 contained NaN in SAR pixels that all three Kuro validity
   mechanisms declare valid. One is reinstated (its four consumed channels are
   clean; only the dropped SL2 epoch is defective, now reported in
   `dropped_source_diagnostics`); one is quarantined by an explicit versioned
   policy. **Neither validation nor test is affected.** See
   `docs/m5-source-data-quarantine.md` and §15.

---

## 14. Shared spatial reprojection

Added 2026-10-09. `src/floodmap/preprocessing/reprojection.py`.

### 14.1 Why one implementation

Training and inference resampling differently is a defect that hides itself:
both paths look individually reasonable, and every metric measured in training
would describe a grid the deployed model never sees. So there is one module,
called by both:

```
M2 production:  raster -> pipeline._reproject_to_grid -> reprojection.reproject_continuous
M5 training:    CanonicalSample -> segmentation.spatial -> reprojection.reproject_continuous
```

Two tests enforce this rather than trusting it: one asserts the training path
and a direct M2-style call produce **bit-identical** values on shared valid
pixels; another is a structural guard that neither `preprocessing/pipeline.py`
nor `segmentation/spatial.py` contains its own `rasterio.warp` call.

### 14.2 Source to target

| | Source (delivered) | Target (canonical) |
|---|---|---|
| CRS | EPSG:3857 | per-sample WGS84/UTM (EPSG:326xx / 327xx) |
| Pixel size | 10 **projected units** | 10.0 m **true ground** |
| True ground | 8.736–9.798 m measured; ~5 m near 60N | 10.0 m |
| Size | 224 × 224 | 224 × 224 (fixed) |

The UTM zone comes from the sample footprint centroid:
`floor((lon + 180) / 6) + 1`, clamped to [1, 60] so longitude 180 cannot index
a zone 61; hemisphere from the sign of latitude with the equator assigned
north. Boundary behaviour is tested explicitly, including ±180 and exact
6-degree multiples.

A **concrete** `AnalysisGrid` is built — CRS, affine transform, width, height,
resolution. Target-grid metadata alone is not reprojection, and the M4 adapter
was explicit that it only recorded metadata.

**Resampling, per C3:** SAR bilinear **in linear sigma0 power**; labels and
validity **nearest neighbour**. `reproject_continuous` refuses an averaging
method on `domain="db"` input, and `reproject_categorical` refuses an averaging
method outright — bilinear across a 0/2 label boundary yields 1, which in this
scheme is "Permanent Waters".

**Fixed tile size.** Resampling a 224 px tile to 10 m gives a footprint-sized
output that varies per activation — measured 219×220, 220×222, 201×203,
216×217, 217×218 — and variable shapes cannot be collated into a batch. 224 is
chosen as the smallest `2**depth`-divisible size that crops **no** activation
(largest natural footprint 222). Area outside the footprint is marked invalid,
never filled. A smaller value such as 208 would waste less area on activation
1111004 but would crop the other four, and cropping discards real labelled
observations whereas padding only adds pixels that `ignore_index` excludes
anyway.

### 14.3 Nodata policy

**The problem.** Kuro stores invalid SAR as `0.0`. Once that enters a bilinear
kernel it is indistinguishable from genuine near-zero backscatter — which is
precisely the open-water signal this model is trying to detect. Masking the
output afterwards is insufficient: it cleans pixels that are *themselves*
invalid while leaving valid pixels that absorbed an invalid neighbour. That was
the behaviour of the previous M2 implementation.

**The policy — `strict`, two independent mechanisms:**

1. Invalid source samples are set to NaN and declared as the warp's
   `src_nodata` **before** interpolation, so they are excluded from the kernel
   rather than averaged in as a value.
2. The validity mask is independently warped as a float coverage field using
   the **same** method as the data. A target pixel is valid only if its
   interpolated weight is 1.0 within tolerance — i.e. its entire kernel drew on
   valid source pixels.

The second mechanism is what makes the guarantee provable rather than a
property of GDAL's internal nodata handling.

`majority` (weight > 0.5) is implemented and **not used**, because selecting it
should be a recorded decision. Measured on one tile per activation, `strict`
costs only **0.0–0.8%** more valid pixels than `majority` — the conservative
choice is nearly free.

Invalid output carries −9999.0 (the frozen continuous nodata, C9), never 0.0,
so a sentinel can never be read back as zero backscatter. Tests assert that no
surviving valid pixel holds a value between the sentinel and the true source
value.

### 14.4 Alignment

All four SAR channels, the label raster and the validity mask are warped onto
the **same** concrete grid. The final mask is the **intersection** of the SAR
and label valid footprints: the two use different resampling methods, so their
footprints differ at the fringe, and intersecting is what keeps image, target
and mask describing the same pixels. Tests assert one shared validity footprint
across all four channels and the target.

### 14.5 M2 ordering defect found and fixed

`preprocess_sentinel1_pair` converted linear backscatter to decibels **before**
calling `_reproject_to_grid`, so bilinear/average resampling averaged
logarithms. That is not the mean of the underlying power and is biased low —
exactly what `configs/preprocessing.yaml` has always declared
(`sar_domain: linear_power`, `sar_resample_before_db_conversion: true`, pinned
by `test_contract_freeze.py:286`). The implementation contradicted its own
declared policy.

Fixed by resampling in linear power and converting afterwards via
`linear_to_db_after_resampling`, which delegates to the existing M3
`numerics.linear_to_db`. The ordering is recorded in the artifact's
`preprocessing_operations` (`resample_in_linear_power` before
`linear_to_db_after_resampling`), so it is auditable from the output rather
than only from a source comment.

**Magnitude, measured.** Averaging 0.01 and 1.0 linear gives 0.505 → −2.967 dB.
Averaging their decibel forms (−20, 0) gives −10 dB. A **7.03 dB** error, in
the biased-low direction — toward the water-like values the model keys on. The
regression test asserts the correct value and explicitly asserts the result is
*not* the defective one.

### 14.6 Statistics implication

`fit_training_statistics` now accepts and records the reprojection settings.
This is not bookkeeping: bilinear resampling changes the variance of the field
it is applied to, so a std fitted before resampling does not describe
post-resampling input. A test asserts the fitted std actually differs between
the two paths, so the recorded setting cannot be vacuous.

**Consequence for Phase 5:** statistics must be fitted with the same
`reprojection` block the training run will use. They remain unfitted.

---

## 15. Fitted training statistics — 2026-10-09

**The production artifact now exists.** `configs/experiments/m5_kuro_sar_unet.yaml`
references it and the config is runnable.

### 15.1 How it was produced

```bash
FLOODMAP_KURO_SIWO_ROOT=… python3 scripts/fit_statistics.py \
    --config m5_kuro_sar_unet \
    --output artifacts/statistics/kuro_sar_fitted
```

The spatial and channel pipeline comes from the **experiment config**, not from
flags. That is deliberate: bilinear resampling changes the variance of the
field it is applied to, so a std fitted on the delivered source grid does not
describe a reprojected training input, and the mismatch would be invisible.
The artifact records its own spatial settings, and a test asserts they match
the config's.

| | |
|---|---|
| Discovered (official `train`) | 7,052 |
| Quarantined | 1 (`nan_on_declared_valid_consumed_sar`) |
| **Fitted over** | **7,051** |
| Valid pixels | **279,310,909** |
| Activations | 1111004, 1111005, 1111006 (all official `train`) |
| Spatial | reprojected, 10.0 m UTM, tile 224, bilinear/nearest, nodata `strict` |
| Representation | linear sigma0, clipped 0.15, **no dB conversion** |
| Aggregation | per valid pixel, pooled across tiles |
| Kind | `fitted` |
| Runtime | 2.8 min (~43 tiles/s) |

### 15.2 Values

| Channel | mean | std |
|---|---|---|
| `pre_vv` | 0.09870434 | 0.04048222 |
| `pre_vh` | 0.02862864 | 0.01623316 |
| `post_vv` | 0.10876928 | 0.04195391 |
| `post_vh` | 0.03405827 | 0.02028676 |

### 15.3 Cross-check against the published reference

C5 permits a **tolerance-banded cross-check** over the full training split and
explicitly forbids a per-raster equality gate. Against Kuro's published
`data_mean`/`data_std` (VV 0.0953 / 0.0427, VH 0.0264 / 0.0215):

| Channel | mean Δ | std Δ |
|---|---|---|
| `pre_vv` | **+3.6%** | −5.2% |
| `pre_vh` | +8.4% | −24.5% |
| `post_vv` | +14.1% | −1.7% |
| `post_vh` | +29.0% | −5.6% |

`pre_vv` at +3.6% is the closest agreement observed so far — the earlier
150-tile post-event-only subset recorded in `configs/data.yaml` was +24%. The
pre-event channels agree more closely than the post-event ones.

**This is a consistency signal, not a validation.** The reference's pre/post
basis, split basis and aggregation basis are all unstated upstream, so the
bands cannot be interpreted as error. A test asserts the reference values were
not adopted as our parameters. The `pre_vh` std at −24.5% is the largest
deviation and is the one worth revisiting if transfer behaviour looks wrong.

### 15.4 Why the artifact is not committed

`artifacts/` is git-ignored: the artifact is derived from a corpus this
repository must not redistribute (C11). On a fresh checkout the path will not
resolve and the run **refuses to start** with the remedy — better than silently
normalising with something else. A test asserts that a missing artifact raises
rather than falling back, and the artifact-dependent tests are skip-gated on
its presence.

### 15.5 What the frozen contract still says, and why

`configs/segmentation.yaml -> features.normalisation` still declares
`fitted_statistics_exist: false` and `statistics_path: null`, and
`configs/data.yaml -> m4_contract.statistics.fitted` still declares
`exists: false`. **These were deliberately not changed.**

The reasoning: those files are the committed, frozen record. The artifact is a
local build product that is *not* in the repository, so from the repository's
own point of view no fitted artifact exists. Declaring one in a committed
config while the file is absent from the checkout would create exactly the
"declared artifact that does not exist" failure C5 was written to prevent.

The experiment config — the file that records what one *run* did — is the right
place for the path, and that is where it is. **Flagged for review:** if the
project later decides the frozen block should track the existence of local
fitted artifacts, that is a contract amendment and should be made deliberately,
not as a side effect of a fitting run.

---

## 16. Next step

Pipeline state is no longer blocking. Remaining before a **reportable** result:

1. validation is one activation / 19 tiles, below the minimum of 2 recorded in
   `configs/evaluation.yaml` — any flood IoU produced now is pipeline evidence,
   not a measurement;
2. the speckle train/serve mismatch must be quantified by ablation before any
   transfer claim.

The baseline run itself can now be launched with
`scripts/train_segmentation.py --config m5_kuro_sar_unet`.
