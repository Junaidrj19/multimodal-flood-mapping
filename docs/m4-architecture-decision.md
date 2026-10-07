# M4 Architecture Decision — Scientific Design

Multimodal AI Hackathon 2026 — Track B.

**Status: DECIDED, NOT IMPLEMENTED.** This record fixes the segmentation
architecture *before* M4 begins, so the design follows the evidence rather than
being rationalised after the code exists. No model has been trained and no
performance has been measured.

**Decision date basis:** the dataset audit in `docs/dataset-registry.md` §1.5,
conducted against both corpora's own repositories, LICENSE files and papers.

---

## 1. The question

Kuro Siwo and Sen1Floods11 are SAR-oriented corpora. Our production pipeline
(M1–M3) can generate Sentinel-1, Sentinel-2 and DEM features. So:

> What is the scientifically defensible way to use multimodal production
> features during segmentation, given what the permitted training labels
> actually supervise?

The answer is constrained by facts, not preference. The governing ones:

| Fact | Source |
|---|---|
| Neither corpus labels debris or sediment | both class lists, §1.5.6 |
| Kuro Siwo separates permanent water from flood water; Sen1Floods11 does not | §1.5.1, §1.5.2 |
| Kuro Siwo has **no optical data**; Sen1Floods11 has **L1C TOA** optical | §1.5.1, §1.5.2 |
| Sen1Floods11 has **no pre-event image** | §1.5.2 |
| Kuro Siwo is **linear σ⁰**; Sen1Floods11 is **dB** | §1.5.3 |
| Neither corpus bundles Copernicus WorldDEM-30 | §1.5.3 |
| Neither is documented as containing Himalayan terrain | §5.3 |

---

## 2. Options evaluated

Scored against the criteria the task requires: modality compatibility, label
compatibility, spatial alignment, temporal alignment, domain shift, leakage,
scientific defensibility, challenge compliance, explainability.

### Option A — SAR-only segmentation

Train on permitted SAR labels, apply to production Sentinel-1.

| Criterion | Assessment |
|---|---|
| Modality compatibility | **Strong.** SAR is the common denominator of both corpora and of our pipeline. |
| Label compatibility | **Strong.** The labels *are* SAR labels. |
| Temporal alignment | **Good with Kuro Siwo** (pre/post triplet matches our pre/post pair in kind). Sen1Floods11 contributes no change signal. |
| Domain shift | Representation/speckle mismatch is real but **bounded and correctable** by an adapter. |
| Leakage | Low risk. |
| Defensibility | **High.** Nothing is claimed that labels do not support. |
| Compliance | Full. |
| Explainability | High — one modality, one learned signal. |
| **Weakness** | **Discards Sentinel-2 and DEM entirely**, which the challenge explicitly permits and which carry genuine independent information. |

### Option B — Fully multimodal learned model (S1 + S2 + DEM channels)

| Criterion | Assessment |
|---|---|
| Modality compatibility | **Fails.** Kuro Siwo — the *required* corpus — has no optical data at all. Only Sen1Floods11 could supervise optical channels. |
| Label compatibility | Weak for optical. Its only ground-truth optical portion is **446 hand-labelled chips**. The 4,385 `S2IndexLabelWeak` chips are *derived from a spectral-index threshold*, so training on them teaches the model to reproduce that rule and embeds a decision threshold in the weights — which this project deliberately keeps explicit and configurable. |
| Spatial alignment | Acceptable (both 10 m, co-registered within each corpus). |
| Temporal alignment | **Problem.** Sen1Floods11's S1 and S2 dates differ by up to 2 days, and it has no pre-event image, so optical *change* cannot be supervised at all. |
| Domain shift | **Severe and silent.** Training optical on **L1C TOA** and serving **L2A BOA** shifts NDWI/MNDWI systematically, because atmospheric path radiance affects visible bands far more than NIR/SWIR. |
| DEM | Kuro Siwo bundles **SRTM 1Sec**; production is **WorldDEM-30**. Different product, different epoch, different error structure. |
| Operational | **Fails hardest here.** A model with required optical channels cannot run when there is no usable post-event optical scene — the expected case for a monsoon Himalayan event (`docs/scientific-assumptions.md` §8). Imputing the missing channels would violate "not observed ≠ not flooded". |
| **Verdict** | **Rejected as the baseline.** |

### Option C — SAR-learned segmentation + multimodal deterministic evidence

Sentinel-1 carries the learned flood signal. Sentinel-2 and DEM contribute
*deterministic, registry-defined* evidence layers downstream of the model,
never as unsupervised input channels.

| Criterion | Assessment |
|---|---|
| Modality compatibility | **Strong.** Learned part uses only what is supervised; permitted-but-unsupervised modalities are used in a way that needs no labels. |
| Label compatibility | **Strong** — no modality is fed to the model without supervision. |
| Domain shift | Bounded on the learned path; the deterministic path has **no training distribution to shift from**. |
| Leakage | Low. Evidence layers are computed, not fitted. |
| Operational robustness | **Strong.** Cloud-obscured optical degrades the *confirmation* to "unconfirmed" rather than breaking inference. |
| Defensibility | **Highest.** Every claim traces to either a supervised model output or an explicit, cited formula. |
| Explainability | **Highest.** "SAR model says flood; optical agrees; terrain is plausible" is auditable layer by layer — exactly the `AGENTS.md` §22 evidence chain. |
| Compliance | Full. Uses the permitted Sentinel-2 and DEM without over-claiming. |

### Option D — Transfer / fine-tuning

Not a competing architecture but a **training schedule** usable inside A or C.

- Sen1Floods11 (4,831 chips, global) → Kuro Siwo (202,470 samples, 43 events).
- Legitimate **only after** SAR representations are harmonised: pre-training in
  dB and fine-tuning in linear σ⁰ would present the same physics on two scales.
- Upstream test activations must be excluded from both stages (§2.4 of the
  evaluation protocol).
- Pre-trained weights must be **disclosed** (`model.pretrained_weights`), since
  the encoder may have seen related imagery.

---

## 3. Decision

> **Adopt Option C, with Option A as its learned core and Option D as an
> optional, declared training schedule.**

```text
  PRODUCTION FEATURES (M3 registry)
  ├─ Sentinel-1: level, log-ratio change, co/cross ratio ──► LEARNED SEGMENTATION
  │                                                              (supervised)
  │                                                                  │
  │                                                      flood-water probability
  │                                                          + class mask
  │                                                                  │
  ├─ Sentinel-2: NDWI/MNDWI/NDVI pre, post, change ──► OPTICAL CONFIRMATION
  │                                                     (deterministic, cloud-aware)
  │                                                                  │
  └─ DEM: elevation, slope ─────────────────────────► TERRAIN PLAUSIBILITY
                                                        (deterministic)
                                                                     │
                                                                     ▼
                                                      EVIDENCE-TIERED OUTPUT
                                                   (model / confirmed / plausible)
```

**Learned core (Option A):** SAR-only segmentation.

- **Primary corpus:** Kuro Siwo, using its native 3-class scheme
  `{No water, Permanent Waters, Floods}`. Keeping all three classes is a
  deliberate scientific choice, not convenience: the permanent-water class is
  the only defence against reporting the Trishuli river itself as flood on every
  run (`docs/scientific-assumptions.md` §7). Collapsing to binary would discard
  the single most valuable property of the required corpus.
- **Input channels:** the M3 Sentinel-1 features — per-polarisation pre and
  post levels, `s1_<pol>_change_db`, and the co/cross ratio where dual
  polarisation exists. Kuro Siwo's pre/post triplet supervises change features
  directly; its *second* pre-image has no production counterpart and must be
  dropped by the adapter rather than faked.
- **Sen1Floods11:** optional, under a **separate binary head or a separate
  pre-training stage** — never unioned into the 3-class head (evaluation
  protocol §2.5). It supplies no change features.

**Why not simply add S2/DEM channels anyway:** an unsupervised input channel
does not become informative by being present. With no optical labels in the
required corpus, the model would either ignore those channels or fit noise, and
either way the resulting number would be uninterpretable — while the operational
cost (failing when cloud removes optical) would be real.

---

## 4. Sentinel-2 role

**Not discarded, and not fed to the model.** Sentinel-2 becomes a deterministic
confirmation and independent-change layer, computed from the M3 registry
features that already exist.

| Use | How | Status |
|---|---|---|
| **Optical confirmation** | Per-pixel agreement between the SAR prediction and water-sensitive index evidence (`s2_ndwi_change`, `s2_mndwi_change`). | Deterministic, cited formulas |
| **Independent change indicator** | `s2_mndwi_change` and `s2_nir_change` are computed from a different physical mechanism than SAR backscatter, so agreement is genuinely corroborative rather than circular. | Deterministic |
| **Permanent-water cross-check** | Pre-event NDWI/MNDWI establishes the baseline water extent independently of the model's permanent-water class. | Deterministic |
| **Confidence tiering** | Raises reported evidence strength where optical agrees; **never** creates a detection where SAR did not. | Deterministic |
| **Sediment/turbidity indication** | Visible-band brightening over water is consistent with sediment load. **Unvalidated** — see §6. | Candidate only |

**Rules:**

1. **Sentinel-2 may never create a positive detection.** It can corroborate or
   fail to corroborate the SAR prediction. Asymmetric by design: the learned
   signal is the one with a measured error rate.
2. **Cloud-aware by construction.** M3 already propagates per-feature validity.
   Where optical is invalid, the output is `UNCONFIRMED` — explicitly distinct
   from `CONTRADICTED`. Collapsing those two would make cloud look like
   disagreement.
3. **No threshold is set in M3 or here.** Any agreement threshold is an
   evaluation-milestone parameter, selected on validation and recorded.
4. **No optical index is reported as a water mask.** An index is evidence
   (`docs/data-contract.md` §0.6.4).

**If a supervised optical path is attempted later** it must be a *separately
reported experiment*, trained only on Sen1Floods11's 446 hand-labelled chips,
with the **L1C-TOA vs L2A-BOA** domain shift stated and ideally quantified. It
must not be merged into the primary result.

---

## 5. DEM role

**Not supervised by flood labels, and not pretended to be.** Three defensible
roles, all deterministic:

| Role | Justification | Milestone |
|---|---|---|
| **Terrain plausibility** | Standing water is physically implausible on steep faces. Slope constrains where an inundation prediction is reasonable. | M4 post-processing |
| **Error stratification** | SAR geometry distortion — shadow, layover, foreshortening — is a function of local slope and aspect. Radar shadow in steep terrain is a *systematic* false-positive mechanism for water detection. Stratifying metrics by slope is how that failure mode becomes visible instead of averaged away. | Evaluation (`docs/evaluation-protocol.md` §6.1) |
| **Downstream flood-path / connectivity context** | Terrain-based inference for the hydrology bonus and settlement exposure. | M5+ (`architecture.md` §10) |

**Rules:**

1. **Not a model input channel in the baseline.** Kuro Siwo bundles SRTM 1Sec
   while production uses WorldDEM-30 — a different product with a different
   epoch and error structure. A model that learned SRTM-specific terrain
   statistics would be served something else at inference. If DEM channels are
   tried, it must be as a **declared ablation** with the DEM mismatch stated.
2. **Terrain never creates or deletes a detection silently.** Slope-based
   implausibility *lowers reported confidence and is recorded*; it does not mask
   pixels out invisibly. Suppressing a true detection on a steep slope would be
   as harmful as a false positive.
3. **The DEM is a pre-event prior, not an observation.** An avalanche and debris
   flow changes real terrain (`docs/scientific-assumptions.md` §10.1). Anything
   terrain-derived describes *pre-event* routing.
4. **No hydrodynamic claim.** Terrain-based inference only (`AGENTS.md` §10).

---

## 6. Debris — feasibility and decision

**Finding: supervised debris/sediment segmentation is not possible within the
challenge rules.**

- Kuro Siwo: `{0: No water, 1: Permanent Waters, 2: Floods}` (validity separate)
- Sen1Floods11: `{-1: No Data, 0: Not Water, 1: Water}`

Both lists are exhaustive; neither contains debris, sediment or mud. The
permitted training list is **closed** (`AGENTS.md` §3), so no permitted route to
debris supervision exists.

### Options considered

| Option | Verdict |
|---|---|
| Separately learned debris class | **Impossible.** No labels in any permitted corpus. |
| Rule-based post-segmentation indicator | **Permitted, with strict framing** — see below. |
| Infrastructure-impact category | **Permitted and useful.** Debris-affected exposure can be reported as a *qualitative* category in the impact layer without a pixel-level debris class. |
| Narrow the product claim | **Adopted as the primary decision.** |

### Decision

1. **The primary product claim narrows to flood-water mapping.** The segmented,
   quantitatively evaluated output is **flood water** (with permanent water
   distinguished). `PRD.md` FR-06 asks for *"flood/debris classes supported by
   the training labels"* — and the labels support water only. README, PRD
   framing and the dashboard must say *flood water*, not *flood/debris*,
   wherever they describe the model output.
2. **A debris/sediment indicator may be offered as a clearly-separated,
   explicitly unvalidated candidate layer**, derived from registry features
   (for example persistent post-event backscatter change that is *not*
   water-like, combined with vegetation loss and terrain context). It must be:
   - labelled **"candidate debris/sediment — unvalidated, no ground truth"**;
   - visually and structurally separate from the segmentation output;
   - **excluded from every segmentation metric**, because there is nothing to
     measure it against;
   - described by its rule, not by an accuracy figure.
3. **No debris accuracy, IoU, precision or recall may ever be reported.**
   Reporting a metric would imply a reference that does not exist.
4. **Do not describe the system as performing debris segmentation.** It does
   not. Saying so would be the exact overreach `docs/scientific-assumptions.md`
   exists to prevent.

> An ice-and-rock avalanche footprint may also be unlike anything in either
> corpus even for the *water* class, which is an additional reason the Himalayan
> performance gap is expected to be large (§5.3 of the registry).

---

## 7. Required adapter — the training/production boundary

`docs/data-contract.md` §0.6.8 requires training corpora to reach the feature
contract through an adapter rather than being special-cased inside the
production feature generator. The audit fixes what that adapter must do.

| Adapter obligation | Kuro Siwo | Sen1Floods11 |
|---|---|---|
| Declare SAR representation | `linear` σ⁰ — **VERIFIED** by published statistics `mean[VV,VH]=[0.0953, 0.0264]`, clip at `0.15` | `decibel` (stated) |
| Bind band roles | `vv`→VV, `vh`→VH | `vv`→band 0, `vh`→band 1 |
| Reproject to a projected metre grid | from EPSG:3857 | from EPSG:4326 |
| Emit per-band valid masks | from the **separate** `valid_mask` raster (`0=invalid`) | from the **in-band** label value `-1` |
| Temporal mapping | use **one** pre-image + post; **drop the second pre-image** | **no change features possible** |
| Speckle filtering | already Lee Sigma 7×7 | **none applied** |
| Label mapping | native 3-class | binary; separate head only |
| Exclude upstream test activations | required | required |

**Two adapter rules that prevent silent corruption:**

1. **The representation declaration is per corpus.** M3's
   `sentinel1.backscatter_representation` gate exists for exactly this: Kuro
   Siwo and Sen1Floods11 must be read under *different* settings, and the gate
   makes that explicit instead of silent.
2. **The speckle-filter mismatch must be recorded as a known domain shift.**
   Kuro Siwo is Lee-Sigma filtered, Sen1Floods11 is unfiltered, and our M2 does
   not filter. Filtered and unfiltered backscatter have different texture
   statistics, which a CNN will notice. Options are to filter consistently
   everywhere or to report the mismatch — but not to ignore it.

---

## 7a. Canonical SAR representation — making training and inference the same thing

The requirement is that these two paths produce features with identical
semantics:

```text
Kuro Siwo  -> adapter -> M2/M3 -> M4 training
Trishuli   ->            M2/M3 -> M4 inference
```

They can be made genuinely compatible, but not by the adapter alone. Seven
properties must be pinned, and two of them cannot be fully reconciled and must
instead be disclosed.

### 7a.1 The canonical representation

**Decided: decibels, at 10 m, on a projected metre CRS, 1 pre + 1 post.**

| Property | Canonical value | Kuro Siwo | Trishuli (M2) |
|---|---|---|---|
| Backscatter scale | **dB** | linear σ⁰ → convert | `radiometric_calibration: linear_to_db` |
| Polarisations | VV, VH (as bound roles) | VV, VH | TODO(verify) from product |
| Resolution | **10 m true ground** | 10 units in EPSG:3857 — see §7a.3 | 10 m in a UTM zone |
| CRS family | **projected, metre units** | EPSG:3857 → reproject | EPSG:326xx |
| Temporal depth | **1 pre + 1 post** | 2 pre + 1 post → drop one | 1 pre + 1 post |
| Speckle filter | **Lee Sigma 7×7** | already applied | **must be added to M2** |
| Upper clip | **0.15 linear ≡ −8.24 dB** | applied | must be applied identically |

**Why dB rather than linear**, given Kuro Siwo ships linear:

1. The primary change feature is already decibel-valued in both branches. In dB
   the log-ratio *is* the plain difference, so `s1_<pol>_change_db` needs no
   special case.
2. Linear σ⁰ is strongly right-skewed; dB is far closer to symmetric. Z-score
   normalisation on linear values is dominated by the tail, which is why the
   upstream pipeline needs a hard clip at 0.15 to be trainable at all.
3. Sen1Floods11 is already dB, so choosing dB costs nothing on that branch and
   avoids an unnecessary inverse transform.
4. `linear_to_db` is already implemented in `floodmap.features.numerics` with
   the correct strictly-positive domain guard, and M3's per-corpus
   `backscatter_representation` gate already exists to route it.

The conversion is monotone and lossless for strictly positive σ⁰. Zero and
negative values are **invalid**, not floored — the existing guard handles that,
and it matters because flooring would manufacture a water-like dark value.

### 7a.2 The clip must move with the representation

Kuro Siwo's published pipeline clips input backscatter at **0.15 linear**, then
z-score normalises. If we train on that distribution we must reproduce the clip
at inference, in the canonical representation:

```text
10 * log10(0.15) = -8.24 dB
```

An upper clip in linear power is an upper clip in dB, so the transform commutes
and the bound is exact. Two consequences:

- The clip is a **declared representation parameter**, not a feature-level
  clamp. M3's `numerics.clip_to_valid_range` is deliberately off and requires a
  written justification; matching a training distribution is a legitimate
  justification, but it must be recorded as such with the −8.24 dB value and
  applied to **both** paths or **neither**.
- Clipping at −8.24 dB discards the bright tail. For flood detection that is
  mostly harmless, because the signal of interest is *dark*. It does, however,
  flatten the bright double-bounce response of flooded vegetation and urban
  areas — exactly the cases the Kuro Siwo authors list as hard. Worth an
  ablation rather than acceptance by default.

### 7a.3 Flagged risk: EPSG:3857 pixel spacing is not true ground metres

Kuro Siwo's terrain correction sets `pixelSpacingInMeter 10.0` with a target CRS
of **EPSG:3857 (Web Mercator)**. Web Mercator's scale factor is `1/cos(lat)`, so
a fixed 10-unit pixel covers *less* ground as latitude increases:

| Latitude | `1/cos(lat)` | 10 projected units ≈ |
|---|---|---|
| 0° (equator) | 1.000 | 10.00 m ground |
| 28.06° (our AOI) | 1.133 | **8.82 m ground** |
| 45° | 1.414 | 7.07 m ground |
| ~60° (their Sweden event) | 2.000 | **5.00 m ground** |

If the spacing is nominal projected units, the corpus carries a **~2× variation
in true ground sampling distance** between its tropical and northern events, and
none of it is 10 m except at the equator. A model trained across that corpus has
seen a mixture of effective resolutions, and our 10 m UTM inference grid matches
none of them exactly.

**This is a flagged risk, not an established fact.** It depends on whether SNAP
interpreted that parameter as projected units or compensated for local scale.
**REQUIRED before training:** read the transform and centre latitude from a
delivered Kuro Siwo raster and compute the implied ground spacing. If the
variation is real, the options are to resample the corpus to constant true
ground spacing, to restrict training to a latitude band near the AOI, or to
disclose it as an irreducible domain shift. Guessing which is unnecessary — one
raster settles it.

### 7a.4 Speckle filtering must be added to M2, or deliberately refused

Kuro Siwo applies **Lee Sigma, 7×7 window, 3×3 target, sigma 0.9**.
Sen1Floods11 applies none. Our M2 applies none.

A CNN reads texture, and speckle is texture. Training on filtered imagery and
serving unfiltered imagery is a silent domain shift of exactly the kind that
produces good validation numbers and poor field performance. Three options, in
order of preference:

1. **Implement Lee Sigma in M2** with the same parameters, declared as an
   explicit M2 operation and recorded in provenance. Matches the primary
   corpus. Cost: a real new preprocessing step with its own correctness burden.
2. **Filter neither** — strip the filter from the training path by using the
   unfiltered SLC-derived product if one is available, or accept Sen1Floods11 as
   the primary corpus instead. Cost: discards Kuro Siwo's label quality and its
   permanent-water class.
3. **Disclose the mismatch** and quantify it with an ablation. Cost: a known
   uncorrected domain shift.

Option 1 is recommended. Option 3 is acceptable only if the ablation is actually
run and reported.

### 7a.5 Temporal depth: parity over richness

Kuro Siwo offers two pre-event images; M1 selects one. The baseline uses
**1 pre + 1 post** on both paths, because train/serve parity is worth more than
the extra pre-image:

- A 2-pre model would require M1 to find *two* same-relative-orbit pre-event
  acquisitions, roughly doubling the availability constraint that
  `on_no_same_track_pair: fail` already enforces. For an arbitrary
  judge-selected date that is a materially higher chance of outright failure.
- The adapter therefore **drops** Kuro Siwo's second pre-image rather than
  synthesising a counterpart at inference. Dropping real data is the honest
  direction; fabricating a second pre-image would not be.
- A 2-pre variant remains available as a **declared experiment** with its own
  stricter M1 requirement, reported separately.

### 7a.6 What cannot be reconciled

Two residual mismatches have no clean fix and must be disclosed rather than
engineered away:

| Mismatch | Why it cannot be fixed | Disclosure |
|---|---|---|
| **Terrain-correction DEM** | Kuro Siwo used SRTM 1Sec; we use Copernicus WorldDEM-30. Re-correcting their imagery would mean reprocessing the corpus from L1, which is outside scope and would void their labels' geolocation. | Residual geolocation differences of up to a pixel in steep terrain, concentrated exactly where our AOI is steepest. |
| **Label semantics** | The `Floods` / `Permanent Waters` boundary rests on unpublished photointerpretation keys (§9.2). | Our class semantics inherit a judgement we cannot reconstruct. |

### 7a.7 The compatibility contract, as a checklist

M4 may begin training only once all of these are true and recorded:

1. Canonical representation is **dB**, declared per corpus in
   `configs/features.yaml → sentinel1.backscatter_representation`.
2. Both paths resolve the same M3 feature registry with the same
   `feature_set_version`.
3. The 0.15 linear / −8.24 dB clip is applied to **both** paths or neither, with
   justification recorded.
4. Speckle filtering is identical on both paths, or the mismatch is disclosed
   and ablated.
5. Kuro Siwo's true ground spacing has been measured from a delivered raster
   (§7a.3) and either matched or disclosed.
6. Both paths are on a projected metre CRS at the same ground spacing.
7. Temporal depth is 1 pre + 1 post on both paths.
8. The adapter has verified a delivered Kuro Siwo raster against the published
   statistics `mean[VV,VH] = [0.0953, 0.0264]`, `std = [0.0427, 0.0215]` before
   any conversion. A distribution that does not match means the product is not
   what this contract assumes.

> **The honest summary.** Items 1–3 and 6–8 are mechanical and the existing M3
> gate already supports them. Item 4 needs real work in M2. Item 5 needs one
> measurement. The two entries in §7a.6 are permanent, and they bound how well
> any reported number can be expected to transfer.

---

## 8. Exact M4 requirements

What M4 must do, derived from the above. **M4 is not started.**

**Must:**

1. Implement the dataset adapter per §7, outside the production feature
   generator, with per-corpus representation and band-role declarations.
2. Train a **SAR-only** segmentation model on Kuro Siwo's native 3-class scheme,
   consuming M3 Sentinel-1 features via the registry.
3. Honour upstream splits; exclude Kuro Siwo's test activations from training
   and validation.
4. Exclude invalid pixels from loss and all metrics, reading invalidity from
   Kuro Siwo's **separate** `valid_mask` raster and from Sen1Floods11's
   **in-band** `-1`. The two forms are not interchangeable.
5. Keep the decision threshold outside the model, selected on validation.
6. Emit a `segmentation_prediction` artifact with `ArtifactProvenance` tracing
   to the M3 feature set, its registry, and onward to the M1 manifest.
7. Record `dataset · split · features · model · hyperparameters · seed ·
   metrics · checkpoint`, plus `pretrained_weights` disclosure.
8. Report per-class metrics, confusion matrix, and the in-domain vs
   Himalaya-proxy gap.

**Must not:**

9. Feed Sentinel-2 or DEM as model input channels in the baseline.
10. Union the two corpora under one head.
11. Claim or evaluate a debris class.
12. Read EMSR927 before the freeze gate (`docs/evaluation-protocol.md` §7.3).
13. Fit normalisation statistics on anything but the training split.

**Deferred to M5+:** optical confirmation layer, terrain plausibility layer,
evidence tiering, infrastructure exposure, connectivity, hydrology bonus.

---

## 9. Open scientific blockers

Carried from `docs/dataset-registry.md` §5.2. The two that most affect M4:

1. **RESOLVED — Kuro Siwo is linear σ⁰.** Four independent lines of evidence:
   the SNAP graph (`outputImageScaleInDb=false`, no `LinearToFromdB`); published
   statistics `data_mean[VV,VH] = [0.0953, 0.0264]` and
   `data_std = [0.0427, 0.0215]`, which are linear-power magnitudes rather than
   decibels; the Supplemental Material's *"clipped at a max value of 0.15 and
   normalized to 0 mean and 1 standard deviation"* together with a loader clamp
   of `min=0.0` that would delete dB data; and a default `scale_input` of plain
   z-score `normalize`, with the optional path applying `torch.log` **to** the
   stored values. The published statistics are the adapter's cross-check on a
   delivered raster.
2. **PARTIALLY RESOLVED — the flood definition is unpublished, not merely
   unread.** The Supplemental Material §4 was located and read. It documents the
   *process* — three categories, extracted photointerpretation keys, 1:1000
   annotation scale, five SAR experts with cross-checking under a senior
   scientist — but it does **not publish the keys**, so the operational
   `Floods`-vs-`Permanent Waters` boundary is an undocumented expert judgement.
   Also established: **no external water layer** (JRC/Pekel or similar) was used
   to define "permanent"; the authors raise that only as future work and warn a
   static layer *"could potentially impute noise"*. Whether `Floods` includes wet
   soil or partially submerged vegetation is **unstated**, and sediment-laden or
   muddy water is **never mentioned**. No inter-annotator agreement or label-noise
   figure is reported; the published 51% / 48% IoU figures are agreement with
   CEMS, which the authors treat as the weaker product.

   This is now a **permanent limitation to disclose**, not a task to complete:
   our `Floods` semantics inherit a judgement we cannot fully reconstruct.

Plus: both licences unresolved; Sen1Floods11 weak-label encodings and test split
unknown; no Himalayan labelled data in the permitted set; the Kuro Siwo "Nepal"
AOI geometry unverified.
