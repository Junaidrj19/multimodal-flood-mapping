# M4 Architecture Decision — Scientific Design

Multimodal AI Hackathon 2026 — Track B.

**Status: DECIDED, NOT IMPLEMENTED. CONTRACT FROZEN 2026-10-08 (§10).** This
record fixes the segmentation architecture *before* M4 begins, so the design
follows the evidence rather than being rationalised after the code exists. No
model has been trained and no performance has been measured.

**§10 is the authoritative ledger.** Where an earlier section in this document
described a decision as open, §10 supersedes it, and the superseding is marked
inline. Two decisions freeze a disclosed *limitation* rather than a fix (C4
speckle parity, C6 threshold value); they are labelled as such and must not be
read as resolved science.

**Decision date basis:** the dataset audit in `docs/dataset-registry.md` §1.5,
conducted against both corpora's own repositories, LICENSE files and papers,
plus the delivered-data inspection in §1.7.

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
| Declare SAR representation | `linear` σ⁰ — **VERIFIED(delivered data)**: float32, non-negative, nodata 0.0, max 901.92 | `decibel` (stated upstream; no local delivery) |
| Bind band roles | `vv`→VV, `vh`→VH | `vv`→band 0, `vh`→band 1 |
| Reproject to the canonical UTM grid (C1, C2) | from EPSG:3857, per-activation measured spacing 8.736–9.798 m | from EPSG:4326, per-chip spacing ~0.000090° |
| Emit per-band valid masks | from **three redundant** mechanisms; canonical `MK0_MNA == 1`, cross-checked against `MK0_MLU == 3` and `SAR == 0.0` | from the **single in-band** label value `-1` |
| Temporal mapping | use `SL1` pre + `MS1` post; **record `SL2` as dropped** | **post-only; no pre-event image may be synthesised** |
| Speckle filtering | already Lee Sigma 7×7 upstream; we add nothing (C4) | none upstream; we add nothing (C4) |
| Label mapping | native 3-class `{0,1,2}`, stored `{0,1,2,3}`, nodata `3` | binary `{0,1}`, nodata `-1`; separate head only |
| Exclude upstream test activations | required | required |
| Redistribute from this repository | **never** (C11) | **never** (C11) |

**Two adapter rules that prevent silent corruption:**

1. **The representation declaration is per corpus.** M3's
   `sentinel1.backscatter_representation` gate exists for exactly this: Kuro
   Siwo and Sen1Floods11 must be read under *different* settings, and the gate
   makes that explicit instead of silent.
2. **The speckle-filter mismatch is a disclosed domain shift, not a task.**
   Kuro Siwo is Lee-Sigma filtered, Sen1Floods11 is unfiltered, and our M2
   filters nothing. Filtered and unfiltered backscatter have different texture
   statistics, which a CNN will notice. Because the upstream filter cannot be
   undone, "filter consistently everywhere" is unavailable; C4 therefore freezes
   the uniform no-filter policy and requires the mismatch to be reported and
   ablated. Ignoring it remains forbidden.

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

**Decided: decibels, at 10 m true ground, on a projected metre CRS (UTM),
1 pre + 1 post.** Frozen as C1, C2, C4 and C10 in §10.

| Property | Canonical value | Kuro Siwo | Trishuli (M2) |
|---|---|---|---|
| Backscatter scale | **dB** | linear σ⁰ → convert | `radiometric_calibration: linear_to_db` |
| Polarisations | VV, VH (as bound roles) | VV, VH | TODO(verify) from product |
| Resolution | **10 m true ground** (C2) | 10 units in EPSG:3857, 8.736–9.798 m measured → resample per activation | 10 m in a UTM zone |
| CRS family | **WGS 84 / UTM, zone per sample** (C1) | EPSG:3857 → reproject | EPSG:326xx |
| Temporal depth | **1 pre + 1 post** | 2 pre + 1 post → drop SL2 | 1 pre + 1 post |
| Speckle filter | **none applied by us** (C4) | already Lee Sigma 7×7 upstream — cannot be undone | none |
| Upper clip | **0.15 linear, applied before dB** (C10) | applied at the adapter | must be applied identically |

> **Superseded.** An earlier revision of this table read
> `Speckle filter | Lee Sigma 7×7 | already applied | must be added to M2`.
> That presumed parity was achievable. It is not: Kuro Siwo arrives already
> filtered, so adding Lee Sigma to M2 would double-filter the training corpus
> while leaving inference single-filtered. C4 freezes the uniform policy —
> apply nothing, disclose the residual mismatch, require an ablation.
>
> The same revision recorded the clip as `−8.24 dB`. C10 applies the clip in
> the **linear** domain at `0.15`, because that is the exact published bound;
> `−8.2391 dB` is recorded as its equivalent and is not the value applied.

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
at inference. **Frozen as C10: the clip is applied in the LINEAR domain, at the
adapter boundary, before the dB conversion.**

```text
10 * log10(0.15) = -8.2391 dB      # recorded, NOT the value applied
```

An upper clip in linear power is an upper clip in dB, so the transform commutes
and the same pixels are clipped either way — but not to the same *bound*. `0.15`
is the exact published value; its dB form is a rounded transform of it. Clipping
linear-then-converting applies the exact bound, whereas clipping in dB would
make the training distribution depend on how many decimals we wrote down. Hence
the linear domain, and hence the dB figure is recorded rather than used.

Three consequences:

- The clip is a **declared representation parameter at the adapter boundary**,
  not a feature-level clamp. M3's `numerics.clip_to_valid_range` stays off, and
  C10 records `is_m3_feature_level_clamp: false` so the two cannot be confused.
- It applies to **both** paths or **neither**. The production-side site is
  named (after radiometric calibration, before linear-to-dB) and its status is
  recorded as `PENDING(M2 wiring)` — the contract is fixed, the step is not yet
  implemented.
- Clipping at this bound discards the bright tail. For flood detection that is
  mostly harmless, because the signal of interest is *dark*. It does, however,
  flatten the bright double-bounce response of flooded vegetation and urban
  areas — exactly the cases the Kuro Siwo authors list as hard. Worth an
  ablation rather than acceptance by default (disclosure E5).

### 7a.3 EPSG:3857 pixel spacing is not true ground metres — MEASURED

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

**MEASURED AND CONFIRMED — 2026-10-08. This is no longer a flagged risk.**
A local delivery was inspected read-only (`docs/dataset-registry.md` §1.7). The
geotransform is **uniformly `10.0 × −10.0` projected units across 200 sampled
rasters, with zero variation**. SNAP wrote nominal projected units and did
**not** compensate for local scale. The predicted variation is real:

| actid | centre lat | measured true spacing |
|---|---|---|
| 1111003 | 11.536° | **9.798 m** |
| 1111002 | 12.238° | **9.773 m** |
| 1111006 | −16.421° | **9.592 m** |
| 1111005 | −17.622° | **9.531 m** |
| 1111004 | 29.121° | **8.736 m** |

**Nothing in this corpus is 10 m true ground.** The observed spread is 12%
across five activations alone; the full corpus reaches ~60°N where spacing
would be ~5 m. The earlier 8.82 m estimate at 28.06° is corroborated by
1111004's measured 8.736 m at 29.121°.

**Consequences, now that the measurement exists:**

- `configs/data.yaml` no longer carries a bare `resolution_m: 10`. It records
  `source_pixel_spacing_projected_units: 10`,
  `source_pixel_spacing_is_true_ground_metres: false`, and a
  **per-activation** `true_ground_spacing_m` block with the measured values.
- The adapter must resample **per activation** from that activation's own
  measured spacing to the canonical grid. A single corpus-wide source GSD is
  not available and must not be assumed.
- The remaining 40 activations are unmeasured. Spacing must be derived from
  each tile's `info.json.geom` centre latitude as data arrives, not guessed.
- **RESOLVED as C2 — the canonical GSD is 10.0 m true ground.** The earlier
  `TODO(decide)` here is closed. The reasoning is in §10 ("Why C2 is 10 m when
  no dataset is 10 m"): the production inference source is natively 10 m, and
  every measured Kuro spacing is *finer* than 10 m, so the training path is
  coarsened rather than interpolated up. The cost — ~2× coarsening for the
  unmeasured high-latitude activations — is disclosed as E8.

### 7a.4 Speckle filtering: parity is impossible, so the mismatch is disclosed

Kuro Siwo applies **Lee Sigma, 7×7 window, 3×3 target, sigma 0.9**.
Sen1Floods11 applies none. Our M2 applies none.

A CNN reads texture, and speckle is texture. Training on filtered imagery and
serving unfiltered imagery is a silent domain shift of exactly the kind that
produces good validation numbers and poor field performance. Three options were
considered:

1. **Implement Lee Sigma in M2** with the same parameters. *Rejected.* Kuro Siwo
   is delivered **already filtered**, so this would double-filter the training
   corpus while leaving inference single-filtered. It does not produce parity;
   it produces a different mismatch, and a less obvious one.
2. **Filter neither** on the training path by using an unfiltered SLC-derived
   product, or by making Sen1Floods11 the primary corpus instead. *Rejected.*
   No unfiltered Kuro Siwo product is delivered, and switching primary corpus
   discards the permanent-water class and the pre-event imagery that the whole
   architecture depends on.
3. **Apply nothing uniformly and disclose the residual mismatch**, with an
   ablation required before any transfer claim. *Chosen — frozen as C4.*

> **Superseded.** An earlier revision recommended option 1 and treated option 3
> as a fallback. That ordering assumed the training corpus could be brought into
> alignment. It cannot: the filter is already baked into the delivered product.
> Option 3 is therefore not a concession but the only honest description of the
> situation, and `parity_possible: false` is recorded in the contract so no
> later reader mistakes the policy for an unfinished task.

The residual mismatch is disclosure E7. The ablation is M4 deliverable 6 in
§10.F, and it is a precondition on claiming transfer, not an optional extra.

**The schema half of this decision mattered as much as the policy.** Until this
freeze, `speckle_filter` sat under an unmodelled `sentinel1.steps` block on a
model with `extra="allow"`, so setting it was a measured no-op. Fixed; see §10
("Why C4 freezes a limitation rather than a fix").

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
3. The 0.15 linear clip is applied to **both** paths or neither, in the linear
   domain before the dB conversion, with justification recorded (C10). The
   production-side step is `PENDING(M2 wiring)`.
4. **RESOLVED as C4.** Speckle filtering is *not* identical on both paths and
   cannot be made so: Kuro Siwo arrives already Lee Sigma filtered. The policy
   is to apply nothing uniformly, record `parity_possible: false`, disclose the
   mismatch (E7) and require the ablation before any transfer claim.
5. **RESOLVED as C2.** Kuro Siwo's true ground spacing is measured (§7a.3):
   8.736–9.798 m across five activations, from a uniform 10-projected-unit
   transform. The canonical GSD is **10.0 m true ground**, and the adapter
   resamples per activation from that activation's own measured spacing.
6. **RESOLVED as C1.** Both paths are on **WGS 84 / UTM**, metre units, with the
   zone derived per sample from its own centroid, at the same 10 m ground
   spacing. A single global projected EPSG is explicitly not chosen.
7. Temporal depth is 1 pre + 1 post on both paths. The delivered `info.json`
   makes the roles machine-readable (`MS1.master=true` post, `SL1` crank 1,
   `SL2` crank 2), so the adapter selects `SL1` and records `SL2` as dropped.
8. The adapter has cross-checked a delivered Kuro Siwo raster against the
   reference statistics `mean[VV,VH] = [0.0953, 0.0264]`,
   `std = [0.0427, 0.0215]` **as a tolerance-banded check over the full
   training split — not as a per-raster equality gate.** Measurement on a
   3-activation subset gave clipped VV mean 0.1180 against a published 0.0953
   from a correct product read correctly, so an equality gate would reject
   valid data. The reference values are also **not** usable as our fitted
   normalisation parameters: their pre/post, split and aggregation bases are
   all unstated upstream.

**Additional conditions established by the delivered data (§1.7):**

9. The adapter selects partition `01` only. Partition `00` carries **no label
   raster**, so globbing every `info.json` would ingest label-free samples.
10. The adapter filters `catalogue.exported == 1`. The catalogue is
    source-level (3 rows per grid) and lists more grids than exist on disk.
11. The adapter treats `MK0_MLU == 3` as **nodata, not a class**, while never
    admitting 3 into the semantic class set. Stored values are `{0,1,2,3}`;
    semantic classes are `{0,1,2}`.
12. The adapter derives validity from `MK0_MNA == 1` and **asserts** the
    redundant agreement `MLU==3 ⇔ MNA==0` and `SAR==0.0 ⇔ MNA==0`.
13. The adapter applies the 0.15 linear clip **explicitly**. The delivery is
    not pre-clipped (observed VV max 901.92).
14. The adapter consumes neither `MK0_SLOPE` (not degree-valued; observed max
    4344.96) nor `MK0_DEM` (SRTM 1Sec vs production WorldDEM-30). The baseline
    is SAR-only.

> **The honest summary.** Items 1–3 are mechanical and the existing M3 gate
> already supports them. Items 5 and 6 are **resolved** by C2 and C1, and item 7
> by the machine-readable temporal roles. Item 4 is **resolved as a disclosed
> limitation** rather than fixed, because parity is unobtainable (C4). Item 8
> cannot pass yet on the available subset (D5), which is why it is a
> tolerance-banded check over the full training split rather than a gate. Items
> 9–14 are established by the delivered data and are mechanical once the adapter
> exists. The two entries in §7a.6 are permanent, and they bound how well any
> reported number can be expected to transfer.

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
4. Exclude invalid pixels from loss and all metrics. Kuro Siwo's validity is
   encoded **three redundant ways** (`MK0_MNA == 1` canonical, `MK0_MLU == 3`,
   `SAR == 0.0`); Sen1Floods11's is a single **in-band** `-1`. The forms are not
   interchangeable and the capability model records them as a *list* per corpus,
   never as one enum (C8).
5. Keep the decision threshold outside the model, selected on validation using
   flood-class IoU (C6). With one validation activation available, any threshold
   so selected is `development_only` and not reportable.
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

---

## 10. Contract freeze status — 2026-10-08

> # Contract is FROZEN. M4 may begin.
>
> Every decision M4 needs in order to start the adapter is decided and recorded.
> Nothing below is left for an implementer to reinterpret.
>
> **Frozen is not the same as settled.** Two decisions freeze an honest
> *limitation* rather than a solution, and they are labelled as such: **C4**
> freezes a disclosed train/serve texture mismatch because parity with this
> corpus is impossible, and **C6** freezes a selection criterion whose *value*
> is still blocked on validation data. Both are usable by M4 as they stand.
> Neither should be read as a claim that the underlying science is resolved.

This section is the single authoritative ledger. The machine-readable half is
`configs/data.yaml → m4_contract`, validated by `src/floodmap/utils/contract.py`
with `extra="forbid"`; the guards are `tests/test_contract_freeze.py`.

### A. Frozen facts — established, not inferred

Verified against the delivered product; safe to build on.

| Fact | Value | Evidence |
|---|---|---|
| Representation | linear σ⁰ | float32, non-negative, nodata 0.0, order 0.1, max 901.92 |
| Delivered clipping | **none** — not pre-clipped | VV max 901.92 ≫ 0.15 |
| Source CRS | `EPSG:3857` | uniform, 200 rasters |
| Source pixel spacing | **10 projected units**, not 10 m ground | uniform `10.0 × −10.0`, zero variation |
| True ground spacing | **8.736–9.798 m, per activation** | `1/cos(lat)` on five activations (§7a.3) |
| Semantic classes | `{0: No water, 1: Permanent Waters, 2: Floods}` | upstream + delivered |
| Label stored values | `{0,1,2,3}` | 60-tile census; 3 is 6.48% of pixels |
| Label nodata | **3** | `info.json` declares `MK0_MLU.nodata = 3` |
| Validity mechanisms | three, **redundant** | `MLU==3` ⇔ `MNA==0` ⇔ `SAR==0.0`, zero disagreement over 120 tile pairs |
| Temporal roles | machine-readable | `MS1.master=true`; `SL1` crank 1; `SL2` crank 2 |
| Partitions | `01` labelled · `00` **unlabelled, no label raster** | 10 vs 7 tif; `aoiid` 1 vs null |
| Catalogue | source-level, 3 rows/grid; `exported=1` ⟺ on disk | exact on all five activations |
| Bundled slope | present, **not degree-valued**, max 4344.96 | p50 0.124, 0.001% > π/2 |
| Bundled DEM | present, SRTM 1Sec | registry §1.5.3 |
| Tile size | 224 × 224 | uniform |
| Upstream split | 27 train / 7 val / 10 test by `activation_id`; `1111012` in no split | unchanged, authoritative |

### B. Frozen invariants — binding on the adapter

Recorded per dataset in
`configs/data.yaml → production.training_datasets.datasets[*].adapter_invariants`
so an invariant cannot drift away from the facts it constrains: **18 for Kuro
Siwo (C13)** and **15 for Sen1Floods11 (C14)**, both enumerated in §10.G below.

Four repository-level invariants sit outside those lists because they constrain
*our* code rather than a corpus:

1. Validity propagates as a parallel boolean — no epsilon, no flooring
   (`features/numerics.py`).
2. The adapter lives outside `src/floodmap/features/`
   (`tests/test_features_m3.py::test_no_dataset_specific_identifier_appears_in_the_feature_package`).
3. Normalisation statistics are fitted from the training split only, never from
   validation or test.
4. The official upstream split is authoritative; no activation is ever
   fabricated to fill a local gap.

### C. Frozen decisions — C1 to C14

Every decision carries a status. **FROZEN** means M4 may rely on it.
**FROZEN (limitation)** means the decision is settled but what it settles is a
disclosed gap. **DEFERRED** means a *value* awaits evidence while the policy
around it is fixed — none of these block the adapter.

| # | Decision | Frozen value | Status |
|---|---|---|---|
| **C1** | Canonical projected CRS | **WGS 84 / UTM, metre units**, zone derived per sample from its own centroid. No single global EPSG. Source CRS retained separately. | **FROZEN** |
| **C2** | Canonical GSD | **10.0 m true ground** on the canonical UTM grid. Native to no dataset. | **FROZEN** |
| **C3** | Resampling | Per type: SAR bilinear *in linear power, before dB*; optical bilinear; labels nearest; validity masks nearest; DEM bilinear. No generic method, no default. | **FROZEN** |
| **C4** | Speckle | **No filter applied by our pipeline, on any path.** Parity is impossible, not unimplemented. Mismatch disclosed; ablation required before any transfer claim. | **FROZEN (limitation)** |
| **C5** | Statistics artifact | Schema `floodmap.statistics/1`, sidecar pair, three kinds (reference / fitted / validation) with full provenance. **No fitted values exist and none are invented.** | **FROZEN** (schema) · **DEFERRED** (values) |
| **C6** | Selection criterion | **IoU of the flood class (class 2)**, on the validation role only. Everything else reported, nothing else selected on. | **FROZEN** (criterion) · **DEFERRED** (value, see D2) |
| **C7** | Spatial buffer | **0 m**, justified by activation-level partitioning. Sub-activation splitting forbidden. | **FROZEN** (+1 pending check) |
| **C8** | Capability model | Structured per-dataset blocks: `imagery` / `terrain` / `labels` / `sar_representation` / `splits`, plus `validity_mechanisms` as a **list**. | **FROZEN** |
| **C9** | Output nodata | Class mask `uint8` **255**; probability and continuous `float32` **−9999.0**; loss/metric `ignore_index` **3** (an *input* concern). | **FROZEN** |
| **C10** | Clip site | **Dataset adapter**, on linear σ⁰, **before** the dB conversion. `0.15` linear ≡ −8.2391 dB recorded but not applied. Raw files never modified. | **FROZEN** |
| **C11** | Licensing / redistribution | Nothing redistributed from this repository, either corpus, regardless of outcome. Upstream licences remain **unresolved** and no licence is manufactured. | **FROZEN** (policy) · upstream **UNRESOLVED** |
| **C12** | Local subset sufficiency | Development permitted; production training run, threshold selection and scientific model selection **not** permitted. Official split never redefined by local availability. | **FROZEN** |
| **C13** | Kuro Siwo adapter invariants | 18, numbered and machine-readable. | **FROZEN** |
| **C14** | Sen1Floods11 adapter invariants | 15, numbered and machine-readable. | **FROZEN** |

**No decision in this table is BLOCKED.** Nothing in C1–C14 requires new
evidence before the adapter can be written.

#### Why C1 is a rule and not an EPSG code

A single projected CRS was considered and rejected on evidence rather than
taste. The corpus spans roughly −18° to +60° latitude across all longitudes, and
no projected CRS holds metre fidelity across that. Inheriting the source
`EPSG:3857` would have been the worst option available: it would preserve
exactly the defect §7a.3 measured, since Web Mercator is metre-*labelled* and
not metre-*true*. Inside a UTM zone the scale error is under ~0.1%, which is
what makes areas, lengths and the slope kernel mean what they say.

The Trishuli AOI would fall in UTM 45N (`EPSG:32645`). That is recorded as a
*derived expectation*, not a frozen value, because `data.yaml → aoi` is
deliberately null and the only candidate geometry on disk has no provenance
(registry §1.6).

#### Why C2 is 10 m when no dataset is 10 m

Every candidate is a compromise, so the question is which direction to
compromise in:

- **10 m.** The production Sentinel-1 IW GRDH inference source is natively 10 m,
  so the path we actually have to serve needs no resampling at all. Every
  *measured* Kuro spacing (8.736–9.798 m) is finer than 10 m, so the training
  path is **coarsened** — it never manufactures detail the source lacked.
- **~8.7 m** (the finest measured spacing) would resample the *inference* input
  to match a training artefact. That is the wrong direction: it degrades the
  data the product is actually judged on.

The cost is real and disclosed: unmeasured activations reach ~60°N, where 10
projected units is ~5 m of ground, so harmonising those to 10 m is roughly a 2×
coarsening.

#### Why C4 freezes a limitation rather than a fix

§7a.4 listed three options. The measurement changes which one is available.
Kuro Siwo is delivered **already Lee Sigma filtered** and cannot be un-filtered,
so:

- filtering our side "to match" would **double-filter** the training corpus;
- filtering nothing leaves a genuine texture difference, because a CNN reads
  texture and speckle is texture.

No setting of our own filter makes the two paths identical. So the policy is the
uniform and honest one — apply nothing, everywhere — `parity_possible` is
recorded as `false`, and the residual mismatch must be quantified by ablation
before any claim about transfer. Inventing a parity claim here would overstate
how well any reported number generalises.

**The schema gap mattered as much as the policy.** Until this freeze,
`speckle_filter` sat under an unmodelled `sentinel1.steps` block on a model with
`extra="allow"`. Measured, not assumed: `steps`, `nodata_value`,
`target_grid.resampling`, `sentinel2.steps`, `dem.steps`, `osm` and `qa` were
*all* being parsed into `__pydantic_extra__` and discarded, a mistyped section
name such as `sentinal1:` was accepted, and the real `sentinel1` then fell back
to all-`None` defaults. Setting a speckle policy was a **no-op**. Every block is
now modelled with `extra="forbid"`, and `Sentinel1Config` additionally rejects a
`steps.speckle_filter` that contradicts the `speckle` policy block.

#### Why C6 selects on flood-class IoU

Two of the three classes are easy. "No water" dominates by area and "Permanent
Waters" is a largely static, high-contrast target, so a macro average over the
three can **rise while flood IoU falls** — selecting a checkpoint that is worse
at the only thing this product claims to do. Per-class IoU and F1, macro and
micro averages, and the full confusion matrix are all still reported; they are
simply not what selection optimises.

The *value* is deferred, and deliberately so. One validation activation (19
tiles) cannot separate event-specific behaviour from generalisable behaviour, so
a threshold tuned on it is a property of that event. The contract requires ≥2
distinct validation activations for a **reportable** threshold — a floor, not a
sufficiency claim — and marks anything selected before then
`development_only`. The error being prevented is concrete: selecting on 19 tiles
and reporting the number as though it came from the official 7-activation
validation split.

#### Why C7 is zero and not an invented distance

The official split assigns **whole activations**, so every tile of an activation
lands in exactly one split and there is no train/validation tile adjacency
anywhere in the corpus. A non-zero buffer would be a number with nothing to act
on, and would imply a tile-level split this project does not use. Zero is
therefore flagged as a *decision* (`buffer_is_a_decision_not_an_unset_value`)
rather than left to look like an unfilled placeholder, and the schema refuses a
zero buffer unless activation disjointness is asserted **and** sub-activation
splitting is forbidden.

One gap is disclosed rather than closed: activation-level splitting isolates
**events**, not **geography**. Two activations over the same basin at different
dates could sit in different splits and would share terrain, land cover and
permanent water. Verifying that needs all 45 activation geometries; 5 are
available. Recorded as `PENDING(data)`. It does not block M4 — it cannot make
the split worse than it already is, and it is checkable the moment the
geometries arrive.

#### Why C10 clips in linear and not in dB

The transform commutes, so either order clips the same pixels — but not to the
same bound. `0.15` is the exact published value; `−8.2391 dB` is a rounded
transform of it. Clipping linear-then-converting applies the exact bound;
clipping in dB would make the training distribution depend on how many decimals
someone wrote down. The dB figure is therefore **recorded and not applied**.

Three representations are named separately in the contract because conflating
them is how a clipped training distribution ends up served against an unclipped
inference one:

| Stage | Representation |
|---|---|
| source | linear σ⁰, **unclipped** (observed VV max 901.92) |
| preprocessing | linear σ⁰, clipped at 0.15 |
| model | decibel, upper bound −8.2391 dB |

### D. Production-data limitations

These bound what M4 may *claim*. None of them block the adapter.

| # | Limitation | Detail |
|---|---|---|
| D1 | Corpus coverage | 5 of 45 activations delivered: 3/27 train, 1/7 val, 1/10 test |
| D2 | **Validation is 19 tiles from one activation** | Cannot support a reportable threshold or checkpoint selection. C6's criterion is frozen; its value is deferred and marked `development_only` |
| D3 | Train skew | 7,052 train tiles, but 1111004 alone is 71% of them, at latitude 29.1° |
| D4 | 40 activations unmeasured | Spacing must be derived per tile from `info.json.geom` as data arrives |
| D5 | Reference cross-check not yet passable | Subset means are 24–41% high for legitimate sampling reasons; the check is tolerance-banded over the full training split |
| D6 | `aoi.geojson` unprovenanced | Untracked, never committed; cannot anchor a grid instance |
| D7 | `target_grid` transform/width/height | All `null`. The grid *rule* and *resolution* are frozen; the grid *instance* needs the AOI |

### E. Permanent disclosure limitations

Not tasks. These cannot be closed and must appear in the final report.

1. **The flood definition is unpublished.** Supplemental §4 documents the
   process (five experts, 1:1000 scale, extracted keys) but withholds the keys.
   Wet soil / submerged vegetation `UNSTATED`; sediment-laden water
   `NOT MENTIONED`; no label-noise figure. Our `Floods` semantics inherit a
   judgement we cannot reconstruct.
2. **Terrain-correction DEM mismatch.** SRTM 1Sec upstream vs WorldDEM-30 in
   production. *Mitigated for M4* by excluding DEM from the SAR-only baseline,
   but it returns the moment terrain features are enabled.
3. **No Himalayan data in either permitted corpus.** The only Nepal activation
   is tropical lowland and sits in the upstream test split. Unseen-Himalaya
   evaluation cannot mean metrics against permitted Himalayan labels.
4. **Reference-statistics basis unknown.** Pre/post, train/all-data and
   per-pixel/per-scene bases are all unstated upstream, which is why they are a
   cross-check and never our fitted parameters.
5. **Clipping at −8.2391 dB discards the bright tail**, flattening the
   double-bounce response of flooded vegetation and urban areas — exactly the
   cases the authors list as hard. Worth an ablation, not acceptance by default.
6. **`MK0_SLOPE` contains unexplained values** (max 4344.96, 0.001% above
   π/2). We exclude it rather than explain it; the cause is upstream.
7. **Speckle filtering differs between the corpora and our pipeline** and
   cannot be reconciled (C4). Kuro Siwo is Lee Sigma filtered, Sen1Floods11 is
   not, we filter nothing. Disclosed, with an ablation required.
8. **Kuro Siwo source spacing is not universal 10 m ground spacing** and the
   40 undelivered activations are unmeasured, so the corpus a trained model has
   seen spans a range of effective resolutions.

### F. M4 implementation prerequisites

In order. Prerequisites 1–3 of the previous revision are **complete**.

1. ~~Resolve every item in §C.~~ **Done** — C1–C14 are frozen above and encoded
   in `configs/`.
2. ~~Fix C4's config gap.~~ **Done** — every block in
   `configs/preprocessing.yaml` is modelled with `extra="forbid"`, and
   contradictory speckle declarations are rejected.
3. ~~Resolve D2 before starting.~~ **Re-scoped, not skipped.** D2 blocks a
   *reportable threshold*, not the adapter. C6 freezes the criterion and marks
   any threshold selected on one activation `development_only`, so M4 can be
   built and exercised now and the selection run when validation data arrives.
4. Implement the adapter outside `src/floodmap/features/`, reusing
   `ArtifactProvenance`, `AnalysisGrid`, `FeatureRegistry` and `numerics`
   unmodified, extending only `ArtifactType`. Enforce C13 and C14 at the adapter
   level.
5. Then training: SAR-only, Kuro Siwo native 3-class, no corpus union, no
   debris claim, no S2/DEM input channels, threshold outside the model.
6. Then, when enough validation activations exist, run the C6 selection and the
   C4 speckle ablation, and report both.

### G. Adapter contract invariants in full

Contract, not implementation. No adapter code exists. These are the conditions
the M4 adapter will be audited against, and they are enforced as configuration
in `configs/data.yaml` with per-subject coverage asserted by
`tests/test_contract_freeze.py`.

#### C13 — Kuro Siwo (18)

1. `partition == 01` only; never glob all `info.json`.
2. `catalogue.exported == 1` is a required filter.
3. `MS1` is the post-event master (`sources.MS1.master == true`).
4. `SL1` (`crank == 1`) is the baseline pre-event epoch.
5. `SL2` is never silently substituted for `SL1`.
6. Stored label values are `{0,1,2,3}`.
7. Semantic classes are `{0,1,2}`.
8. Label value `3` is nodata, never a semantic class.
9. Canonical validity is `MK0_MNA == 1`.
10. Assert `MLU==3 ⇔ MNA==0` and `SAR==0.0 ⇔ MNA==0`; do not assume.
11. Source CRS `EPSG:3857` is retained, never overwritten.
12. Source transform is retained.
13. Source spacing is 10 *projected units*, never universal 10 m ground.
14. `MK0_SLOPE` is excluded from the M4 SAR-only baseline.
15. `MK0_DEM` is excluded from the M4 SAR-only baseline.
16. Raw SAR is linear σ⁰, not decibel.
17. The 0.15 linear clip is explicit preprocessing, never assumed applied.
18. Official activation-based split identity is preserved.

#### C14 — Sen1Floods11 (15)

1. S1 is decibel, never linear σ⁰.
2. VV and VH are both available.
3. S2 is TOA reflectance scaled by 10000.
4. Labels are binary `{0,1}`.
5. Label value `-1` is nodata.
6. No permanent-water vs flood semantic separation exists.
7. Post-event only.
8. No pre-event counterpart may be synthesised.
9. No change features may be derived.
10. No DEM capability is assumed.
11. Source CRS `EPSG:4326` is retained, never overwritten.
12. Source transform is retained.
13. Source spacing is ~0.000090 degrees, never universal 10 m ground.
14. No artificial Kuro Siwo semantics are imposed.
15. The dataset stays external to this repository until its licence is resolved.

> **On invariant 8.** This is the one most likely to be violated by accident
> rather than by intent. Synthesising a pre-event image — by reusing the
> post-event one, or by substituting a neighbouring chip — would let a single
> code path serve both corpora, and the resulting change feature would be
> identically zero or pure noise. Nothing downstream could detect it. The
> capability model enforces the condition structurally: `change_features`
> cannot be declared without `temporal_pair`.

### Status

```
CONTRACT FREEZE STATUS: FROZEN
M4 STATUS: READY TO IMPLEMENT (not started)
```

14 contract decisions frozen (**C1–C14**, zero blocked); **7
production-data limitations** (D1–D7, which bound claims rather than
implementation); **8 permanent disclosures** (E1–E8, which must be reported).

---

## 11. The canonical M4 sample contract

What the adapter must present at its boundary, so M4 does not have to infer it.

**Documentation only.** No Python sample class is defined here: constructing the
sample is adapter work and belongs to M4. What is fixed is the *shape* of the
agreement, and in particular which fields a corpus is permitted to omit.

### 11.1 Fields

`required` means every corpus must supply it. `capability-gated` means supply it
if and only if the corpus's C8 capability block says it exists — **absence is a
valid, expected state and must not be filled with a placeholder.**

| Field | Presence | Notes |
|---|---|---|
| `sample_id` | required | Stable and unique within a dataset |
| `dataset_id` | required | `kuro_siwo` \| `sen1floods11` |
| `split` | required | Resolved against the **official** split table, never from local availability |
| `event_id` | capability-gated | Kuro Siwo `activation_id`; Sen1Floods11 event |
| `source_crs` | required | Retained verbatim; never the canonical CRS |
| `source_transform` | required | Retained verbatim |
| `source_spacing` | required | With its **units** — projected units, degrees or metres. Never a bare `10` |
| `target_grid` | required | Canonical CRS, GSD, transform, dimensions (C1, C2) |
| `temporal_roles` | required | Which epochs exist and what each is. Post-only is a valid value |
| `sar_pre` | capability-gated | Absent for Sen1Floods11 |
| `sar_post` | required | Both corpora have one |
| `sar_representation` | required | `linear_sigma0` \| `decibel`, per corpus |
| `optical` | capability-gated | Sen1Floods11 only |
| `dem` | capability-gated | Excluded from the M4 baseline for both |
| `label` | required | In the corpus's own scheme, not a harmonised one |
| `label_scheme` | required | Semantic values, stored values, nodata value and its form |
| `validity` | required | Boolean valid mask derived from the corpus's canonical source |
| `validity_mechanisms` | required | **A list.** Three entries for Kuro Siwo, one for Sen1Floods11 |
| `nodata` | required | Output nodata semantics (C9), distinct from class IDs |
| `capabilities` | required | The corpus's C8 block, so a consumer can ask before assuming |
| `preprocessing_state` | required | Clip applied? Resampled? Converted to dB? Speckle filtered upstream? |
| `provenance` | required | `ArtifactProvenance`, tracing to the source files actually read |
| `dropped_sources` | required | May be empty. Kuro Siwo records `SL2` here |

### 11.2 Two rules that make the contract worth having

**Absence is representable, and is not failure.** `sar_pre` is absent for
Sen1Floods11 and that is the correct, expected state. A consumer must branch on
the capability block, not on a truthiness check against a zero-filled array. The
alternative — forcing Sen1Floods11 into the Kuro Siwo shape — is the specific
failure C14 invariant 8 exists to prevent.

**No corpus is harmonised into the other.** Labels arrive in the corpus's own
scheme with its own nodata value and its own validity mechanisms. Kuro Siwo's
3-class scheme and Sen1Floods11's binary scheme are not unioned under one head
(§8 item 10), and Kuro Siwo's collapse to binary — the only direction that is
lossless-by-discarding rather than invented — is a reporting operation for
external validation, never a training-time merge.

