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

- Kuro Siwo: `{0: No water, 1: Permanent Waters, 2: Floods, 3: Invalid}`
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
| Declare SAR representation | `linear` (**confirm from a delivered raster** — currently inferred) | `decibel` (stated) |
| Bind band roles | `vv`→VV, `vh`→VH | `vv`→band 0, `vh`→band 1 |
| Reproject to a projected metre grid | from EPSG:3857 | from EPSG:4326 |
| Emit per-band valid masks | from class `3` | from class `-1` |
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

## 8. Exact M4 requirements

What M4 must do, derived from the above. **M4 is not started.**

**Must:**

1. Implement the dataset adapter per §7, outside the production feature
   generator, with per-corpus representation and band-role declarations.
2. Train a **SAR-only** segmentation model on Kuro Siwo's native 3-class scheme,
   consuming M3 Sentinel-1 features via the registry.
3. Honour upstream splits; exclude Kuro Siwo's test activations from training
   and validation.
4. Exclude ignore values (`3`, `-1`) from loss and all metrics.
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

1. **Kuro Siwo's SAR representation is inferred, not stated.** Linear σ⁰ comes
   from reading its SNAP graph (`outputImageScaleInDb=false`, no
   `LinearToFromdB` node). A wrong inference silently corrupts every SAR
   feature. **Must be confirmed from a delivered raster's value distribution
   before training.**
2. **Kuro Siwo's flood definition is in an unread Supplemental Material.** Our
   `Floods` class semantics inherit theirs, so the label mapping cannot be
   frozen until it is read.

Plus: both licences unresolved; Sen1Floods11 weak-label encodings and test split
unknown; no Himalayan labelled data in the permitted set; the Kuro Siwo "Nepal"
AOI geometry unverified.
