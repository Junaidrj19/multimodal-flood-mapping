# Kuro Siwo malformed samples — evidence and proposed quarantine policy

**Status: EVIDENCE RECORDED. POLICY PROPOSED, NOT ACTIVATED.**

No sample is excluded from any split, no statistics artifact has been fitted,
and `configs/experiments/m5_kuro_sar_unet.yaml` is unchanged by this
investigation. Activating the policy below is a separate, reviewed decision.

---

## 1. What was found

Two of the 7,103 locally delivered Kuro Siwo samples contain **NaN values in
SAR pixels that all three of Kuro's validity mechanisms declare valid**.

### 1.1 Scope — full corpus scan

Every sample was scanned (all 6 SAR planes, checking finiteness and sign
against `MK0_MNA`):

| Scanned | Affected | Rate |
|---|---|---|
| 7,103 | **2** | 0.028% |

Both are in the `train` split, both in activation `1111004`. No sample anywhere
in the corpus contains a negative SAR value. The two samples are exactly those
named in the audit; the scan found no others.

### 1.2 The two samples are NOT equivalent

This distinction matters for any decision, because the M5 baseline uses only
SL1 (pre) and MS1 (post) — `SL2` is a recorded dropped source.

**Sample A — `1111004:304feeed-d5a7-5ad8-b08d-ea711c073f9d`**
`/…/kuro_siwo/1111004/01/304feeedd5a75ad8b08dea711c073f9d`

| Plane | Role | NaN | NaN on valid pixels |
|---|---|---|---|
| SL1 VV / VH | **baseline** | 0 | 0 |
| MS1 VV / VH | **baseline** | 0 | 0 |
| SL2 VV / VH | dropped | 1 each | 1 each |

**All four baseline channels are completely clean.** The only defect is a
single pixel at row 192, column 0 — the left tile edge — in the epoch this
baseline discards. The adapter nonetheless rejects the sample, because
`load_sample` validates finiteness across all three temporal roles including
the one it then drops.

**Sample B — `1111004:df3d7e17-0a15-5d76-aad9-4571285aae39`**
`/…/kuro_siwo/1111004/01/df3d7e170a155d76aad94571285aae39`

| Plane | Role | NaN | NaN on valid pixels |
|---|---|---|---|
| SL1 VV / VH | **baseline** | 118 each | 118 each |
| MS1 VV / VH | **baseline** | 114 each | 114 each |
| SL2 VV / VH | dropped | 112 each | 112 each |

**464 baseline-channel pixels are NaN on declared-valid positions.** The
footprint is a contiguous block at rows 103–112, columns 187–223 — reaching the
right tile edge. Footprints are near-identical but not equal across planes
(intersection 112, union 118), so this is a per-acquisition geocoding artifact
rather than one shared mask.

Sample B's `MK0_DEM` also contains 6,180 negative values, which is a separate
and unrelated observation: DEM and slope are excluded from this baseline.

### 1.3 Why this is an upstream inconsistency, not something our code can fix

For both samples:

- `MK0_MNA == 1` (valid) on every NaN position;
- `MK0_MLU != 3` (labelled) on every NaN position — and `MLU == 3` exactly
  equals `MNA == 0`, so the label/validity cross-check itself is internally
  consistent;
- SAR is **not** `0.0` on those positions, so the SAR-nodata-zero mechanism
  also reports them valid.

All three declared validity mechanisms say "observed". The pixel value says
"no measurement". **NaN is a fourth state that the frozen contract does not
model**, and the clustering at tile edges is consistent with an upstream
geocoding or resampling artifact at a scene boundary.

Treating NaN as invalid would be inventing a fourth validity mechanism. C8
deliberately models validity as an explicit *list* of declared mechanisms; a
silent fourth one would make the capability model describe something the corpus
does not declare.

### 1.4 Current behaviour

The M4 adapter already refuses both samples:

```
DatasetIntegrityError: Kuro linear SAR is not finite/non-negative: …/SL2_IVV_…tif
DatasetIntegrityError: Kuro linear SAR is not finite/non-negative: …/MS1_IVV_…tif
```

This is a **loud, correct refusal** — nothing is silently skipped and no NaN
can reach a loss function. The consequence is that any full pass over the
training split raises on these two samples, which is the blocker.

---

## 2. Recovery assessment

### 2.1 An official verified copy IS publicly available

The dataset registry records the HuggingFace card as unreadable (HTTP 401).
That note was made against the wrong repository identifier. The actual
distribution points were re-checked on 2026-10-09:

| Resource | Status |
|---|---|
| `github.com/Orion-AI-Lab/KuroSiwo` | HTTP 200 |
| `huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-GeoTIFFs` | **public, `gated: false`, `private: false`** |
| `huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset` | public |

**Secondary finding, reported but not acted on.** The GeoTIFF repository's card
declares `license: cc-by-4.0` explicitly. Registry blocker #1 records an
unresolved MIT-vs-CC-BY discrepancy partly because this card could not be read.
Updating the registry is a scientific-record decision outside this task's
scope, and the conservative no-redistribution policy (C11) is unaffected either
way — nothing is redistributed from this repository regardless.

### 2.2 But targeted retrieval is not possible

The GRD GeoTIFFs are distributed as **35 tar shards totalling 699 GB**
(~20 GB each). `tar` has no central index, so a specific grid cannot be
addressed by HTTP range request; locating two grid directories requires
streaming shards until they are found — **expected ~350 GB, worst case 699 GB,
to recover 0.028% of the corpus.**

The webdataset variant is smaller but stores `.npy` arrays, not GeoTIFFs. It
therefore cannot serve as a drop-in replacement: the M4 adapter requires
GeoTIFFs with CRS, transform and nodata metadata, and synthesising that
metadata around a `.npy` payload would fabricate exactly the geospatial facts
the adapter exists to verify.

### 2.3 Recommendation

**Do not attempt recovery now. Quarantine instead, under the policy in §3.**

Reasoning:

1. The cost is disproportionate — up to 699 GB of transfer for 2 of 7,103
   samples, 0.028% of the corpus and 0.028% of the training split.
2. Recovery may not even help. The NaNs sit at tile edges in a contiguous
   block, with per-acquisition footprints. That is the signature of an upstream
   processing artifact baked into the published product, not of local download
   corruption. A verified copy would most likely reproduce the same NaNs, and
   there is no evidence either way without the download.
3. Nothing is lost by deferring. Quarantine is reversible and fully recorded;
   if a verified copy is later obtained for another reason, the two samples can
   be compared and reinstated.
4. **Sample A may not need recovery at all.** Its four baseline channels are
   clean. See §3.4.

What would change this recommendation: evidence that the published product
differs from the local delivery, or a per-file HTTP endpoint for the GeoTIFF
repository making targeted retrieval cheap.

---

## 3. Proposed quarantine policy — NOT ACTIVATED

### 3.1 Principles

- **Explicit, never silent.** A quarantined sample is listed by ID with a
  reason. Nothing is skipped by a `try/except` around a loader.
- **Loud by default.** The adapter's existing integrity refusal stays. Quarantine
  is an allowlist-of-exclusions consulted *before* loading, not a way to swallow
  the error.
- **No invented data.** No NaN is replaced, no value imputed, no sample
  reconstructed from a neighbour.
- **Frozen validity semantics unchanged.** `MK0_MNA == 1` remains the canonical
  valid pixel. NaN is not promoted to a fourth validity mechanism.
- **Counted in every report.** A quarantined sample is reported as excluded,
  with its reason, wherever sample counts appear.

### 3.2 Proposed artifact

`configs/quarantine/kuro_siwo.yaml`, versioned and committed (it contains
sample IDs and reasons, no pixel data):

```yaml
version: "1.0.0"
dataset_id: "kuro_siwo"
reason_codes:
  nan_on_declared_valid_sar: >-
    One or more SAR planes contain NaN at pixels that MK0_MNA, MK0_MLU and the
    SAR-zero mechanism all declare valid. NaN is a fourth state the frozen
    contract does not model.
samples:
  - sample_id: "1111004:df3d7e17-0a15-5d76-aad9-4571285aae39"
    activation: 1111004
    split: "train"
    reason_code: "nan_on_declared_valid_sar"
    affected_planes: ["SL1_IVV", "SL1_IVH", "MS1_IVV", "MS1_IVH", "SL2_IVV", "SL2_IVH"]
    baseline_nan_on_valid_pixels: 464
    nan_footprint: "rows 103-112, cols 187-223 (right tile edge)"
    recovery_attempted: false
    recovery_blocked_by: "upstream GeoTIFFs are 35 un-indexed tar shards, 699 GB"
    evidence: "docs/m5-source-data-quarantine.md"
```

### 3.3 Proposed scope of effect

| Stage | Behaviour |
|---|---|
| Discovery | Sample stays in the index, flagged `quarantined`, so the corpus inventory remains complete and auditable. |
| Statistics fitting | **Excluded.** A NaN would propagate to the mean and std of every channel. |
| Training | Excluded. |
| Validation / test | Excluded, and reported — but see §3.5: neither affected sample is in those splits. |
| Metrics | Not scored; counted as excluded. |
| Run record | `quarantined_samples` listed by ID with reason, alongside the sample counts. |
| Audit report | Excluded samples and reasons reported, never silently dropped. |

### 3.4 Open question for review: sample A

Sample A's four baseline channels are clean; only the dropped `SL2` epoch is
affected. Two defensible options, and this is **a decision for review, not one
to take quietly**:

- **(a) Quarantine both.** Simplest and most conservative. Costs one clean
  sample out of 7,052.
- **(b) Quarantine only sample B, and narrow the adapter's finiteness check to
  the temporal roles the canonical sample actually exposes.** Sample A then
  loads with four clean channels, and `SL2` remains metadata-only as the
  contract already says it is.

Option (b) is arguably more correct — validating a plane that is then discarded
is an over-broad check — but it changes frozen M4 adapter behaviour, so it
should not be bundled with a quarantine decision. **Recommendation: adopt (a)
now, and consider (b) separately on its own merits.**

### 3.5 Measured cost of the proposal

Under option (a), excluding both samples:

| Split | Total | Quarantined | Remaining |
|---|---|---|---|
| train | 7,052 | 2 | 7,050 |
| validation | 19 | 0 | 19 |
| test | 32 | 0 | 32 |

0.028% of the training split. **Neither validation nor test is affected**, so
no reported metric and no checkpoint-selection number is touched by this
decision either way.

---

## 4. What was deliberately not done

- No file under `/Users/mohammedjunaid/Documents/kuro_siwo` was modified,
  copied, or deleted. The investigation was read-only.
- No NaN was replaced, zeroed, or imputed.
- No sample was silently skipped.
- `MK0_MNA == 1` validity semantics were not changed.
- No quarantine config was created; §3.2 is a proposal.
- No sample was excluded from any experiment config.
- No dataset was downloaded.
