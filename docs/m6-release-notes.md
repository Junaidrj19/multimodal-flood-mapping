# M6 release notes — complete M6 flood-mapping milestone

**Release scope:** M6.1 – M6.17 (acquisition validation gates → real Sentinel-1
RTC change-candidate vertical slice → study-extent adoption → downstream
reconciliation → pre-PR audit and its resolution).
**Baseline:** `main` @ `a1a7a04deced4d90b3467223ee2647158a6795a5`,
`origin/main`, 0 ahead / 0 behind.
**Release branch:** `release/m6-flood-mapping` (created from that baseline; the
complete M6 change set is uncommitted work on top of it — see §3).
**Status:** READY TO PUBLISH — complete M6 release validated (conditions in §6).

Attribution: Contains modified Copernicus Sentinel data 2026. Produced using
Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH
2014–2018 provided under COPERNICUS by the European Union and ESA; all rights
reserved. © OpenStreetMap contributors. HydroRIVERS: Lehner, B., Grill, G.
(2013), *Hydrological Processes* 27(15): 2171–2186.

---

## 1. What this release contains, stage by stage

| Stage | State in this release | Real-data status |
|---|---|---|
| M6.1 acquisition validation | implemented | offline; gates closed, polygon coverage + same-track-pair validation enforced |
| M6.2 RTC / optical preprocessing and registry | implemented | exercised on the local Sentinel-1 RTC pair and Sentinel-2 products |
| M6.3 RTC change candidates | implemented, contract **0.2.0** | **runs on real data**, AOI-scoped; candidate evidence only |
| M6.4 infrastructure exposure | implemented and tested | **BLOCKED** — no authorised pre-event OSM; no real output exists |
| M6.5 network disruption / connectivity | implemented and tested | **BLOCKED** — no real M6.4 outputs; no real output exists |
| M6.6 dashboard (real path) | implemented | presents the real M6.3 artifacts; withholds stale results |
| M6.7–M6.13 AOI investigation and HydroRIVERS construction | done | geometry built and reproduced from local HydroRIVERS + SAFE |
| M6.14 human decision | recorded | D1–D6 approved by the project human decision-maker |
| M6.15 AOI adoption | done | adopted `trishuli-corridor-hydrorivers-v10-m615-2ae4183f`, SHA-256 `2ae4183f…8965e3` |
| M6.16 downstream reconciliation | done | M6.3 regenerated to 0.2.0; blockers re-keyed; dashboard reconciled |
| M6.17 pre-PR audit + resolution | done | findings F1–F10 resolved or documented (§4) |

No stage claims more than its evidence: the change candidates are rule-based
Sentinel-1 γ⁰ darkening observations, not confirmed flooding or damage, and
nothing in this release is a trained-model prediction.

---

## 2. Adopted study extent (the one input every number inherits)

| Field | Value |
|---|---|
| Geometry | `configs/aoi/trishuli-corridor-hydrorivers-v10-m615.geojson` |
| AOI ID | `trishuli-corridor-hydrorivers-v10-m615-2ae4183f` |
| SHA-256 | `2ae4183f0d44f80200a889ed401ba25525c52134ba1f31d65e9f2ebc3d8965e3` |
| Provenance | `…-m615.provenance.yaml`, SHA-256 `a01bc8ac5face68d0b51db772bf31cd851d0a0cf4bc390b80c5cfc35c466c0b9` |
| Construction | HydroRIVERS v1.0 main stem `40771762 → 40801674` (index 49), exact reference cut, 1,000 m half-width, EPSG:32645 build, RFC 7946 EPSG:4326 delivery, 250 m same-track SAFE-swath inset as an acceptance gate |
| Geometry contract | one valid Polygon, zero holes, 2,164 closed-ring coordinates, 206.150650904 km² |
| Observation coverage | delivered **130.641632 km² (63.3719 %)**, unobserved **75.509019 km² (36.6281 %)**, valid in-extent pixel centres **1,306,418 (100 % of delivered)** |
| Candidates (M6.3 0.2.0) | **45,269 px = 4.5269 km²** — 3.465124 % of valid in-extent pixels, 2.195918 % of the extent as a lower bound |
| Provenance status | `unverified` — **deliberately**: adopting a geometry authorises no retrieval |
| Gates | `download.enabled: false`, `authorises_download: false`, exploratory exception absent, candidate-stage `allow_download: false` |

Historical AOIs (`trishuli-corridor-UNVERIFIED.geojson` `9035eeee…9362610`,
`trishuli-corridor-hydrorivers-v10.geojson` `941a4079…64b586`) are preserved
byte-for-byte and are still reproducible via `scripts/build_corridor_aoi.py
--config …`. Their derived outputs are labelled stale, never re-keyed.

---

## 3. Release manifest

### 3.1 Required M6 release files — committed

**Tracked files modified by this release (29):**
`README.md`, `configs/baseline.yaml`, `configs/data.yaml`,
`configs/preprocessing.yaml`, `dashboard/app.py`,
`docs/dashboard-artifact-contract.md`, `docs/data-contract.md`,
`docs/dataset-registry.md`, `pyproject.toml`, `scripts/acquire.py`,
`src/floodmap/acquisition/{aoi,config,manifest,outcomes,planetary,scenes,selection}.py`,
`src/floodmap/acquisition/providers/cdse.py`, `src/floodmap/dashboard/__init__.py`,
`src/floodmap/network/__init__.py`, `src/floodmap/preprocessing/pipeline.py`,
`src/floodmap/utils/{config,provenance}.py`, `tests/test_acquisition_m1.py`,
`tests/test_contract_freeze.py`, `tests/test_dashboard_app.py`,
`tests/test_features_m3.py`, `tests/test_preprocessing_m2.py`, `uv.lock`.

**New files added by this release:**
- scripts (4): `scripts/build_corridor_aoi.py`, `scripts/rtc_change_candidates.py`,
  `scripts/candidate_exposure.py`, `scripts/candidate_disruption.py`
- configuration (10): `configs/aoi/trishuli-corridor-hydrorivers-v10-m615.{geojson,provenance.yaml,yaml}`,
  `configs/aoi/trishuli-corridor-hydrorivers-v10.{geojson,provenance.yaml,yaml}`,
  `configs/aoi/trishuli-corridor-UNVERIFIED.geojson`,
  `configs/rtc_change_candidates.yaml`, `configs/candidate_exposure.yaml`,
  `configs/candidate_disruption.yaml`
- implementation (13): `src/floodmap/acquisition/coverage.py`,
  `src/floodmap/baseline/rtc_change_candidates.py`,
  `src/floodmap/dashboard/{realbundle,realdata,render,workflow}.py`,
  `src/floodmap/hydrology/{coverage,hydrorivers}.py`,
  `src/floodmap/infrastructure/{candidate_exposure,candidate_mask}.py`,
  `src/floodmap/network/candidate_disruption.py`,
  `src/floodmap/preprocessing/{s1_rtc_assembly,s2_optical}.py`
- tests (16 new modules + fixtures, 5 existing modules updated):
  `tests/m63_fixtures.py`, `tests/test_acquisition_validation.py`,
  `tests/test_aoi_adoption_m615.py`, `tests/test_aoi_hydrorivers_corridor.py`,
  `tests/test_aoi_observation_coverage.py`, `tests/test_candidate_disruption.py`,
  `tests/test_candidate_exposure.py`, `tests/test_dashboard_adoption.py`,
  `tests/test_dashboard_realbundle.py`, `tests/test_dashboard_realdata.py`,
  `tests/test_dashboard_render.py`, `tests/test_dashboard_workflow.py`,
  `tests/test_hydrological_corridor.py`, `tests/test_rtc_change_candidates.py`,
  `tests/test_s1_rtc_assembly.py`, `tests/test_s2_optical_validation.py`
- documentation (19): `docs/m6-acquisition-validation.md`,
  `docs/m6.2-catalogue-requery.md`, `docs/m6.2-sentinel2-optical-validation.md`,
  `docs/m6.3-rtc-change-candidates.md`, `docs/m6.4-candidate-infrastructure-exposure.md`,
  `docs/m6.5-candidate-network-disruption.md`, `docs/m6.6-dashboard-integration.md`,
  `docs/m6.7-aoi-provenance-investigation.md`, `docs/m6.8-aoi-source-assessment.md`,
  `docs/m6.9-hydrorivers-corridor-extent.md`, `docs/m6.10-hydrorivers-aoi-construction.md`,
  `docs/m6.11-aoi-adoption-review.md`, `docs/m6.12-aoi-boundary-independence.md`,
  `docs/m6.13-extent-decision-brief.md`, `docs/m6.14-aoi-human-decision.md`,
  `docs/m6.15-aoi-adoption-implementation.md`, `docs/m6.16-downstream-reconciliation.md`,
  `docs/m6.17-pre-pr-audit.md`, `docs/m6-release-notes.md` (this file)

No file was deleted, renamed or moved by this release; `git diff --diff-filter=DR`
is empty.

### 3.2 Pre-existing unrelated work — **not** committed

- `.gitignore` `+ .aider*`: a personal agent-tooling ignore line. It has no
  scientific, pipeline or provenance role, so it is deliberately left unstaged in
  the working tree and excluded from this release.

### 3.3 Local / environment-specific files — **not** committed

- `data/raw/**` (SAFE archives, RTC crops, HydroRIVERS geodatabase),
  `artifacts/**`, `models/**`, `.venv/`, caches: git-ignored, large, or both.
- `configs/data.yaml:646` retains a pre-existing absolute external corpus path
  (`dataset_path` for a local Kuro Siwo root). It is **already committed
  historical content**, not introduced or changed by this release; it is recorded
  here as a known local-path string, and the surrounding M6 changes do not
  depend on it.
- Git-ignored generated artifacts embed an absolute `/Users/…` prefix in raster
  tags and in `rtc_change_summary.json`. They are not published.

### 3.4 Generated or sensitive artifacts — **not** published

- `artifacts/m6_3/rtc_change_candidates/*` (7 files, regenerated 0.2.0),
  M6.4/M6.5 blocker JSONs, M6.2 evidence, the legacy
  `artifacts/acquisition_manifest_trishuli{,_l2a}.json`
  (`trishuli_emsr927_corridor` label, null hash — history only) and
  `artifacts/dashboard_demo/` (synthetic, unchanged).
- No credential, token or secret appears in any published file; the only
  credential strings are environment-variable *names*.
- Five absolute local paths that had been written into publishable prose
  (`docs/m6.7`, `docs/dataset-registry`, `docs/m6-acquisition-validation`) were
  sanitized to `<sibling-checkout>/…` and `$KURO_SIWO_ROOT` before publication.

### 3.5 Files requiring further investigation — recorded, none blocking

- `.gitignore` `+.aider*` (owner decision: keep, drop, or commit separately).
- Pre-existing absolute corpus path in `configs/data.yaml` (out of scope here;
  fix only if the project wants machine-independent configuration).
- `configs/aoi/trishuli-corridor-UNVERIFIED.geojson` is an unprovenanced
  candidate geometry that three preservation tests depend on. It is published so
  those tests remain meaningful and so the preservation invariant is auditable;
  its status is documented in `docs/dataset-registry.md §1.6`.

---

## 4. Release blockers reviewed (M6.17 F1–F10) and their resolution

Classification used: **critical** = would publish a wrong result, a broken
contract or a data-integrity hazard; **important** = would mislead a reviewer or
leak machine-specific data; **non-blocking** = hygiene.

| # | Finding | Class | Resolution in this release |
|---|---|---|---|
| F1 | `docs/m6.6-dashboard-integration.md` described six required files, stage `0.1.0`, and the replaced 132.93 km² AOI as the scope of the area figure | **critical (documentation of a live contract)** | fixed: seven required files incl. `rtc_aoi_mask.tif`, stage `0.2.0`, AOI-mask/band/grid and AOI-scope invariants, and a four-notion area table with the current values |
| F2 | `docs/m6.3-rtc-change-candidates.md` presented the superseded window-scoped 275,839 px / 27.5839 km² as the observed result | **critical (numbers a reviewer would quote)** | fixed: the 0.1.0 table is labelled window-scope and superseded; the 0.2.0 AOI-scoped table (45,269 px / 4.5269 km²) and the 83.59 % outside-AOI split are stated |
| F3 | `docs/m6.4`, `docs/m6.5`: stale stage literal, superseded numbers, and “promote `provenance_status` to VERIFIED” guidance that M6.16 had already corrected in the scripts | important | fixed: stage/numbers labelled, blocker state re-stated from the regenerated artifacts, and the real prerequisite (separate OSM authorisation) named, with the unobserved-area claim limit |
| F4 | `docs/dashboard-artifact-contract.md` never stated the 0.2.0 contract | important | fixed: new subsection lists the seven required files, the four rasters' agreement checks, the AOI-scoped invariants and the required `aoi_scope` block |
| F5 | `LIMITATIONS` strings in `candidate_exposure.py` / `candidate_disruption.py` claimed the extent's “origin is not established” — false since M6.15 recorded producer, source and parameters | important (would be emitted into future artifacts) | fixed in both modules; limitation counts unchanged (9 / 10) and tests still pass |
| F6 | Five absolute local paths in three publishable docs | important (privacy/portability) | fixed: replaced with `<sibling-checkout>/…` and `$KURO_SIWO_ROOT` |
| F7 | Dashboard showed a geometric delivered area next to a pixel-centre valid estimate (valid marginally > delivered) with no label | important (presentational ambiguity) | fixed: the row is labelled “pixel-centre estimate” and a caption states the measurement-type difference and that no rate is a whole-extent prevalence rate; asserted in `tests/test_dashboard_app.py` |
| F8 | Trailing whitespace in `docs/m6.14`, `docs/m6.8` | non-blocking | fixed (the two intentional Markdown hard-break lines were converted to blank-line-separated paragraphs) |
| F9 | M6.15/M6.16 asserted whitespace cleanliness from an exit status that cannot prove it | non-blocking (evidence accuracy) | fixed: both reports now state that `git diff --no-index --check` exits 1 whenever the inputs differ and that cleanliness requires inspecting the output |
| F10 | README status table lacked M6.1/M6.2/M6.15/M6.16/M6.17 rows and mis-phrased the AOI status | non-blocking | fixed |

**No critical defect was found in code, configuration, geometry, artifacts or
their numbers.** Every critical/important finding was documentation or emitted
text; all are resolved. Nothing was discarded to resolve a discrepancy, and the
adopted AOI, the historical AOIs and all pre-existing work are preserved.

---

## 5. Validation evidence (post-fix, this release)

| Check | Command | Result |
|---|---|---|
| Full suite | `.venv/bin/python -m pytest tests/ -q -p no:randomly` | **1466 passed, 18 skipped, exit 0** (65.73 s) |
| Focused suite | dashboard app + adoption + M6.4/M6.5 + M6.3 + AOI adoption modules | **196 passed, exit 0** (34.30 s) |
| AOI builder, adopted | `scripts/build_corridor_aoi.py --check` | **MATCH** `2ae4183f…8965e3`, exit 0 |
| AOI builder, historical | `scripts/build_corridor_aoi.py --config aoi/trishuli-corridor-hydrorivers-v10 --check` | **MATCH** `941a4079…64b586`, exit 0 |
| AOI digest enforcement | `from_geojson_file(..., expected_sha256="0"*64)` | raises `ValueError` (refuses to load) |
| Configuration agreement | both `configs/data.yaml` AOI blocks | name/path/SHA/status/gaps agree; `aoi_id` only in `acquisition.aoi` |
| Gates | `load_acquisition_config()` | `unverified`, `authorises_download=false`, `download.enabled=false`, exploratory absent |
| M6.3 mask invariants | raster-level recomputation | candidate 45,269 / valid 1,306,418 / AOI 1,306,418; `candidate ⊆ valid ⊆ AOI` with **0** violations; `255` exactly where `valid ≠ 1`; **0** nodata leaks into valid support and **0** non-nodata values outside it |
| Area separation | summary + dashboard | delivered 130.641632, unobserved 75.509019, valid 130.6418 (pixel-centre estimate), candidates 4.5269 km²; delivered + unobserved = extent exactly |
| Dashboard | in-process `discover_m63_artifacts()` | `state=valid`, `usable=true`, **36 checks, 0 failed** |
| Syntax | `ast.parse` over 137 project modules | 0 errors |
| Whitespace | `git diff --check` (tracked) and output-inspected `git diff --no-index --check` (all changed files) | clean |
| Diff hygiene | `git diff --diff-filter=DR --name-status` | empty — no deletions or renames |
| Formatting | `.venv/bin/python -m black --version` / `-m ruff --version` | **UNAVAILABLE** — `No module named black` / `no module named ruff`. No formatting result is claimed and no formatter was installed |

The 18 skips are environmental (absent external Kuro Siwo root, no CUDA, absent
git-ignored fitted artifacts, one ungenerated M5 selection artifact) and are
unchanged from the pre-release baseline.

---

## 6. Known limitations and remaining blockers

1. **M6.4 is blocked** — no authorised pre-event OSM for the adopted extent;
   `data/raw/osm/` does not exist. No exposure number exists.
2. **M6.5 is blocked behind M6.4** — no road/settlement/connectivity result
   exists. Nothing was synthesised or substituted.
3. **Partial satellite observation coverage** — 36.6281 % of the corridor has no
   delivered RTC pixels; the RTC crop was cut to the replaced AOI's envelope.
   Closing it needs RTC reprocessing from raw GRDs plus a DEM that is not present,
   and processing authorisation that has not been granted.
4. **No DEM is present locally**, so no terrain-based flood-path tracing and no
   DEM-derived substitute is available.
5. **No model is trained and no inference was run.** `models/` contains only
   `.gitkeep`; the γ⁰ candidate rule is explicitly not equivalent to the M5 σ⁰
   baseline, and no accuracy or skill metric exists.
6. **FalkorDB is design-only** — nothing created, connected or populated.
7. **`provenance_status` remains `unverified`.** The extent is a human-approved
   project corridor, not an organiser-endorsed boundary; competition
   acceptability is unresolved and no endorsement is invented.
8. **Historical M6.2 evidence files** remain keyed to the replaced AOIs and are
   preserved as history; M6.16 §11 lists an optional adopted-AOI M6.2 analysis
   record as non-blocking.
9. **Formatting gate absent** — Black/Ruff are not installed in this environment.

---

## 7. Reproducing this release from a clean checkout

```bash
git clone https://github.com/Junaidrj19/multimodal-flood-mapping.git
cd multimodal-flood-mapping
git switch release/m6-flood-mapping        # or main once merged
uv sync                                    # or: python -m venv .venv && pip install -e .
.venv/bin/python -m pytest tests/ -q -p no:randomly
```

* **Code, configuration, contracts and documentation are fully reproducible from
  the repository alone.** The adopted AOI binding loads from tracked files, so
  `load_acquisition_config()` and the dashboard's AOI disclosure work on a clean
  checkout with no `data/` directory.
* **Data-dependent evidence is not reproducible from the repository alone, by
  design** (`AGENTS.md` §15): the SAFE archives, RTC crops, HydroRIVERS
  geodatabase and all `artifacts/` outputs are git-ignored. On a clean checkout
  the M6.3 tests that need those inputs skip and the dashboard reports the real
  artifacts as missing rather than substituting synthetic data; the AOI builder
  `--check` cannot run without the HydroRIVERS source and the same-track SAFE
  annotations.
* Streamlit dashboard: `.venv/bin/python -m streamlit run dashboard/app.py`
  (requires the `dashboard` extra: `streamlit`, `streamlit-folium`, `folium`).
* The synthetic demo bundle remains available for a data-free visual check and is
  always labelled `SYNTHETIC PLACEHOLDER`.

---

## 8. Merge recommendation

Merge `release/m6-flood-mapping` into `main` **without squash**, so the
milestone-level commit history is preserved, after a reviewer confirms:

1. this manifest (§3) contains no artifact, dataset, secret or unrelated file;
2. M6.4/M6.5 are still described as blocked everywhere (§6.1–6.2) and no
   infrastructure/connectivity claim is implied;
3. the partial-coverage disclosure appears verbatim on the dashboard, in the
   M6.3 artifacts and in the reports;
4. `provenance_status: unverified` and all closed acquisition gates are
   unchanged.

No merge is performed by this release preparation.
