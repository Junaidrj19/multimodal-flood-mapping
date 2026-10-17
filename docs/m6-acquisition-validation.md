# M6.1 — Acquisition selection validation

**Scope.** The mandatory gates that stand between a catalogue result and a scene
this project is willing to treat as an observation: AOI footprint coverage,
Sentinel-1 polarisation, Sentinel-2 pair compatibility, and the policy for a
Sentinel-1 same-track failure.

**Status.** Implemented and tested. Acquisition of products remains blocked on
the AOI geometry — its *provenance*, not its availability — and not on this code
(see §7).

**Why this document exists.** The previous implementation could report a *usable*
pre/post pair having verified almost nothing: the AOI footprint check was a
constant `True`, Sentinel-1 polarisations were never inspected, and tile /
processing-level agreement was recorded only as prose. A reviewer could not tell
from a manifest what had actually been checked. Every gate below is therefore
both enforced and recorded, and the record is what the report cites.

---

## 1. What makes a pair usable

`SelectionOutcome.is_usable` is derived from a recorded validation result
(`pair_validation.passed`), **not** from the presence of two scene references.
This matters because `is_usable` gates the download loop
(`scripts/acquire.py`) and the provenance `BeforeAfterPair`
(`manifest._selected_pair`). A gate that fails cannot produce a usable outcome,
and a hand-built outcome without a validation record is not usable either.

Gates, in the order they are applied:

| # | Gate | Applies to | Failure reason(s) |
|---|---|---|---|
| 1 | Acquired inside the configured window | S1, S2 | `outside_search_window` |
| 2 | Scene cloud reported, and within the configured limit | S2 | `cloud_unknown`, `cloud_above_configured_limit` |
| 3 | Processing-level tier (only when a preference is set) | S2 | `processing_level_excluded` |
| 4 | **AOI footprint coverage** | S1, S2 | `footprint_does_not_intersect_aoi`, `footprint_coverage_below_minimum`, `footprint_unverifiable` |
| 5 | **Required polarisations present** | S1 | `polarisation_mismatch`, `polarisation_unverifiable` |
| 6 | Relative orbit + pass direction agree | S1 | `orbit_unknown`, `orbit_mismatch`, `orbit_direction_mismatch` |
| 7 | **MGRS tile compatibility** (pair-level) | S2 | `tile_mismatch`, `tile_unverifiable` |
| 8 | Mixed processing levels permissible (pair-level) | S2 | `mixed_processing_levels_not_allowed` |

Gates 7 and 8 are inherently pair-level and are evaluated over candidate
*pairs*, not per scene. Every discarded candidate carries a reason record
(`rejected_candidates[]`, with `scene_id`, `side`, `reason`, `rationale` and the
measured `coverage_fraction` where one exists), and the manifest also folds the
rationales into `limitations`.

---

## 2. Coverage semantics

Implemented in `src/floodmap/acquisition/coverage.py`.

```
coverage_fraction = area( footprint ∩ AOI ) / area( AOI )
```

**Measure.** Both geometries are scaled into a local plane about the AOI
centroid latitude — the equirectangular approximation
`(lon · cos(lat0), lat)` — and the ratio is taken there. Two consequences:

* the number is a locally correct **metric** ratio, not a ratio of
  degree-squared areas, which is not an area at all;
* both geometries are scaled by the same factor, so the ratio does not inherit
  the cos(lat) distortion that a raw degree-area ratio would carry.

**Approximation and its bound.** Scaling by `cos(lat0)` ignores the change in
meridional convergence across the AOI. The residual error on the ratio grows
with `Δlat² · tan(lat0)`. For the Trishuli corridor (`Δlat` ≈ 0.44°, centred at
28.0°N) it is under ~0.3%, which is immaterial against a threshold stated to one
significant figure. An AOI spanning more than about a degree, or one at very
high latitude (> 89°), must be measured in a projected CRS instead; a polar AOI
is reported `unverifiable` rather than silently producing nonsense.

**Statuses.** `full_coverage`, `partial_coverage`, `no_coverage`,
`unverifiable`. "Full" is `fraction ≥ 1 - 1e-9`, a floating-point tolerance at
coordinate magnitudes of ~85°, not a scientific one.

**Threshold.** `minimum_aoi_coverage`, configured per sensor in
`configs/data.yaml`, stated as a required fraction in `(0, 1]`.

* `0.9` (both sensors). Justification: a scene clipping any part of the corridor
  yields a flood extent bounded by the image edge rather than by the flood, so
  even ~10% of missing coverage is a real result-affecting limitation. 0.9
  confines that to the tile seam. The value is deliberately the same as the
  existing `min_aoi_coverage` on the Sentinel-1 RTC route
  (`configs/baseline.yaml`), so the project's two acquisition paths do not
  disagree about what "covering" means.
* `null` means **intersection is required and the measured fraction is recorded
  but not thresholded**. No threshold is invented, in line with the project's
  rule that an unmade scientific decision must not look like a made one.

**Zero-area AOI.** A degenerate AOI (a collinear ring) raises `ValueError`
rather than rejecting every candidate. A zero-area AOI is a caller error; if it
were reported as "no usable scenes" it would be indistinguishable from a
data-availability finding.

---

## 3. CRS assumptions

* The AOI is `EPSG:4326` only, enforced at construction
  (`aoi.AreaOfInterest._crs_is_supported`) because the provider spatial filter
  is defined in it.
* `SceneFootprint` now carries an explicit `crs`. The CDSE OData `GeoFootprint`
  is the geography type and is WGS84 by definition, and the query itself is
  issued as `SRID=4326`, so `providers/cdse.py` records `EPSG:4326` at parse
  time.
* The coverage gate accepts `EPSG:4326` and **rejects anything else** as
  `footprint_unverifiable`. There is no implicit reprojection: comparing
  coordinates in two CRSs without an explicit, documented transform is exactly
  the silent unit error `AGENTS.md` §7 exists to prevent.
* Footprint **holes are not transmitted** to the provider query (only the
  exterior ring is sent, which can only widen the candidate set), but they *are*
  honoured when measuring coverage, because ignoring a hole would overstate how
  much of the study area a scene covers.

---

## 4. AOI polygon versus bounding box

These are **not interchangeable**, and the difference is scientific, not
cosmetic.

* The case-study corridor is ~133 km², roughly 1 km wide and ~66 km long. Its
  bounding box (`lon 85.05814–85.38789`, `lat 27.84397–28.28743`) is an
  envelope **11.98× larger in area** (1,592.91 km² against 132.93 km²: 91.7% of
  the envelope is not corridor). Measured, not estimated —
  `docs/dataset-registry.md` §1.6.2.
* A query issued with the bbox asks "intersects this rectangle". A scene that
  clips a corner of the envelope but touches none of the corridor satisfies it.
* Coverage measured against the bbox measures the wrong quantity. Since
  `coverage_fraction = area(footprint ∩ AOI) / area(AOI)`, substituting the
  envelope moves the 0.9 gate itself, and it moves in **both** directions.
  Measured with the project's own `evaluate_footprint_coverage` over 16,488
  tile-scale footprints (construction stated in `docs/dataset-registry.md`
  §1.6.2): 157 were **envelope-accepted but corridor-rejected** and 205 the other
  way round. The worst case is a 1.0° square rotated 20° that scores 0.9144
  against the envelope (accepted) and 0.7712 against the corridor (rejected) —
  the envelope would admit a product leaving **23%** of the real corridor
  outside the image. An envelope AOI therefore does not merely over-include
  area, and the disagreement is not an argument: it is a regression test
  (`test_envelope_aoi_admits_a_product_that_leaves_the_corridor_partly_outside`).

Therefore:

* `configs/data.yaml → acquisition.aoi` resolves to the located corridor
  polygon (hash-pinned, `provenance_status: unverified`) rather than to the
  envelope, so the coverage gate measures the study area and not a rectangle
  around it. It is configured on an implementing-agent decision and is **not**
  an operator approval (§7).
* `--aoi-bbox` remains available, and remains a mistake in production: it is
  marked synthetic only when the AOI id contains "synthetic", so an envelope AOI
  is **not** automatically flagged, and its coverage numbers are over-inclusive
  in one direction and over-strict in the other. Any run using it must say so.
* The polygon's *bytes* are present and pinned; its *provenance* is not
  established, which is a different deficiency and the one that still blocks
  acquisition (§7).

---

## 5. Tile compatibility and processing-level policy (Sentinel-2)

**Tile compatibility.** `require_same_tile: true`. Pre and post must report the
same MGRS tile. Matching tiles do not *prove* coverage — the footprint gate
does that, and `tile_consistency` says so in its own note — but differing tiles
are positive evidence the two scenes were not produced on the same grid, so they
are not a like-for-like comparison. A candidate whose tile is absent (catalogues
do omit it) is rejected as `tile_unverifiable` rather than assumed compatible.
Setting `require_same_tile: false` accepts the pair and still records the
mismatch, so the decision is visible either way.

**Processing levels.** `allow_mixed_processing_levels: true`, meaning mixed
levels are *permitted but deprioritised and disclosed*:

* same-level pairs are ranked above mixed-level pairs;
* when a mixed-level pair is selected, `pair_validation.mixed_processing_levels`
  is `true`, the rationale says so, and the manifest adds a limitation stating
  that surface (BOA) and top-of-atmosphere (TOA) reflectance are not
  interchangeable in a spectral index.

With `false`, a mixed-level pair is ineligible and the outcome is
`no_compatible_pair` with reason `mixed_processing_levels_not_allowed`. This is
a real incompatibility and not a stylistic preference: NDWI/MNDWI computed
across BOA and TOA carry a systematic offset, so the comparison is not
like-for-like (`docs/m4-architecture-decision.md`).

**Pair ranking.** Sides are ranked independently and then reconciled as pairs,
ranked by their **worse** side first. A pair is limited by its weaker
observation, so the best individual scene on each side does not automatically
form the selected pair.

---

## 6. Failure behaviour

The governing rule is: **unverifiable is a rejection, not an assumption.**

| Condition | Behaviour |
|---|---|
| A footprint is absent, `Unknown`, empty, malformed, zero-area, self-intersecting, or in an unsupported CRS | Rejected, `footprint_unverifiable`. Never accepted. |
| A footprint does not intersect the AOI | Rejected, `footprint_does_not_intersect_aoi`, measured fraction `0.0`. |
| A footprint intersects but covers less than the threshold | Rejected, `footprint_coverage_below_minimum`, with the measured fraction. |
| Sentinel-1 polarisations are unreported | Rejected, `polarisation_unverifiable`. A single-channel product would leave a model input empty rather than fail. |
| Sentinel-1 is missing a required channel | Rejected, `polarisation_mismatch`. |
| `aoi` is not passed to a selector | `TypeError`. There is deliberately no path that selects without measuring coverage. |
| The AOI itself is degenerate | `ValueError`, loudly. Not "no usable scenes". |
| No compatible pair exists, though both sides had eligible candidates | Status `no_compatible_pair` — distinct from `no_usable_scene`, so an operator can tell a pairing problem from a data problem. |
| No Sentinel-1 same-track pair | `on_no_same_track_pair` is now **enforced**: `fail` raises `NoSameTrackPairError` and writes no manifest; `warn_and_continue` warns on stderr and records; `record` records. |

**Invalid geometry is not repaired.** `make_valid` is deliberately not called: a
self-intersecting ring means the footprint is not what the provider claims, and
repairing it would substitute our guess for their record. One exception is
documented and deliberate — a `MultiPolygon` whose parts meet along an edge is
reported invalid by GEOS under the OGC ring rules although its meaning ("the
union of these parts") is unambiguous, so parts are validated individually and
then combined. A part that is itself invalid is still rejected.

**Nothing fails open on a missing AOI.** `aoi` is a required keyword argument on
both selectors.

---

## 7. Blocking prerequisite: the AOI geometry

**The authoritative Trishuli corridor polygon is still not established.** What
changed in M6.2 is *where the candidate is*, not whether it is trustworthy.

**Located (M6.2).** The `aoi.geojson` candidate is not in this repository and
never has been, but it was found in a sibling checkout of this same project at
`<sibling-checkout>/aoi.geojson`. It is
byte-identical to the artifact §1.6 measured (sha256 `9035eeee…9362610`, 79,228
bytes, same mtime), untracked there as well, and absent from every git object in
either checkout. Its geometry was re-verified independently here: valid, simple,
closed, 1,515 vertices, 132.9284 km² in EPSG:32645 against a declared 132.93.
Full record: `docs/dataset-registry.md` §1.6.1.

**Configured provisionally (M6.2).** The located bytes are referenced by
`configs/data.yaml` (`configs/aoi/trishuli-corridor-UNVERIFIED.geojson`, pinned
by the sha256 recorded in §1.6, `provenance_status: unverified`, and the open
questions recorded as five `provenance_gaps` in the configuration and in the
manifest limitations). This is an **implementing-agent decision, not an
operator approval**: it exists so metadata-only analysis runs against the real
corridor instead of the envelope, and it is refused for anything stronger.
`AreaOfInterest.authorises_download` is `false`, and `scripts/acquire.py`
refuses `--download` on that basis, so no raster — the input every downstream
flood claim is computed from — can be fetched against an unestablished extent.
Full record: `docs/dataset-registry.md` §1.6.3.

**Still blocked, for the same reason as before.** Finding a file does not supply
its provenance. The registry status is **unchanged: `UNVERIFIED (provenance
unknown)`**, not adopted as the authoritative study extent, because the
producer, the centreline it was buffered from (and that centreline's OSM
snapshot date), the 1 km buffer justification and the "upperstream" extent are
all still unrecorded. A second untracked copy on the same machine is not
corroboration, and no documented retrieval source or regeneration procedure
exists in either repository.

Acquisition therefore remains **blocked on an operator decision**. Exactly one of
these is required:

1. **Approve the located geometry's provenance.** Record who produced it, from
   which centreline (and, if OSM-derived, the centreline's snapshot date,
   `AGENTS.md` §3), and why a 1 km buffer; then set `provenance_status: verified`
   and commit the file. This is the only option that unblocks downloads.
2. **Replace it with an authoritative corridor polygon** as a GeoJSON `Polygon`,
   `Feature` or single-feature `FeatureCollection` in `EPSG:4326`, with its
   provenance, re-pin the sha256 and re-run the selection.
3. **Explicitly accept the bounding box as an envelope-only approximation**, in
   writing, recorded in the manifest and the report — acknowledging §4's
   quantified 11.98× over-inclusion and its consequence for selection and for
   optical valid-pixel statistics.

Silence, a successful dry run, or the existence of a previous run is **not**
approval of any of these. Until one is taken, this milestone is
`NOT READY — BLOCKED ON AOI APPROVAL`, and a catalogue requery may only proceed
as metadata-only analysis whose result states the AOI caveat.

**Ingestion defect fixed (M6.2).** `--aoi-geojson` accepted a bare `Polygon`
geometry or a `Feature`, but **not** a `FeatureCollection` — which is the
`ogr2ogr` default output and the exact shape of the one candidate artifact. The
CLI therefore exited 2 with "only GeoJSON Polygon is supported, got
'FeatureCollection'" on the only AOI file this project has. `_geometry_from_geojson`
now unwraps a single-feature collection and still refuses a multi-feature one
rather than adopting its first member or a union, since either would choose an
analysis extent nobody specified.

---

## 8. Manifest and download status

`status` and `download_status` are deliberately separate fields.

* `status` (`ManifestStatus`) answers *"did selection find a usable pair?"* — a
  finding about the world and about our query.
* `download_status` (`DownloadSummary`) answers *"did we obtain the bytes?"* —
  `not_attempted` | `complete` | `partial` | `failed`.

Collapsing them allowed a manifest to read `complete` while every download
failed. A download failure now downgrades `download_status` and never rewrites
`status`.

When a fetch is authorised, the manifest is written **before** the download loop
with status `selection_complete_download_pending`, then rewritten with per-scene
download outcomes. An interrupted run therefore leaves a truthful record of what
was selected and attempted, under a status that cannot be mistaken for a
finished acquisition.

Each selection also records `selection_policy` — AOI id/WKT/CRS, coverage
threshold, required polarisations, tile and processing-level policy, strategy,
max cloud, and the **search window actually used**. The config ships those
windows as `null`, so without the manifest value the run that admitted a pair
could not be reconstructed.

---

## 9. Verification

Measured on 2026-10-15 in this checkout (`.venv`, Python 3.12); exit statuses
captured with `set -o pipefail` and reported as observed.

```
.venv/bin/python -m pytest -q tests/test_acquisition_validation.py
# 63 passed, exit 0

.venv/bin/python -m pytest -q tests/test_acquisition_m1.py
# 24 passed, exit 0

.venv/bin/python -m pytest -q tests/test_contract_freeze.py
# 139 passed, exit 0

.venv/bin/python -m pytest -q --ignore=tests/test_dashboard_app.py
# 990 passed, 18 skipped, exit 0

.venv/bin/python -m pytest -q tests/test_dashboard_app.py
# 1 error during collection, exit 2
```

`tests/test_dashboard_app.py` cannot be collected without the `dashboard` extra
(`streamlit`), so it is excluded above; the failure is
`ModuleNotFoundError: No module named 'streamlit'` at import of
`streamlit.testing.v1`, a missing dependency rather than a logic defect. The 18
skips are the M5/dataset prerequisites that are absent in this environment
(external Kuro Siwo root, fitted statistics artifact, an ungenerated selection
artifact), not skipped acquisition tests. The 990 figure applies **only** with
that module excluded; it is not the whole suite.

A dry run exercises the CLI without network I/O; manifests for verification are
written to a temporary path, never into `artifacts/`. This one takes the AOI
from the configuration rather than from a flag, which is what makes it a check
on the configured study extent:

```
.venv/bin/python scripts/acquire.py --dry-run \
    --before-days 12 --after-days 12 --manifest "$TMPDIR/manifest.json"
# exit 0; status dry_run; download_not_attempted
# aoi.provenance_status = unverified; aoi.source_sha256 = 9035eeee…9362610
# limitations include the UNVERIFIED-provenance line, the sha256 pin and the
# five recorded provenance gaps
```

The two download gates were exercised as well, both refusing with exit 2 and no
manifest written:

```
.venv/bin/python scripts/acquire.py --download --before-days 12 --after-days 12
# exit 2: "--download was passed but acquisition.download.enabled is false"
```

With `download.enabled` forced true (a config copy, not the committed file), the
provenance gate refuses in turn: `_authorise_download` raises `AcquisitionError`
with `provenance_status 'unverified' rather than 'verified'` and lists the open
questions. That second gate is asserted in
`test_unverified_configured_aoi_is_disclosed_and_cannot_authorise_a_download`.

**A dry run proves only that queries can be built and a manifest written.** It
issues no catalogue request, so it is not evidence that any product is
discoverable, downloadable, or has usable pixels.

`tests/test_acquisition_validation.py` covers: full / partial / non-intersecting
coverage, threshold boundaries, `MultiPolygon` (touching and disjoint parts), and
missing / unknown / empty / null / malformed / zero-area / self-intersecting /
unsupported-CRS geometry; latitude-corrected measurement; orbit and polarisation
gates; pair-level tile and processing-level decisions; rejection provenance;
`is_usable` consistency; the same-track failure policy; download summary and
manifest status separation; `typing.get_type_hints(SelectionOutcome)`; the AOI
GeoJSON container rules (§7) — single-feature `FeatureCollection` accepted,
multi-feature refused rather than reduced, `MultiPolygon` still refused — and
the configured-extent contract: the sha256 pin refuses a swapped file, the
provisional AOI's caveat reaches the manifest while a download is refused, and
the pinned corridor **artifact itself** is re-derived as geometry — valid,
simple, closed, 1,515 coordinate pairs, bounds inside the documented envelope,
and 132.9284 km² in EPSG:32645 against a declared 132.93 — so §1.6's record and
the file cannot drift apart silently. One further test pins §4's central claim:
the envelope-vs-corridor disagreement is asserted on a fixed footprint rather
than quoted, so "the bounding box can wrongly admit a product" is checked by the
suite and not merely stated in prose.
