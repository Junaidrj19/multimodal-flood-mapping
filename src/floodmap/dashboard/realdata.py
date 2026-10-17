"""Real-data dashboard source: discovery and validation of the M6.3 artifacts.

Why this module exists
----------------------
``AGENTS.md`` §16 makes the dashboard a presentation layer, but a presentation
layer is only honest if it reads *real* generated artifacts, proves they are the
artifacts they claim to be, and refuses to invent a number when they are absent.
Before M6.6 the dashboard could only show the synthetic demo bundle. This module
adds the missing half: it locates the real M6.3 change-candidate output, validates
it against its declared metadata, and exposes exactly what was found.

What it does *not* do
---------------------
It computes no flood extent, applies no threshold and derives no classification.
Reading a mask raster and counting its class values is bookkeeping, not science:
the authoritative figures are the ones M6.3 wrote, and this module's job is to
check the rasters agree with them rather than to replace them. It is deliberately
free of any ``streamlit``/``folium`` import so it can be unit-tested on a machine
without the dashboard extra installed.

Rules it enforces
---------------------
1. **Absolute paths.** Artifact locations resolve against the repository root (or
   ``FLOODMAP_PROJECT_ROOT``), never the process working directory, so a dashboard
   launched from any cwd reads the same files.
2. **Failure is explicit.** A missing, unreadable, malformed or internally
    inconsistent artifact set is reported as such, with the specific problem. The
    caller is never handed a plausible-looking default.
3. **Workspace adoption is bound.** A configured non-synthetic acquisition AOI
   must match the artifact's recorded ID and source hash. Older results cannot
   become current results by changing a label or omitting the hash.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import yaml

from ..acquisition.aoi import AreaOfInterest
from ..utils.provenance import ArtifactProvenance, ArtifactType

__all__ = [
    "AOI_BAND",
    "M63_RELATIVE_DIR",
    "M63_STAGE",
    "PROJECT_ROOT_ENV",
    "REQUIRED_AOI_SCOPE_FIELDS",
    "REQUIRED_FILES",
    "ArtifactFile",
    "Check",
    "ConfiguredAOI",
    "M63Artifacts",
    "RasterInfo",
    "ValidationState",
    "discover_m63_artifacts",
    "resolve_project_root",
]

#: Stage tag M6.3 writes into the rasters and the summary.
M63_STAGE = "m6.3-rtc-change-candidates-0.2.0"
#: Expected artifact directory, relative to the project root.
M63_RELATIVE_DIR = Path("artifacts") / "m6_3" / "rtc_change_candidates"
#: Environment override for the project root, used by tests and non-standard layouts.
PROJECT_ROOT_ENV = "FLOODMAP_PROJECT_ROOT"

#: Radiometric convention the candidate stage is allowed to consume.
RTC_CONVENTION = "gamma0_linear_power"

#: Required M6.3 files, keyed by a stable role name.
REQUIRED_FILES: Dict[str, str] = {
    "features": "rtc_change_features.tif",
    "candidate_mask": "rtc_change_candidates.tif",
    "valid_mask": "rtc_valid_mask.tif",
    "aoi_mask": "rtc_aoi_mask.tif",
    "summary": "rtc_change_summary.json",
    "features_provenance": "rtc_change_features_provenance.yaml",
    "candidate_provenance": "rtc_change_candidates_provenance.yaml",
}

#: Provenance records must declare the artifact type of the layer they describe.
EXPECTED_PROVENANCE_TYPES: Dict[str, ArtifactType] = {
    "candidate_provenance": ArtifactType.CHANGE_CANDIDATE_MASK,
    "features_provenance": ArtifactType.FEATURE_STACK,
}

#: Candidate-mask class values (packed alongside an explicit not-observed value).
CANDIDATE_NOT_OBSERVED = 255

#: Feature band names written by the M6.3 stage.
FEATURE_BANDS: Tuple[str, str] = ("vv_change_db", "vh_change_db")
CANDIDATE_BAND = "change_candidate"
VALID_BAND = "valid_change_observation"
AOI_BAND = "inside_configured_aoi"

#: AOI-scoped figures the summary must carry so the dashboard never has to divide
#: a candidate count by the raster window and call the result an AOI rate.
REQUIRED_AOI_SCOPE_FIELDS: Tuple[str, ...] = (
    "aoi_geometric_area_km2",
    "aoi_delivered_area_km2",
    "aoi_unobserved_area_km2",
    "aoi_delivered_fraction",
    "aoi_unobserved_fraction",
    "aoi_pixel_centers_in_window",
    "aoi_valid_pixels",
    "aoi_valid_area_km2",
    "candidate_pixels",
    "candidate_area_km2",
    "candidate_fraction_of_aoi_valid_pixels",
    "window_pixels_outside_aoi",
    "denominator_definitions",
    "disclosure",
)

#: Absolute/relative comparison tolerances for the raster-vs-summary cross-check.
_ABS_TOL = 1e-6
_REL_TOL = 1e-6


class ValidationState(str, Enum):
    """Outcome of validating one artifact set.

    The values are distinct on purpose so the UI can say *why* real results are
    not shown instead of collapsing every failure into "no data".
    """

    #: Files are internally consistent and any workspace AOI binding matches.
    VALID = "valid"
    #: One or more required files are absent.
    MISSING = "missing"
    #: A file exists but cannot be parsed/opened (bad JSON, unreadable raster).
    MALFORMED = "malformed"
    #: Files are readable but disagree with each other or with the summary.
    INCONSISTENT = "inconsistent"
    #: Output cannot be bound to the current configured study extent.
    STALE = "stale"


@dataclass(frozen=True)
class ArtifactFile:
    """One expected file and whether it was found."""

    key: str
    path: Path
    exists: bool


@dataclass(frozen=True)
class Check:
    """A single named consistency check."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class RasterInfo:
    """The grid metadata read back from one raster, for validation and display."""

    path: Path
    crs: Optional[str]
    transform: Tuple[float, ...]
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]
    band_names: Tuple[Optional[str], ...]
    resolution: Tuple[float, float]
    bounds: Tuple[float, float, float, float]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class ConfiguredAOI:
    """Read-only workspace extent and coverage disclosure; never a generated result.

    Only non-synthetic workspaces with ``configs/data.yaml`` opt into this
    binding. Standalone artifact directories keep their existing discovery rules.
    Coverage numbers and disclosure are copied from the geometry's provenance,
    never inferred from a raster bounding box or its valid-pixel percentage.
    """

    aoi_id: Optional[str] = None
    source_sha256: Optional[str] = None
    provenance_status: Optional[str] = None
    geojson_path: Optional[Path] = None
    bounds_4326: Optional[Tuple[float, float, float, float]] = None
    provenance_path: Optional[Path] = None
    coverage: Dict[str, Any] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)


def _configured_aoi(root: Path) -> Optional[ConfiguredAOI]:
    path = root / "configs" / "data.yaml"
    if not path.is_file():
        return None
    result = ConfiguredAOI()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw = data["acquisition"]["aoi"]
        if raw.get("is_synthetic") is True or raw.get("provenance_status") == "synthetic":
            return None
        result.aoi_id = raw.get("aoi_id")
        result.source_sha256 = raw.get("sha256")
        result.provenance_status = raw.get("provenance_status", "unverified")
        if not isinstance(result.aoi_id, str) or not result.aoi_id.strip():
            raise ValueError("acquisition.aoi.aoi_id is missing")
        digest = result.source_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("acquisition.aoi.sha256 is missing or invalid")
        geometry_path = root / raw["geojson_path"]
        aoi = AreaOfInterest.from_geojson_file(
            result.aoi_id, geometry_path, expected_sha256=digest
        )
        result.geojson_path = geometry_path
        result.bounds_4326 = tuple(aoi.bounds().as_list())
        provenance_path = (
            root / raw["provenance_path"]
            if raw.get("provenance_path")
            else geometry_path.with_suffix(".provenance.yaml")
        )
        result.provenance_path = provenance_path
        if provenance_path.is_file():
            provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
            if provenance["aoi_id"] != result.aoi_id or provenance["geometry"]["sha256"] != digest:
                raise ValueError("AOI provenance ID/hash does not match acquisition.aoi")
            coverage = provenance.get("coverage", {})
            if not isinstance(coverage, dict):
                raise ValueError("AOI provenance coverage must be a mapping")
            result.coverage = coverage
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
        result.problems.append(f"Configured AOI binding unavailable ({path}): {exc}")
    return result


def _aoi_binding_check(configured: ConfiguredAOI, summary: Dict[str, Any]) -> Check:
    """Require recorded identity AND source bytes; an ID alone is not evidence.

    Accept the explicit AOI hash at summary level or on its pair metadata. Never
    infer it from an ID suffix, a file name, or the currently configured geometry.
    """
    problems = list(configured.problems)
    pair = summary.get("pair")
    pair = pair if isinstance(pair, dict) else {}
    if pair.get("aoi_id") != configured.aoi_id:
        problems.append(
            f"artifact AOI {pair.get('aoi_id')!r} != configured AOI {configured.aoi_id!r}"
        )
    hashes = [
        record[key]
        for record in (summary, pair)
        for key in ("aoi_source_sha256", "aoi_sha256")
        if record.get(key) is not None
    ]
    if pair.get("source_sha256") is not None:
        hashes.append(pair["source_sha256"])
    if not hashes:
        problems.append("artifact AOI source hash is missing; a matching ID alone is insufficient")
    elif any(digest != configured.source_sha256 for digest in hashes):
        problems.append("artifact AOI source hash does not match configured acquisition.aoi.sha256")
    return Check(
        "configured_aoi.binding",
        not problems,
        "; ".join(problems) if problems else "artifact AOI ID and source hash match configuration",
    )


@dataclass
class M63Artifacts:
    """The result of discovering and validating the real M6.3 artifact set."""

    directory: Path
    state: ValidationState
    files: Dict[str, ArtifactFile] = field(default_factory=dict)
    rasters: Dict[str, RasterInfo] = field(default_factory=dict)
    summary: Optional[Dict[str, Any]] = None
    #: Figures recomputed from the rasters themselves.
    computed: Dict[str, Any] = field(default_factory=dict)
    #: Figures copied from ``rtc_change_summary.json``.
    recorded: Dict[str, Any] = field(default_factory=dict)
    checks: List[Check] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    provenance: Dict[str, ArtifactProvenance] = field(default_factory=dict)

    #: Extra provenance copies the stage did not relate to a dashboard layer.
    provenance_unmapped: Dict[str, str] = field(default_factory=dict)
    configured_aoi: Optional[ConfiguredAOI] = None

    @property
    def is_usable(self) -> bool:
        """True only for a consistent set bound to any current workspace AOI."""
        return self.state is ValidationState.VALID

    def check(self, name: str) -> Optional[Check]:
        for item in self.checks:
            if item.name == name:
                return item
        return None

    def failed_checks(self) -> List[Check]:
        return [item for item in self.checks if not item.ok]

    def problems_text(self) -> str:
        return "; ".join(self.problems) if self.problems else "no problems recorded"


def resolve_project_root(explicit: Optional[Path] = None) -> Path:
    """Return the repository root, independent of the process working directory.

    Precedence: an explicit path, then ``FLOODMAP_PROJECT_ROOT``, then the
    repository root inferred from this file's location (``src/floodmap/dashboard``
    -> up three levels). The inferred candidate is sanity-checked against a known
    marker so a relocated module fails loudly rather than resolving to ``/``.
    """
    if explicit is not None:
        return Path(explicit).resolve()
    env = os.environ.get(PROJECT_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[3]
    if not (candidate / "pyproject.toml").is_file():
        raise FileNotFoundError(
            "could not infer the project root from the module location "
            f"({candidate}); set {PROJECT_ROOT_ENV} or pass an explicit path"
        )
    return candidate


def _relative_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def _numbers_close(a: float, b: float) -> bool:
    return abs(a - b) <= _ABS_TOL or _relative_error(a, b) <= _REL_TOL


def _read_raster_info(path: Path) -> RasterInfo:
    with rasterio.open(path) as src:
        names = tuple((desc.strip() or None) if desc else None for desc in src.descriptions)
        return RasterInfo(
            path=path,
            crs=src.crs.to_string() if src.crs else None,
            transform=tuple(float(v) for v in src.transform[:6]),
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0] if src.dtypes else "",
            nodata=(float(src.nodata) if src.nodata is not None else None),
            band_names=names,
            resolution=(abs(float(src.transform.a)), abs(float(src.transform.e))),
            bounds=(float(src.bounds.left), float(src.bounds.bottom),
                    float(src.bounds.right), float(src.bounds.top)),
        )


def _grids_match(a: RasterInfo, b: RasterInfo) -> Tuple[bool, str]:
    """Crude-but-strict grid comparison: CRS, size and transform must agree."""
    if a.crs != b.crs:
        return False, f"CRS differs ({a.crs} vs {b.crs})"
    if a.shape != b.shape:
        return False, f"shape differs ({a.shape} vs {b.shape})"
    if not all(_numbers_close(x, y) for x, y in zip(a.transform, b.transform)):
        return False, f"transform differs ({a.transform} vs {b.transform})"
    return True, "crs, shape and transform agree"


def _counts_from_rasters(
    candidate: np.ndarray,
    valid: np.ndarray,
    aoi: np.ndarray,
    pixel_area_m2: float,
) -> Tuple[Dict[str, Any], List[str]]:
    """Recount class values from the rasters. Mechanical bookkeeping only."""
    problems: List[str] = []
    total = int(candidate.size)

    candidate_values = candidate.reshape(-1)
    counts = np.bincount(candidate_values, minlength=256).tolist()
    n_candidate = int(counts[1]) if len(counts) > 1 else 0
    n_not_candidate = int(counts[0])
    n_not_observed = int(counts[CANDIDATE_NOT_OBSERVED])

    observed_classes = n_candidate + n_not_candidate + n_not_observed
    if observed_classes != total:
        unexpected = sorted(
            int(v) for v in np.unique(candidate_values)
            if int(v) not in (0, 1, CANDIDATE_NOT_OBSERVED)
        )
        problems.append(
            f"candidate mask contains unexpected class values {unexpected}"
        )

    valid_values = valid.reshape(-1)
    n_valid = int(np.count_nonzero(valid_values == 1))
    n_invalid = int(np.count_nonzero(valid_values == 0))
    if n_valid + n_invalid != total:
        unexpected = sorted(
            int(v) for v in np.unique(valid_values) if int(v) not in (0, 1)
        )
        problems.append(f"valid mask contains unexpected values {unexpected}")

    aoi_values = aoi.reshape(-1)
    n_in_aoi = int(np.count_nonzero(aoi_values == 1))
    n_out_aoi = int(np.count_nonzero(aoi_values == 0))
    if n_in_aoi + n_out_aoi != total:
        unexpected = sorted(int(v) for v in np.unique(aoi_values) if int(v) not in (0, 1))
        problems.append(f"AOI mask contains unexpected values {unexpected}")
    if not n_in_aoi:
        problems.append(
            "AOI mask marks no pixel inside the configured extent; the products "
            "cannot describe the study area"
        )

    # A candidate pixel must be a valid observation; if not, the validity mask and
    # the candidate mask describe different things and nothing can be trusted.
    leaked = int(np.count_nonzero((candidate_values == 1) & (valid_values != 1)))
    if leaked:
        problems.append(
            f"{leaked} candidate pixels fall outside the valid-data mask"
        )

    # And it must be inside the study extent, otherwise out-of-extent pixels would
    # be reported as findings about the corridor.
    out_of_extent = int(np.count_nonzero((candidate_values == 1) & (aoi_values != 1)))
    if out_of_extent:
        problems.append(
            f"{out_of_extent} candidate pixels fall outside the configured AOI mask"
        )
    valid_out_of_extent = int(np.count_nonzero((valid_values == 1) & (aoi_values != 1)))
    if valid_out_of_extent:
        problems.append(
            f"{valid_out_of_extent} valid pixels fall outside the configured AOI mask; "
            "the validity layer is not AOI-scoped"
        )

    computed = {
        "total_pixels": total,
        "valid_pixels": n_valid,
        "invalid_pixels": n_invalid,
        "valid_fraction": float(n_valid / total) if total else 0.0,
        "candidate_pixels": n_candidate,
        "not_candidate_pixels": n_not_candidate,
        "not_observed_pixels": n_not_observed,
        "candidate_fraction_of_valid": (float(n_candidate / n_valid) if n_valid else 0.0),
        "candidate_fraction_of_grid": (float(n_candidate / total) if total else 0.0),
        "candidate_grid_area_km2": float(n_candidate * pixel_area_m2 / 1_000_000.0),
        "pixel_area_m2": float(pixel_area_m2),
        # Area of the whole assembled grid (bounding window), so the candidate
        # area can be read against its denominator rather than in isolation.
        "grid_area_km2": float(total * pixel_area_m2 / 1_000_000.0),
        # --- AOI scope, recomputed from the masks rather than trusted ---------
        "aoi_pixel_centers_in_window": n_in_aoi,
        "window_pixels_outside_aoi": n_out_aoi,
        "aoi_valid_pixels": n_valid,
        "aoi_valid_area_km2": float(n_valid * pixel_area_m2 / 1_000_000.0),
        "candidate_area_km2": float(n_candidate * pixel_area_m2 / 1_000_000.0),
        "candidate_fraction_of_aoi_valid_pixels": (
            float(n_candidate / n_valid) if n_valid else 0.0
        ),
    }
    return computed, problems


#: AOI-scope fields that must agree with the numbers recomputed from the rasters.
_AOI_SCOPE_COMPARABLE: Tuple[str, ...] = (
    "aoi_pixel_centers_in_window",
    "window_pixels_outside_aoi",
    "aoi_valid_pixels",
    "aoi_valid_area_km2",
    "candidate_pixels",
    "candidate_area_km2",
    "candidate_fraction_of_aoi_valid_pixels",
)


def _aoi_scope_checks(
    summary: Dict[str, Any], computed: Dict[str, Any]
) -> Tuple[List[Check], List[str]]:
    """Require a complete, self-consistent AOI scope block.

    The dashboard must never derive an AOI rate itself: it would have only the
    raster window as a denominator, and the window is 92% outside the extent.
    So the stage has to supply the AOI-scoped figures, and they have to match
    what the masks actually contain.
    """
    checks: List[Check] = []
    problems: List[str] = []
    scope = summary.get("aoi_scope")
    if not isinstance(scope, dict):
        checks.append(Check("summary.aoi_scope", False, "absent or not an object"))
        problems.append(
            "summary has no aoi_scope block; AOI-relative figures and their "
            "denominators are unavailable and nothing may be shown as an AOI result"
        )
        return checks, problems

    missing = [key for key in REQUIRED_AOI_SCOPE_FIELDS if scope.get(key) is None]
    ok = not missing
    checks.append(
        Check("summary.aoi_scope", ok, "complete" if ok else f"missing {missing}")
    )
    if not ok:
        problems.append(f"summary.aoi_scope is missing required fields: {missing}")

    for key in _AOI_SCOPE_COMPARABLE:
        if scope.get(key) is None:
            continue
        try:
            rec = float(scope[key])
            comp = float(computed[key])
        except (TypeError, ValueError, KeyError):
            checks.append(Check(f"aoi_scope.{key}", False, "not numeric"))
            problems.append(f"summary.aoi_scope.{key} is not numeric")
            continue
        agree = _numbers_close(rec, comp)
        detail = f"summary={rec!r} recomputed={comp!r}"
        checks.append(Check(f"aoi_scope.{key}", agree, detail))
        if not agree:
            problems.append(f"summary.aoi_scope.{key} disagrees with the rasters ({detail})")

    # The full extent must never be presented as observed. Delivered and valid
    # area are both strictly smaller than the AOI here, and silently equating
    # them is the specific error the coverage decision was written to prevent.
    try:
        full = float(scope["aoi_geometric_area_km2"])
        delivered = float(scope["aoi_delivered_area_km2"])
        valid_area = float(scope["aoi_valid_area_km2"])
    except (KeyError, TypeError, ValueError):
        return checks, problems
    for label, observed in (("delivered", delivered), ("valid", valid_area)):
        sane = observed <= full * (1.0 + 1e-6)
        checks.append(
            Check(
                f"aoi_scope.{label}_within_extent",
                sane,
                f"{label}={observed!r} full_extent={full!r}",
            )
        )
        if not sane:
            problems.append(
                f"summary.aoi_scope reports {label} area {observed} km2 larger than the "
                f"full AOI area {full} km2; the denominators are inconsistent"
            )
    return checks, problems


def _cross_check_summary(
    recorded: Dict[str, Any], computed: Dict[str, Any]
) -> Tuple[List[Check], List[str]]:
    """Compare summary figures against the values recomputed from the rasters."""
    checks: List[Check] = []
    problems: List[str] = []

    comparable = [
        "total_pixels",
        "valid_pixels",
        "invalid_pixels",
        "candidate_pixels",
        "candidate_fraction_of_valid",
        "candidate_fraction_of_grid",
        "candidate_grid_area_km2",
    ]
    for key in comparable:
        if key not in recorded or recorded[key] is None:
            checks.append(Check(f"summary.{key}", False, "absent from summary"))
            problems.append(f"summary is missing {key}")
            continue
        try:
            rec = float(recorded[key])
            comp = float(computed[key])
        except (TypeError, ValueError):
            checks.append(Check(f"summary.{key}", False, "not numeric"))
            problems.append(f"summary.{key} is not numeric")
            continue
        ok = _numbers_close(rec, comp)
        detail = f"summary={rec!r} recomputed={comp!r}"
        checks.append(Check(f"summary.{key}", ok, detail))
        if not ok:
            problems.append(f"summary.{key} disagrees with the rasters ({detail})")
    return checks, problems


def _load_provenance_records(
    files: Dict[str, ArtifactFile],
) -> Tuple[Dict[str, ArtifactProvenance], List[Check], List[str]]:
    records: Dict[str, ArtifactProvenance] = {}
    checks: List[Check] = []
    problems: List[str] = []
    for role, expected in EXPECTED_PROVENANCE_TYPES.items():
        file = files.get(role)
        if file is None or not file.exists:
            checks.append(Check(f"provenance.{role}", False, "file absent"))
            problems.append(f"missing provenance file for {role}")
            continue
        try:
            text = file.path.read_text(encoding="utf-8")
            record = ArtifactProvenance.from_yaml(text)
        except Exception as exc:  # unreadable, malformed YAML, wrong schema
            checks.append(Check(f"provenance.{role}", False, f"unreadable: {exc}"))
            problems.append(f"provenance file {file.path} is unreadable: {exc}")
            continue
        if record.artifact_type is not expected:
            checks.append(
                Check(
                    f"provenance.{role}",
                    False,
                    f"artifact_type {record.artifact_type.value} != {expected.value}",
                )
            )
            problems.append(
                f"provenance {file.path} declares artifact_type "
                f"{record.artifact_type.value}, expected {expected.value}"
            )
            continue
        records[role] = record
        checks.append(Check(f"provenance.{role}", True, record.artifact_type.value))
    return records, checks, problems


def discover_m63_artifacts(
    project_root: Optional[Path] = None,
    *,
    directory: Optional[Path] = None,
) -> M63Artifacts:
    """Locate and validate the M6.3 candidate artifacts.

    Paths resolve against the project root, never the working directory. Every
    failure mode (absent, unreadable, malformed, inconsistent) is recorded on the
    returned object with a machine-readable :class:`ValidationState` and a human
    readable ``problems`` list. In a non-synthetic workspace, results must also
    record the configured acquisition AOI ID and source SHA-256. Nothing is
    fabricated or relabelled when that binding is absent or stale.
    """
    root = resolve_project_root(project_root)
    configured_aoi = _configured_aoi(root)
    directory = Path(directory) if directory is not None else root / M63_RELATIVE_DIR

    files: Dict[str, ArtifactFile] = {
        key: ArtifactFile(key=key, path=directory / name, exists=(directory / name).is_file())
        for key, name in REQUIRED_FILES.items()
    }
    problems: List[str] = []
    checks: List[Check] = []

    missing = [key for key, f in files.items() if not f.exists]
    if missing:
        problems.append(
            f"missing required M6.3 files in {directory}: "
            + ", ".join(f"{key} ({REQUIRED_FILES[key]})" for key in missing)
        )
        return M63Artifacts(
            directory=directory,
            state=ValidationState.MISSING,
            files=files,
            checks=[Check("files.present", False, f"missing {missing}")],
            problems=problems,
            configured_aoi=configured_aoi,
        )

    # From here every required file exists; any failure is malformed/inconsistent.
    summary: Optional[Dict[str, Any]] = None
    try:
        summary = json.loads(files["summary"].path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("summary JSON is not an object")
        checks.append(Check("summary.parse", True, "parsed"))
    except Exception as exc:
        problems.append(f"rtc_change_summary.json is malformed: {exc}")
        return M63Artifacts(
            directory=directory,
            state=ValidationState.MALFORMED,
            files=files,
            checks=[Check("summary.parse", False, str(exc))],
            problems=problems,
            configured_aoi=configured_aoi,
        )

    if configured_aoi is not None:
        binding = _aoi_binding_check(configured_aoi, summary)
        checks.append(binding)
        if not binding.ok:
            return M63Artifacts(
                directory=directory,
                state=ValidationState.STALE,
                files=files,
                summary=summary,
                checks=checks,
                problems=["Stale or unbound M6.3 results: " + binding.detail],
                configured_aoi=configured_aoi,
            )

    rasters: Dict[str, RasterInfo] = {}
    for role in ("candidate_mask", "valid_mask", "aoi_mask", "features"):
        try:
            rasters[role] = _read_raster_info(files[role].path)
            checks.append(Check(f"raster.{role}.readable", True, "opened"))
        except Exception as exc:
            problems.append(f"{REQUIRED_FILES[role]} is unreadable: {exc}")
            return M63Artifacts(
                directory=directory,
                state=ValidationState.MALFORMED,
                files=files,
                summary=summary,
                rasters=rasters,
                checks=checks + [Check(f"raster.{role}.readable", False, str(exc))],
                problems=problems,
                configured_aoi=configured_aoi,
            )

    # --- band-name and stage-tag expectations --------------------------------
    candidate = rasters["candidate_mask"]
    valid = rasters["valid_mask"]
    features = rasters["features"]

    def _band_check(role: str, expected: Tuple[Optional[str], ...]) -> None:
        info = rasters[role]
        ok = info.count == len(expected) and info.band_names == expected
        checks.append(
            Check(
                f"bands.{role}",
                ok,
                f"names={info.band_names} expected={expected}",
            )
        )
        if not ok:
            problems.append(
                f"{REQUIRED_FILES[role]} band names {info.band_names} "
                f"!= expected {expected}"
            )

    _band_check("candidate_mask", (CANDIDATE_BAND,))
    _band_check("valid_mask", (VALID_BAND,))
    _band_check("aoi_mask", (AOI_BAND,))
    _band_check("features", FEATURE_BANDS)

    # --- grid alignment across the four rasters ------------------------------
    for left, right in (
        ("candidate_mask", "valid_mask"),
        ("candidate_mask", "aoi_mask"),
        ("candidate_mask", "features"),
    ):
        ok, detail = _grids_match(rasters[left], rasters[right])
        checks.append(Check(f"grid.{left}~{right}", ok, detail))
        if not ok:
            problems.append(f"{left} and {right} grids do not match: {detail}")

    # --- declared radiometric convention and stage --------------------------
    recorded_convention = summary.get("radiometric_convention")
    conv_ok = recorded_convention == RTC_CONVENTION
    checks.append(
        Check(
            "summary.radiometric_convention",
            conv_ok,
            f"summary={recorded_convention!r} expected={RTC_CONVENTION!r}",
        )
    )
    if not conv_ok:
        problems.append(
            f"summary radiometric_convention {recorded_convention!r} is not "
            f"{RTC_CONVENTION!r}; the M6.3 candidate feature is gamma0-specific"
        )

    recorded_stage = summary.get("stage")
    stage_ok = recorded_stage == M63_STAGE
    checks.append(
        Check("summary.stage", stage_ok, f"summary={recorded_stage!r} expected={M63_STAGE!r}")
    )
    if not stage_ok:
        problems.append(
            f"summary stage {recorded_stage!r} is not {M63_STAGE!r}; unknown provenance"
        )

    # --- recount class values from the rasters -------------------------------
    recomputed: Dict[str, Any] = {}
    try:
        with rasterio.open(files["candidate_mask"].path) as src:
            candidate_array = src.read(1)
        with rasterio.open(files["valid_mask"].path) as src:
            valid_array = src.read(1)
        with rasterio.open(files["aoi_mask"].path) as src:
            aoi_array = src.read(1)
        pixel_area = abs(candidate.transform[0]) * abs(candidate.transform[4])
        recomputed, count_problems = _counts_from_rasters(
            candidate_array, valid_array, aoi_array, pixel_area
        )
        problems.extend(count_problems)
    except Exception as exc:
        problems.append(f"could not recompute class counts: {exc}")

    recorded: Dict[str, Any] = {}
    if recomputed:
        cross_checks, cross_problems = _cross_check_summary(summary, recomputed)
        checks.extend(cross_checks)
        problems.extend(cross_problems)
        scope_checks, scope_problems = _aoi_scope_checks(summary, recomputed)
        checks.extend(scope_checks)
        problems.extend(scope_problems)
        recorded = {key: summary.get(key) for key in recomputed if key in summary}

    # --- provenance ---------------------------------------------------------
    provenance, prov_checks, prov_problems = _load_provenance_records(files)
    checks.extend(prov_checks)
    problems.extend(prov_problems)

    if configured_aoi is not None:
        for role, record in provenance.items():
            ok = record.aoi == configured_aoi.aoi_id
            checks.append(Check(f"configured_aoi.{role}", ok, f"artifact AOI: {record.aoi}"))
            if not ok:
                problems.append(f"{role} AOI does not match configured acquisition.aoi.aoi_id")

    state = ValidationState.VALID if not problems else ValidationState.INCONSISTENT
    return M63Artifacts(
        directory=directory,
        state=state,
        files=files,
        rasters=rasters,
        summary=summary,
        computed=recomputed,
        recorded=recorded,
        checks=checks,
        problems=problems,
        provenance=provenance,
        configured_aoi=configured_aoi,
    )
