"""M6.5 — road-network disruption and settlement connectivity under
hypothetical closure of M6.4 candidate-exposed roads.

Scientific responsibility
-------------------------
Turn per-road candidate exposure (M6.4) into a graph-level statement about
which settlements could lose their modelled road route to a reference hub
(``architecture.md`` §8/§9, ``AGENTS.md`` §9).

This is a scenario, not an observation
--------------------------------------
M6.4 measured how much of each pre-event OSM road lies on M6.3 Sentinel-1 RTC
darkening-candidate pixels. That measurement does **not** establish that such a
road is impassable: a mask intersecting a road is evidence of spatial exposure,
not of physical destruction (``AGENTS.md`` §8). The M6.4 contract carries no
passability field, so this stage does not "apply observed damage". It states a
closure rule, applies it, and compares the result against the untouched
baseline graph:

====================================== ======================================
Scenario                               Pieces removed from the graph
====================================== ======================================
``baseline``                           none
``candidate_closure``                  closure units whose candidate-
                                       intersecting length reaches the
                                       threshold
``candidate_and_unobserved_closure``   the above, plus units whose
                                       not-observed length reaches it
====================================== ======================================

Every reachability number here is conditional on that rule. A settlement
reported ``SCENARIO_UNREACHABLE`` is a settlement that *would* lose its modelled
road route *if* candidate-intersecting roads were impassable. It is not a
finding that anybody is cut off.

Closure unit
------------
The threshold acts on one of two units, declared in
``configs/candidate_disruption.yaml``:

* ``road`` — the unit M6.4 actually measured. A whole OSM way closes when its
  own ``candidate_m`` reaches the threshold. Needs no raster, but it is coarse
  and pessimistic: 10 m of intersection closes the entire way.
* ``piece`` — each graph piece is re-measured against the M6.3 candidate raster
  with the same kernel M6.4 used (:func:`...candidate_exposure.assess_lines`)
  and closes on its own ``candidate_m``. Exact at piece resolution, and
  requires the raster.

Per-piece quantities are never obtained by dividing a road's ``candidate_m``
between its pieces. Where along a way the intersection falls is not recorded,
and inventing a distribution would manufacture evidence.

Missing data
------------
``not_observed_m`` is **not** unaffected road length. It drives the pessimistic
scenario and is reported under ``coverage_uncertainty``, never folded into any
"open" or "unaffected" total. ``NO_CANDIDATE_INTERSECTION`` length is likewise
reported as "the rule did not fire here", not as intact road.

Reuse
-----
The graph, the removal rule, multi-source Dijkstra, the status logic and route
tracing are the tested implementations in :mod:`floodmap.network.graph` and
:mod:`floodmap.network.connectivity`. Only the assembly is new, because
``assess_connectivity`` names its outputs for a predicted flood
(``*_km_after``, "after the flood"), and an M6.5 artifact must not describe a
radar-change scenario in flood vocabulary. Statuses are translated the same way
M6.4 translates the exposure kernel's.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS

from floodmap.infrastructure.candidate_exposure import (
    CANDIDATE_INTERSECTING,
    CANDIDATE_STATUSES,
    GEOJSON_DEFAULT_CRS,
    NO_CANDIDATE_INTERSECTION,
    OUTSIDE_CANDIDATE_GRID,
    UNOBSERVED,
    IntersectionRules,
    assess_lines,
    scene_pair_from_summary,
)
from floodmap.infrastructure.candidate_mask import CandidateMask
from floodmap.infrastructure.osm import OSM_ATTRIBUTION, OSM_LICENSE, parse_timestamp
from floodmap.network.connectivity import (
    CONNECTED,
    INSUFFICIENT_DATA as KERNEL_INSUFFICIENT_DATA,
    POTENTIALLY_CUT_OFF,
    POTENTIALLY_DISRUPTED,
    REASON_NO_BASELINE_ROUTE,
    REASON_NO_HUBS,
    REASON_NO_ROAD,
    UNKNOWN as KERNEL_UNKNOWN,
    Reach,
    classify,
    reach_from_hubs,
    removed_pieces,
    route_geometry,
)
from floodmap.network.graph import RoadNetwork, Snap, build_road_network
from floodmap.utils.config import load_config
from floodmap.utils.provenance import (
    ArtifactProvenance,
    ArtifactType,
    ProductionInput,
)

CANDIDATE_DISRUPTION_VERSION = "m6.5-candidate-network-disruption-0.1.0"

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

#: The untouched graph. Every comparison is against this.
BASELINE = "baseline"
#: Closure units whose candidate-intersecting length reaches the threshold are
#: removed. The question this stage exists to answer.
CANDIDATE_CLOSURE = "candidate_closure"
#: The above plus units whose not-observed length reaches the threshold, so a
#: verdict that survives both is not an artefact of incomplete observation.
CANDIDATE_AND_UNOBSERVED_CLOSURE = "candidate_and_unobserved_closure"

SCENARIOS: Tuple[str, ...] = (BASELINE, CANDIDATE_CLOSURE, CANDIDATE_AND_UNOBSERVED_CLOSURE)

SCENARIO_MEANINGS: Dict[str, str] = {
    BASELINE: (
        "the road graph as mapped in pre-event OpenStreetMap, with nothing removed; "
        "the reference every scenario number is compared against"
    ),
    CANDIDATE_CLOSURE: (
        "hypothetical: every closure unit whose M6.3 candidate-intersecting length "
        "reaches the threshold is treated as impassable. Candidate intersection is "
        "radar-change evidence, not observed impassability, so this is a stated "
        "what-if and not a measurement"
    ),
    CANDIDATE_AND_UNOBSERVED_CLOSURE: (
        "hypothetical: as candidate_closure, and additionally every closure unit "
        "whose not-observed length reaches the threshold. A unit closed only here is "
        "a limit of observation coverage, not modelled disruption"
    ),
}

# ---------------------------------------------------------------------------
# Status vocabulary
#
# Named for the scenario, not for a real-world consequence. The kernel's
# statuses are reused through a translation table rather than restated, so the
# tested classification logic stays single-sourced.
# ---------------------------------------------------------------------------

#: Reaches a hub in both closure scenarios, by a route no longer than
#: ``detour_ratio`` times the baseline route.
SCENARIO_REACHABLE = "SCENARIO_REACHABLE"
#: Still reaches a hub under candidate closure, but the shortest route grows by
#: at least ``detour_ratio``.
SCENARIO_DETOURED = "SCENARIO_DETOURED"
#: Had a baseline route and reaches no hub of the group under candidate closure.
SCENARIO_UNREACHABLE = "SCENARIO_UNREACHABLE"
#: Reaches a hub under candidate closure but not when not-observed units are
#: also closed. An observation-coverage limit, not a modelled loss of access.
UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED = "UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED"
#: No mapped road nearby, no baseline route to any hub of the group, or no hub
#: of the group near a mapped road. The scenario cannot say anything.
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

SETTLEMENT_STATUSES: Tuple[str, ...] = (
    SCENARIO_REACHABLE,
    SCENARIO_DETOURED,
    SCENARIO_UNREACHABLE,
    UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED,
    INSUFFICIENT_DATA,
)

STATUS_MEANINGS: Dict[str, str] = {
    SCENARIO_REACHABLE: (
        "reaches a hub of this group in both closure scenarios, by a route no longer "
        "than detour_ratio times the baseline route"
    ),
    SCENARIO_DETOURED: (
        "still reaches a hub under candidate closure, but the shortest modelled route "
        "grows by at least detour_ratio"
    ),
    SCENARIO_UNREACHABLE: (
        "had a baseline road route and reaches no hub of this group once "
        "candidate-intersecting closure units are removed. This is a modelled "
        "consequence of the stated closure rule, NOT confirmed isolation of any "
        "settlement or person (architecture.md §18)"
    ),
    UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED: (
        "reachable under candidate closure and unreachable when not-observed units "
        "are also closed; the outcome is undetermined because of observation "
        "coverage, not because of modelled disruption"
    ),
    INSUFFICIENT_DATA: (
        "no mapped road within snap_max_m, no baseline route to any hub of this "
        "group, or no hub of this group near a mapped road. Many Himalayan hamlets "
        "have no mapped road at all, so this is never reported as cut off"
    ),
}

#: Translation of the connectivity kernel's flood-framed vocabulary.
_STATUS_FROM_KERNEL: Dict[str, str] = {
    CONNECTED: SCENARIO_REACHABLE,
    POTENTIALLY_DISRUPTED: SCENARIO_DETOURED,
    POTENTIALLY_CUT_OFF: SCENARIO_UNREACHABLE,
    KERNEL_UNKNOWN: UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED,
    KERNEL_INSUFFICIENT_DATA: INSUFFICIENT_DATA,
}

#: Scenario-framed reasons. The kernel's UNKNOWN reason names a flood; an M6.5
#: artifact must describe the closure scenario it actually ran.
REASON_NO_MAPPED_ROAD = REASON_NO_ROAD
REASON_NO_BASELINE_ROUTE_SCENARIO = "no road route to any hub of this group in the baseline graph"
REASON_NO_HUB_ON_NETWORK = REASON_NO_HUBS
REASON_UNOBSERVED_ONLY = (
    "unreachable only when not-observed closure units are also closed; an observation-"
    "coverage limit rather than modelled disruption"
)

_REASON_FROM_KERNEL: Dict[str, str] = {
    REASON_NO_ROAD: REASON_NO_MAPPED_ROAD,
    REASON_NO_BASELINE_ROUTE: REASON_NO_BASELINE_ROUTE_SCENARIO,
    REASON_NO_HUBS: REASON_NO_HUB_ON_NETWORK,
    "cut off only if unobserved pixels hide flood": REASON_UNOBSERVED_ONLY,
}

# ---------------------------------------------------------------------------
# Per-piece closure labels
# ---------------------------------------------------------------------------

#: Removed in both closure scenarios.
CLOSED_IN_CANDIDATE_SCENARIO = "CLOSED_IN_CANDIDATE_SCENARIO"
#: Removed only when not-observed units are closed too.
CLOSED_ONLY_IF_UNOBSERVED_CLOSED = "CLOSED_ONLY_IF_UNOBSERVED_CLOSED"
#: Left in the graph by both scenarios. Not a statement that the road is intact.
OPEN_IN_BOTH_SCENARIOS = "OPEN_IN_BOTH_SCENARIOS"

CLOSURE_LABELS: Tuple[str, ...] = (
    CLOSED_IN_CANDIDATE_SCENARIO,
    CLOSED_ONLY_IF_UNOBSERVED_CLOSED,
    OPEN_IN_BOTH_SCENARIOS,
)

#: Closure units supported by :class:`ClosureRules`.
CLOSURE_UNITS: Tuple[str, ...] = ("road", "piece")

#: Columns the M6.4 road artifact must carry for this stage to run.
REQUIRED_ROAD_COLUMNS: Tuple[str, ...] = (
    "osm_id",
    "candidate_status",
    "length_m",
    "candidate_m",
    "not_observed_m",
)
#: Columns the M6.4 settlement artifact must carry.
REQUIRED_SETTLEMENT_COLUMNS: Tuple[str, ...] = ("osm_id", "candidate_status")

#: Column names that only exist on a flood-mask exposure artifact. Their
#: presence means the wrong input was supplied, and silently accepting it would
#: relabel a predicted flood length as a radar-change measurement.
FORBIDDEN_ROAD_COLUMNS: Tuple[str, ...] = ("flooded_m", "exposure_status", "permanent_water_m")

#: Hub layer columns required for hub reachability.
REQUIRED_HUB_COLUMNS: Tuple[str, ...] = ("hub_group",)


class CandidateDisruptionError(ValueError):
    """The M6.5 inputs or configuration are not usable."""


class MissingNetworkInputs(CandidateDisruptionError):
    """Required M6.5 inputs are not present on disk.

    Raised rather than worked around. Synthesising a road network, a settlement
    or a hub would make every reachability verdict unattributable.
    """

    def __init__(self, message: str, *, missing: Sequence[str], searched: Path) -> None:
        super().__init__(message)
        self.missing = list(missing)
        self.searched = Path(searched)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureRules:
    """Explicit M6.5 graph and closure assumptions.

    Loaded from ``configs/candidate_disruption.yaml`` so the report can state
    them and sensitivity analysis can vary them (``AGENTS.md`` §9, §14).
    """

    snap_max_m: float = 500.0
    closure_unit: str = "road"
    closure_min_candidate_m: float = 10.0
    sensitivity_closure_m: Tuple[float, ...] = (10.0, 30.0, 100.0)
    detour_ratio: float = 1.5
    close_outside_grid_roads: bool = False

    def __post_init__(self) -> None:
        if self.snap_max_m <= 0:
            raise CandidateDisruptionError("snap_max_m must be positive")
        if self.closure_unit not in CLOSURE_UNITS:
            raise CandidateDisruptionError(
                f"closure_unit must be one of {CLOSURE_UNITS}, got {self.closure_unit!r}"
            )
        if self.closure_min_candidate_m <= 0:
            raise CandidateDisruptionError("closure_min_candidate_m must be positive")
        if not self.sensitivity_closure_m:
            raise CandidateDisruptionError(
                "sensitivity_closure_m must list at least one threshold; a single-threshold "
                "conclusion cannot be shown to be threshold-independent (AGENTS.md §9)"
            )
        if any(t <= 0 for t in self.sensitivity_closure_m):
            raise CandidateDisruptionError("sensitivity_closure_m thresholds must be positive")
        if self.detour_ratio < 1.0:
            raise CandidateDisruptionError("detour_ratio must be at least 1.0")

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "ClosureRules":
        block = (config or load_config("candidate_disruption"))["network"]
        return cls(
            snap_max_m=float(block["snap_max_m"]),
            closure_unit=str(block["closure_unit"]),
            closure_min_candidate_m=float(block["closure_min_candidate_m"]),
            sensitivity_closure_m=tuple(float(t) for t in block["sensitivity_closure_m"]),
            detour_ratio=float(block["detour_ratio"]),
            close_outside_grid_roads=bool(block.get("close_outside_grid_roads", False)),
        )

    def as_intersection_rules(self) -> IntersectionRules:
        """M6.4 kernel rules for re-measuring pieces under ``closure_unit=piece``."""
        return IntersectionRules(
            line_sample_step_m=self.closure_min_candidate_m / 2.0,
            line_min_candidate_m=self.closure_min_candidate_m,
        )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class NetworkInputs:
    """M6.4 road and settlement artifacts, plus an optional hub layer."""

    directory: Path
    roads: gpd.GeoDataFrame
    settlements: gpd.GeoDataFrame
    hubs: Optional[gpd.GeoDataFrame] = None
    hubs_path: Optional[Path] = None
    exposure_summary: Optional[Dict[str, Any]] = None
    osm_provenance: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def hub_reachability_possible(self) -> bool:
        return self.hubs is not None and len(self.hubs) > 0

    def attribution(self) -> List[str]:
        """Attribution recorded by the upstream stages, with OSM guaranteed."""
        values: List[str] = []
        for source in (self.exposure_summary or {}, self.osm_provenance or {}):
            recorded = (source.get("osm") or {}).get("attribution") or source.get("attribution")
            if isinstance(recorded, str):
                values.append(recorded)
            elif isinstance(recorded, Sequence):
                values.extend(str(v) for v in recorded if v)
        if not any(OSM_ATTRIBUTION in value for value in values):
            values.append(OSM_ATTRIBUTION)
        return list(dict.fromkeys(values))

    def osm_snapshot(self) -> Optional[str]:
        """OSM snapshot recorded by M6.4, falling back to the hub sidecar."""
        recorded = ((self.exposure_summary or {}).get("osm") or {}).get("snapshot")
        if recorded:
            return str(recorded)
        sidecar = (self.osm_provenance or {}).get("snapshot")
        return str(sidecar) if sidecar else None


def _read_vector(path: Path, label: str, notes: List[str]) -> gpd.GeoDataFrame:
    try:
        frame = gpd.read_file(path)
    except Exception as exc:
        raise CandidateDisruptionError(f"could not read {label} from {path}: {exc}") from exc
    if frame.crs is None:
        frame = frame.set_crs(GEOJSON_DEFAULT_CRS)
        notes.append(
            f"{path.name} declared no CRS; RFC 7946 CRS84 ({GEOJSON_DEFAULT_CRS}) was assumed"
        )
    return frame


def _require_columns(frame: gpd.GeoDataFrame, required: Sequence[str], label: str) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise CandidateDisruptionError(
            f"{label} is missing required column(s) {missing}; expected the M6.4 "
            f"artifact contract {list(required)}"
        )


def _reject_flood_vocabulary(frame: gpd.GeoDataFrame, label: str) -> None:
    present = [name for name in FORBIDDEN_ROAD_COLUMNS if name in frame.columns]
    if present:
        raise CandidateDisruptionError(
            f"{label} carries flood-mask exposure column(s) {present}. That is the "
            "configs/impact.yaml exposure artifact, whose quantities are predicted "
            "inundation lengths; this stage consumes M6.3 radar-change candidate "
            "lengths and must not relabel one as the other."
        )


def _validate_statuses(frame: gpd.GeoDataFrame, column: str, label: str) -> None:
    values = frame[column].dropna().unique().tolist() if len(frame) else []
    unknown = sorted({str(v) for v in values} - set(CANDIDATE_STATUSES))
    if unknown:
        raise CandidateDisruptionError(
            f"{label} contains unrecognised {column} value(s) {unknown}; the M6.4 "
            f"vocabulary is {list(CANDIDATE_STATUSES)}"
        )


def load_candidate_exposure_roads(path: Path | str, notes: Optional[List[str]] = None):
    """Read and validate ``candidate_exposure_roads.geojson``."""
    path = Path(path)
    notes = [] if notes is None else notes
    if not path.is_file():
        raise MissingNetworkInputs(
            f"M6.4 road exposure artifact not found: {path}",
            missing=[path.name],
            searched=path.parent,
        )
    frame = _read_vector(path, "candidate exposure roads", notes)
    _reject_flood_vocabulary(frame, str(path))
    # Checked before the column contract: an empty GeoJSON FeatureCollection
    # carries no properties at all, so its schema cannot be verified and the
    # column error would hide the real problem.
    if len(frame) == 0:
        raise CandidateDisruptionError(
            f"{path} contains no road features. A connectivity verdict from an empty "
            "network would report every settlement as having no mapped road, which is a "
            "statement about the input rather than about the area. Establish why the road "
            "layer is empty before running this stage."
        )
    _require_columns(frame, REQUIRED_ROAD_COLUMNS, str(path))
    _validate_statuses(frame, "candidate_status", str(path))
    for optional in ("highway", "name", "tags", "not_candidate_m", "outside_grid_m"):
        if optional not in frame.columns:
            frame[optional] = None
            notes.append(
                f"{path.name} carries no {optional!r} column; it is reported as null rather "
                "than inferred"
            )
    return frame, notes


def load_candidate_exposure_settlements(path: Path | str, notes: Optional[List[str]] = None):
    """Read and validate ``candidate_exposure_settlements.geojson``."""
    path = Path(path)
    notes = [] if notes is None else notes
    if not path.is_file():
        raise MissingNetworkInputs(
            f"M6.4 settlement exposure artifact not found: {path}",
            missing=[path.name],
            searched=path.parent,
        )
    frame = _read_vector(path, "candidate exposure settlements", notes)
    if len(frame) == 0:
        # An empty FeatureCollection carries no properties, so the column
        # contract cannot be checked. Road disruption is still reportable; the
        # absence of settlements to assess is recorded, not inferred away.
        notes.append(
            f"{path.name} contains no settlement features; no settlement reachability "
            "verdict is produced and none is implied"
        )
        for column in (*REQUIRED_SETTLEMENT_COLUMNS, "name", "place"):
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame, notes
    _require_columns(frame, REQUIRED_SETTLEMENT_COLUMNS, str(path))
    _validate_statuses(frame, "candidate_status", str(path))
    for optional in ("name", "place"):
        if optional not in frame.columns:
            frame[optional] = None
            notes.append(f"{path.name} carries no {optional!r} column; settlements are unnamed")
    return frame, notes


def load_hub_layer(
    path: Path | str,
    notes: Optional[List[str]] = None,
    *,
    sidecar: Optional[Mapping[str, Any]] = None,
    event_date: Optional[str] = None,
    require_pre_event_snapshot: bool = True,
) -> gpd.GeoDataFrame:
    """Read the pre-event OSM hub layer that M6.4 did not assess.

    Hubs are reference destinations, not exposure subjects, so M6.4 carries no
    hub artifact. They are read from the same pre-event OSM directory, under the
    same temporal rule (``AGENTS.md`` §3): a post-event snapshot is refused.
    """
    path = Path(path)
    notes = [] if notes is None else notes
    if not path.is_file():
        raise MissingNetworkInputs(
            f"pre-event OSM hub layer not found: {path}",
            missing=[path.name],
            searched=path.parent,
        )
    recorded_snapshot = (sidecar or {}).get("snapshot")
    if require_pre_event_snapshot and recorded_snapshot and event_date:
        snapshot = parse_timestamp(str(recorded_snapshot))
        if snapshot >= parse_timestamp(f"{event_date}T00:00:00Z"):
            raise CandidateDisruptionError(
                f"the hub layer's OSM snapshot {snapshot.isoformat()} is not strictly before "
                f"the event day {event_date}; post-event OSM is validation-only "
                "(AGENTS.md §3)"
            )
    elif require_pre_event_snapshot and event_date and not recorded_snapshot:
        notes.append(
            f"{path.name} has no recorded OSM snapshot beside it; the pre-event rule could "
            "not be checked for the hub layer"
        )

    frame = _read_vector(path, "osm hubs", notes)
    _require_columns(frame, REQUIRED_HUB_COLUMNS, str(path))
    if "name" not in frame.columns:
        frame["name"] = None
        notes.append(f"{path.name} carries no 'name' column; hubs are reported unnamed")
    return frame


def load_network_inputs(
    exposure_dir: Path | str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    osm_dir: Optional[Path | str] = None,
    require_hubs: Optional[bool] = None,
) -> NetworkInputs:
    """Read the M6.4 artifacts, and the hub layer when one is available.

    No retrieval: every input must already exist on disk, for the same reason
    M6.4 performs none (the AOI is ``UNVERIFIED`` and
    ``AreaOfInterest.authorises_download`` is ``false``).
    """
    config = config or load_config("candidate_disruption")
    inputs = config["inputs"]
    osm_config = inputs.get("osm") or {}
    if require_hubs is None:
        require_hubs = bool(osm_config.get("require_hubs", False))

    exposure_dir = Path(exposure_dir)
    if not exposure_dir.is_dir():
        raise MissingNetworkInputs(
            f"M6.4 candidate-exposure artifact directory does not exist: {exposure_dir}",
            missing=[str(inputs["roads_file"]), str(inputs["settlements_file"])],
            searched=exposure_dir,
        )

    notes: List[str] = []
    roads_path = exposure_dir / str(inputs["roads_file"])
    settlements_path = exposure_dir / str(inputs["settlements_file"])
    missing = [p.name for p in (roads_path, settlements_path) if not p.is_file()]
    if missing:
        raise MissingNetworkInputs(
            "required M6.4 candidate-exposure artifacts are not present on disk: "
            + ", ".join(missing),
            missing=missing,
            searched=exposure_dir,
        )

    summary_path = exposure_dir / str(inputs["summary_file"])
    exposure_summary = _load_exposure_summary(
        summary_path, expected_stage=str(inputs["candidate_exposure_stage"]), notes=notes
    )
    event_date = _event_date(exposure_summary)

    roads, notes = load_candidate_exposure_roads(roads_path, notes)
    settlements, notes = load_candidate_exposure_settlements(settlements_path, notes)

    hubs: Optional[gpd.GeoDataFrame] = None
    hubs_path: Optional[Path] = None
    osm_provenance: Optional[Dict[str, Any]] = None
    if osm_dir is not None:
        osm_dir = Path(osm_dir)
        hubs_path = osm_dir / str(osm_config.get("hubs_file", "osm_hubs.geojson"))
        sidecar_path = osm_dir / "osm_provenance.json"
        if sidecar_path.is_file():
            try:
                loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise CandidateDisruptionError(f"could not read {sidecar_path}: {exc}") from exc
            if isinstance(loaded, Mapping):
                osm_provenance = dict(loaded)
        try:
            hubs = load_hub_layer(
                hubs_path,
                notes,
                sidecar=osm_provenance,
                event_date=event_date,
                require_pre_event_snapshot=bool(
                    osm_config.get("require_pre_event_snapshot", True)
                ),
            )
        except MissingNetworkInputs:
            if require_hubs:
                raise
            hubs = None
            notes.append(
                f"hub layer {hubs_path} is absent; hub reachability is recorded as not "
                "assessed rather than replaced by a different question"
            )
    elif require_hubs:
        raise MissingNetworkInputs(
            "hub reachability is required by configuration but no --osm-dir was supplied",
            missing=[str(osm_config.get("hubs_file", "osm_hubs.geojson"))],
            searched=exposure_dir,
        )
    else:
        notes.append(
            "no pre-event OSM hub layer was supplied; hub reachability is not assessed and "
            "only road disruption and connected-component change are reported"
        )

    return NetworkInputs(
        directory=exposure_dir,
        roads=roads,
        settlements=settlements,
        hubs=hubs,
        hubs_path=hubs_path if hubs is not None else None,
        exposure_summary=exposure_summary,
        osm_provenance=osm_provenance,
        notes=notes,
    )


def _load_exposure_summary(
    path: Path, *, expected_stage: str, notes: List[str]
) -> Optional[Dict[str, Any]]:
    """Read ``candidate_exposure_summary.json`` and verify the stage tag."""
    if not path.is_file():
        notes.append(
            f"{path.name} is absent; the analysis CRS, source Sentinel-1 pair and OSM "
            "snapshot cannot be carried forward from M6.4 and must be supplied explicitly"
        )
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateDisruptionError(f"could not read {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise CandidateDisruptionError(f"{path} must contain a JSON object")
    stage = loaded.get("stage")
    if stage != expected_stage:
        raise CandidateDisruptionError(
            f"{path} declares stage {stage!r} but this stage consumes {expected_stage!r}; "
            "an artifact from another stage is refused rather than reinterpreted"
        )
    return dict(loaded)


def _event_date(exposure_summary: Optional[Mapping[str, Any]]) -> Optional[str]:
    pair = (exposure_summary or {}).get("source_pair") or {}
    value = pair.get("event_date")
    return str(value) if value else None


def analysis_crs(
    exposure_summary: Optional[Mapping[str, Any]], override: Optional[str] = None
) -> CRS:
    """Metric CRS the graph is built in.

    Defaults to the candidate mask's CRS recorded by M6.4, so lengths here and
    the ``candidate_m`` they are compared against were measured on the same
    grid. A geographic CRS is refused: piece lengths and thresholds are metres
    (``AGENTS.md`` §7).
    """
    if override is not None:
        crs = CRS.from_user_input(override)
    else:
        recorded = (
            ((exposure_summary or {}).get("data_coverage") or {}).get("candidate_mask") or {}
        ).get("crs")
        if not recorded:
            raise CandidateDisruptionError(
                "the analysis CRS could not be read from the M6.4 summary "
                "(data_coverage.candidate_mask.crs). Pass it explicitly rather than "
                "defaulting to one, because a wrong projection silently changes every "
                "length and threshold."
            )
        crs = CRS.from_user_input(str(recorded))
    if crs.is_geographic:
        raise CandidateDisruptionError(
            f"the analysis CRS {crs.to_string()} is geographic; road lengths and the "
            "closure threshold are metres, so a projected metric CRS is required"
        )
    return crs


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_candidate_road_network(
    roads: gpd.GeoDataFrame,
    crs: CRS,
    *,
    settlements: Optional[gpd.GeoDataFrame] = None,
    hubs: Optional[gpd.GeoDataFrame] = None,
    snap_max_m: float = 500.0,
) -> Tuple[RoadNetwork, Snap, Snap]:
    """Build the road graph, attaching settlements and hubs to it.

    Delegates to :func:`floodmap.network.graph.build_road_network` unchanged, so
    piece segmentation, junction identity and snapping semantics are those
    already documented and tested. The only addition is that settlements and
    hubs arrive as M6.4/OSM frames rather than as prepared point series.
    """
    if "osm_id" not in roads.columns or "highway" not in roads.columns:
        raise CandidateDisruptionError(
            "the road graph builder reads 'osm_id' and 'highway'; both must be present "
            "even if 'highway' is null"
        )
    point_sets: List[gpd.GeoSeries] = []
    for frame in (settlements, hubs):
        if frame is None or len(frame) == 0:
            point_sets.append(gpd.GeoSeries([], crs=crs))
        else:
            point_sets.append(frame.to_crs(crs).representative_point())
    network, (settlement_snap, hub_snap) = build_road_network(
        roads, crs, point_sets, snap_max_m
    )
    return network, settlement_snap, hub_snap


def scenario_graph(graph: nx.MultiGraph, closed: FrozenSet) -> nx.MultiGraph:
    """The graph with ``closed`` piece ids removed.

    Routing hides closed edges through the Dijkstra weight function, which never
    mutates the graph. Component counting needs a real subgraph, so this builds
    one. Nodes are kept even when isolated, so a settlement's node still exists
    after every road around it closes.
    """
    scenario = nx.MultiGraph()
    scenario.add_nodes_from(graph.nodes)
    scenario.add_edges_from(
        (u, v, key, dict(attrs))
        for u, v, key, attrs in graph.edges(keys=True, data=True)
        if key not in closed
    )
    return scenario


def component_labels(graph: nx.MultiGraph) -> Dict[int, int]:
    """Node to component id, ordered by component size then lowest node id.

    Deterministic, so a run is reproducible. The id is a **label within one
    scenario**, not an identity across scenarios: the largest component always
    takes id 0, so an unchanged id does not mean an unchanged neighbourhood.
    Use :func:`component_sizes` to compare scenarios.
    """
    components = sorted(
        (sorted(int(n) for n in component) for component in nx.connected_components(graph)),
        key=lambda nodes: (-len(nodes), nodes[0] if nodes else 0),
    )
    return {node: index for index, nodes in enumerate(components) for node in nodes}


def component_sizes(graph: nx.MultiGraph) -> Dict[int, int]:
    """Node to the number of nodes in its component.

    Comparable between scenarios, unlike a component label: a settlement whose
    component shrinks has lost reachable network, whatever the component is
    called.
    """
    return {
        int(node): len(component)
        for component in nx.connected_components(graph)
        for node in component
    }


# ---------------------------------------------------------------------------
# Closure rule
# ---------------------------------------------------------------------------


def closure_table(
    network: RoadNetwork,
    roads: gpd.GeoDataFrame,
    rules: ClosureRules,
    *,
    mask: Optional[CandidateMask] = None,
) -> pd.DataFrame:
    """Per-piece table the closure threshold is applied to.

    Columns: ``piece_id``, ``piece_length_m``, ``osm_id``, ``highway``,
    ``source_candidate_status``, ``unit_length_m``, ``candidate_m``,
    ``not_observed_m``.

    ``unit_length_m`` is the length the threshold is compared against, which is
    what distinguishes the two closure units:

    * ``road`` — the parent way's own total length and measured quantities are
      repeated onto each of its pieces, so the whole way closes or none of it
      does. No per-piece quantity is invented.
    * ``piece`` — each piece is re-measured against the candidate raster, so
      ``unit_length_m`` is the piece's own length.
    """
    pieces = network.pieces
    if rules.closure_unit == "piece":
        if mask is None:
            raise CandidateDisruptionError(
                "closure_unit='piece' re-measures every graph piece against the M6.3 "
                "candidate raster, so the raster must be supplied. Per-piece quantities "
                "are never derived by dividing a road's candidate_m between its pieces: "
                "where along the way the intersection falls is not recorded."
            )
        measured = assess_lines(
            pieces[["piece_id", "geometry"]], mask, rules.as_intersection_rules()
        )
        table = pd.DataFrame(
            {
                "piece_id": pieces["piece_id"].to_numpy(),
                "piece_length_m": pieces["length_m"].to_numpy(dtype=float),
                "osm_id": pieces["osm_id"].to_numpy(),
                "highway": pieces["highway"].to_numpy(),
                "unit_length_m": pieces["length_m"].to_numpy(dtype=float),
                "candidate_m": measured["candidate_m"].to_numpy(dtype=float),
                "not_observed_m": measured["not_observed_m"].to_numpy(dtype=float),
                "source_candidate_status": measured["candidate_status"].to_numpy(),
            }
        )
        return table

    per_road = _road_quantities(roads)
    osm_ids = pd.Series(pieces["osm_id"].to_numpy(), name="osm_id")
    joined = per_road.reindex(osm_ids.to_numpy())
    unmatched = int(joined["length_m"].isna().sum())
    if unmatched:
        raise CandidateDisruptionError(
            f"{unmatched} graph piece(s) have an osm_id that is absent from the M6.4 road "
            "measurements. The graph and the exposure table must describe the same roads."
        )
    return pd.DataFrame(
        {
            "piece_id": pieces["piece_id"].to_numpy(),
            "piece_length_m": pieces["length_m"].to_numpy(dtype=float),
            "osm_id": osm_ids.to_numpy(),
            "highway": pieces["highway"].to_numpy(),
            "unit_length_m": joined["length_m"].to_numpy(dtype=float),
            "candidate_m": joined["candidate_m"].to_numpy(dtype=float),
            "not_observed_m": joined["not_observed_m"].to_numpy(dtype=float),
            "source_candidate_status": joined["candidate_status"].to_numpy(),
        }
    )


def _road_quantities(roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per ``osm_id``, summing duplicate rows for the same way.

    An ohsome response can carry a way more than once (for example as separate
    geometry parts). Summing keeps the measured total intact; taking the first
    row would silently drop measured candidate length.
    """
    numeric = ["length_m", "candidate_m", "not_observed_m"]
    frame = pd.DataFrame(
        {
            "osm_id": roads["osm_id"].to_numpy(),
            **{
                column: pd.to_numeric(roads[column], errors="coerce").fillna(0.0).to_numpy()
                for column in numeric
            },
            "candidate_status": roads["candidate_status"].to_numpy(),
        }
    )
    grouped = frame.groupby("osm_id", dropna=False)
    out = grouped[numeric].sum()
    # Worst-status wins when a way appears more than once: an intersection
    # measured on any part of it is an intersection on the way.
    priority = {
        CANDIDATE_INTERSECTING: 0,
        UNOBSERVED: 1,
        NO_CANDIDATE_INTERSECTION: 2,
        OUTSIDE_CANDIDATE_GRID: 3,
    }
    out["candidate_status"] = grouped["candidate_status"].agg(
        lambda values: min(
            (str(v) for v in values.dropna()), key=lambda v: priority.get(v, 99), default=None
        )
    )
    return out


def closed_pieces(
    table: pd.DataFrame, threshold_m: float, *, close_unobserved: bool, rules: ClosureRules
) -> FrozenSet:
    """Piece ids removed under one threshold.

    The threshold rule itself is :func:`floodmap.network.connectivity.removed_pieces`
    — a unit closes when its candidate-intersecting length reaches the
    threshold, or all of it when the unit is shorter. Reused rather than
    restated so both stages cannot drift apart.
    """
    kernel = pd.DataFrame(
        {
            "piece_id": table["piece_id"].to_numpy(),
            "length_m": table["unit_length_m"].to_numpy(dtype=float),
            "flooded_m": table["candidate_m"].to_numpy(dtype=float),
            "unknown_m": table["not_observed_m"].to_numpy(dtype=float),
        }
    )
    if not rules.close_outside_grid_roads:
        outside = table["source_candidate_status"].to_numpy() == OUTSIDE_CANDIDATE_GRID
        if outside.any():
            kernel.loc[outside, ["flooded_m", "unknown_m"]] = 0.0
    return removed_pieces(kernel, threshold_m, close_unobserved)


def closure_status(
    table: pd.DataFrame, candidate_closed: FrozenSet, pessimistic_closed: FrozenSet
) -> np.ndarray:
    """Per-piece closure label for the two reported scenarios."""
    piece_ids = table["piece_id"].to_numpy()
    if piece_ids.size == 0:
        return np.array([], dtype=object)
    return np.select(
        [
            np.isin(piece_ids, list(candidate_closed)),
            np.isin(piece_ids, list(pessimistic_closed)),
        ],
        [CLOSED_IN_CANDIDATE_SCENARIO, CLOSED_ONLY_IF_UNOBSERVED_CLOSED],
        default=OPEN_IN_BOTH_SCENARIOS,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _text(value: Any) -> Optional[str]:
    """Missing names arrive as NaN, which is not valid JSON."""
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _km(reach: Reach, node: int) -> Optional[float]:
    if node < 0 or not reach.reachable(node):
        return None
    return round(reach.distance[node] / 1000.0, 3)


def _hub_label(hub_names: Mapping[int, Optional[str]], reach: Reach, node: int) -> Optional[str]:
    if node < 0 or not reach.reachable(node):
        return None
    return hub_names.get(reach.path_from_hub(node)[0])


def _closed_km(table: pd.DataFrame, closed: FrozenSet) -> float:
    selected = table[table["piece_id"].isin(closed)]
    return round(float(selected["piece_length_m"].sum()) / 1000.0, 3)


def _closed_km_by_highway(table: pd.DataFrame, closed: FrozenSet) -> Dict[str, float]:
    selected = table[table["piece_id"].isin(closed)].copy()
    if selected.empty:
        return {}
    selected["highway"] = [
        _text(value) or "unrecorded" for value in selected["highway"].to_numpy()
    ]
    by_class = selected.groupby("highway")["piece_length_m"].sum() / 1000.0
    return {
        str(k): round(float(v), 3) for k, v in by_class.sort_values(ascending=False).items()
    }


def _component_sizes_km(
    graph: nx.MultiGraph, table: pd.DataFrame, closed: FrozenSet, top: int = 5
) -> List[Dict[str, Any]]:
    """Largest components of a scenario graph, by node count and road length."""
    scenario = scenario_graph(graph, closed)
    lengths = dict(
        zip(table["piece_id"].to_numpy().tolist(), table["piece_length_m"].to_numpy().tolist())
    )
    out: List[Dict[str, Any]] = []
    for nodes in nx.connected_components(scenario):
        subgraph = scenario.subgraph(nodes)
        km = sum(lengths.get(key, 0.0) for _, _, key in subgraph.edges(keys=True)) / 1000.0
        out.append({"nodes": subgraph.number_of_nodes(), "road_km": round(km, 3)})
    out.sort(key=lambda entry: (-entry["road_km"], -entry["nodes"]))
    return out[:top]


def assess_candidate_disruption(
    network: RoadNetwork,
    table: pd.DataFrame,
    inputs: NetworkInputs,
    settlement_snap: Snap,
    hub_snap: Snap,
    rules: ClosureRules,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
    """Run the closure scenarios and classify every settlement.

    Returns per-piece closure labels, per-settlement scenario results, route
    lines (all in EPSG:4326) and a machine-readable summary.
    """
    graph = network.graph
    threshold = rules.closure_min_candidate_m
    candidate_closed = closed_pieces(table, threshold, close_unobserved=False, rules=rules)
    pessimistic_closed = closed_pieces(table, threshold, close_unobserved=True, rules=rules)

    pieces_out = _piece_output(network, table, candidate_closed, pessimistic_closed)
    candidate_scenario = scenario_graph(graph, candidate_closed)
    baseline_components = component_labels(graph)
    candidate_components = component_labels(candidate_scenario)
    baseline_sizes = component_sizes(graph)
    candidate_sizes = component_sizes(candidate_scenario)

    settlements = inputs.settlements
    nodes = settlement_snap.node
    results = gpd.GeoDataFrame(
        {
            "osm_id": settlements["osm_id"].to_numpy(),
            "name": [_text(v) for v in settlements["name"].to_numpy()],
            "place": [_text(v) for v in settlements["place"].to_numpy()],
            "candidate_status": settlements["candidate_status"].to_numpy(),
            "graph_node": [int(n) for n in nodes],
            "snap_distance_m": np.round(settlement_snap.distance_m, 1),
            "component_baseline": [
                baseline_components.get(int(n)) if int(n) >= 0 else None for n in nodes
            ],
            "component_candidate_closure": [
                candidate_components.get(int(n)) if int(n) >= 0 else None for n in nodes
            ],
            "component_nodes_baseline": [
                baseline_sizes.get(int(n)) if int(n) >= 0 else None for n in nodes
            ],
            "component_nodes_candidate_closure": [
                candidate_sizes.get(int(n)) if int(n) >= 0 else None for n in nodes
            ],
        },
        geometry=settlements.to_crs(network.crs)
        .representative_point()
        .to_crs(GEOJSON_DEFAULT_CRS)
        .values,
        crs=GEOJSON_DEFAULT_CRS,
    )

    hubs = inputs.hubs
    groups = list(dict.fromkeys(hubs["hub_group"])) if inputs.hub_reachability_possible else []
    group_summary: Dict[str, Any] = {}
    sensitivity: Dict[str, Any] = {}
    routes: List[Dict[str, Any]] = []
    route_geoms: List[Any] = []

    for group in groups:
        in_group = (hubs["hub_group"] == group).to_numpy()
        group_nodes = hub_snap.node[in_group]
        hub_names: Dict[int, Optional[str]] = {}
        for node, name in zip(group_nodes, hubs["name"].to_numpy()[in_group]):
            if node >= 0 and hub_names.get(int(node)) is None:
                hub_names[int(node)] = _text(name)

        baseline = reach_from_hubs(graph, group_nodes, frozenset())
        candidate = reach_from_hubs(graph, group_nodes, candidate_closed)
        pessimistic = reach_from_hubs(graph, group_nodes, pessimistic_closed)

        statuses: List[str] = []
        reasons: List[Optional[str]] = []
        for node in nodes:
            kernel_status, kernel_reason = classify(
                int(node), baseline, candidate, pessimistic, rules.detour_ratio
            )
            status = _STATUS_FROM_KERNEL[kernel_status]
            reason = _REASON_FROM_KERNEL.get(kernel_reason, kernel_reason)
            if status == INSUFFICIENT_DATA and not hub_names:
                reason = REASON_NO_HUB_ON_NETWORK
            statuses.append(status)
            reasons.append(reason)

        results[f"{group}_status"] = statuses
        results[f"{group}_reason"] = reasons
        results[f"{group}_hub_baseline"] = [_hub_label(hub_names, baseline, int(n)) for n in nodes]
        results[f"{group}_km_baseline"] = [_km(baseline, int(n)) for n in nodes]
        results[f"{group}_hub_candidate_closure"] = [
            _hub_label(hub_names, candidate, int(n)) for n in nodes
        ]
        results[f"{group}_km_candidate_closure"] = [_km(candidate, int(n)) for n in nodes]
        results[f"{group}_km_candidate_and_unobserved_closure"] = [
            _km(pessimistic, int(n)) for n in nodes
        ]

        for position, node in enumerate(nodes):
            node = int(node)
            for scenario, reach in ((BASELINE, baseline), (CANDIDATE_CLOSURE, candidate)):
                if node < 0 or not reach.reachable(node):
                    continue
                geometry = route_geometry(network, reach, node)
                if geometry is None:
                    continue
                routes.append(
                    {
                        "settlement_osm_id": results.loc[position, "osm_id"],
                        "settlement_name": results.loc[position, "name"],
                        "hub_group": group,
                        "scenario": scenario,
                        "hub_name": _hub_label(hub_names, reach, node),
                        "length_km": _km(reach, node),
                    }
                )
                route_geoms.append(geometry)

        counts = pd.Series(statuses).value_counts()
        group_summary[group] = {
            "hubs_total": int(in_group.sum()),
            "hubs_on_network": int((group_nodes >= 0).sum()),
            "status_counts": {s: int(counts.get(s, 0)) for s in SETTLEMENT_STATUSES},
            "scenario_unreachable": [
                {
                    "osm_id": results.loc[position, "osm_id"],
                    "name": results.loc[position, "name"],
                    "km_baseline": results.loc[position, f"{group}_km_baseline"],
                }
                for position, status in enumerate(statuses)
                if status == SCENARIO_UNREACHABLE
            ],
        }

        per_threshold: Dict[str, Any] = {}
        for sensitivity_threshold in rules.sensitivity_closure_m:
            optimistic = reach_from_hubs(
                graph,
                group_nodes,
                closed_pieces(table, sensitivity_threshold, close_unobserved=False, rules=rules),
            )
            worst = reach_from_hubs(
                graph,
                group_nodes,
                closed_pieces(table, sensitivity_threshold, close_unobserved=True, rules=rules),
            )
            tally = pd.Series(
                [
                    _STATUS_FROM_KERNEL[
                        classify(int(n), baseline, optimistic, worst, rules.detour_ratio)[0]
                    ]
                    for n in nodes
                ]
            ).value_counts()
            per_threshold[f"{sensitivity_threshold:g}"] = {
                status: int(tally.get(status, 0))
                for status in (
                    SCENARIO_UNREACHABLE,
                    UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED,
                    SCENARIO_DETOURED,
                )
            }
        sensitivity[group] = per_threshold

    routes_frame = _routes_frame(routes, route_geoms, network.crs)

    summary = _summarize(
        network=network,
        table=table,
        inputs=inputs,
        rules=rules,
        results=results,
        settlement_snap=settlement_snap,
        hub_snap=hub_snap,
        candidate_closed=candidate_closed,
        pessimistic_closed=pessimistic_closed,
        group_summary=group_summary,
        sensitivity=sensitivity,
    )
    return pieces_out, results, routes_frame, summary


#: Column contract of ``candidate_disruption_routes.geojson``, kept stable so a
#: run that produces no route still writes a readable artifact.
ROUTE_COLUMNS: Tuple[str, ...] = (
    "settlement_osm_id",
    "settlement_name",
    "hub_group",
    "scenario",
    "hub_name",
    "length_km",
)


def _routes_frame(
    routes: Sequence[Mapping[str, Any]], geometries: Sequence[Any], crs: CRS
) -> gpd.GeoDataFrame:
    if not routes:
        empty = gpd.GeoDataFrame(
            {name: pd.Series(dtype="object") for name in ROUTE_COLUMNS},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
        return empty.to_crs(GEOJSON_DEFAULT_CRS)
    return gpd.GeoDataFrame(list(routes), geometry=list(geometries), crs=crs).to_crs(
        GEOJSON_DEFAULT_CRS
    )


def _piece_output(
    network: RoadNetwork,
    table: pd.DataFrame,
    candidate_closed: FrozenSet,
    pessimistic_closed: FrozenSet,
) -> gpd.GeoDataFrame:
    pieces = network.pieces
    routable = (pieces["u"].to_numpy() != pieces["v"].to_numpy()) & (
        pieces["length_m"].to_numpy() > 0
    )
    out = gpd.GeoDataFrame(
        {
            "piece_id": table["piece_id"].to_numpy(),
            "osm_id": table["osm_id"].to_numpy(),
            "highway": [_text(v) for v in table["highway"].to_numpy()],
            "length_m": np.round(table["piece_length_m"].to_numpy(dtype=float), 3),
            "closure_unit_length_m": np.round(table["unit_length_m"].to_numpy(dtype=float), 3),
            "candidate_m": np.round(table["candidate_m"].to_numpy(dtype=float), 3),
            "not_observed_m": np.round(table["not_observed_m"].to_numpy(dtype=float), 3),
            "source_candidate_status": table["source_candidate_status"].to_numpy(),
            "closure_status": closure_status(table, candidate_closed, pessimistic_closed),
            "routable": routable,
            "u": pieces["u"].to_numpy(),
            "v": pieces["v"].to_numpy(),
        },
        geometry=pieces.geometry.values,
        crs=network.crs,
    )
    return out.to_crs(GEOJSON_DEFAULT_CRS)


def _summarize(
    *,
    network: RoadNetwork,
    table: pd.DataFrame,
    inputs: NetworkInputs,
    rules: ClosureRules,
    results: gpd.GeoDataFrame,
    settlement_snap: Snap,
    hub_snap: Snap,
    candidate_closed: FrozenSet,
    pessimistic_closed: FrozenSet,
    group_summary: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> Dict[str, Any]:
    graph = network.graph
    roads = inputs.roads
    piece_km = float(table["piece_length_m"].sum()) / 1000.0
    status_counts = roads["candidate_status"].value_counts()
    nodes = settlement_snap.node

    numeric = {
        column: pd.to_numeric(roads[column], errors="coerce").fillna(0.0)
        for column in ("length_m", "candidate_m", "not_observed_m")
    }

    def road_km(predicate) -> float:
        """Mapped length of the roads a predicate selects, in km."""
        return round(float(numeric["length_m"][predicate].sum()) / 1000.0, 3)

    # None rather than 0.0 when M6.4 recorded no outside-grid column at all:
    # "not measured" and "measured as zero" are different statements.
    outside_recorded = "outside_grid_m" in roads.columns and bool(
        roads["outside_grid_m"].notna().any()
    )
    outside_km = (
        round(
            float(pd.to_numeric(roads["outside_grid_m"], errors="coerce").fillna(0.0).sum())
            / 1000.0,
            3,
        )
        if outside_recorded
        else None
    )

    only_pessimistic = pessimistic_closed - candidate_closed
    summary: Dict[str, Any] = {
        "stage": CANDIDATE_DISRUPTION_VERSION,
        "status": "hypothetical_closure_scenario",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": dict(SCENARIO_MEANINGS),
        "closure_rule": {
            "unit": rules.closure_unit,
            "threshold_m": rules.closure_min_candidate_m,
            "rule": (
                "a closure unit is treated as impassable when its candidate-intersecting "
                "length reaches threshold_m, or all of it when the unit is shorter than "
                "threshold_m"
            ),
            "unit_meaning": (
                "the whole OSM way, measured by M6.4; 10 m of candidate intersection "
                "closes the entire way, so the scenario overstates disruption on long ways"
                if rules.closure_unit == "road"
                else "one graph piece between junctions, re-measured against the M6.3 "
                "candidate raster with the M6.4 kernel"
            ),
            "evidence_basis": (
                "M6.3 Sentinel-1 RTC dual-polarisation darkening candidates. Candidate "
                "intersection is NOT observed impassability; closure is an assumption "
                "this stage states and tests, not a measurement (AGENTS.md §8)."
            ),
            "outside_candidate_grid_roads_closed": rules.close_outside_grid_roads,
        },
        "rules": asdict(rules),
        "statuses": dict(STATUS_MEANINGS),
        "closure_labels": {
            CLOSED_IN_CANDIDATE_SCENARIO: "removed in both closure scenarios",
            CLOSED_ONLY_IF_UNOBSERVED_CLOSED: (
                "removed only when not-observed units are closed too"
            ),
            OPEN_IN_BOTH_SCENARIOS: (
                "left in the graph by both scenarios; this is not a statement that the "
                "road is intact or undamaged"
            ),
        },
        "graph": {
            "source_roads": int(len(roads)),
            "nodes": int(graph.number_of_nodes()),
            "pieces_total": int(len(table)),
            "pieces_routable": int(graph.number_of_edges()),
            "road_km": round(piece_km, 3),
            "connected_components_baseline": int(nx.number_connected_components(graph)),
            "largest_components_baseline": _component_sizes_km(graph, table, frozenset()),
            "analysis_crs": network.crs.to_string(),
            "segmentation": (
                "edges are road pieces between junctions; a junction is a way end, an OSM "
                "node shared by two ways, or the road vertex a settlement or hub attaches to"
            ),
        },
        "disruption": {
            CANDIDATE_CLOSURE: {
                "pieces_closed": len(candidate_closed),
                "closed_km": _closed_km(table, candidate_closed),
                "closed_km_by_highway": _closed_km_by_highway(table, candidate_closed),
                "connected_components": int(
                    nx.number_connected_components(scenario_graph(graph, candidate_closed))
                ),
                "largest_components": _component_sizes_km(graph, table, candidate_closed),
            },
            CANDIDATE_AND_UNOBSERVED_CLOSURE: {
                "pieces_closed": len(pessimistic_closed),
                "closed_km": _closed_km(table, pessimistic_closed),
                "connected_components": int(
                    nx.number_connected_components(scenario_graph(graph, pessimistic_closed))
                ),
            },
        },
        "road_length_accounting": {
            "total_km": round(float(numeric["length_m"].sum()) / 1000.0, 3),
            "candidate_intersecting_km": round(float(numeric["candidate_m"].sum()) / 1000.0, 3),
            "not_observed_km": round(float(numeric["not_observed_m"].sum()) / 1000.0, 3),
            "outside_candidate_grid_km": outside_km,
            "km_by_m6_4_status": {
                status: road_km(roads["candidate_status"] == status)
                for status in CANDIDATE_STATUSES
            },
            "note": (
                "candidate_intersecting_km is length measured on radar-change candidate "
                "pixels, not flooded, damaged or impassable length. not_observed_km is "
                "missing information and is never counted as unaffected road. "
                "NO_CANDIDATE_INTERSECTION length means the M6.3 rule did not fire there, "
                "which is not evidence the road was unaffected."
            ),
        },
        "coverage_uncertainty": {
            "roads_not_observed_status": int(status_counts.get(UNOBSERVED, 0)),
            "roads_outside_candidate_grid": int(status_counts.get(OUTSIDE_CANDIDATE_GRID, 0)),
            "not_observed_km": round(float(numeric["not_observed_m"].sum()) / 1000.0, 3),
            "outside_candidate_grid_km": outside_km,
            "pieces_closed_only_if_unobserved_closed": len(only_pessimistic),
            "km_closed_only_if_unobserved_closed": _closed_km(table, frozenset(only_pessimistic)),
            "pieces_not_routable": int(len(table) - graph.number_of_edges()),
            "settlements_without_mapped_road": int((nodes < 0).sum()),
            "hubs_without_mapped_road": (
                int((hub_snap.node < 0).sum()) if inputs.hub_reachability_possible else None
            ),
            "note": (
                "These are limits of observation and of OpenStreetMap coverage. They are "
                "reported separately from modelled disruption and must not be added to it "
                "or subtracted from it."
            ),
            "input_notes": list(inputs.notes),
        },
        "settlements": {
            "total": int(len(results)),
            "on_network": int((nodes >= 0).sum()),
            "representation": (
                "one representative point per settlement, attached to the nearest road "
                "vertex within snap_max_m; the same point M6.4 used for its settlement "
                "verdict"
            ),
            "component_shrunk_under_candidate_closure": int(
                sum(
                    1
                    for before, after in zip(
                        results["component_nodes_baseline"],
                        results["component_nodes_candidate_closure"],
                    )
                    if before is not None and after is not None and after < before
                )
            ),
            "component_note": (
                "component_baseline and component_candidate_closure are labels within one "
                "scenario, ordered largest first; they are not identities across scenarios. "
                "component_nodes_* is the comparable quantity: a settlement whose component "
                "shrinks has lost reachable mapped network under the closure rule."
            ),
        },
        "hub_reachability": {
            "assessed": bool(group_summary),
            "reason": (
                None
                if group_summary
                else (
                    "no pre-event OSM hub layer was available, so settlement-to-hub "
                    "reachability was not computed. Road disruption and connected-component "
                    "change are reported instead; they are not a substitute for a hub "
                    "accessibility verdict."
                )
            ),
            "groups": dict(group_summary),
            "sensitivity_by_closure_m": dict(sensitivity),
            "routing": (
                "shortest path by length to the nearest hub of each group, from a virtual "
                "source joined to every hub node at zero cost; undirected, oneway ignored"
            ),
        },
        "assumptions": list(ASSUMPTIONS),
        "interpretation": (
            "Every reachability number is conditional on the stated closure rule. A "
            "settlement reported SCENARIO_UNREACHABLE would lose its modelled road route "
            "to this hub group IF candidate-intersecting roads were impassable. Candidate "
            "intersection is radar-change evidence, not observed impassability, so this is "
            "not a finding that any road is destroyed or that any settlement or person is "
            "cut off (AGENTS.md §8, architecture.md §18)."
        ),
        "limitations": list(LIMITATIONS),
        "inputs": {
            "candidate_exposure_directory": str(inputs.directory),
            "candidate_exposure_stage": (inputs.exposure_summary or {}).get("stage"),
            "hub_layer": str(inputs.hubs_path) if inputs.hubs_path else None,
            "retrieval_performed": False,
            "retrieval_disabled_reason": (
                "the study AOI is UNVERIFIED and does not authorise a download; this stage "
                "reads road, settlement and hub geometry only from artifacts already on disk"
            ),
            "osm_snapshot": inputs.osm_snapshot(),
            "source_pair": (inputs.exposure_summary or {}).get("source_pair"),
            "aoi_provenance_status": (inputs.exposure_summary or {}).get(
                "aoi_provenance_status"
            ),
            "attribution": inputs.attribution(),
            "license": OSM_LICENSE,
        },
    }
    return summary


ASSUMPTIONS: Tuple[str, ...] = (
    "Road pieces run between junctions: way ends, OSM nodes shared by ways, and the road "
    "vertex each settlement or hub attaches to. Two ways that cross without sharing a node "
    "stay unconnected, as in OpenStreetMap.",
    "The graph is an undirected multigraph weighted by length; oneway tags are ignored, "
    "because in an emergency the question is whether a road exists.",
    "Closure is hypothetical. A unit is closed when its candidate-intersecting length "
    "reaches the threshold, or all of it when the unit is shorter. No M6.4 field states "
    "that a candidate-exposed road is impassable.",
    "Under closure_unit='road' the threshold acts on the whole OSM way, so any "
    "candidate intersection closes the entire way. That overstates disruption on long "
    "ways and is an upper bound, not a best estimate.",
    "Per-piece candidate length is never derived by dividing a road's candidate_m between "
    "its pieces; piece-level closure requires re-measuring against the candidate raster.",
    "Roads left open in both scenarios are assumed passable. That is an assumption, not an "
    "observation: the M6.3 rule can miss inundation under vegetation or in built-up areas.",
    "Roads entirely outside the candidate raster extent are assumed passable unless "
    "close_outside_grid_roads is set, because they were never observed by the change "
    "detector and closing them would remove the routes to out-of-AOI hubs.",
    "Not-observed length drives a second, pessimistic scenario; an outcome that differs "
    "between the two scenarios is reported as undetermined rather than resolved.",
    "Settlements and hubs are single representative points attached to the nearest road "
    "vertex within snap_max_m; a settlement farther than that gets no verdict.",
    "Settlements with no baseline route to a hub are INSUFFICIENT_DATA, never unreachable: "
    "the scenario did not cause that, and many Himalayan hamlets have no mapped road.",
    "Footpaths and footbridges are not in the road graph, so foot access is not modelled.",
    "Hub groups are the OSM feature classes in the pre-event hub layer; hubs absent from "
    "OpenStreetMap cannot be reached by definition.",
)

LIMITATIONS: Tuple[str, ...] = (
    "Closure is a stated scenario, not an observation. M6.3 candidates are rule-based "
    "Sentinel-1 RTC gamma0 darkening evidence, and nothing in the M6.4 contract establishes "
    "that a candidate-exposed road is impassable (AGENTS.md §8).",
    "Radar darkening has causes other than inundation, and inundation under vegetation or "
    "in built-up areas can be missed, so the closure set carries both false positives and "
    "false negatives.",
    "A SCENARIO_UNREACHABLE verdict is modelled road inaccessibility under the stated rule. "
    "It is not confirmed isolation of a settlement or of any person "
    "(architecture.md §18).",
    "Not-observed road length is missing information, reported under coverage_uncertainty "
    "and never counted as unaffected road.",
    "Road, settlement and hub completeness is limited by OpenStreetMap coverage at the "
    "pre-event snapshot; unmapped roads cannot carry a route and unmapped settlements "
    "cannot appear.",
    "The graph ignores road capacity, surface, gradient, seasonal passability, bridge "
    "load limits and landslide risk, all of which matter in a Himalayan corridor.",
    "Routes are shortest paths by length, not by travel time, and ignore direction of "
    "travel.",
    "Settlement and hub positions are single representative points, so a settlement's "
    "built-up extent and multiple access roads are not represented.",
    "The study extent is a human-approved project corridor whose producer, source and "
    "construction parameters are recorded, but its provenance_status remains "
    "UNVERIFIED: adopting the geometry authorised no retrieval, and every count is "
    "relative to a project-defined analytical extent rather than an official boundary.",
    "No DEM-based flood-path reasoning, segmentation checkpoint, validation product or "
    "ground truth is used by this stage.",
)


# ---------------------------------------------------------------------------
# Provenance and writers
# ---------------------------------------------------------------------------


def candidate_disruption_provenance(
    *,
    inputs: NetworkInputs,
    summary: Mapping[str, Any],
    config_version: str,
) -> ArtifactProvenance:
    """Machine-readable provenance for the M6.5 artifact."""
    pair = (inputs.exposure_summary or {}).get("source_pair") or {}
    assessed = bool((summary.get("hub_reachability") or {}).get("assessed"))
    product_ids = [
        str((pair.get("pre") or {}).get("scene_id", "")),
        str((pair.get("post") or {}).get("scene_id", "")),
    ]
    operations = [
        "load_m6_4_candidate_exposure_roads_and_settlements_from_disk_without_retrieval",
        "verify_m6_4_stage_tag_and_candidate_status_vocabulary",
        "refuse_flood_mask_exposure_columns",
        "build_undirected_road_multigraph_of_pieces_between_junctions",
        "snap_settlements_and_hubs_to_nearest_road_vertex_within_snap_max_m",
        f"apply_hypothetical_closure_rule_unit_{summary['closure_rule']['unit']}",
        "compare_connected_components_baseline_vs_closure_scenarios",
    ]
    if assessed:
        operations += [
            "multi_source_dijkstra_to_nearest_hub_per_group_per_scenario",
            "classify_settlement_scenario_reachability_and_threshold_sensitivity",
        ]
    return ArtifactProvenance(
        artifact_type=(
            ArtifactType.SETTLEMENT_CONNECTIVITY if assessed else ArtifactType.ROAD_DISRUPTION
        ),
        artifact_id=(
            f"candidate-network-disruption-{pair.get('aoi_id', 'unknown-aoi')}-"
            f"{pair.get('event_date', 'unknown-date')}"
        ),
        aoi=pair.get("aoi_id"),
        aoi_crs=GEOJSON_DEFAULT_CRS,
        event_date=pair.get("event_date"),
        # Carried forward rather than reduced to scene ids, so the acquisition
        # timestamps and relative orbit behind every verdict stay inspectable.
        sentinel1=scene_pair_from_summary(pair),
        osm_snapshot_date=inputs.osm_snapshot(),
        production_inputs=[ProductionInput.SENTINEL1, ProductionInput.OSM_PRE_EVENT],
        source_product_ids=[pid for pid in product_ids if pid],
        preprocessing_operations=operations,
        quality=dict(summary),
        preprocessing_version=CANDIDATE_DISRUPTION_VERSION,
        config_version=config_version,
        limitations=list(LIMITATIONS),
        notes=(
            "Road-network disruption and settlement reachability under a hypothetical "
            "closure of M6.4 candidate-exposed roads. No download was performed: the AOI "
            "is UNVERIFIED and does not authorise one. Closure is a stated scenario, not "
            "an observation, and no segmentation checkpoint, DEM, validation product or "
            "ground truth was used."
        ),
        attribution=inputs.attribution(),
    )


def write_candidate_disruption_artifacts(
    pieces: gpd.GeoDataFrame,
    settlements: gpd.GeoDataFrame,
    routes: gpd.GeoDataFrame,
    summary: Mapping[str, Any],
    output_dir: Path | str,
    *,
    provenance: Optional[ArtifactProvenance] = None,
) -> Dict[str, Path]:
    """Write graph, settlement, route, summary and provenance artifacts.

    GeoJSON is written in EPSG:4326 so it opens directly in QGIS or geojson.io.
    Pieces left open are retained and labelled rather than dropped, so a reader
    can see the whole modelled network and not only its closed part.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, frame in (
        ("pieces", pieces),
        ("settlements", settlements),
        ("routes", routes),
    ):
        path = output_dir / f"candidate_disruption_{name}.geojson"
        path.write_text(frame.to_json(drop_id=True), encoding="utf-8")
        written[name] = path

    summary_path = output_dir / "candidate_disruption_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    written["summary"] = summary_path

    if provenance is not None:
        provenance_path = output_dir / "candidate_disruption_provenance.yaml"
        provenance_path.write_text(provenance.to_yaml(), encoding="utf-8")
        written["provenance"] = provenance_path
    return written


def disruption_blocker_report(
    *,
    reason: str,
    missing: Sequence[str],
    searched: Path | str,
    upstream: Optional[Mapping[str, Any]] = None,
    next_step: Sequence[str] = (),
) -> Dict[str, Any]:
    """A machine-readable record of why M6.5 could not complete.

    Written instead of connectivity numbers. A blocker artifact is a result: it
    states which permitted input is absent, which authorisation gate prevents
    obtaining it, and what would unblock it.
    """
    pair = (upstream or {}).get("source_pair") or {}
    return {
        "stage": CANDIDATE_DISRUPTION_VERSION,
        "status": "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "missing_inputs": list(missing),
        "searched_directory": str(searched),
        "retrieval_performed": False,
        "authorisation": {
            "aoi_provenance_status": (upstream or {}).get("aoi_provenance_status", "UNVERIFIED"),
            "authorises_download": False,
            "gate": (
                "AoiProvenanceStatus.UNVERIFIED permits geometry inspection and catalogue "
                "metadata analysis only; it must not be used to authorise a download "
                "(src/floodmap/acquisition/aoi.py)."
            ),
        },
        "inputs_available": {
            "candidate_exposure_stage": (upstream or {}).get("stage"),
            "source_pair": dict(pair) if pair else None,
        },
        "next_step": list(next_step),
        "not_done": [
            "No road network, settlement or hub was synthesised, approximated or substituted.",
            "No disruption, reachability or cut-off number was produced.",
        ],
    }


def write_disruption_blocker_report(report: Mapping[str, Any], output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_disruption_blocker.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ASSUMPTIONS",
    "BASELINE",
    "CANDIDATE_AND_UNOBSERVED_CLOSURE",
    "CANDIDATE_CLOSURE",
    "CANDIDATE_DISRUPTION_VERSION",
    "CLOSED_IN_CANDIDATE_SCENARIO",
    "CLOSED_ONLY_IF_UNOBSERVED_CLOSED",
    "CLOSURE_LABELS",
    "CLOSURE_UNITS",
    "CandidateDisruptionError",
    "ClosureRules",
    "FORBIDDEN_ROAD_COLUMNS",
    "INSUFFICIENT_DATA",
    "LIMITATIONS",
    "MissingNetworkInputs",
    "NetworkInputs",
    "OPEN_IN_BOTH_SCENARIOS",
    "REASON_NO_BASELINE_ROUTE_SCENARIO",
    "REASON_NO_HUB_ON_NETWORK",
    "REASON_NO_MAPPED_ROAD",
    "REASON_UNOBSERVED_ONLY",
    "REQUIRED_HUB_COLUMNS",
    "REQUIRED_ROAD_COLUMNS",
    "REQUIRED_SETTLEMENT_COLUMNS",
    "ROUTE_COLUMNS",
    "SCENARIOS",
    "SCENARIO_DETOURED",
    "SCENARIO_MEANINGS",
    "SCENARIO_REACHABLE",
    "SCENARIO_UNREACHABLE",
    "SETTLEMENT_STATUSES",
    "STATUS_MEANINGS",
    "UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED",
    "analysis_crs",
    "assess_candidate_disruption",
    "build_candidate_road_network",
    "candidate_disruption_provenance",
    "closed_pieces",
    "closure_status",
    "closure_table",
    "component_labels",
    "component_sizes",
    "disruption_blocker_report",
    "load_candidate_exposure_roads",
    "load_candidate_exposure_settlements",
    "load_hub_layer",
    "load_network_inputs",
    "scenario_graph",
    "write_candidate_disruption_artifacts",
    "write_disruption_blocker_report",
]
