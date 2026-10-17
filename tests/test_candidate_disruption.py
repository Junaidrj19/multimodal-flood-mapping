"""Tests for the M6.5 candidate road-network disruption stage (offline).

Every fixture is synthetic: a few straight roads in UTM 45N metres around
(X0, Y0), written to disk in the exact M6.4 artifact layout, with per-road
candidate quantities set by hand so each expected closure, component count and
settlement status can be checked by arithmetic.

No test asserts a real-world conclusion. Nothing here touches real imagery,
real OpenStreetMap data or a real settlement, and the synthetic numbers must
never be presented as a case-study result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from shapely.geometry import LineString, Point

from floodmap.infrastructure.candidate_exposure import (
    CANDIDATE_EXPOSURE_VERSION,
    CANDIDATE_INTERSECTING,
    NO_CANDIDATE_INTERSECTION,
    OUTSIDE_CANDIDATE_GRID,
    UNOBSERVED,
)
from floodmap.infrastructure.candidate_mask import (
    CANDIDATE,
    CANDIDATE_SEMANTIC_VALUES_TAG,
    NOT_CANDIDATE,
    NOT_OBSERVED_VALUE,
    VALID_MASK_SEMANTICS_TAG,
    load_candidate_mask,
)
from floodmap.network.candidate_disruption import (
    ASSUMPTIONS,
    CANDIDATE_AND_UNOBSERVED_CLOSURE,
    CANDIDATE_CLOSURE,
    CANDIDATE_DISRUPTION_VERSION,
    CLOSED_IN_CANDIDATE_SCENARIO,
    CLOSED_ONLY_IF_UNOBSERVED_CLOSED,
    INSUFFICIENT_DATA,
    LIMITATIONS,
    OPEN_IN_BOTH_SCENARIOS,
    REASON_NO_BASELINE_ROUTE_SCENARIO,
    REASON_NO_HUB_ON_NETWORK,
    REASON_NO_MAPPED_ROAD,
    REASON_UNOBSERVED_ONLY,
    ROUTE_COLUMNS,
    SCENARIO_DETOURED,
    SCENARIO_REACHABLE,
    SCENARIO_UNREACHABLE,
    SETTLEMENT_STATUSES,
    UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED,
    CandidateDisruptionError,
    ClosureRules,
    MissingNetworkInputs,
    analysis_crs,
    assess_candidate_disruption,
    build_candidate_road_network,
    candidate_disruption_provenance,
    closed_pieces,
    closure_table,
    component_labels,
    disruption_blocker_report,
    load_candidate_exposure_roads,
    load_network_inputs,
    scenario_graph,
    write_candidate_disruption_artifacts,
    write_disruption_blocker_report,
)
from floodmap.utils.config import load_config
from floodmap.utils.provenance import ArtifactType
from tests.conftest import REPO_ROOT

UTM = "EPSG:32645"
WGS84 = "EPSG:4326"
X0, Y0 = 300_000.0, 3_100_000.0
EVENT_DATE = "2026-08-26"
OSM_SNAPSHOT = "2026-07-27T09:00:00Z"
M63_STAGE = "m6.3-rtc-change-candidates-0.2.0"

RULES = ClosureRules(
    snap_max_m=500.0,
    closure_unit="road",
    closure_min_candidate_m=10.0,
    sensitivity_closure_m=(10.0, 30.0, 100.0),
    detour_ratio=1.5,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _line(*points) -> LineString:
    return LineString([(X0 + x, Y0 + y) for x, y in points])


def _pt(x: float, y: float) -> Point:
    return Point(X0 + x, Y0 + y)


#: The only route from the hub to the settlement, 3 km.
ROAD_A = _line((0, 0), (3000, 0))
#: An alternative, 7 km: ratio 2.33, a detour.
DETOUR_B = _line((0, 0), (0, 2000), (3000, 2000), (3000, 0))
#: An alternative, 3.16 km: ratio 1.05, not a detour.
SHORT_C = _line((0, 0), (1500, 500), (3000, 0))
#: A road with an interior vertex, so a snapped settlement splits it in two.
SPLIT_D = _line((0, -5), (1500, -5), (3000, -5))

HUB = (_pt(0, -30), "hospital", "Hub One")
SETTLEMENT = _pt(3000, 50)
ALL = float("inf")


# ---------------------------------------------------------------------------
# M6.4 artifact fixtures
# ---------------------------------------------------------------------------


def _road_records(
    lines: Sequence[LineString],
    candidate_m: Optional[Dict[str, float]] = None,
    not_observed_m: Optional[Dict[str, float]] = None,
    statuses: Optional[Dict[str, str]] = None,
    highway: str = "track",
) -> List[Dict[str, Any]]:
    """M6.4 per-road measurements, assigned by hand.

    ``candidate_m`` and ``not_observed_m`` are capped at the road's length, as
    the M6.4 kernel guarantees. The status is derived with the same rule M6.4
    applies unless overridden, so the fixture cannot disagree with the contract
    it is imitating.
    """
    records = []
    for index, line in enumerate(lines):
        osm_id = f"way/{index}"
        length = float(line.length)
        candidate = min(float((candidate_m or {}).get(osm_id, 0.0)), length)
        unobserved = min(float((not_observed_m or {}).get(osm_id, 0.0)), length)
        threshold = min(10.0, length) - 1e-6
        if candidate >= threshold:
            status = CANDIDATE_INTERSECTING
        elif unobserved >= threshold:
            status = UNOBSERVED
        else:
            status = NO_CANDIDATE_INTERSECTION
        records.append(
            {
                "osm_id": osm_id,
                "name": f"road {index}",
                "highway": highway,
                "tags": json.dumps({"highway": highway}),
                "length_m": round(length, 3),
                "candidate_m": round(candidate, 3),
                "not_observed_m": round(unobserved, 3),
                "not_candidate_m": round(max(length - candidate - unobserved, 0.0), 3),
                "outside_grid_m": 0.0,
                "candidate_status": (statuses or {}).get(osm_id, status),
                "geometry": line,
            }
        )
    return records


def _settlement_records(points: Sequence[Point], statuses: Optional[Sequence[str]] = None):
    return [
        {
            "osm_id": f"node/s{index}",
            "name": f"S{index}",
            "place": "hamlet",
            "candidate_status": (
                statuses[index] if statuses else NO_CANDIDATE_INTERSECTION
            ),
            "settlement_buffer_m": 0.0,
            "geometry": point,
        }
        for index, point in enumerate(points)
    ]


def _hub_records(entries: Iterable[tuple]):
    return [
        {
            "osm_id": f"node/h{index}",
            "name": entry[2],
            "hub_group": entry[1],
            "amenity": entry[1],
            "geometry": entry[0],
        }
        for index, entry in enumerate(entries)
    ]


def _exposure_summary(crs: str = UTM) -> Dict[str, Any]:
    """The subset of the M6.4 summary this stage reads."""
    return {
        "stage": CANDIDATE_EXPOSURE_VERSION,
        "status": "candidate_exposure_only",
        "aoi_provenance_status": "UNVERIFIED",
        "source_pair": {
            "aoi_id": "synthetic-test-aoi",
            "event_date": EVENT_DATE,
            "aoi_provenance_status": "UNVERIFIED",
            "pre": {
                "scene_id": "pre_rtc",
                "acquired_at": "2026-08-16T12:21:53.557589Z",
                "relative_orbit": 85,
                "orbit_direction": "ASCENDING",
            },
            "post": {
                "scene_id": "post_rtc",
                "acquired_at": "2026-08-28T12:21:54.261291Z",
                "relative_orbit": 85,
                "orbit_direction": "ASCENDING",
            },
        },
        "osm": {
            "snapshot": OSM_SNAPSHOT,
            "attribution": ["© OpenStreetMap contributors."],
        },
        "data_coverage": {"candidate_mask": {"crs": crs, "stage": M63_STAGE}},
    }


def _write_frame(records: Sequence[Dict[str, Any]], path: Path, columns=None) -> Path:
    """Write a GeoJSON in EPSG:4326, as the M6.4 writer does."""
    if records:
        frame = gpd.GeoDataFrame(list(records), crs=UTM).to_crs(WGS84)
    else:
        frame = gpd.GeoDataFrame(
            {name: pd.Series(dtype="object") for name in (columns or ["osm_id"])},
            geometry=gpd.GeoSeries([], crs=WGS84),
            crs=WGS84,
        )
    path.write_text(frame.to_json(drop_id=True), encoding="utf-8")
    return path


def _exposure_dir(
    tmp_path: Path,
    roads: Sequence[Dict[str, Any]],
    settlements: Sequence[Dict[str, Any]],
    *,
    summary: Optional[Dict[str, Any]] = None,
    name: str = "m6_4",
) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    _write_frame(
        roads,
        directory / "candidate_exposure_roads.geojson",
        columns=[
            "osm_id",
            "name",
            "highway",
            "length_m",
            "candidate_m",
            "not_observed_m",
            "candidate_status",
        ],
    )
    _write_frame(
        settlements,
        directory / "candidate_exposure_settlements.geojson",
        columns=["osm_id", "name", "place", "candidate_status"],
    )
    if summary is not False:
        (directory / "candidate_exposure_summary.json").write_text(
            json.dumps(summary if summary is not None else _exposure_summary()),
            encoding="utf-8",
        )
    return directory


def _osm_dir(
    tmp_path: Path, hubs: Sequence[Dict[str, Any]], *, snapshot: str = OSM_SNAPSHOT
) -> Path:
    directory = tmp_path / "osm"
    directory.mkdir(parents=True, exist_ok=True)
    _write_frame(hubs, directory / "osm_hubs.geojson")
    (directory / "osm_provenance.json").write_text(
        json.dumps(
            {
                "snapshot": snapshot,
                "event_date": EVENT_DATE,
                "attribution": "© OpenStreetMap contributors.",
            }
        ),
        encoding="utf-8",
    )
    return directory


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------


def _run(
    tmp_path: Path,
    lines: Sequence[LineString],
    settlement_points: Sequence[Point],
    hub_entries: Sequence[tuple] = (HUB,),
    *,
    candidate_m: Optional[Dict[str, float]] = None,
    not_observed_m: Optional[Dict[str, float]] = None,
    statuses: Optional[Dict[str, str]] = None,
    rules: ClosureRules = RULES,
    with_hubs: bool = True,
    mask=None,
):
    roads = _road_records(lines, candidate_m, not_observed_m, statuses)
    exposure = _exposure_dir(tmp_path, roads, _settlement_records(settlement_points))
    osm = _osm_dir(tmp_path, _hub_records(hub_entries)) if with_hubs else None
    inputs = load_network_inputs(exposure, osm_dir=osm)
    crs = analysis_crs(inputs.exposure_summary)
    network, settlement_snap, hub_snap = build_candidate_road_network(
        inputs.roads,
        crs,
        settlements=inputs.settlements,
        hubs=inputs.hubs,
        snap_max_m=rules.snap_max_m,
    )
    table = closure_table(network, inputs.roads, rules, mask=mask)
    pieces, results, routes, summary = assess_candidate_disruption(
        network, table, inputs, settlement_snap, hub_snap, rules
    )
    return {
        "inputs": inputs,
        "network": network,
        "table": table,
        "pieces": pieces,
        "results": results,
        "routes": routes,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Candidate raster fixture, for closure_unit='piece'
# ---------------------------------------------------------------------------

#: 10 m grid spanning the split road, candidate only east of X0 + 1500.
_MASK_TRANSFORM = Affine(10.0, 0.0, X0 - 100.0, 0.0, -10.0, Y0 + 100.0)
_MASK_WIDTH, _MASK_HEIGHT = 330, 20


def _candidate_classes() -> np.ndarray:
    classes = np.full((_MASK_HEIGHT, _MASK_WIDTH), NOT_CANDIDATE, dtype="uint8")
    first_candidate_column = int((X0 + 1500.0 - (X0 - 100.0)) / 10.0)
    classes[:, first_candidate_column:] = CANDIDATE
    return classes


def _write_mask(tmp_path: Path):
    classes = _candidate_classes()
    mask_path = tmp_path / "rtc_change_candidates.tif"
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        width=_MASK_WIDTH,
        height=_MASK_HEIGHT,
        count=1,
        dtype="uint8",
        crs=UTM,
        transform=_MASK_TRANSFORM,
        nodata=float(NOT_OBSERVED_VALUE),
    ) as dst:
        dst.write(classes, 1)
        dst.set_band_description(1, "change_candidate")
        dst.update_tags(
            artifact_stage=M63_STAGE, semantic_values=CANDIDATE_SEMANTIC_VALUES_TAG
        )
    valid_path = tmp_path / "rtc_valid_mask.tif"
    with rasterio.open(
        valid_path,
        "w",
        driver="GTiff",
        width=_MASK_WIDTH,
        height=_MASK_HEIGHT,
        count=1,
        dtype="uint8",
        crs=UTM,
        transform=_MASK_TRANSFORM,
        nodata=0,
    ) as dst:
        dst.write(np.ones((_MASK_HEIGHT, _MASK_WIDTH), dtype="uint8"), 1)
        dst.set_band_description(1, "valid_change_observation")
        dst.update_tags(mask_semantics=VALID_MASK_SEMANTICS_TAG)
    return load_candidate_mask(mask_path, valid_path, expected_stage=M63_STAGE)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class TestInputs:
    def test_reads_the_m6_4_contract(self, tmp_path: Path):
        exposure = _exposure_dir(
            tmp_path, _road_records([ROAD_A]), _settlement_records([SETTLEMENT])
        )
        inputs = load_network_inputs(exposure, osm_dir=_osm_dir(tmp_path, _hub_records([HUB])))
        assert list(inputs.roads["osm_id"]) == ["way/0"]
        assert inputs.hub_reachability_possible
        assert inputs.osm_snapshot() == OSM_SNAPSHOT
        assert "© OpenStreetMap contributors." in inputs.attribution()

    def test_absent_directory_is_a_blocker_not_a_zero_result(self, tmp_path: Path):
        with pytest.raises(MissingNetworkInputs) as excinfo:
            load_network_inputs(tmp_path / "nothing-here")
        assert "candidate_exposure_roads.geojson" in excinfo.value.missing

    def test_absent_road_artifact_is_a_blocker(self, tmp_path: Path):
        exposure = _exposure_dir(
            tmp_path, _road_records([ROAD_A]), _settlement_records([SETTLEMENT])
        )
        (exposure / "candidate_exposure_roads.geojson").unlink()
        with pytest.raises(MissingNetworkInputs) as excinfo:
            load_network_inputs(exposure)
        assert excinfo.value.missing == ["candidate_exposure_roads.geojson"]

    def test_flood_exposure_artifact_is_refused(self, tmp_path: Path):
        """The impact.py artifact measures predicted inundation, not radar change."""
        roads = _road_records([ROAD_A])
        roads[0]["flooded_m"] = 40.0
        roads[0]["exposure_status"] = "EXPOSED"
        exposure = _exposure_dir(tmp_path, roads, _settlement_records([SETTLEMENT]))
        with pytest.raises(CandidateDisruptionError, match="flood-mask exposure column"):
            load_network_inputs(exposure)

    def test_missing_required_column_is_refused(self, tmp_path: Path):
        roads = _road_records([ROAD_A])
        del roads[0]["candidate_m"]
        exposure = _exposure_dir(tmp_path, roads, _settlement_records([SETTLEMENT]))
        with pytest.raises(CandidateDisruptionError, match="candidate_m"):
            load_network_inputs(exposure)

    def test_unknown_status_vocabulary_is_refused(self, tmp_path: Path):
        roads = _road_records([ROAD_A], statuses={"way/0": "FLOODED"})
        exposure = _exposure_dir(tmp_path, roads, _settlement_records([SETTLEMENT]))
        with pytest.raises(CandidateDisruptionError, match="unrecognised candidate_status"):
            load_network_inputs(exposure)

    def test_empty_road_layer_is_refused_rather_than_reported_as_no_access(
        self, tmp_path: Path
    ):
        exposure = _exposure_dir(tmp_path, [], _settlement_records([SETTLEMENT]))
        with pytest.raises(CandidateDisruptionError, match="no road features"):
            load_network_inputs(exposure)

    def test_empty_settlement_layer_still_allows_road_disruption(self, tmp_path: Path):
        """No settlements to assess is a fact about the input, not a verdict."""
        exposure = _exposure_dir(tmp_path, _road_records([ROAD_A]), [])
        inputs = load_network_inputs(exposure, osm_dir=_osm_dir(tmp_path, _hub_records([HUB])))
        assert len(inputs.settlements) == 0
        assert any("no settlement features" in note for note in inputs.notes)
        network, settlement_snap, hub_snap = build_candidate_road_network(
            inputs.roads,
            analysis_crs(inputs.exposure_summary),
            settlements=inputs.settlements,
            hubs=inputs.hubs,
            snap_max_m=RULES.snap_max_m,
        )
        table = closure_table(network, inputs.roads, RULES)
        _, results, routes, summary = assess_candidate_disruption(
            network, table, inputs, settlement_snap, hub_snap, RULES
        )
        assert len(results) == 0 and len(routes) == 0
        assert summary["settlements"]["total"] == 0
        assert summary["graph"]["road_km"] == pytest.approx(3.0, abs=1e-3)

    def test_missing_optional_columns_are_nulled_and_noted(self, tmp_path: Path):
        roads = _road_records([ROAD_A])
        del roads[0]["highway"]
        del roads[0]["name"]
        frame, notes = load_candidate_exposure_roads(
            _write_frame(roads, tmp_path / "roads.geojson")
        )
        assert frame["highway"].isna().all()
        assert any("highway" in note for note in notes)

    def test_wrong_upstream_stage_is_refused(self, tmp_path: Path):
        summary = _exposure_summary()
        summary["stage"] = "m6.4-candidate-infrastructure-exposure-0.0.1"
        exposure = _exposure_dir(
            tmp_path,
            _road_records([ROAD_A]),
            _settlement_records([SETTLEMENT]),
            summary=summary,
        )
        with pytest.raises(CandidateDisruptionError, match="declares stage"):
            load_network_inputs(exposure)

    def test_absent_hub_layer_is_recorded_not_substituted(self, tmp_path: Path):
        exposure = _exposure_dir(
            tmp_path, _road_records([ROAD_A]), _settlement_records([SETTLEMENT])
        )
        empty_osm = tmp_path / "osm-empty"
        empty_osm.mkdir()
        inputs = load_network_inputs(exposure, osm_dir=empty_osm)
        assert inputs.hubs is None
        assert any("hub layer" in note for note in inputs.notes)

    def test_required_hubs_raise_when_absent(self, tmp_path: Path):
        exposure = _exposure_dir(
            tmp_path, _road_records([ROAD_A]), _settlement_records([SETTLEMENT])
        )
        empty_osm = tmp_path / "osm-empty"
        empty_osm.mkdir()
        with pytest.raises(MissingNetworkInputs):
            load_network_inputs(exposure, osm_dir=empty_osm, require_hubs=True)

    def test_post_event_hub_snapshot_is_refused(self, tmp_path: Path):
        exposure = _exposure_dir(
            tmp_path, _road_records([ROAD_A]), _settlement_records([SETTLEMENT])
        )
        osm = _osm_dir(tmp_path, _hub_records([HUB]), snapshot="2026-08-30T00:00:00Z")
        with pytest.raises(CandidateDisruptionError, match="strictly before"):
            load_network_inputs(exposure, osm_dir=osm)


class TestAnalysisCrs:
    def test_defaults_to_the_candidate_mask_crs(self):
        assert analysis_crs(_exposure_summary()).to_epsg() == 32645

    def test_geographic_crs_is_refused(self):
        with pytest.raises(CandidateDisruptionError, match="geographic"):
            analysis_crs(_exposure_summary(), WGS84)

    def test_absent_record_is_refused_rather_than_guessed(self):
        with pytest.raises(CandidateDisruptionError, match="analysis CRS"):
            analysis_crs({"stage": CANDIDATE_EXPOSURE_VERSION})


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestGraph:
    def test_pieces_partition_the_mapped_road_length(self, tmp_path: Path):
        lines = [ROAD_A, DETOUR_B, SHORT_C]
        run = _run(tmp_path, lines, [SETTLEMENT])
        assert run["network"].pieces["length_m"].sum() == pytest.approx(
            sum(line.length for line in lines)
        )
        assert run["summary"]["graph"]["road_km"] == pytest.approx(
            sum(line.length for line in lines) / 1000.0, abs=1e-3
        )

    def test_roads_sharing_a_node_connect(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A, _line((3000, 0), (6000, 0))], [SETTLEMENT])
        assert run["summary"]["graph"]["connected_components_baseline"] == 1

    def test_crossing_without_a_shared_node_stays_unconnected(self, tmp_path: Path):
        crossing = [_line((-500, 0), (500, 0)), _line((0, -500), (0, 500))]
        run = _run(tmp_path, crossing, [_pt(400, 20)])
        assert run["summary"]["graph"]["connected_components_baseline"] == 2

    def test_snapped_settlement_vertex_becomes_a_junction(self, tmp_path: Path):
        run = _run(tmp_path, [SPLIT_D], [_pt(1500, 40)])
        assert len(run["network"].pieces) == 2

    def test_builder_requires_the_osm_attributes_it_reads(self):
        roads = gpd.GeoDataFrame({"length_m": [1.0]}, geometry=[ROAD_A], crs=UTM)
        with pytest.raises(CandidateDisruptionError, match="osm_id"):
            build_candidate_road_network(roads, UTM)

    def test_graph_and_exposure_must_describe_the_same_roads(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        other = run["inputs"].roads.copy()
        other["osm_id"] = ["way/elsewhere"]
        with pytest.raises(CandidateDisruptionError, match="absent from the M6.4"):
            closure_table(run["network"], other, RULES)


class TestComponents:
    def test_scenario_graph_keeps_isolated_nodes(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        graph = run["network"].graph
        closed = frozenset(run["table"]["piece_id"].tolist())
        scenario = scenario_graph(graph, closed)
        assert scenario.number_of_nodes() == graph.number_of_nodes()
        assert scenario.number_of_edges() == 0

    def test_component_ids_are_largest_first_and_deterministic(self, tmp_path: Path):
        lines = [ROAD_A, _line((10_000, 0), (10_100, 0))]
        run = _run(tmp_path, lines, [SETTLEMENT])
        labels = component_labels(run["network"].graph)
        assert set(labels.values()) == {0, 1}
        sizes = pd.Series(list(labels.values())).value_counts().sort_index()
        assert sizes.iloc[0] >= sizes.iloc[1]
        assert labels == component_labels(run["network"].graph)

    def test_closure_fragments_the_network(self, tmp_path: Path):
        """Closing the middle piece of a chain splits one component into two."""
        run = _run(
            tmp_path,
            [_line((0, 0), (1000, 0)), _line((1000, 0), (2000, 0)), _line((2000, 0), (3000, 0))],
            [SETTLEMENT],
            candidate_m={"way/1": ALL},
        )
        disruption = run["summary"]["disruption"]
        assert run["summary"]["graph"]["connected_components_baseline"] == 1
        assert disruption[CANDIDATE_CLOSURE]["connected_components"] == 2
        assert disruption[CANDIDATE_CLOSURE]["pieces_closed"] == 1

    def test_component_labels_are_scenario_local_not_identities(self, tmp_path: Path):
        """Guard against reading an unchanged label as an unchanged neighbourhood.

        Closing the hub-side road leaves the settlement in the largest surviving
        component, which keeps label 0 even though the network it can reach has
        shrunk. The comparable quantity is the component's node count.
        """
        run = _run(
            tmp_path,
            [ROAD_A, _line((3000, 0), (6000, 0))],
            [_pt(6000, 20)],
            candidate_m={"way/0": ALL},
        )
        row = run["results"].iloc[0]
        assert row["component_baseline"] == row["component_candidate_closure"] == 0
        assert row["component_nodes_candidate_closure"] < row["component_nodes_baseline"]
        assert run["summary"]["settlements"]["component_shrunk_under_candidate_closure"] == 1

    def test_component_sizes_are_unchanged_when_nothing_closes(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A, _line((3000, 0), (6000, 0))], [_pt(6000, 20)])
        row = run["results"].iloc[0]
        assert row["component_nodes_baseline"] == row["component_nodes_candidate_closure"]
        assert run["summary"]["settlements"]["component_shrunk_under_candidate_closure"] == 0


# ---------------------------------------------------------------------------
# Closure rule
# ---------------------------------------------------------------------------


class TestClosureRule:
    def test_road_unit_closes_the_whole_way(self, tmp_path: Path):
        """The documented coarseness: 10 m of intersection closes 3 km of road."""
        run = _run(tmp_path, [SPLIT_D], [_pt(1500, 40)], candidate_m={"way/0": 10.0})
        closure = run["pieces"]["closure_status"].tolist()
        assert closure == [CLOSED_IN_CANDIDATE_SCENARIO] * 2
        assert run["summary"]["disruption"][CANDIDATE_CLOSURE]["closed_km"] == pytest.approx(
            3.0, abs=1e-3
        )

    def test_below_threshold_closes_nothing(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], candidate_m={"way/0": 9.0})
        assert run["summary"]["disruption"][CANDIDATE_CLOSURE]["pieces_closed"] == 0
        assert run["pieces"]["closure_status"].tolist() == [OPEN_IN_BOTH_SCENARIOS]

    def test_short_road_intersecting_end_to_end_closes(self, tmp_path: Path):
        short = _line((0, 0), (6, 0))
        run = _run(tmp_path, [ROAD_A, short], [SETTLEMENT], candidate_m={"way/1": 6.0})
        closed = run["pieces"][run["pieces"]["closure_status"] == CLOSED_IN_CANDIDATE_SCENARIO]
        assert closed["osm_id"].tolist() == ["way/1"]

    def test_not_observed_closes_only_in_the_pessimistic_scenario(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], not_observed_m={"way/0": 200.0})
        disruption = run["summary"]["disruption"]
        assert disruption[CANDIDATE_CLOSURE]["pieces_closed"] == 0
        assert disruption[CANDIDATE_AND_UNOBSERVED_CLOSURE]["pieces_closed"] == 1
        assert run["pieces"]["closure_status"].tolist() == [CLOSED_ONLY_IF_UNOBSERVED_CLOSED]

    def test_outside_grid_roads_are_not_closed_by_default(self, tmp_path: Path):
        """Never observed by the change detector, so the rule has no evidence.

        The flag exists so the choice is explicit rather than buried.
        """
        roads = _road_records([ROAD_A], statuses={"way/0": OUTSIDE_CANDIDATE_GRID})
        roads[0]["not_observed_m"] = roads[0]["length_m"]
        table = _table_for(tmp_path, roads)
        assert closed_pieces(table, 10.0, close_unobserved=True, rules=RULES) == frozenset()
        permissive = ClosureRules(close_outside_grid_roads=True)
        assert closed_pieces(table, 10.0, close_unobserved=True, rules=permissive)

    def test_piece_unit_requires_the_raster(self, tmp_path: Path):
        run = _run(tmp_path, [SPLIT_D], [_pt(1500, 40)])
        piece_rules = ClosureRules(closure_unit="piece")
        with pytest.raises(CandidateDisruptionError, match="candidate raster"):
            closure_table(run["network"], run["inputs"].roads, piece_rules)

    def test_piece_unit_closes_only_the_intersecting_piece(self, tmp_path: Path):
        """Re-measuring is exact where whole-way closure is an upper bound."""
        mask = _write_mask(tmp_path)
        piece_rules = ClosureRules(closure_unit="piece")
        run = _run(
            tmp_path,
            [SPLIT_D],
            [_pt(1500, 40)],
            candidate_m={"way/0": 1500.0},
            rules=piece_rules,
            mask=mask,
        )
        pieces = run["pieces"].sort_values("piece_id")
        assert pieces["closure_status"].tolist() == [
            OPEN_IN_BOTH_SCENARIOS,
            CLOSED_IN_CANDIDATE_SCENARIO,
        ]
        assert run["summary"]["disruption"][CANDIDATE_CLOSURE]["closed_km"] == pytest.approx(
            1.5, abs=1e-2
        )

    def test_piece_unit_and_road_unit_disagree_as_documented(self, tmp_path: Path):
        mask = _write_mask(tmp_path)
        road_run = _run(
            tmp_path, [SPLIT_D], [_pt(1500, 40)], candidate_m={"way/0": 1500.0}
        )
        piece_run = _run(
            tmp_path,
            [SPLIT_D],
            [_pt(1500, 40)],
            candidate_m={"way/0": 1500.0},
            rules=ClosureRules(closure_unit="piece"),
            mask=mask,
        )
        road_km = road_run["summary"]["disruption"][CANDIDATE_CLOSURE]["closed_km"]
        piece_km = piece_run["summary"]["disruption"][CANDIDATE_CLOSURE]["closed_km"]
        assert road_km > piece_km


def _nested_keys(value: Any) -> List[str]:
    """Every mapping key in a nested structure, for vocabulary checks."""
    keys: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_nested_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.extend(_nested_keys(child))
    return keys


def _table_for(tmp_path: Path, roads: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Closure table for hand-written road records, without the full run."""
    exposure = _exposure_dir(
        tmp_path, roads, _settlement_records([SETTLEMENT]), name="m6_4_table"
    )
    inputs = load_network_inputs(exposure)
    network, _, _ = build_candidate_road_network(
        inputs.roads, analysis_crs(inputs.exposure_summary), settlements=inputs.settlements
    )
    return closure_table(network, inputs.roads, RULES)


# ---------------------------------------------------------------------------
# Settlement reachability
# ---------------------------------------------------------------------------


class TestSettlementReachability:
    def test_reachable_when_nothing_closes(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        row = run["results"].iloc[0]
        assert row["hospital_status"] == SCENARIO_REACHABLE
        assert row["hospital_km_baseline"] == pytest.approx(3.0)
        assert row["hospital_hub_baseline"] == "Hub One"

    def test_closing_the_only_road_makes_it_unreachable_in_the_scenario(
        self, tmp_path: Path
    ):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], candidate_m={"way/0": 40.0})
        row = run["results"].iloc[0]
        assert row["hospital_status"] == SCENARIO_UNREACHABLE
        assert row["hospital_km_candidate_closure"] is None
        group = run["summary"]["hub_reachability"]["groups"]["hospital"]
        assert group["scenario_unreachable"][0]["name"] == "S0"

    def test_long_detour_is_reported_as_a_detour(self, tmp_path: Path):
        run = _run(
            tmp_path, [ROAD_A, DETOUR_B], [SETTLEMENT], candidate_m={"way/0": ALL}
        )
        row = run["results"].iloc[0]
        assert row["hospital_status"] == SCENARIO_DETOURED
        assert row["hospital_km_candidate_closure"] == pytest.approx(7.0)

    def test_short_detour_stays_reachable(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A, SHORT_C], [SETTLEMENT], candidate_m={"way/0": ALL})
        assert run["results"].iloc[0]["hospital_status"] == SCENARIO_REACHABLE

    def test_not_observed_only_is_undetermined_not_safe(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], not_observed_m={"way/0": 200.0})
        row = run["results"].iloc[0]
        assert row["hospital_status"] == UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED
        assert row["hospital_reason"] == REASON_UNOBSERVED_ONLY
        assert row["hospital_km_candidate_closure"] == pytest.approx(3.0)
        assert row["hospital_km_candidate_and_unobserved_closure"] is None

    def test_no_baseline_route_is_insufficient_data_not_unreachable(self, tmp_path: Path):
        island = _line((10_000, 0), (10_500, 0), (11_000, 0))
        run = _run(
            tmp_path,
            [ROAD_A, island],
            [_pt(10_500, 20)],
            candidate_m={"way/1": ALL},
        )
        row = run["results"].iloc[0]
        assert row["hospital_status"] == INSUFFICIENT_DATA
        assert row["hospital_reason"] == REASON_NO_BASELINE_ROUTE_SCENARIO

    def test_settlement_with_no_mapped_road_gets_no_verdict(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [_pt(5000, 5000)])
        row = run["results"].iloc[0]
        assert row["hospital_status"] == INSUFFICIENT_DATA
        assert row["hospital_reason"] == REASON_NO_MAPPED_ROAD
        assert run["summary"]["coverage_uncertainty"]["settlements_without_mapped_road"] == 1

    def test_hub_off_the_network_is_reported_as_such(self, tmp_path: Path):
        far = (_pt(50_000, 50_000), "hospital", "Far")
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], [far])
        assert run["results"].iloc[0]["hospital_reason"] == REASON_NO_HUB_ON_NETWORK
        assert (
            run["summary"]["hub_reachability"]["groups"]["hospital"]["hubs_on_network"] == 0
        )

    def test_hub_groups_are_classified_separately(self, tmp_path: Path):
        hubs = [HUB, (_pt(3000, -2030), "town", "Town")]
        run = _run(
            tmp_path,
            [ROAD_A, _line((3000, 0), (3000, -2000))],
            [SETTLEMENT],
            hubs,
            candidate_m={"way/0": ALL},
        )
        row = run["results"].iloc[0]
        assert row["hospital_status"] == SCENARIO_UNREACHABLE
        assert row["town_status"] == SCENARIO_REACHABLE

    def test_nearest_hub_changes_when_the_route_closes(self, tmp_path: Path):
        hubs = [HUB, (_pt(10_000, 30), "hospital", "Hub Two")]
        run = _run(
            tmp_path,
            [ROAD_A, _line((3000, 0), (10_000, 0))],
            [SETTLEMENT],
            hubs,
            candidate_m={"way/0": ALL},
        )
        row = run["results"].iloc[0]
        assert (row["hospital_hub_baseline"], row["hospital_hub_candidate_closure"]) == (
            "Hub One",
            "Hub Two",
        )
        assert row["hospital_status"] == SCENARIO_DETOURED

    def test_without_hubs_reachability_is_not_assessed_rather_than_assumed(
        self, tmp_path: Path
    ):
        run = _run(
            tmp_path,
            [ROAD_A],
            [SETTLEMENT],
            candidate_m={"way/0": ALL},
            with_hubs=False,
        )
        reachability = run["summary"]["hub_reachability"]
        assert reachability["assessed"] is False
        assert "not computed" in reachability["reason"]
        assert reachability["groups"] == {}
        # Road disruption is still a real result.
        assert run["summary"]["disruption"][CANDIDATE_CLOSURE]["pieces_closed"] == 1
        assert "hospital_status" not in run["results"].columns

    def test_settlements_without_names_do_not_break_the_output(self, tmp_path: Path):
        settlements = _settlement_records([SETTLEMENT])
        del settlements[0]["name"]
        del settlements[0]["place"]
        exposure = _exposure_dir(tmp_path, _road_records([ROAD_A]), settlements)
        inputs = load_network_inputs(exposure, osm_dir=_osm_dir(tmp_path, _hub_records([HUB])))
        network, settlement_snap, hub_snap = build_candidate_road_network(
            inputs.roads,
            analysis_crs(inputs.exposure_summary),
            settlements=inputs.settlements,
            hubs=inputs.hubs,
            snap_max_m=RULES.snap_max_m,
        )
        table = closure_table(network, inputs.roads, RULES)
        _, results, _, summary = assess_candidate_disruption(
            network, table, inputs, settlement_snap, hub_snap, RULES
        )
        assert results.iloc[0]["name"] is None
        assert summary["settlements"]["total"] == 1


class TestSensitivity:
    def test_unreachable_count_follows_the_threshold(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], candidate_m={"way/0": 50.0})
        per_threshold = run["summary"]["hub_reachability"]["sensitivity_by_closure_m"][
            "hospital"
        ]
        assert [
            per_threshold[t][SCENARIO_UNREACHABLE] for t in ("10", "30", "100")
        ] == [1, 1, 0]

    def test_a_single_threshold_configuration_is_refused(self):
        with pytest.raises(CandidateDisruptionError, match="at least one threshold"):
            ClosureRules(sensitivity_closure_m=())


# ---------------------------------------------------------------------------
# Outputs, accounting and vocabulary
# ---------------------------------------------------------------------------


class TestOutputs:
    def test_open_pieces_are_retained_and_labelled(self, tmp_path: Path):
        run = _run(
            tmp_path, [ROAD_A, DETOUR_B], [SETTLEMENT], candidate_m={"way/0": ALL}
        )
        pieces = run["pieces"]
        assert len(pieces) == len(run["network"].pieces)
        assert set(pieces["closure_status"]) == {
            CLOSED_IN_CANDIDATE_SCENARIO,
            OPEN_IN_BOTH_SCENARIOS,
        }
        assert pieces.crs.to_epsg() == 4326

    def test_routes_match_the_reported_distances(self, tmp_path: Path):
        run = _run(
            tmp_path, [ROAD_A, DETOUR_B], [SETTLEMENT], candidate_m={"way/0": ALL}
        )
        routes = run["routes"]
        assert routes.crs.to_epsg() == 4326
        lengths = routes.set_index("scenario").to_crs(UTM).length / 1000.0
        row = run["results"].iloc[0]
        assert lengths["baseline"] == pytest.approx(row["hospital_km_baseline"], abs=1e-3)
        assert lengths[CANDIDATE_CLOSURE] == pytest.approx(
            row["hospital_km_candidate_closure"], abs=1e-3
        )

    def test_route_schema_is_stable_when_no_route_exists(self, tmp_path: Path):
        run = _run(
            tmp_path,
            [ROAD_A],
            [SETTLEMENT],
            with_hubs=False,
        )
        assert list(run["routes"].columns) == list(ROUTE_COLUMNS) + ["geometry"]
        assert len(run["routes"]) == 0

    def test_not_observed_length_is_reported_separately_from_disruption(
        self, tmp_path: Path
    ):
        run = _run(
            tmp_path,
            [ROAD_A, DETOUR_B],
            [SETTLEMENT],
            candidate_m={"way/0": 40.0},
            not_observed_m={"way/1": 100.0},
        )
        summary = run["summary"]
        accounting = summary["road_length_accounting"]
        uncertainty = summary["coverage_uncertainty"]
        assert accounting["not_observed_km"] == pytest.approx(0.1, abs=1e-3)
        assert uncertainty["not_observed_km"] == accounting["not_observed_km"]
        assert uncertainty["pieces_closed_only_if_unobserved_closed"] == 1
        # Not-observed length is not folded into the modelled closure total.
        assert summary["disruption"][CANDIDATE_CLOSURE]["closed_km"] == pytest.approx(
            3.0, abs=1e-3
        )
        assert "unaffected" not in json.dumps(
            {k: v for k, v in accounting.items() if k != "note"}
        )
        assert "never counted as unaffected road" in accounting["note"]

    def test_accounting_keeps_the_four_m6_4_statuses_distinct(self, tmp_path: Path):
        run = _run(
            tmp_path,
            [ROAD_A, DETOUR_B, SHORT_C],
            [SETTLEMENT],
            candidate_m={"way/0": 40.0},
            not_observed_m={"way/1": 100.0},
        )
        by_status = run["summary"]["road_length_accounting"]["km_by_m6_4_status"]
        assert by_status[CANDIDATE_INTERSECTING] == pytest.approx(3.0, abs=1e-3)
        assert by_status[UNOBSERVED] == pytest.approx(7.0, abs=1e-3)
        assert by_status[NO_CANDIDATE_INTERSECTION] == pytest.approx(
            SHORT_C.length / 1000.0, abs=1e-3
        )
        assert by_status[OUTSIDE_CANDIDATE_GRID] == 0.0

    def test_summary_states_the_closure_rule_and_its_evidence_basis(self, tmp_path: Path):
        summary = _run(tmp_path, [ROAD_A], [SETTLEMENT])["summary"]
        rule = summary["closure_rule"]
        assert rule["unit"] == "road"
        assert rule["threshold_m"] == 10.0
        assert "NOT observed impassability" in rule["evidence_basis"]
        assert summary["status"] == "hypothetical_closure_scenario"
        assert len(summary["assumptions"]) == len(ASSUMPTIONS) >= 10
        assert len(summary["limitations"]) == len(LIMITATIONS) >= 8
        assert "conditional on the stated closure rule" in summary["interpretation"]
        assert summary["inputs"]["retrieval_performed"] is False

    def test_no_output_field_or_status_value_uses_flood_vocabulary(self, tmp_path: Path):
        """M6.4 refuses flood vocabulary in its fields; M6.5 must not reintroduce it.

        Scoped to field names, status values and summary keys — the places a
        reader would take as a measurement. Prose that explicitly disclaims
        ("not flooded length") is required elsewhere and is checked separately.
        """
        run = _run(
            tmp_path, [ROAD_A, DETOUR_B], [SETTLEMENT], candidate_m={"way/0": ALL}
        )
        banned = ("flood", "damage", "destroyed", "permanent_water", "isolated")
        names = [
            *run["pieces"].columns,
            *run["results"].columns,
            *run["routes"].columns,
            *run["pieces"]["closure_status"].astype(str),
            *run["pieces"]["source_candidate_status"].astype(str),
            *run["results"]["hospital_status"].astype(str),
            *_nested_keys(run["summary"]),
            *SETTLEMENT_STATUSES,
        ]
        for name in names:
            assert not any(word in name.lower() for word in banned), name

    def test_prose_disclaims_the_stronger_claims(self, tmp_path: Path):
        summary = _run(tmp_path, [ROAD_A], [SETTLEMENT])["summary"]
        assert "not flooded" in summary["road_length_accounting"]["note"]
        assert "never counted as unaffected road" in summary["road_length_accounting"]["note"]
        assert "not a finding that any road is destroyed" in summary["interpretation"]

    def test_artifacts_are_written_and_reparseable(self, tmp_path: Path):
        run = _run(
            tmp_path, [ROAD_A, DETOUR_B], [SETTLEMENT], candidate_m={"way/0": ALL}
        )
        out = tmp_path / "out"
        provenance = candidate_disruption_provenance(
            inputs=run["inputs"], summary=run["summary"], config_version="0.1.0"
        )
        paths = write_candidate_disruption_artifacts(
            run["pieces"],
            run["results"],
            run["routes"],
            run["summary"],
            out,
            provenance=provenance,
        )
        for label, path in paths.items():
            if path.suffix == ".json" or path.suffix == ".geojson":
                json.loads(path.read_text(encoding="utf-8"))

    def test_writer_emits_the_documented_artifact_set(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        out = tmp_path / "out"
        provenance = candidate_disruption_provenance(
            inputs=run["inputs"], summary=run["summary"], config_version="0.1.0"
        )
        paths = write_candidate_disruption_artifacts(
            run["pieces"],
            run["results"],
            run["routes"],
            run["summary"],
            out,
            provenance=provenance,
        )
        assert set(paths) == {"pieces", "settlements", "routes", "summary", "provenance"}
        for path in paths.values():
            assert path.is_file() and path.stat().st_size > 0
        assert json.loads(paths["summary"].read_text(encoding="utf-8"))["stage"] == (
            CANDIDATE_DISRUPTION_VERSION
        )
        for name in ("pieces", "settlements", "routes"):
            assert json.loads(paths[name].read_text(encoding="utf-8"))["type"] == (
                "FeatureCollection"
            )


class TestProvenance:
    def test_connectivity_run_is_a_settlement_connectivity_artifact(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        record = candidate_disruption_provenance(
            inputs=run["inputs"], summary=run["summary"], config_version="0.1.0"
        )
        assert record.artifact_type == ArtifactType.SETTLEMENT_CONNECTIVITY
        assert record.event_date == EVENT_DATE
        assert record.osm_snapshot_date == OSM_SNAPSHOT
        assert sorted(record.source_product_ids) == ["post_rtc", "pre_rtc"]
        assert "© OpenStreetMap contributors." in record.attribution
        assert any("dijkstra" in op for op in record.preprocessing_operations)

    def test_the_sentinel1_pair_is_carried_forward_not_reduced_to_ids(
        self, tmp_path: Path
    ):
        """AGENTS.md §4/§22 — acquisition times and orbit must stay traversable."""
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        record = candidate_disruption_provenance(
            inputs=run["inputs"], summary=run["summary"], config_version="0.1.0"
        )
        pair = record.sentinel1
        assert pair is not None
        assert pair.before.scene_id == "pre_rtc" and pair.after.scene_id == "post_rtc"
        assert pair.before.relative_orbit == pair.after.relative_orbit == 85
        assert pair.same_relative_orbit is True
        assert str(pair.before.acquired_at).startswith("2026-08-16")

    def test_run_without_hubs_is_a_road_disruption_artifact(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], with_hubs=False)
        record = candidate_disruption_provenance(
            inputs=run["inputs"], summary=run["summary"], config_version="0.1.0"
        )
        assert record.artifact_type == ArtifactType.ROAD_DISRUPTION
        assert not any("dijkstra" in op for op in record.preprocessing_operations)

    def test_provenance_records_the_closure_unit_actually_used(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        record = candidate_disruption_provenance(
            inputs=run["inputs"], summary=run["summary"], config_version="0.1.0"
        )
        assert "apply_hypothetical_closure_rule_unit_road" in record.preprocessing_operations

    def test_provenance_yaml_round_trips(self, tmp_path: Path):
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT])
        record = candidate_disruption_provenance(
            inputs=run["inputs"], summary=run["summary"], config_version="0.1.0"
        )
        assert "settlement_connectivity" in record.to_yaml()


class TestBlocker:
    def test_blocker_names_the_gate_and_the_next_step(self):
        report = disruption_blocker_report(
            reason="required M6.4 artifacts are not present on disk",
            missing=["candidate_exposure_roads.geojson"],
            searched="artifacts/m6_4/candidate_infrastructure_exposure",
            upstream={"stage": CANDIDATE_EXPOSURE_VERSION},
            next_step=("establish AOI provenance",),
        )
        assert report["status"] == "blocked"
        assert report["retrieval_performed"] is False
        assert report["authorisation"]["authorises_download"] is False
        assert report["authorisation"]["aoi_provenance_status"] == "UNVERIFIED"
        assert report["next_step"] == ["establish AOI provenance"]
        assert any("No road network" in line for line in report["not_done"])

    def test_blocker_is_written_as_json(self, tmp_path: Path):
        path = write_disruption_blocker_report(
            disruption_blocker_report(reason="r", missing=[], searched=tmp_path), tmp_path
        )
        assert path.name == "candidate_disruption_blocker.json"
        assert json.loads(path.read_text(encoding="utf-8"))["stage"] == (
            CANDIDATE_DISRUPTION_VERSION
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_config_loads_into_rules(self):
        assert ClosureRules.from_config() == RULES

    def test_default_threshold_matches_the_m6_4_intersection_threshold(self):
        """So the default scenario reproduces M6.4's own CANDIDATE_INTERSECTING."""
        assert (
            load_config("candidate_disruption")["network"]["closure_min_candidate_m"]
            == load_config("candidate_exposure")["intersection"]["line_min_candidate_m"]
        )

    def test_snap_distance_matches_the_impact_connectivity_stage(self):
        assert (
            load_config("candidate_disruption")["network"]["snap_max_m"]
            == load_config("impact")["network"]["snap_max_m"]
        )

    def test_retrieval_is_disabled_in_configuration(self):
        assert load_config("candidate_disruption")["inputs"]["osm"]["allow_download"] is False

    def test_invalid_rules_are_refused(self):
        with pytest.raises(CandidateDisruptionError, match="closure_unit"):
            ClosureRules(closure_unit="segment")
        with pytest.raises(CandidateDisruptionError, match="detour_ratio"):
            ClosureRules(detour_ratio=0.5)
        with pytest.raises(CandidateDisruptionError, match="snap_max_m"):
            ClosureRules(snap_max_m=0.0)
        with pytest.raises(CandidateDisruptionError, match="closure_min_candidate_m"):
            ClosureRules(closure_min_candidate_m=0.0)


class TestVocabulary:
    def test_status_set_is_closed_and_documented(self):
        from floodmap.network.candidate_disruption import STATUS_MEANINGS

        assert set(STATUS_MEANINGS) == set(SETTLEMENT_STATUSES)
        assert all(STATUS_MEANINGS[status] for status in SETTLEMENT_STATUSES)

    def test_unreachable_is_described_as_modelled_not_confirmed(self):
        from floodmap.network.candidate_disruption import STATUS_MEANINGS

        meaning = STATUS_MEANINGS[SCENARIO_UNREACHABLE]
        assert "NOT confirmed isolation" in meaning

    def test_scenario_names_do_not_claim_an_observation(self):
        from floodmap.network.candidate_disruption import SCENARIO_MEANINGS

        assert "hypothetical" in SCENARIO_MEANINGS[CANDIDATE_CLOSURE]
        assert "hypothetical" in SCENARIO_MEANINGS[CANDIDATE_AND_UNOBSERVED_CLOSURE]

    def test_reused_kernel_reasons_are_translated(self, tmp_path: Path):
        """The connectivity kernel's UNKNOWN reason names a flood; ours must not."""
        run = _run(tmp_path, [ROAD_A], [SETTLEMENT], not_observed_m={"way/0": 200.0})
        reason = run["results"].iloc[0]["hospital_reason"]
        assert "flood" not in reason.lower()
        assert reason == REASON_UNOBSERVED_ONLY


class TestNoNetworkMutation:
    def test_routing_leaves_the_graph_unchanged(self, tmp_path: Path):
        """reach_from_hubs attaches a virtual source; it must clean up after itself."""
        run = _run(
            tmp_path, [ROAD_A, DETOUR_B], [SETTLEMENT], candidate_m={"way/0": ALL}
        )
        graph = run["network"].graph
        assert -1 not in graph.nodes
        assert nx.number_connected_components(graph) == 1


class TestCommandLine:
    """The CLI contract: exit codes, blocker artifact, no retrieval."""

    @staticmethod
    def _main():
        import importlib.util

        path = REPO_ROOT / "scripts" / "candidate_disruption.py"
        spec = importlib.util.spec_from_file_location("m6_5_cli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_absent_inputs_exit_two_and_write_a_blocker(self, tmp_path: Path):
        out = tmp_path / "out"
        code = self._main().main(
            ["--exposure-dir", str(tmp_path / "nothing"), "--out-dir", str(out)]
        )
        assert code == 2
        report = json.loads(
            (out / "candidate_disruption_blocker.json").read_text(encoding="utf-8")
        )
        assert report["status"] == "blocked"
        assert report["retrieval_performed"] is False
        assert report["next_step"]

    def test_full_run_exits_zero_and_writes_every_artifact(self, tmp_path: Path):
        exposure = _exposure_dir(
            tmp_path,
            _road_records([ROAD_A, DETOUR_B], candidate_m={"way/0": ALL}),
            _settlement_records([SETTLEMENT]),
        )
        osm = _osm_dir(tmp_path, _hub_records([HUB]))
        out = tmp_path / "out"
        code = self._main().main(
            [
                "--exposure-dir",
                str(exposure),
                "--osm-dir",
                str(osm),
                "--out-dir",
                str(out),
            ]
        )
        assert code == 0
        for name in (
            "candidate_disruption_pieces.geojson",
            "candidate_disruption_settlements.geojson",
            "candidate_disruption_routes.geojson",
            "candidate_disruption_summary.json",
            "candidate_disruption_provenance.yaml",
        ):
            assert (out / name).is_file()
        summary = json.loads(
            (out / "candidate_disruption_summary.json").read_text(encoding="utf-8")
        )
        assert summary["disruption"][CANDIDATE_CLOSURE]["pieces_closed"] >= 1
        assert summary["hub_reachability"]["assessed"] is True

    def test_piece_unit_without_a_raster_exits_one(self, tmp_path: Path):
        exposure = _exposure_dir(
            tmp_path, _road_records([ROAD_A]), _settlement_records([SETTLEMENT])
        )
        code = self._main().main(
            [
                "--exposure-dir",
                str(exposure),
                "--closure-unit",
                "piece",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert code == 1

    def test_bad_input_exits_one_without_writing_results(self, tmp_path: Path):
        roads = _road_records([ROAD_A])
        roads[0]["flooded_m"] = 1.0
        exposure = _exposure_dir(tmp_path, roads, _settlement_records([SETTLEMENT]))
        out = tmp_path / "out"
        assert (
            self._main().main(["--exposure-dir", str(exposure), "--out-dir", str(out)]) == 1
        )
        assert not (out / "candidate_disruption_summary.json").exists()
