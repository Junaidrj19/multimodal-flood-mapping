"""Bounded hydrological selection and the transient flat-cap hairpin regression.

The real-data cases read only the locally delivered HydroRIVERS source. Their
small spatial subset contains the approved upper stem and downstream context;
it is not a production extent definition or an observation footprint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import substring

from floodmap.hydrology import hydrorivers as hr


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (85.37778, 28.27778)
START = 40771762
TERMINUS = 40801674


def _reach(hyriv_id: int, next_down: int = 0) -> hr.Reach:
    return hr.Reach(hyriv_id, next_down, 100.0, 5, 4, 0, 1.0, 10.0)


def _geographic(geometry):
    return hr.reproject_geometry(geometry, src_crs=hr.STUDY_AREA_CRS, dst_crs=hr.DELIVERY_CRS)


@pytest.fixture(params=[1, -1], ids=["eastward", "westward"])
def metric_network(request):
    """A reference between source vertices; a tributary and downstream cycle excluded."""
    direction = request.param
    x, y = 340_000.0, 3_120_000.0
    geometries = {
        40000001: _geographic(LineString([(x, y), (x + direction * 1_000, y)])),
        40000002: _geographic(
            LineString([(x + direction * 1_000, y), (x + direction * 2_000, y)])
        ),
        40000004: _geographic(
            LineString([(x + direction * 1_000, y + 1_000), (x + direction * 1_000, y)])
        ),
    }
    network = hr.HydroRiversNetwork(
        [
            _reach(40000001, 40000002),
            _reach(40000002, 40000003),
            _reach(40000003, 40000003),
            _reach(40000004, 40000002),
        ]
    )
    reference = tuple(_geographic(Point(x + direction * 250, y + 30)).coords[0])
    return network, geometries, reference


def test_hydrological_selection_trims_exactly_and_stops_without_downstream_geometry(metric_network):
    network, geometries, reference = metric_network
    selection = hr.select_hydrological_corridor(
        network,
        geometries,
        reference,
        terminus_reach_id=40000002,
        terminus_index=1,
        buffer_m=100,
    )
    assert selection.path.reach_ids == (40000001, 40000002)
    assert selection.centreline_length_m == pytest.approx(1_750.0, abs=1e-6)
    assert selection.upstream_trim_m == pytest.approx(250.0, abs=1e-6)
    assert selection.reference_offset_m == pytest.approx(30.0, abs=1e-6)
    assert selection.polygon.area == pytest.approx(350_000.0, abs=1e-3)
    ref_utm = hr.reproject_geometry(Point(reference))
    first = hr.reproject_geometry(geometries[40000001])
    projected = first.interpolate(first.project(ref_utm))
    assert selection.polygon.exterior.distance(projected) < 1e-6
    assert selection.polygon.geom_type == "Polygon"
    assert selection.polygon.is_valid
    record = selection.to_record()
    assert record["downstream_stop_reason"] == "caller_stop_reach"
    assert record["terminus_index"] == 1
    assert record["coverage_fraction"] is None
    assert selection.coverage_fraction is None
    assert "hydrological" in record["selection_rule"]
    assert "acceptance checks" in record["selection_rule"]
    assert "furthest downstream" not in record["selection_rule"]


@pytest.mark.parametrize("reverse", [False, True])
def test_nearest_reach_tie_is_broken_by_id_not_mapping_order(reverse):
    line = LineString([(85.0, 28.0), (85.03, 28.0), (85.03, 27.98)])
    pairs = [(40000009, line), (40000001, line)]
    geometries = dict(reversed(pairs) if reverse else pairs)
    network = hr.HydroRiversNetwork([_reach(40000009), _reach(40000001)])
    selection = hr.select_hydrological_corridor(
        network,
        geometries,
        (85.01, 28.0),
        terminus_reach_id=40000001,
        terminus_index=0,
    )
    assert selection.path.reach_ids == (40000001,)
    assert selection.downstream_stop_reason == "caller_stop_reach"
    assert selection.buffer_m == 1_000
    assert selection.cap_style == "flat"
    assert selection.join_style == "round"
    # The public selector's default bend resolution is 32, independent of the
    # legacy helper's default of 8.
    projected = hr.reproject_geometry(line)
    trimmed = hr.trim_line_to_point(projected, hr.reproject_geometry(Point(85.01, 28.0)))
    expected = hr.buffer_corridor(trimmed, 1_000, quad_segs=32)
    assert selection.polygon.wkb == expected.wkb
    assert selection.polygon.wkb != hr.buffer_corridor(trimmed, 1_000, quad_segs=8).wkb


@pytest.mark.parametrize(
    "reach,index", [(40000002, 0), (40000002, 2), (40000001, 1), (40000004, 1)]
)
def test_wrong_terminus_index_or_branch_is_refused(metric_network, reach, index):
    network, geometries, reference = metric_network
    with pytest.raises(ValueError, match="terminus_index .*does not correspond"):
        hr.select_hydrological_corridor(
            network, geometries, reference, terminus_reach_id=reach, terminus_index=index
        )


@pytest.mark.parametrize("index", [-1, 1.5, True])
def test_invalid_index_is_refused(metric_network, index):
    network, geometries, reference = metric_network
    with pytest.raises(ValueError, match="non-negative zero-based integer"):
        hr.select_hydrological_corridor(
            network, geometries, reference, terminus_reach_id=40000002, terminus_index=index
        )


def test_absent_terminus_is_refused(metric_network):
    network, geometries, reference = metric_network
    with pytest.raises(ValueError, match="terminus reach .*not in the loaded network"):
        hr.select_hydrological_corridor(
            network, geometries, reference, terminus_reach_id=49999999, terminus_index=1
        )


def test_missing_selected_geometry_is_refused(metric_network):
    network, geometries, reference = metric_network
    del geometries[40000002]
    with pytest.raises(ValueError, match="no geometry supplied"):
        hr.select_hydrological_corridor(
            network, geometries, reference, terminus_reach_id=40000002, terminus_index=1
        )


def test_reference_outside_the_local_network_is_refused(metric_network):
    network, geometries, _ = metric_network
    with pytest.raises(ValueError, match="no HydroRIVERS reach within"):
        hr.select_hydrological_corridor(
            network, geometries, (86.5, 29.0), terminus_reach_id=40000002, terminus_index=1
        )


@pytest.fixture
def hairpin(monkeypatch):
    """Real GEOS multipart buffer, in a tiny metric-only selection fixture.

    Projection is the identity here so the flat-cap topology can be isolated
    from coordinate conversion. Buffering and all validity checks are real.
    """
    monkeypatch.setattr(hr, "reproject_geometry", lambda geometry, **kwargs: geometry)
    geometries = {
        1: LineString([(-20, 0), (0, 0)]),
        2: LineString([(0, 0), (100, 0), (100, 10), (0, 10)]),
        3: LineString([(0, 10), (0, 500)]),
    }
    network = hr.HydroRiversNetwork([_reach(1, 2), _reach(2, 3), _reach(3)])
    return network, geometries, (-0.01, 0.0), box(-1_000, -1_000, 1_000, 1_000)


def test_legacy_walk_skips_a_real_multipart_prefix_and_delivers_the_whole_polygon(hairpin):
    network, geometries, reference, footprint = hairpin
    prefix = hr.trim_line_to_point(hr.build_corridor_line(geometries, [1, 2]), Point(reference))
    raw = prefix.buffer(200, cap_style="flat", join_style="round", quad_segs=32)
    assert raw.geom_type == "MultiPolygon"
    assert len(raw.geoms) == 2
    assert min(part.area for part in raw.geoms) > 0
    with pytest.raises(ValueError, match="MultiPolygon.*2 parts"):
        hr.buffer_corridor(prefix, 200, quad_segs=32)

    selection = hr.select_coverage_bounded_corridor(
        network, geometries, reference, footprint, buffer_m=200, quad_segs=32
    )
    assert selection.path.reach_ids == (1, 2, 3)
    assert selection.polygon.is_valid
    expected = hr.buffer_corridor(
        hr.trim_line_to_point(hr.build_corridor_line(geometries, [1, 2, 3]), Point(reference)),
        200,
        quad_segs=32,
    )
    assert selection.polygon.equals(expected)
    assert selection.coverage_fraction == pytest.approx(1.0)
    # Legacy provenance and its constructor API retain their existing defaults.
    record = selection.to_record()
    assert "furthest downstream reach" in record["selection_rule"]
    assert "terminus_index" not in record


@pytest.mark.parametrize("policy", ["coverage", "hydrological"])
def test_a_real_multipart_final_corridor_is_still_rejected(hairpin, policy):
    _, geometries, reference, footprint = hairpin
    network = hr.HydroRiversNetwork([_reach(1, 2), _reach(2)])
    with pytest.raises(ValueError, match="MultiPolygon.*largest part"):
        if policy == "coverage":
            hr.select_coverage_bounded_corridor(
                network, geometries, reference, footprint, buffer_m=200, quad_segs=32
            )
        else:
            hr.select_hydrological_corridor(
                network,
                geometries,
                reference,
                terminus_reach_id=2,
                terminus_index=1,
                buffer_m=200,
            )


@pytest.mark.parametrize(
    "bad_buffer,message",
    [
        (Polygon(), "empty polygon"),
        (Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)]), "invalid Polygon"),
    ],
)
def test_other_geometry_failures_at_intermediate_prefixes_are_not_skipped(
    hairpin, monkeypatch, bad_buffer, message
):
    network, geometries, reference, footprint = hairpin
    monkeypatch.setattr(LineString, "buffer", lambda *args, **kwargs: bad_buffer)
    with pytest.raises(ValueError, match=message):
        hr.select_coverage_bounded_corridor(network, geometries, reference, footprint, buffer_m=200)


def test_invalid_buffer_parameters_at_intermediate_prefixes_are_not_skipped(hairpin):
    network, geometries, reference, footprint = hairpin
    with pytest.raises(ValueError, match="must be positive"):
        hr.select_coverage_bounded_corridor(network, geometries, reference, footprint, buffer_m=0)


@pytest.fixture(scope="module")
def local_source():
    source = ROOT / "data/raw/hydrorivers/HydroRIVERS_v10_as.gdb"
    if not source.exists():
        pytest.skip(f"local HydroRIVERS source missing (git-ignored): {source}")
    import pyogrio

    frame = pyogrio.read_dataframe(
        source,
        layer="HydroRIVERS_v10_as",
        columns=list(hr.HYDRORIVERS_COLUMNS),
        bbox=(84.0, 27.0, 86.0, 29.0),
    )
    network = hr.HydroRiversNetwork(
        hr.Reach(
            int(r.HYRIV_ID),
            int(r.NEXT_DOWN),
            float(r.UPLAND_SKM),
            int(r.ORD_STRA),
            int(r.ORD_CLAS),
            int(r.ENDORHEIC),
            float(r.LENGTH_KM),
            float(r.DIS_AV_CMS),
        )
        for r in frame.itertuples()
    )
    geometries = dict(zip(frame.HYRIV_ID.astype(int), frame.geometry))
    path, _ = network.downstream_path(START)
    assert path[49] == TERMINUS
    return network, geometries, path


def _legacy_prefixes(geometries, path):
    """Reproduce the legacy prefix construction, including floating-point cuts."""
    ref = hr.reproject_geometry(Point(REFERENCE))
    merged = hr.reproject_geometry(hr.build_corridor_line(geometries, path))
    trim_offset = merged.project(ref)
    line = hr.trim_line_to_point(merged, ref)
    cumulative = np.cumsum([hr.reproject_geometry(geometries[h]).length for h in path])
    return [substring(line, 0, min(end - trim_offset, line.length)) for end in cumulative]


def test_local_2000m_failure_is_only_the_index1_hairpin_and_walk_continues(local_source):
    network, geometries, path = local_source
    prefixes = _legacy_prefixes(geometries, path)
    raw = prefixes[1].buffer(2_000, cap_style="flat", join_style="round", quad_segs=32)
    assert raw.geom_type == "MultiPolygon"
    assert len(raw.geoms) == 3
    detached = sum(p.area for p in sorted(raw.geoms, key=lambda p: -p.area)[1:])
    assert detached == pytest.approx(1_315_257.6, abs=0.1)
    with pytest.raises(ValueError, match="MultiPolygon with 3 parts"):
        hr.buffer_corridor(prefixes[1], 2_000, quad_segs=32)
    assert hr.buffer_corridor(prefixes[2], 2_000, quad_segs=32).is_valid
    assert hr.buffer_corridor(prefixes[49], 2_000, quad_segs=32).is_valid

    # Deliberately generous synthetic containment, not an observation assertion.
    selection = hr.select_coverage_bounded_corridor(
        network,
        geometries,
        REFERENCE,
        box(83, 26, 87, 30),
        buffer_m=2_000,
        quad_segs=32,
    )
    assert selection.path.reach_ids == tuple(path)
    assert selection.polygon.geom_type == "Polygon"
    assert selection.polygon.is_valid


@pytest.mark.parametrize("width,area", [(500, 104.7079), (1_000, 206.1509), (2_000, 400.9262)])
def test_local_hydrological_corridor_matches_m613_at_each_width(local_source, width, area):
    network, geometries, path = local_source
    selection = hr.select_hydrological_corridor(
        network,
        geometries,
        REFERENCE,
        terminus_reach_id=TERMINUS,
        terminus_index=49,
        buffer_m=width,
    )
    assert selection.path.reach_ids == tuple(path[:50])
    assert selection.path.start_reach_id == START
    assert selection.path.end_reach_id == TERMINUS
    assert selection.centreline_length_m == pytest.approx(105_858, abs=0.5)
    # The brief rounded a cumulative-distance prefix of the full downstream
    # line; a bounded merge uses the exact reach endpoint. GEOS flat-cap
    # intersections differ by sub-millimetre cuts (less than 0.001 km² here).
    assert selection.area_km2 == pytest.approx(area, abs=0.001)
    assert selection.polygon.geom_type == "Polygon"
    assert selection.polygon.is_valid
    assert selection.coverage_fraction is None
    assert selection.downstream_stop_reason == "caller_stop_reach"
    assert all(network.get(h).ord_stra == 5 for h in selection.path.reach_ids[1:])
    assert network.get(network.get(TERMINUS).next_down).ord_stra == 6


@pytest.mark.parametrize("reach,index", [(TERMINUS, 50), (40801919, 49)])
def test_local_approved_reach_and_index_must_agree(local_source, reach, index):
    network, geometries, _ = local_source
    with pytest.raises(ValueError, match="terminus_index .*does not correspond"):
        hr.select_hydrological_corridor(
            network, geometries, REFERENCE, terminus_reach_id=reach, terminus_index=index
        )


def test_legacy_1000m_geometry_generation_is_byte_identical(local_source):
    network, geometries, _ = local_source
    # Original delivered RTC window, explicitly retained only for this legacy
    # reproducibility regression. It is not an input to hydrological selection.
    footprint = _geographic(box(308_770, 3_080_970, 341_900, 3_130_580))
    selection = hr.select_coverage_bounded_corridor(
        network, geometries, REFERENCE, footprint, buffer_m=1_000, quad_segs=32
    )
    assert selection.terminus_reach_id == 40799268
    original = json.loads(
        (ROOT / "configs/aoi/trishuli-corridor-hydrorivers-v10.geojson").read_text()
    )["features"][0]["geometry"]
    generated = mapping(_geographic(selection.polygon))
    assert json.dumps(generated, separators=(",", ":")) == json.dumps(
        original, separators=(",", ":")
    )
