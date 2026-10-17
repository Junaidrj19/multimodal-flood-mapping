"""HydroRIVERS main-stem corridor selection and construction.

What this module is for
-----------------------
Milestone M6.10 replaces the provenance-unknown Trishuli corridor with a
**derived, traceable** study extent: the main stem of the river the challenge
names ("Bhote Koshi--Trishuli corridor"), taken from HydroRIVERS v1.0 and
buffered into a corridor polygon.

This module produces *geometry and its provenance inputs*.  It does **not**
authorise anything: it never downloads, never queries a catalogue, and the
result it returns carries no acquisition permission.  Authorisation is a
separate operator decision (`floodmap.acquisition.aoi.AoiProvenanceStatus`).

Data source and verified schema
-------------------------------
HydroRIVERS v1.0, regional tile ``HydroRIVERS_v10_as`` (Central and South-East
Asia), delivered as an ESRI file geodatabase and distributed in WGS84
geographic coordinates (``GCS_WGS_1984``).  Attribute semantics below are read
from the provider's own *HydroRIVERS Technical Documentation v1.0* §2.1, §3.1,
§3.2:

===========================  ====================================================
``HYRIV_ID``                 unique reach id, 8 digits; first digit = region
                             (4 = Asia), remaining 7 identify the reach
``NEXT_DOWN``                ``HYRIV_ID`` of the next downstream segment;
                             ``0`` = no downstream connection
``MAIN_RIV``                 ``HYRIV_ID`` of the most downstream reach of the
                             connected river basin
``LENGTH_KM``                reach length, km
``DIST_DN_KM`` / ``DIST_UP_KM``  distances to the downstream/upstream network end
``CATCH_SKM`` / ``UPLAND_SKM``   direct and total upstream area, km²
``ENDORHEIC``                1 = endorheic (inland) basin
``DIS_AV_CMS``               long-term average discharge, m³/s
``ORD_STRA``                 Strahler order
``ORD_CLAS``                 classical order; order 1 = "backbone" main stem
``ORD_FLOW``                 discharge size class 1..10
``HYBAS_L12``                containing HydroBASINS level-12 sub-basin id
===========================  ====================================================

The delivered layer also carries ESRI's ``Shape_Length`` (degrees, because the
delivery CRS is geographic).  It is ignored here: it is not a provider
attribute and its unit is not metres.

Why the corridor is a *single downstream path*
----------------------------------------------
The approved policy for this milestone is **main stem only, no tributaries**.
A downstream walk is therefore the whole selection rule: at every junction the
network's ``NEXT_DOWN`` pointer leads to the *merged* river, so following it can
never branch and can never wander onto a tributary.  The only place a branch
choice is needed is *upstream* of the chosen start reach, which is outside the
corridor.  That makes the corridor deterministic once its two terminal nodes are
named, and it removes the usual "which branch is the main stem?" failure mode
from the part of the network that actually ends up in the AOI.

Terminal rules are *inputs*, not assumptions
--------------------------------------------
The documented reference and downstream policy are caller inputs. This module
never invents an extent decision:

* the upstream terminal is derived from the caller's documented geographic
  reference (e.g. a border crossing), projected onto the nearest mapped reach,
  and must be recorded in the provenance;
* the downstream terminal is chosen by the caller from a policy stated in the
  milestone record (coverage bound, hydrological node, ...) and must be recorded
  in the provenance. Hydrological selection verifies a named reach/index pair
  without a footprint; swath coverage and inset remain caller acceptance checks.

That separation is deliberate.  `AGENTS.md` §11 forbids the model from being the
source of truth for a numerical or extent decision, and `docs/m6.9-...md` §8
lists the terminal choices as decisions rather than measurements.

Geospatial discipline
---------------------
The delivery CRS is geographic, so **no metric operation happens in it**:
geometry is reprojected to EPSG:32645 before buffering, and the buffer width is
therefore metres on the ground (`AGENTS.md` §7; M6.9 §5.2 measured the UTM 45N
scale-factor error at 1 km as 0.092 m).  Buffering in degrees would be 1.125×
anisotropic at this latitude.

Simplification and validity repair are deliberately **not** applied: the AOI is
hash-pinned, so any vertex-count change is a geometry change, and an invalid
input is a finding to report rather than something to silently repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, substring, unary_union

__all__ = [
    "Reach",
    "HydroRiversNetwork",
    "CorridorPath",
    "CorridorSelection",
    "STUDY_AREA_CRS",
    "DELIVERY_CRS",
    "choose_upstream_branch",
    "build_corridor_line",
    "drop_repeated_vertices",
    "trim_line_to_point",
    "buffer_corridor",
    "reproject_geometry",
    "load_hydrorivers",
    "select_coverage_bounded_corridor",
    "select_hydrological_corridor",
]

#: CRS the AOI is delivered in.  `AreaOfInterest` accepts only this
#: (`floodmap.acquisition.aoi.SUPPORTED_AOI_CRS`).
DELIVERY_CRS = "EPSG:4326"

#: CRS used for every metric operation.  UTM zone 45N covers 84-90 E, which
#: contains the whole corridor, and it is already the project's analysis grid.
STUDY_AREA_CRS = "EPSG:32645"


@dataclass(frozen=True)
class Reach:
    """One HydroRIVERS reach, network attributes only (no geometry).

    Geometry is kept out of this record so the topology of a 1.4-million-reach
    continental layer can be loaded without reading 1.4 million polylines.
    """

    hyriv_id: int
    next_down: int
    upland_skm: float
    ord_stra: int
    ord_clas: int
    endorheic: int
    length_km: float
    dis_av_cms: float

    @property
    def region_digit(self) -> int:
        """First digit of ``HYRIV_ID`` — the provider's region code (4 = Asia)."""
        return int(str(int(self.hyriv_id))[0])


def choose_upstream_branch(candidates: Sequence[Reach]) -> Reach:
    """Pick the main-stem branch at a fork, with a fixed tie-break order.

    The order matters more than the rule: it has to be decided *before* the run
    and recorded, or the selection is not reproducible (`AGENTS.md` §14).

    1. larger total upstream area (``UPLAND_SKM``) — the conventional main-stem
       criterion;
    2. then higher Strahler order (``ORD_STRA``);
    3. then lower ``HYRIV_ID``, so the result cannot depend on iteration order.

    Raises:
        ValueError: if ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("no upstream candidates to choose from")
    return sorted(candidates, key=lambda r: (-r.upland_skm, -r.ord_stra, r.hyriv_id))[0]


class HydroRiversNetwork:
    """The ``NEXT_DOWN`` topology over a set of reaches.

    The network is deliberately *incomplete* by design: callers may load only the
    attributes they need, and every traversal method reports whether it stopped
    because it reached a network end, hit a reach that is absent from the loaded
    set, or hit the caller's own stop condition.  An absent reach is never
    silently treated as "no downstream connection".
    """

    def __init__(self, reaches: Iterable[Reach]) -> None:
        by_id: dict[int, Reach] = {}
        for reach in reaches:
            if reach.hyriv_id in by_id:
                raise ValueError(f"duplicate HYRIV_ID {reach.hyriv_id} in the reach set")
            by_id[reach.hyriv_id] = reach
        self._by_id = by_id
        parents: dict[int, list[int]] = {}
        for reach in by_id.values():
            parents.setdefault(reach.next_down, []).append(reach.hyriv_id)
        self._parents = {k: tuple(sorted(v)) for k, v in parents.items()}

    # ---- accessors -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, hyriv_id: object) -> bool:
        return hyriv_id in self._by_id

    def get(self, hyriv_id: int) -> Reach:
        try:
            return self._by_id[hyriv_id]
        except KeyError as exc:  # pragma: no cover - message detail only
            raise KeyError(f"HYRIV_ID {hyriv_id} is not in the loaded network") from exc

    def parents(self, hyriv_id: int) -> tuple[int, ...]:
        """Upstream neighbours whose ``NEXT_DOWN`` points at ``hyriv_id``."""
        return self._parents.get(hyriv_id, ())

    def reach_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_id))

    # ---- traversals ------------------------------------------------------
    def upstream_path(self, seed: int, *, max_steps: int = 10_000) -> list[int]:
        """Main-stem path from ``seed`` to the most upstream reach.

        Follows :func:`choose_upstream_branch` at every fork.  Returns
        ``[seed, parent, ..., headwater]``.
        """
        if seed not in self._by_id:
            raise KeyError(f"seed HYRIV_ID {seed} is not in the loaded network")
        path = [seed]
        seen = {seed}
        for _ in range(max_steps):
            candidates = [self._by_id[p] for p in self.parents(path[-1]) if p in self._by_id]
            if not candidates:
                break
            nxt = choose_upstream_branch(candidates)
            if nxt.hyriv_id in seen:
                raise ValueError(
                    f"cycle detected in the upstream traversal at HYRIV_ID {nxt.hyriv_id}; "
                    "the delivered topology is not a tree and must not be silently truncated"
                )
            path.append(nxt.hyriv_id)
            seen.add(nxt.hyriv_id)
        return path

    def downstream_path(
        self,
        seed: int,
        *,
        stop_at: Optional[Iterable[int]] = None,
        max_steps: int = 100_000,
    ) -> tuple[list[int], str]:
        """Path from ``seed`` following ``NEXT_DOWN``, with why it stopped.

        Returns ``(reach_ids, stop_reason)``.  ``stop_at`` names reach ids to
        include and then stop at (inclusive), so a caller can express "the last
        reach inside the coverage gate" without this method needing to know
        anything about rasters or footprints.

        The stop reason is one of ``"caller_stop_reach"``,
        ``"no_downstream_connection"`` (``NEXT_DOWN == 0``) or
        ``"reach_<id>_absent_from_loaded_set"``.  A reach missing from the loaded
        set is **not** treated as the river's end, and the reason string says so.
        """
        if seed not in self._by_id:
            raise KeyError(f"seed HYRIV_ID {seed} is not in the loaded network")
        stops = set(stop_at or ())
        path = [seed]
        seen = {seed}
        stop_reason = "network_end"
        for _ in range(max_steps):
            nxt = self.get(path[-1]).next_down
            if nxt == 0:
                stop_reason = "no_downstream_connection"
                break
            if nxt not in self._by_id:
                stop_reason = f"reach_{nxt}_absent_from_loaded_set"
                break
            if nxt in seen:
                raise ValueError(
                    f"cycle detected in the downstream traversal at HYRIV_ID {nxt}"
                )
            path.append(nxt)
            seen.add(nxt)
            if nxt in stops:
                stop_reason = "caller_stop_reach"
                break
        return path, stop_reason

    def summarise_path(self, path: Sequence[int]) -> "CorridorPath":
        """Build the auditable :class:`CorridorPath` record for ``path``."""
        if not path:
            raise ValueError("cannot summarise an empty path")
        reaches = [self.get(h) for h in path]
        return CorridorPath(
            reach_ids=tuple(path),
            start_reach_id=int(path[0]),
            end_reach_id=int(path[-1]),
            length_km=float(sum(r.length_km for r in reaches)),
            upstream_area_start_skm=float(reaches[0].upland_skm),
            upstream_area_end_skm=float(reaches[-1].upland_skm),
            endorheic_reaches=tuple(r.hyriv_id for r in reaches if r.endorheic != 0),
            non_asia_reaches=tuple(r.hyriv_id for r in reaches if r.region_digit != 4),
        )


@dataclass(frozen=True)
class CorridorPath:
    """A selected main-stem path plus the audit trail that justifies it."""

    reach_ids: tuple[int, ...]
    start_reach_id: int
    end_reach_id: int
    length_km: float
    upstream_area_start_skm: float
    upstream_area_end_skm: float
    endorheic_reaches: tuple[int, ...]
    non_asia_reaches: tuple[int, ...]

    def to_record(self) -> dict:
        return {
            "start_reach_id": self.start_reach_id,
            "end_reach_id": self.end_reach_id,
            "reach_count": len(self.reach_ids),
            "reach_ids": list(self.reach_ids),
            "length_km": round(self.length_km, 3),
            "upstream_area_start_skm": round(self.upstream_area_start_skm, 3),
            "upstream_area_end_skm": round(self.upstream_area_end_skm, 3),
            "endorheic_reaches": list(self.endorheic_reaches),
            "non_asia_region_reaches": list(self.non_asia_reaches),
        }


def build_corridor_line(
    geometries: Mapping[int, LineString],
    path: Sequence[int],
    *,
    require_single_part: bool = True,
) -> LineString:
    """Merge an ordered reach path into one continuous line.

    Every reach must have geometry, the merged result must be a single
    ``LineString``, and the path must be contiguous.  A discontinuous merge means
    the path is not a single river course, which is a construction error rather
    than something to fix by snapping.
    """
    missing = [h for h in path if h not in geometries]
    if missing:
        raise ValueError(f"no geometry supplied for reaches {missing[:5]}")
    parts = [geometries[h] for h in path]
    for h, geom in zip(path, parts):
        if geom.is_empty:
            raise ValueError(f"reach {h} has empty geometry")
        if geom.geom_type not in ("LineString", "MultiLineString"):
            raise ValueError(f"reach {h} has geometry type {geom.geom_type}, expected a line")
    union = unary_union([_as_line(g) for g in parts])
    # A single-part path is already one line; ``linemerge`` refuses a bare
    # LineString, so the trivial case is returned before it is asked.
    merged = union if union.geom_type == "LineString" else linemerge(union)
    if merged.geom_type == "MultiLineString":
        if require_single_part:
            raise ValueError(
                f"the selected path merges into {len(merged.geoms)} disconnected parts; "
                "the path is not one continuous river course"
            )
        merged = max(merged.geoms, key=lambda g: g.length)
    return drop_repeated_vertices(merged)


def drop_repeated_vertices(line: LineString) -> LineString:
    """Remove consecutive duplicate coordinates from a line.

    Reach boundaries and cut points routinely produce the same coordinate twice.
    A zero-length segment carries no geometric information, but buffering it with
    a *flat* cap makes GEOS emit a degenerate zero-area sliver as a separate part
    of a ``MultiPolygon`` — which the AOI contract (one ``Polygon``) then refuses,
    for a reason that has nothing to do with the corridor. Dropping the repeats is
    a degeneracy guard, not a geometry repair: no vertex that describes shape is
    removed, and the line's path length is unchanged.
    """
    coords = list(line.coords)
    kept = [coords[0]]
    for xy in coords[1:]:
        if xy != kept[-1]:
            kept.append(xy)
    if len(kept) < 2:
        raise ValueError("the line collapses to a single point after removing repeated vertices")
    return LineString(kept) if len(kept) != len(coords) else line


def _as_line(geom) -> LineString:
    if geom.geom_type == "MultiLineString":
        return linemerge(geom) if len(geom.geoms) > 1 else LineString(geom.geoms[0].coords)
    return geom


def trim_line_to_point(line: LineString, point: Point) -> LineString:
    """Cut ``line`` where ``point`` projects onto it, keeping the downstream part.

    "Downstream" means towards larger distance along the merged line, which is
    the direction the reaches were assembled in (upstream -> downstream).

    The cut is placed at the *projected* point, not snapped to the nearest
    vertex.  Snapping would move the corridor's start by up to half a reach
    vertex spacing, which on a coarse source like HydroRIVERS is hundreds of
    metres - the provenance would then claim an endpoint the geometry does not
    have.  The reference is a documented *place* rather than a surveyed station,
    so the perpendicular offset from the line is reported separately (see
    :func:`nearest_point_on_line`) instead of being absorbed into the cut.
    """
    if line.is_empty:
        raise ValueError("cannot trim an empty line")
    offset = float(line.project(point))
    if offset <= 0.0:
        raise ValueError(
            "the cut point projects to the upstream end of the line; the trim would "
            "remove nothing and change the corridor's length silently"
        )
    if offset >= line.length:
        raise ValueError("the cut point falls on the last vertex; nothing would remain")
    return drop_repeated_vertices(substring(line, offset, line.length))


def nearest_point_on_line(line: LineString, point: Point) -> tuple[float, Point]:
    """Return ``(offset_towards, projected_point)`` for ``point`` on ``line``.

    The offset is the distance along the line from its start, in the line's own
    units (metres when the line is in a projected CRS).
    """
    projected = line.interpolate(line.project(point))
    return float(line.project(point)), projected


def reproject_geometry(
    geometry,
    *,
    src_crs: str = DELIVERY_CRS,
    dst_crs: str = STUDY_AREA_CRS,
):
    """Reproject with an explicit, recorded transform (never an implicit one)."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely.transform(
        geometry, lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1]))
    )


class _NonPolygonCorridorError(ValueError):
    """A buffer violates the single-Polygon contract, possibly only at a prefix."""


def buffer_corridor(
    line_in_study_crs: LineString,
    buffer_m: float,
    *,
    cap_style: str = "flat",
    join_style: str = "round",
    quad_segs: int = 8,
) -> Polygon:
    """Buffer a single line into the corridor polygon, in metres.

    ``cap_style="flat"`` is deliberate.  A round cap bulges one buffer radius
    beyond each chosen terminal node (about 1.57 km² per terminal at 1 km), which
    would move the corridor past the endpoint the provenance claims, and would
    push a coverage-inset terminal back outside the inset it was chosen for.
    Flat caps make the corridor end exactly at the terminal cross-section, which
    is perpendicular to the local line direction, and keep area linear in length.
    """
    if buffer_m <= 0:
        raise ValueError(f"buffer distance must be positive, got {buffer_m}")
    # Guard at the point of use: a prefix cut at a vertex can end with a
    # zero-length segment, and a flat cap over one produces a zero-area sliver
    # that makes the result a MultiPolygon for reasons unrelated to the corridor.
    line_in_study_crs = drop_repeated_vertices(line_in_study_crs)
    polygon = line_in_study_crs.buffer(
        buffer_m, cap_style=cap_style, join_style=join_style, quad_segs=quad_segs
    )
    if polygon.is_empty:
        raise ValueError("buffering produced an empty polygon")
    if polygon.geom_type != "Polygon":
        parts = sorted(getattr(polygon, "geoms", [polygon]), key=lambda g: -g.area)
        detached_m2 = sum(p.area for p in parts[1:])
        raise _NonPolygonCorridorError(
            f"the buffered corridor is a {polygon.geom_type} with {len(parts)} parts "
            f"({detached_m2:.1f} m^2 outside the largest), but the AOI contract accepts a "
            "single Polygon; a MultiPolygon must be reported, never reduced to its "
            "largest part"
        )
    if not polygon.is_valid:
        raise ValueError("buffering produced an invalid Polygon; geometry must not be repaired silently")
    return polygon


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

#: Attribute columns read from the delivered file geodatabase.  Reading only
#: these keeps a continental layer tractable; ``Shape_Length`` is excluded on
#: purpose (see the module docstring).
HYDRORIVERS_COLUMNS = (
    "HYRIV_ID",
    "NEXT_DOWN",
    "UPLAND_SKM",
    "ORD_STRA",
    "ORD_CLAS",
    "ENDORHEIC",
    "LENGTH_KM",
    "DIS_AV_CMS",
)


def load_hydrorivers(
    gdb_path,
    *,
    layer: Optional[str] = None,
) -> tuple["HydroRiversNetwork", dict]:
    """Read the HydroRIVERS file geodatabase into topology and geometry.

    An ESRI file geodatabase can only be read through GDAL, so ``pyogrio`` (the
    GDAL Python binding used elsewhere in this project's geo stack) is imported
    lazily.  Requiring it at module import would make every consumer of this
    module depend on the optional geo extra.

    Returns ``(network, geometries)`` with geometries in the delivery CRS
    (WGS84 geographic).  Nothing is reprojected or repaired here.
    """
    try:  # pragma: no cover - import failure is environment-specific
        import pyogrio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "reading HydroRIVERS requires pyogrio/GDAL; install the project's 'geo' "
            "extra before constructing a corridor from the delivered geodatabase"
        ) from exc

    from pathlib import Path

    path = Path(gdb_path)
    if not path.exists():
        raise FileNotFoundError(f"HydroRIVERS source not found: {path}")
    if layer is None:
        layers = pyogrio.list_layers(path)
        if not len(layers):
            raise ValueError(f"{path} contains no layers")
        layer = layers[0][0]

    frame = pyogrio.read_dataframe(path, layer=layer, columns=list(HYDRORIVERS_COLUMNS))
    reaches: dict[int, Reach] = {}
    geometries: dict[int, object] = {}
    for row, geometry in zip(frame.itertuples(index=False), frame.geometry):
        hyriv_id = int(row.HYRIV_ID)
        reaches[hyriv_id] = Reach(
            hyriv_id=hyriv_id,
            next_down=int(row.NEXT_DOWN),
            upland_skm=float(row.UPLAND_SKM),
            ord_stra=int(row.ORD_STRA),
            ord_clas=int(row.ORD_CLAS),
            endorheic=int(row.ENDORHEIC),
            length_km=float(row.LENGTH_KM),
            dis_av_cms=float(row.DIS_AV_CMS),
        )
        geometries[hyriv_id] = geometry
    return HydroRiversNetwork(reaches.values()), geometries


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorridorSelection:
    """A constructed corridor: geometry, the path behind it, and its policy.

    All distances are metres and ``polygon`` is in :data:`STUDY_AREA_CRS`, so
    ``polygon.area`` is in square metres.  Delivery/CRS conversion is the
    caller's job and is recorded separately.

    ``coverage_fraction=None`` means coverage has not been measured. Hydrological
    selection has no footprint input; its caller must separately check the common
    same-track swath and any required inset before accepting the geometry.
    """

    path: CorridorPath
    terminus_reach_id: int
    buffer_m: float
    cap_style: str
    join_style: str
    reference_offset_m: float
    #: How much of the start reach lies *above* the reference cut. The corridor
    #: begins at the reference, so ``path.length_km`` (the summed source reach
    #: lengths) is this much longer than the corridor itself.
    upstream_trim_m: float
    centreline_length_m: float
    polygon: Polygon
    coverage_fraction: Optional[float]
    downstream_stop_reason: str
    selection_rule: str = (
        "main-stem downstream path from the documented reference; terminus is the "
        "furthest downstream reach whose fully buffered corridor lies inside the "
        "delivered product footprint"
    )
    terminus_index: Optional[int] = None

    @property
    def area_km2(self) -> float:
        return float(self.polygon.area) / 1e6

    def to_record(self) -> dict:
        record = {
            "selection_rule": self.selection_rule,
            "selected_reaches": self.path.to_record(),
            "terminus_reach_id": self.terminus_reach_id,
            "terminus_is_river_end": False,
            "buffer_m": self.buffer_m,
            "cap_style": self.cap_style,
            "join_style": self.join_style,
            "reference_offset_m": round(self.reference_offset_m, 2),
            "upstream_trim_m": round(self.upstream_trim_m, 2),
            "centreline_length_m": round(self.centreline_length_m, 2),
            "area_km2": round(self.area_km2, 4),
            "coverage_fraction": (
                round(self.coverage_fraction, 6) if self.coverage_fraction is not None else None
            ),
            "downstream_stop_reason": self.downstream_stop_reason,
        }
        if self.terminus_index is not None:
            record["terminus_index"] = self.terminus_index
        return record


def select_hydrological_corridor(
    network: HydroRiversNetwork,
    geometries: Mapping[int, object],
    reference_point: tuple[float, float],
    *,
    terminus_reach_id: int,
    terminus_index: int,
    buffer_m: float = 1_000.0,
    cap_style: str = "flat",
    join_style: str = "round",
    quad_segs: int = 32,
) -> CorridorSelection:
    """Construct a main-stem corridor ending at a caller-named hydrological reach.

    Inputs are in EPSG:4326; trimming, distances and buffering use EPSG:32645.
    The start is the mapped reach nearest the documented reference within the
    local 0.05-degree search window, ranked by metric distance then HYRIV_ID.
    ``terminus_index`` is zero-based from that start, and must identify exactly
    ``terminus_reach_id`` along NEXT_DOWN (inclusive). Neither a tributary nor
    any continuation below the named terminus is included or needed.

    The reference is projected onto the continuous, downstream-oriented line,
    never snapped to a source vertex. For a single reach its delivered coordinate
    order supplies the downstream direction; multi-reach order is checked against
    the first and last reaches rather than inferred from GEOS's merge direction.

    No footprint participates in this selection. Coverage and the 250 m inset
    against an independently derived common same-track swath are acceptance-only
    checks for the caller. A returned geometry alone asserts neither of them.
    """
    if (
        isinstance(terminus_index, bool)
        or not isinstance(terminus_index, Integral)
        or terminus_index < 0
    ):
        raise ValueError("terminus_index must be a non-negative zero-based integer")
    if isinstance(terminus_reach_id, bool) or not isinstance(terminus_reach_id, Integral):
        raise ValueError("terminus_reach_id must be an integer HYRIV_ID")
    if terminus_reach_id not in network:
        raise ValueError(f"terminus reach {terminus_reach_id} is not in the loaded network")

    ref_lon, ref_lat = reference_point
    ref_utm = reproject_geometry(Point(reference_point))
    nearby = [
        h
        for h, g in geometries.items()
        if h in network
        and g.bounds[0] <= ref_lon + 0.05
        and g.bounds[2] >= ref_lon - 0.05
        and g.bounds[1] <= ref_lat + 0.05
        and g.bounds[3] >= ref_lat - 0.05
    ]
    if not nearby:
        raise ValueError(
            f"no HydroRIVERS reach within 0.05 degrees of ({ref_lon}, {ref_lat}); "
            "the reference does not correspond to a mapped river reach"
        )
    start = min(
        nearby,
        key=lambda h: (float(reproject_geometry(geometries[h]).distance(ref_utm)), h),
    )
    # Bound the traversal itself: missing geometry or a topology defect below
    # the approved terminus cannot change a hydrologically selected extent.
    if terminus_index == 0:
        path = [start]
    else:
        path, _ = network.downstream_path(
            start, stop_at=[terminus_reach_id], max_steps=int(terminus_index)
        )
    if len(path) != terminus_index + 1 or path[-1] != terminus_reach_id:
        raise ValueError(
            f"terminus_index {terminus_index} does not correspond to terminus_reach_id "
            f"{terminus_reach_id} along NEXT_DOWN from {start}; traversal reached "
            f"{path[-1]} at index {len(path) - 1}"
        )

    merged = reproject_geometry(build_corridor_line(geometries, path))
    if len(path) > 1:
        first = reproject_geometry(_as_line(geometries[path[0]]))
        last = reproject_geometry(_as_line(geometries[path[-1]]))
        if merged.project(first.interpolate(0.5, normalized=True)) > merged.project(
            last.interpolate(0.5, normalized=True)
        ):
            merged = LineString(list(merged.coords)[::-1])
    trim_offset = float(merged.project(ref_utm))
    reference_offset_m = float(merged.distance(ref_utm))
    # An exact upstream-end reference already defines the required cross-section.
    part = trim_line_to_point(merged, ref_utm) if trim_offset > 0.0 else merged
    polygon = buffer_corridor(
        part, buffer_m, cap_style=cap_style, join_style=join_style, quad_segs=quad_segs
    )
    return CorridorSelection(
        path=network.summarise_path(path),
        terminus_reach_id=int(terminus_reach_id),
        buffer_m=float(buffer_m),
        cap_style=cap_style,
        join_style=join_style,
        reference_offset_m=reference_offset_m,
        upstream_trim_m=trim_offset,
        centreline_length_m=float(part.length),
        polygon=polygon,
        coverage_fraction=None,
        downstream_stop_reason="caller_stop_reach",
        selection_rule=(
            "main-stem downstream NEXT_DOWN path from the documented reference, cut at its "
            "exact projected point; terminus is the caller-named hydrological reach at the "
            "verified zero-based index; no tributaries or downstream trunk expansion; "
            "swath coverage and inset are separate acceptance checks"
        ),
        terminus_index=int(terminus_index),
    )


def select_coverage_bounded_corridor(
    network: HydroRiversNetwork,
    geometries: Mapping[int, object],
    reference_point: tuple[float, float],
    footprint,
    *,
    buffer_m: float = 1_000.0,
    cap_style: str = "flat",
    join_style: str = "round",
    quad_segs: int = 8,
    search_window_degrees: float = 0.05,
) -> CorridorSelection:
    """Build the corridor from a documented reference point and a data footprint.

    The rule, in full:

    1. the **start** is the HydroRIVERS reach nearest the documented reference,
       cut at the reference itself (its offset from the mapped river is reported,
       because a documented place is not a surveyed station);
    2. the **course** is the downstream ``NEXT_DOWN`` path from that reach, so no
       tributary can enter it and no branch choice is needed;
    3. the **terminus** is the furthest downstream reach whose *entire* buffered
       corridor lies inside ``footprint``.  The walk stops at the first reach
       that does not fit, which is recorded, so the corridor never includes a
       stretch the products do not observe;
    4. coverage is then measured, and returned, rather than assumed to be 1.

    ``footprint`` is the intersection of the delivered product footprints, in
    EPSG:4326.  Passing ``footprint=None`` is refused: an unbounded corridor
    would run the length of the river system, which is not this milestone's AOI.
    """
    if footprint is None:
        raise ValueError(
            "a coverage footprint is required: without one the corridor would run down the "
            "entire river system instead of the observed reach"
        )
    ref = Point(reference_point)
    ref_lon, ref_lat = reference_point

    # Cheap pre-filter on bounds, then exact ranking on the small remainder.
    window = float(search_window_degrees)
    nearby = [
        h
        for h, g in geometries.items()
        if g.bounds[0] <= ref_lon + window
        and g.bounds[2] >= ref_lon - window
        and g.bounds[1] <= ref_lat + window
        and g.bounds[3] >= ref_lat - window
    ]
    if not nearby:
        raise ValueError(
            f"no HydroRIVERS reach within {search_window_degrees} degrees of "
            f"({ref_lon}, {ref_lat}); the reference does not correspond to a mapped river reach"
        )
    ref_utm = reproject_geometry(ref)

    def _distance_m(hyriv_id: int) -> float:
        """Exact distance from the reference to one candidate reach, metres."""
        return float(reproject_geometry(geometries[hyriv_id]).distance(ref_utm))

    start = min(nearby, key=_distance_m)
    reference_offset_m = _distance_m(start)

    path, stop_reason = network.downstream_path(start)
    merged = reproject_geometry(build_corridor_line(geometries, path))
    trim_offset = float(merged.project(ref_utm))
    # Measured against the untrimmed line: after the cut the reference lies *on*
    # the corridor boundary by construction, so a post-trim distance would be a
    # rounding artefact rather than the offset it claims to report.
    reference_offset_m = float(merged.distance(ref_utm))
    corridor_start = trim_line_to_point(merged, ref_utm)

    footprint_utm = reproject_geometry(footprint)
    cumulative = np.concatenate(
        [[0.0], np.cumsum([reproject_geometry(geometries[h]).length for h in path])]
    )

    best = None
    for index in range(len(path)):
        end_along = float(cumulative[index + 1]) - trim_offset
        if end_along <= 0.0:
            continue
        part = substring(corridor_start, 0.0, min(end_along, corridor_start.length))
        try:
            polygon = buffer_corridor(
                part, buffer_m, cap_style=cap_style, join_style=join_style, quad_segs=quad_segs
            )
        except _NonPolygonCorridorError:
            # A short flat-capped hairpin can be multipart even when a later
            # prefix is a valid Polygon. It is not a deliverable candidate, but
            # must not abort the walk. The final prefix still fails the contract;
            # empty/invalid buffers and all unrelated failures always propagate.
            if index == len(path) - 1:
                raise
            continue
        if polygon.within(footprint_utm):
            best = (index, part, polygon)
        elif best is not None:
            break

    if best is None:
        raise ValueError(
            f"no reach of the main stem has a fully covered {buffer_m:.0f} m corridor inside the "
            "delivered footprint; the footprint is too small for this buffer distance"
        )

    index, part, polygon = best
    coverage = float(polygon.intersection(footprint_utm).area / polygon.area) if polygon.area else 0.0
    trimmed_path = network.summarise_path(path[: index + 1])
    return CorridorSelection(
        path=trimmed_path,
        terminus_reach_id=int(path[index]),
        buffer_m=float(buffer_m),
        cap_style=cap_style,
        join_style=join_style,
        reference_offset_m=reference_offset_m,
        upstream_trim_m=trim_offset,
        centreline_length_m=float(part.length),
        polygon=polygon,
        coverage_fraction=coverage,
        downstream_stop_reason=stop_reason,
    )
