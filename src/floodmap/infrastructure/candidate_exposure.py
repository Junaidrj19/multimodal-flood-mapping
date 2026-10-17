"""M6.4 — infrastructure intersection with M6.3 SAR change candidates.

Scientific responsibility
-------------------------
Report which pre-event OpenStreetMap roads, buildings and settlements lie on
pixels that the M6.3 Sentinel-1 RTC stage flagged as dual-polarisation
darkening **candidates**, and which lie on pixels that were never observed.

This is a *candidate-exposure* analysis. A candidate pixel is radar change
evidence; radar darkening is not unique to inundation, and the M6.3 mask is not
ground truth. So every verdict here is named for what was actually measured — a
spatial intersection with a candidate region — and never as flooding, damage or
disruption (``AGENTS.md`` §8).

What this stage does NOT do
---------------------------
* It does not download anything. Pre-event OSM is read from GeoJSON already on
  disk. The study AOI is configured ``UNVERIFIED``
  (``AreaOfInterest.authorises_download`` is ``false``), and an UNVERIFIED
  extent must not authorise a retrieval. When the required layers are absent
  this stage emits an explicit blocker (:func:`blocker_report`) instead of
  substituting, simplifying or synthesising infrastructure.
* It does not run the M5 segmentation model, and it does not convert the
  candidate mask into the three-class segmentation contract. See
  :mod:`floodmap.infrastructure.candidate_mask`.
* It does not route, and it draws no connectivity conclusion. It preserves the
  OSM identifiers and attributes M6.5 needs so that road-network connectivity
  can consume these results unchanged.

Measurement
-----------
Geometry measurement reuses the tested kernel in
:mod:`floodmap.infrastructure.exposure` rather than reimplementing it:

* **Lines** (roads, bridges) are sampled every ``line_sample_step_m`` metres in
  the candidate mask's projected CRS; each sample carries an equal share of the
  line's length and takes the class of the pixel containing it.
* **Polygons** (buildings) are rasterised onto the candidate grid with
  ``all_touched=True``, so any pixel the footprint touches counts.
* **Settlements** are point locations. The verdict uses the pixel containing
  the settlement's representative point, which is the same point M6.5 snaps to
  the road graph. A positive ``settlement_buffer_m`` turns that point into a
  disc and is measured as a polygon.

Geometries are reprojected into the candidate mask's projected CRS before any
measurement, so lengths and areas are in metres and no comparison relies on a
bounding box.

The kernel speaks of a "flood" class because it was written for the
segmentation mask. The class it tests is supplied by
:func:`floodmap.infrastructure.candidate_mask.candidate_semantics`, and every
column and status it returns is translated to candidate vocabulary here, so no
M6.4 artifact contains the word "flooded".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from floodmap.infrastructure.candidate_mask import (
    CANDIDATE,
    NOT_OBSERVED_VALUE,
    CandidateMask,
)
from floodmap.infrastructure.exposure import (
    EXPOSED,
    NOT_EXPOSED,
    NOT_OBSERVED,
    UNKNOWN,
    ExposureRules,
    line_status,
    measure_lines,
    measure_polygons,
    polygon_status,
    sample_classes,
)
from floodmap.infrastructure.osm import OSM_ATTRIBUTION, OSM_LICENSE, parse_timestamp
from floodmap.utils.config import load_config
from floodmap.utils.provenance import (
    ArtifactProvenance,
    ArtifactType,
    BeforeAfterPair,
    ProductionInput,
    SceneReference,
)

CANDIDATE_EXPOSURE_VERSION = "m6.4-candidate-infrastructure-exposure-0.1.0"

# ---------------------------------------------------------------------------
# Status vocabulary
#
# Named for the measurement, not for an inferred consequence. "Intersects a
# change candidate" is a spatial statement about a rule-based SAR observation.
# ---------------------------------------------------------------------------

#: Enough of the feature lies on candidate pixels to meet the configured rule.
CANDIDATE_INTERSECTING = "CANDIDATE_INTERSECTING"
#: No candidate intersection on observed pixels, but enough of the feature lies
#: on not-observed pixels that a candidate there could not have been seen.
UNOBSERVED = "UNOBSERVED"
#: Observed across the feature, and the candidate rule did not fire.
NO_CANDIDATE_INTERSECTION = "NO_CANDIDATE_INTERSECTION"
#: The feature lies entirely outside the candidate raster's extent.
OUTSIDE_CANDIDATE_GRID = "OUTSIDE_CANDIDATE_GRID"

CANDIDATE_STATUSES: Tuple[str, ...] = (
    CANDIDATE_INTERSECTING,
    UNOBSERVED,
    NO_CANDIDATE_INTERSECTION,
    OUTSIDE_CANDIDATE_GRID,
)

STATUS_MEANINGS: Dict[str, str] = {
    CANDIDATE_INTERSECTING: (
        "overlaps M6.3 Sentinel-1 RTC darkening candidates by at least the configured "
        "rule threshold; this is spatial coincidence with radar change evidence, not "
        "confirmed flooding or damage"
    ),
    UNOBSERVED: (
        "no candidate intersection on observed pixels, but enough of the feature lies on "
        "not-observed pixels that a candidate region there could not have been detected"
    ),
    NO_CANDIDATE_INTERSECTION: (
        "observed across the feature, and the M6.3 candidate rule did not fire; this is "
        "not evidence that the feature was unaffected by the event"
    ),
    OUTSIDE_CANDIDATE_GRID: "lies entirely outside the candidate raster extent",
}

#: Translation of the shared kernel's segmentation vocabulary.
_STATUS_FROM_KERNEL = {
    EXPOSED: CANDIDATE_INTERSECTING,
    UNKNOWN: UNOBSERVED,
    NOT_EXPOSED: NO_CANDIDATE_INTERSECTION,
    NOT_OBSERVED: OUTSIDE_CANDIDATE_GRID,
}

#: Kernel line columns kept, under candidate names.
_LINE_COLUMNS = {
    "length_m": "length_m",
    "flooded_m": "candidate_m",
    "unknown_m": "not_observed_m",
    "outside_m": "outside_grid_m",
}
#: Kernel polygon columns kept, under candidate names.
_POLYGON_COLUMNS = {
    "pixels": "pixels_touched",
    "flood_px": "candidate_px",
    "unknown_px": "not_observed_px",
}

# ---------------------------------------------------------------------------
# Local pre-event OSM input
# ---------------------------------------------------------------------------

#: Filenames written by ``scripts/impact.py::write_osm_layers``. M6.4 reads that
#: existing on-disk format rather than defining a second one.
LAYER_FILENAMES: Dict[str, str] = {
    "roads": "osm_roads.geojson",
    "bridges": "osm_bridges.geojson",
    "buildings": "osm_buildings.geojson",
    "settlements": "osm_settlements.geojson",
    "hubs": "osm_hubs.geojson",
}
PROVENANCE_SIDECAR = "osm_provenance.json"

#: RFC 7946 fixes GeoJSON coordinates to CRS84. Used only when a reader reports
#: no CRS, and recorded in the output notes whenever it is relied upon.
GEOJSON_DEFAULT_CRS = "EPSG:4326"

#: OSM attributes carried through to the outputs so M6.5 can build the road
#: graph and name settlements without re-reading the source layers.
CARRIED_ATTRIBUTES: Tuple[str, ...] = (
    "osm_id",
    "name",
    "osm_version",
    "highway",
    "bridge",
    "building",
    "place",
    "amenity",
    "hub_group",
    "tags",
)


class CandidateExposureError(ValueError):
    """The M6.4 inputs or configuration are not usable."""


class MissingInfrastructureLayers(CandidateExposureError):
    """Required pre-event OSM layers are not present on disk.

    Raised rather than worked around. Fabricating or approximating
    infrastructure would make every count in the report unattributable.
    """

    def __init__(self, message: str, *, missing: Sequence[str], searched: Path) -> None:
        super().__init__(message)
        self.missing = list(missing)
        self.searched = Path(searched)


@dataclass(frozen=True)
class IntersectionRules:
    """Explicit M6.4 intersection thresholds (``configs/candidate_exposure.yaml``)."""

    line_sample_step_m: float = 5.0
    line_min_candidate_m: float = 10.0
    polygon_min_candidate_px: int = 1
    settlement_buffer_m: float = 0.0

    def __post_init__(self) -> None:
        if self.line_sample_step_m <= 0:
            raise CandidateExposureError("line_sample_step_m must be positive")
        if self.line_min_candidate_m <= 0:
            raise CandidateExposureError("line_min_candidate_m must be positive")
        if self.polygon_min_candidate_px < 1:
            raise CandidateExposureError("polygon_min_candidate_px must be at least 1")
        if self.settlement_buffer_m < 0:
            raise CandidateExposureError("settlement_buffer_m must not be negative")

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "IntersectionRules":
        block = (config or load_config("candidate_exposure"))["intersection"]
        return cls(
            line_sample_step_m=float(block["line_sample_step_m"]),
            line_min_candidate_m=float(block["line_min_candidate_m"]),
            polygon_min_candidate_px=int(block["polygon_min_candidate_px"]),
            settlement_buffer_m=float(block.get("settlement_buffer_m", 0.0)),
        )

    def as_kernel_rules(self) -> ExposureRules:
        """The same thresholds in the shared measurement kernel's field names."""
        return ExposureRules(
            line_sample_step_m=self.line_sample_step_m,
            line_min_flooded_m=self.line_min_candidate_m,
            building_min_flood_px=self.polygon_min_candidate_px,
        )


@dataclass
class LocalOsmLayers:
    """Pre-event OSM layers read from disk, with their recorded provenance."""

    directory: Path
    layers: Dict[str, gpd.GeoDataFrame]
    provenance: Dict[str, Any]
    snapshot: Optional[datetime]
    missing_optional: List[str]
    notes: List[str]

    def layer_report(self) -> Dict[str, Any]:
        return {
            name: {
                "path": str(self.directory / LAYER_FILENAMES[name]),
                "features": int(len(frame)),
                "crs": str(frame.crs),
                "geometry_types": sorted(
                    {str(value) for value in frame.geom_type.dropna().unique().tolist()}
                ),
            }
            for name, frame in self.layers.items()
        }

    def attribution(self) -> List[str]:
        """Attribution strings recorded with the layers, as a list.

        ``scripts/impact.py`` writes a single string; a hand-assembled sidecar
        may hold a list. Both are accepted, and the required OSM attribution is
        added when the sidecar carries none.
        """
        recorded = self.provenance.get("attribution")
        if isinstance(recorded, str):
            values = [recorded]
        elif isinstance(recorded, Sequence):
            values = [str(v) for v in recorded if v]
        else:
            values = []
        if not any(OSM_ATTRIBUTION in value for value in values):
            values.append(OSM_ATTRIBUTION)
        return list(dict.fromkeys(values))


def _read_layer(path: Path, name: str, notes: List[str]) -> gpd.GeoDataFrame:
    try:
        frame = gpd.read_file(path)
    except Exception as exc:
        raise CandidateExposureError(f"could not read OSM layer {name} from {path}: {exc}") from exc
    if frame.crs is None:
        frame = frame.set_crs(GEOJSON_DEFAULT_CRS)
        notes.append(
            f"{path.name} declared no CRS; RFC 7946 CRS84 ({GEOJSON_DEFAULT_CRS}) was assumed"
        )
    return frame


def load_local_osm_layers(
    directory: Path | str,
    *,
    required_layers: Sequence[str] = ("roads", "buildings", "settlements"),
    optional_layers: Sequence[str] = ("bridges", "hubs"),
    require_provenance_sidecar: bool = True,
    event_date: Optional[str] = None,
    require_pre_event_snapshot: bool = True,
) -> LocalOsmLayers:
    """Read pre-event OSM layers that already exist on disk. No retrieval.

    Raises
    ------
    MissingInfrastructureLayers
        When the directory or any required layer is absent. The caller is
        expected to surface this as a blocker, not to continue.
    CandidateExposureError
        When the provenance sidecar is missing, or records a snapshot that is
        not strictly before the event (``AGENTS.md`` §3 — post-event OSM is
        validation-only).
    """
    directory = Path(directory)
    requested = set(required_layers) | set(optional_layers)
    unknown = requested - set(LAYER_FILENAMES)
    if unknown:
        raise CandidateExposureError(f"unknown OSM layer names requested: {sorted(unknown)}")

    if not directory.is_dir():
        raise MissingInfrastructureLayers(
            f"pre-event OSM layer directory does not exist: {directory}",
            missing=list(required_layers),
            searched=directory,
        )

    notes: List[str] = []
    missing_required = [
        name for name in required_layers if not (directory / LAYER_FILENAMES[name]).is_file()
    ]
    if missing_required:
        raise MissingInfrastructureLayers(
            "required pre-event OSM layers are not present on disk: "
            + ", ".join(f"{n} ({LAYER_FILENAMES[n]})" for n in missing_required),
            missing=missing_required,
            searched=directory,
        )

    sidecar_path = directory / PROVENANCE_SIDECAR
    sidecar: Dict[str, Any] = {}
    if sidecar_path.is_file():
        try:
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CandidateExposureError(f"could not read {sidecar_path}: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise CandidateExposureError(f"{sidecar_path} must contain a JSON object")
        sidecar = dict(loaded)
    elif require_provenance_sidecar:
        raise CandidateExposureError(
            f"{sidecar_path} is missing. The OSM snapshot date and the AOI the layers "
            "belong to must be recorded; layers of unknown origin cannot support an "
            "attributable exposure result (AGENTS.md §14)."
        )
    else:
        notes.append(f"{PROVENANCE_SIDECAR} absent; OSM snapshot and AOI are unrecorded")

    snapshot: Optional[datetime] = None
    raw_snapshot = sidecar.get("snapshot")
    if raw_snapshot:
        try:
            snapshot = parse_timestamp(str(raw_snapshot))
        except Exception as exc:
            raise CandidateExposureError(
                f"{sidecar_path} records an unparseable OSM snapshot {raw_snapshot!r}: {exc}"
            ) from exc
    elif require_provenance_sidecar:
        raise CandidateExposureError(
            f"{sidecar_path} does not record an OSM 'snapshot'; the pre-event rule "
            "cannot be checked"
        )

    recorded_event = sidecar.get("event_date")
    if event_date is not None and recorded_event is not None and recorded_event != event_date:
        raise CandidateExposureError(
            f"OSM layers were fetched for event_date {recorded_event!r} but the candidate "
            f"mask belongs to {event_date!r}; mixing temporal windows is refused "
            "(AGENTS.md §4)"
        )
    if require_pre_event_snapshot and snapshot is not None and event_date is not None:
        event_start = parse_timestamp(f"{event_date}T00:00:00Z")
        if snapshot >= event_start:
            raise CandidateExposureError(
                f"OSM snapshot {snapshot.isoformat()} is not strictly before the event day "
                f"{event_date}; post-event OSM is validation-only (AGENTS.md §3)"
            )

    layers: Dict[str, gpd.GeoDataFrame] = {}
    for name in (*required_layers, *optional_layers):
        path = directory / LAYER_FILENAMES[name]
        if path.is_file():
            layers[name] = _read_layer(path, name, notes)

    missing_optional = [name for name in optional_layers if name not in layers]
    for name in missing_optional:
        notes.append(f"optional layer {name!r} absent; it is excluded from the summary")

    return LocalOsmLayers(
        directory=directory,
        layers=layers,
        provenance=sidecar,
        snapshot=snapshot,
        missing_optional=missing_optional,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# AOI association
# ---------------------------------------------------------------------------


def check_layers_overlap_mask(osm: LocalOsmLayers, mask: CandidateMask) -> Dict[str, Any]:
    """Confirm the OSM extent and the candidate raster describe the same place.

    A non-overlapping pair is almost always a wrong-AOI mistake, and would
    otherwise produce a confident report of zero exposure. The per-layer counts
    use a real geometric intersection against the raster footprint; the bounding
    boxes are recorded for the reader but are not the exposure test.
    """
    footprint = (
        gpd.GeoSeries([box(*mask.bounds)], crs=mask.crs).to_crs(GEOJSON_DEFAULT_CRS).iloc[0]
    )
    recorded_bbox = osm.provenance.get("aoi_bbox")
    report: Dict[str, Any] = {
        "candidate_mask_bbox_epsg4326": [round(v, 6) for v in footprint.bounds],
        "osm_recorded_aoi_bbox_epsg4326": recorded_bbox,
        "osm_fetch_bbox_epsg4326": osm.provenance.get("fetch_bbox"),
        "overlaps_recorded_aoi": None,
    }
    if recorded_bbox is not None:
        recorded = box(*[float(v) for v in recorded_bbox])
        report["overlaps_recorded_aoi"] = bool(footprint.intersects(recorded))
        if not report["overlaps_recorded_aoi"]:
            raise CandidateExposureError(
                f"the OSM layers record AOI bbox {recorded_bbox} which does not overlap the "
                f"candidate raster extent {report['candidate_mask_bbox_epsg4326']}; these "
                "inputs describe different areas"
            )

    mask_footprint_projected = box(*mask.bounds)
    per_layer: Dict[str, Any] = {}
    for name, frame in osm.layers.items():
        if len(frame) == 0:
            per_layer[name] = {"features": 0, "intersecting_mask_extent": 0}
            continue
        in_extent = int(frame.to_crs(mask.crs).intersects(mask_footprint_projected).sum())
        per_layer[name] = {"features": int(len(frame)), "intersecting_mask_extent": in_extent}
    report["layers"] = per_layer
    return report


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_points(geoms: gpd.GeoSeries, mask: CandidateMask) -> pd.DataFrame:
    """Class of the pixel containing each point.

    ``geoms`` must already be in the mask CRS and hold exactly one coordinate
    per geometry. Columns match the polygon table so the two compose.
    """
    index = geoms.index
    measured = pd.DataFrame(
        {
            "pixels_touched": pd.Series(0, index=index, dtype="int64"),
            "candidate_px": pd.Series(0, index=index, dtype="int64"),
            "not_observed_px": pd.Series(0, index=index, dtype="int64"),
        }
    )
    if len(geoms) == 0:
        return measured

    coords = shapely.get_coordinates(geoms.values)
    if coords.shape[0] != len(geoms):
        raise CandidateExposureError(
            "measure_points expects exactly one coordinate per geometry; pass "
            "representative points rather than lines or polygons"
        )
    classes = sample_classes(mask, coords[:, 0], coords[:, 1])
    measured["pixels_touched"] = (classes >= 0).astype("int64")
    measured["candidate_px"] = (classes == CANDIDATE).astype("int64")
    measured["not_observed_px"] = (classes == NOT_OBSERVED_VALUE).astype("int64")
    return measured


def _check_no_permanent_water(measured: pd.DataFrame) -> None:
    """The candidate contract has no permanent-water class.

    The shared kernel still measures one. It must come back empty; a non-zero
    value would mean the mask semantics were built wrongly. M6.4 never publishes
    a permanent-water quantity, because it never measured one.
    """
    for column in ("permanent_water_m", "permanent_water_px"):
        if column in measured and float(np.abs(measured[column].to_numpy()).sum()) != 0.0:
            raise CandidateExposureError(
                "the candidate mask produced a non-zero permanent-water measurement, which "
                "is impossible under the M6.3 contract; the mask semantics are wrong"
            )


def _empty_polygon_measurements(index: pd.Index) -> pd.DataFrame:
    """Kernel-shaped polygon table for an empty layer.

    ``measure_polygons`` builds its frame from a list of row dicts, so with no
    rows it returns a frame with no columns at all. This keeps the column
    contract stable for empty inputs.
    """
    return pd.DataFrame(
        {name: pd.Series(0, index=index, dtype="int64") for name in _POLYGON_COLUMNS}
    )


def assess_lines(
    frame: gpd.GeoDataFrame, mask: CandidateMask, rules: IntersectionRules
) -> gpd.GeoDataFrame:
    """Candidate intersection for a line layer (roads, bridges)."""
    kernel_rules = rules.as_kernel_rules()
    projected = frame.geometry.to_crs(mask.crs)
    measured = measure_lines(projected, mask, kernel_rules.line_sample_step_m)
    _check_no_permanent_water(measured)
    status = line_status(measured, kernel_rules)
    renamed = measured[list(_LINE_COLUMNS)].rename(columns=dict(_LINE_COLUMNS))
    renamed["not_candidate_m"] = (
        renamed["length_m"]
        - renamed["candidate_m"]
        - renamed["not_observed_m"]
        - renamed["outside_grid_m"]
    ).clip(lower=0.0)
    out = frame.copy()
    for column in renamed.columns:
        out[column] = np.round(renamed[column].to_numpy(dtype=float), 3)
    out["candidate_status"] = [_STATUS_FROM_KERNEL[value] for value in status]
    return out


def assess_polygons(
    frame: gpd.GeoDataFrame, mask: CandidateMask, rules: IntersectionRules
) -> gpd.GeoDataFrame:
    """Candidate intersection for a polygon layer (buildings)."""
    kernel_rules = rules.as_kernel_rules()
    projected = frame.geometry.to_crs(mask.crs)
    if len(frame) == 0:
        measured = _empty_polygon_measurements(frame.index)
    else:
        measured = measure_polygons(projected, mask)
        _check_no_permanent_water(measured)
    status = polygon_status(measured, kernel_rules)
    renamed = measured[list(_POLYGON_COLUMNS)].rename(columns=dict(_POLYGON_COLUMNS))
    renamed["not_candidate_px"] = (
        renamed["pixels_touched"] - renamed["candidate_px"] - renamed["not_observed_px"]
    ).clip(lower=0)
    out = frame.copy()
    for column in renamed.columns:
        out[column] = renamed[column].to_numpy()
    out["candidate_status"] = [_STATUS_FROM_KERNEL[value] for value in status]
    return out


def assess_settlements(
    frame: gpd.GeoDataFrame, mask: CandidateMask, rules: IntersectionRules
) -> gpd.GeoDataFrame:
    """Candidate intersection for settlement features.

    Settlements are treated as point locations: the verdict uses the pixel
    containing each settlement's representative point, which is the same point
    M6.5 snaps to the road graph, so the two stages cannot disagree about where
    a settlement is. With a positive ``settlement_buffer_m`` the point becomes a
    disc and is measured as a polygon.
    """
    points = frame.to_crs(mask.crs).representative_point()
    if rules.settlement_buffer_m > 0:
        discs = points.buffer(rules.settlement_buffer_m)
        measured = (
            _empty_polygon_measurements(frame.index)
            if len(frame) == 0
            else measure_polygons(discs, mask)
        )
        if len(frame):
            _check_no_permanent_water(measured)
        status = [
            _STATUS_FROM_KERNEL[value]
            for value in polygon_status(measured, rules.as_kernel_rules())
        ]
        measured = measured[list(_POLYGON_COLUMNS)].rename(columns=dict(_POLYGON_COLUMNS))
    else:
        measured = measure_points(points, mask)
        status = list(
            np.select(
                [
                    measured["pixels_touched"].to_numpy() == 0,
                    measured["candidate_px"].to_numpy() > 0,
                    measured["not_observed_px"].to_numpy() > 0,
                ],
                [OUTSIDE_CANDIDATE_GRID, CANDIDATE_INTERSECTING, UNOBSERVED],
                default=NO_CANDIDATE_INTERSECTION,
            )
        )

    measured = measured.copy()
    measured["not_candidate_px"] = (
        measured["pixels_touched"] - measured["candidate_px"] - measured["not_observed_px"]
    ).clip(lower=0)
    out = frame.copy()
    for column in measured.columns:
        out[column] = measured[column].to_numpy()
    out["candidate_status"] = status
    out["settlement_buffer_m"] = rules.settlement_buffer_m
    return out


#: Which assessor handles which layer, and therefore which layers M6.4 reports.
LAYER_KINDS: Dict[str, str] = {
    "roads": "line",
    "bridges": "line",
    "buildings": "polygon",
    "settlements": "point",
}

_ASSESSORS = {"line": assess_lines, "polygon": assess_polygons, "point": assess_settlements}


def assess_candidate_intersections(
    layers: Mapping[str, gpd.GeoDataFrame], mask: CandidateMask, rules: IntersectionRules
) -> Dict[str, gpd.GeoDataFrame]:
    """Assess every supported layer present in ``layers``."""
    assessed: Dict[str, gpd.GeoDataFrame] = {}
    for name, kind in LAYER_KINDS.items():
        if name in layers:
            assessed[name] = _ASSESSORS[kind](layers[name], mask, rules)
    return assessed


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _optional_value(row: pd.Series, column: str) -> Any:
    """Safely read an optional OSM attribute from a row."""
    if column not in row.index:
        return None
    value = row[column]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value if isinstance(value, (str, int, bool)) else str(value)


def summarize_candidate_exposure(
    assessed: Mapping[str, gpd.GeoDataFrame],
    mask: CandidateMask,
    rules: IntersectionRules,
    *,
    osm: Optional[LocalOsmLayers] = None,
    aoi_report: Optional[Mapping[str, Any]] = None,
    pair: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Counts, lengths and data-coverage caveats, with the rules they came from."""
    layer_summaries: Dict[str, Any] = {}
    for name, frame in assessed.items():
        counts = frame["candidate_status"].value_counts()
        entry: Dict[str, Any] = {
            "kind": LAYER_KINDS[name],
            "features_total": int(len(frame)),
            "count_by_status": {s: int(counts.get(s, 0)) for s in CANDIDATE_STATUSES},
            "preserved_attributes": [c for c in CARRIED_ATTRIBUTES if c in frame.columns],
        }
        if "length_m" in frame:
            by_status = frame.groupby("candidate_status")["length_m"].sum() / 1000.0
            entry["length_km_by_status"] = {
                s: round(float(by_status.get(s, 0.0)), 3) for s in CANDIDATE_STATUSES
            }
            entry["total_length_km"] = round(float(frame["length_m"].sum()) / 1000.0, 3)
            entry["candidate_intersecting_length_km"] = round(
                float(frame["candidate_m"].sum()) / 1000.0, 3
            )
            entry["not_observed_length_km"] = round(
                float(frame["not_observed_m"].sum()) / 1000.0, 3
            )
            entry["outside_grid_length_km"] = round(
                float(frame["outside_grid_m"].sum()) / 1000.0, 3
            )
            if "highway" in frame:
                hit = frame[frame["candidate_status"] == CANDIDATE_INTERSECTING]
                by_class = hit.groupby("highway")["candidate_m"].sum() / 1000.0
                entry["candidate_intersecting_km_by_highway"] = {
                    str(k): round(float(v), 3)
                    for k, v in by_class.sort_values(ascending=False).items()
                }
        if "candidate_px" in frame:
            entry["candidate_pixels_touched_total"] = int(frame["candidate_px"].sum())
            entry["not_observed_pixels_touched_total"] = int(frame["not_observed_px"].sum())
        if name == "settlements":
            hit = frame[frame["candidate_status"] == CANDIDATE_INTERSECTING]
            entry["candidate_intersecting"] = [
                {
                    "osm_id": _optional_value(row, "osm_id"),
                    "name": _optional_value(row, "name"),
                    "place": _optional_value(row, "place"),
                }
                for _, row in hit.iterrows()
            ]
            entry["non_point_geometries"] = (
                int((~frame.geom_type.isin(["Point"])).sum()) if len(frame) else 0
            )
        layer_summaries[name] = entry

    coverage: Dict[str, Any] = {
        "candidate_mask": mask.describe(),
        "optional_layers_absent": list(osm.missing_optional) if osm else [],
        "layers_read_but_not_assessed": sorted(set(osm.layers) - set(assessed)) if osm else [],
        "input_notes": list(osm.notes) if osm else [],
    }
    if aoi_report is not None:
        coverage["aoi_association"] = dict(aoi_report)

    summary: Dict[str, Any] = {
        "stage": CANDIDATE_EXPOSURE_VERSION,
        "status": "candidate_exposure_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": asdict(rules),
        "statuses": dict(STATUS_MEANINGS),
        "interpretation": (
            "Results report spatial intersection between pre-event OSM infrastructure and "
            "M6.3 Sentinel-1 RTC dual-polarisation darkening candidates. Radar darkening "
            "is not unique to inundation. These are not confirmed flood extents, not "
            "verified damage, and not evidence that any road, building or settlement was "
            "destroyed, structurally failed or made inaccessible (AGENTS.md §8)."
        ),
        "layers": layer_summaries,
        "data_coverage": coverage,
        "limitations": list(LIMITATIONS),
        "osm": _osm_summary(osm),
    }
    if pair is not None:
        summary["source_pair"] = dict(pair)
        summary["aoi_provenance_status"] = pair.get("aoi_provenance_status")
    return summary


def _osm_summary(osm: Optional[LocalOsmLayers]) -> Optional[Dict[str, Any]]:
    if osm is None:
        return None
    return {
        "retrieval_performed": False,
        "retrieval_disabled_reason": (
            "the study AOI is UNVERIFIED and does not authorise a download; M6.4 reads "
            "pre-event OSM only from layers already on disk"
        ),
        "directory": str(osm.directory),
        "snapshot": osm.snapshot.isoformat() if osm.snapshot else None,
        "recorded_event_date": osm.provenance.get("event_date"),
        "source": osm.provenance.get("source"),
        "endpoint": osm.provenance.get("endpoint"),
        "attribution": osm.attribution(),
        "license": osm.provenance.get("license", OSM_LICENSE),
        "layers_read": osm.layer_report(),
    }


LIMITATIONS: Tuple[str, ...] = (
    "M6.3 candidates are rule-based Sentinel-1 RTC gamma0 darkening evidence, not "
    "confirmed flood, damage or ground truth; intersection with them is spatial "
    "coincidence only.",
    "Radar darkening has causes other than inundation, and flooded vegetation or "
    "built-up areas can be missed entirely, so both false positives and false "
    "negatives are expected in the candidate mask.",
    "A candidate region intersecting a road or building is evidence of spatial "
    "exposure, not of physical destruction or structural failure (AGENTS.md §8).",
    "Not-observed pixels are reported as UNOBSERVED and never as unaffected; a "
    "feature in an unobserved area may or may not intersect a candidate region.",
    "Infrastructure completeness is limited by OpenStreetMap coverage at the "
    "pre-event snapshot; unmapped roads, buildings and settlements cannot appear.",
    "Settlement verdicts use a single representative point per settlement, so they "
    "describe that point and not the full built-up extent of the settlement.",
    "The 10 m candidate grid cannot resolve features narrower than one pixel, and "
    "building footprints are measured with all_touched rasterisation, which is "
    "inclusive at footprint edges.",
    "The study extent is a human-approved project corridor whose producer, source and "
    "construction parameters are recorded, but its provenance_status remains "
    "UNVERIFIED: adopting the geometry authorised no retrieval, and every count is "
    "relative to a project-defined analytical extent rather than an official boundary.",
    "No connectivity, routing or accessibility conclusion is drawn by this stage.",
)


# ---------------------------------------------------------------------------
# M6.3 linkage
# ---------------------------------------------------------------------------


def load_candidate_pair_metadata(summary_path: Path | str) -> Dict[str, Any]:
    """Read the M6.3 ``rtc_change_summary.json`` pair block.

    Carrying the scene identifiers, acquisition timestamps and orbit forward is
    what makes the evidence chain from an exposed road back to the source
    Sentinel-1 scenes traversable (``AGENTS.md`` §22).
    """
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise CandidateExposureError(
            f"M6.3 summary not found: {summary_path}. The candidate mask's source scenes, "
            "acquisition timestamps and AOI cannot be recorded without it."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateExposureError(f"could not read {summary_path}: {exc}") from exc
    pair = summary.get("pair")
    if not isinstance(pair, Mapping):
        raise CandidateExposureError(f"{summary_path} does not contain a 'pair' block")
    record = dict(pair)
    record["m6_3_stage"] = summary.get("stage")
    record["m6_3_candidate_rule"] = summary.get("candidate_rule")
    record["m6_3_radiometric_convention"] = summary.get("radiometric_convention")
    record["m6_3_summary_path"] = str(summary_path)
    return record


def scene_pair_from_summary(pair: Mapping[str, Any]) -> Optional[BeforeAfterPair]:
    """The M6.3 ``pair`` block as a structured before/after scene reference.

    Public so downstream stages (M6.5) can carry the same acquisition
    timestamps and orbit forward instead of recording only scene ids: the
    evidence chain from a derived result back to the source scenes has to stay
    traversable (``AGENTS.md`` §4, §22).
    """
    pre, post = pair.get("pre"), pair.get("post")
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping):
        return None

    def scene(block: Mapping[str, Any]) -> SceneReference:
        return SceneReference(
            scene_id=str(block.get("scene_id", "")),
            acquired_at=block.get("acquired_at"),
            relative_orbit=block.get("relative_orbit"),
            orbit_direction=block.get("orbit_direction"),
            processing_level="sentinel-1-rtc",
            product_type="sentinel-1-rtc",
            source=ProductionInput.SENTINEL1,
        )

    same_orbit = (
        True
        if isinstance(pre.get("relative_orbit"), int)
        and pre.get("relative_orbit") == post.get("relative_orbit")
        else None
    )
    return BeforeAfterPair(before=scene(pre), after=scene(post), same_relative_orbit=same_orbit)


def candidate_exposure_provenance(
    *,
    mask: CandidateMask,
    osm: LocalOsmLayers,
    summary: Mapping[str, Any],
    pair: Mapping[str, Any],
    config_version: str,
) -> ArtifactProvenance:
    """Machine-readable provenance for the M6.4 exposure artifact."""
    product_ids = [
        str((pair.get("pre") or {}).get("scene_id", "")),
        str((pair.get("post") or {}).get("scene_id", "")),
    ]
    return ArtifactProvenance(
        artifact_type=ArtifactType.INFRASTRUCTURE_EXPOSURE,
        artifact_id=(
            f"candidate-infrastructure-exposure-{pair.get('aoi_id', 'unknown-aoi')}-"
            f"{pair.get('event_date', 'unknown-date')}"
        ),
        aoi=pair.get("aoi_id"),
        aoi_crs=GEOJSON_DEFAULT_CRS,
        event_date=pair.get("event_date"),
        sentinel1=scene_pair_from_summary(pair),
        osm_snapshot_date=osm.snapshot.isoformat() if osm.snapshot else None,
        production_inputs=[ProductionInput.SENTINEL1, ProductionInput.OSM_PRE_EVENT],
        source_product_ids=[pid for pid in product_ids if pid],
        preprocessing_operations=[
            "load_m6_3_change_candidate_mask_and_valid_mask",
            "verify_candidate_mask_stage_tag_and_class_semantics",
            "verify_candidate_and_valid_mask_grid_alignment",
            "load_pre_event_osm_layers_from_disk_without_retrieval",
            "verify_osm_snapshot_precedes_event_date",
            "verify_osm_extent_overlaps_candidate_raster_extent",
            "reproject_osm_geometries_to_candidate_mask_crs",
            "sample_lines_and_rasterise_polygons_on_candidate_grid",
            "classify_features_as_candidate_intersecting_unobserved_or_not_intersecting",
        ],
        quality=dict(summary),
        preprocessing_version=CANDIDATE_EXPOSURE_VERSION,
        config_version=config_version,
        limitations=list(LIMITATIONS),
        notes=(
            "Infrastructure intersection with M6.3 Sentinel-1 RTC change candidates. "
            "No download was performed: the AOI is UNVERIFIED and does not authorise "
            "one, so pre-event OSM was read from layers already on disk. No "
            "segmentation checkpoint, DEM, validation product or ground truth was "
            "used, and no connectivity conclusion is drawn."
        ),
        attribution=osm.attribution(),
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_candidate_exposure_artifacts(
    assessed: Mapping[str, gpd.GeoDataFrame],
    summary: Mapping[str, Any],
    output_dir: Path | str,
    *,
    provenance: Optional[ArtifactProvenance] = None,
) -> Dict[str, Path]:
    """Write per-layer GeoJSON, the summary and provenance.

    GeoJSON is written in EPSG:4326 so it opens directly in QGIS or geojson.io.
    Features outside the candidate raster are retained and labelled rather than
    dropped, so the reader can see what was excluded and why.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, frame in assessed.items():
        path = output_dir / f"candidate_exposure_{name}.geojson"
        path.write_text(frame.to_crs(GEOJSON_DEFAULT_CRS).to_json(drop_id=True), encoding="utf-8")
        written[name] = path

    summary_path = output_dir / "candidate_exposure_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written["summary"] = summary_path

    if provenance is not None:
        provenance_path = output_dir / "candidate_exposure_provenance.yaml"
        provenance_path.write_text(provenance.to_yaml(), encoding="utf-8")
        written["provenance"] = provenance_path
    return written


def blocker_report(
    *,
    reason: str,
    missing: Sequence[str],
    searched: Path | str,
    mask: Optional[CandidateMask] = None,
    pair: Optional[Mapping[str, Any]] = None,
    next_step: Sequence[str] = (),
) -> Dict[str, Any]:
    """A machine-readable record of why M6.4 could not complete.

    Written instead of exposure numbers. A blocker artifact is a result: it
    states exactly which permitted input is absent, which authorisation gate
    prevents obtaining it, and what would unblock it.
    """
    return {
        "stage": CANDIDATE_EXPOSURE_VERSION,
        "status": "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "missing_layers": list(missing),
        "searched_directory": str(searched),
        "expected_filenames": {
            name: LAYER_FILENAMES[name] for name in missing if name in LAYER_FILENAMES
        },
        "retrieval_performed": False,
        "authorisation": {
            "aoi_provenance_status": (pair or {}).get("aoi_provenance_status", "UNVERIFIED"),
            "authorises_download": False,
            "gate": (
                "AoiProvenanceStatus.UNVERIFIED permits geometry inspection and catalogue "
                "metadata analysis only; it must not be used to authorise a download "
                "(src/floodmap/acquisition/aoi.py)."
            ),
        },
        "inputs_available": {
            "candidate_mask": mask.describe() if mask is not None else None,
            "source_pair": dict(pair) if pair is not None else None,
        },
        "next_step": list(next_step),
        "not_done": [
            "No infrastructure layer was synthesised, approximated or substituted.",
            "No exposure, disruption or connectivity number was produced.",
        ],
    }


def write_blocker_report(report: Mapping[str, Any], output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_exposure_blocker.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CANDIDATE_EXPOSURE_VERSION",
    "CANDIDATE_INTERSECTING",
    "CANDIDATE_STATUSES",
    "CARRIED_ATTRIBUTES",
    "CandidateExposureError",
    "GEOJSON_DEFAULT_CRS",
    "IntersectionRules",
    "LAYER_FILENAMES",
    "LAYER_KINDS",
    "LIMITATIONS",
    "LocalOsmLayers",
    "MissingInfrastructureLayers",
    "NO_CANDIDATE_INTERSECTION",
    "OUTSIDE_CANDIDATE_GRID",
    "PROVENANCE_SIDECAR",
    "STATUS_MEANINGS",
    "UNOBSERVED",
    "assess_candidate_intersections",
    "assess_lines",
    "assess_polygons",
    "assess_settlements",
    "blocker_report",
    "candidate_exposure_provenance",
    "check_layers_overlap_mask",
    "load_candidate_pair_metadata",
    "load_local_osm_layers",
    "scene_pair_from_summary",
    "measure_points",
    "summarize_candidate_exposure",
    "write_blocker_report",
    "write_candidate_exposure_artifacts",
]
