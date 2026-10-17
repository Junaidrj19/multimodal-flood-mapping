#!/usr/bin/env python
"""Build/check the approved Trishuli AOI locally; historical configurations supported.

Defaults to ``configs/aoi/trishuli-corridor-hydrorivers-v10-m615.yaml`` and
rebuilds from local HydroRIVERS, SAFE annotations and RTC observations. The
historical M6.10 configuration remains available through ``--config``.

What this script does NOT do
----------------------------
* It does not download anything. If the geodatabase is missing it stops and
  names the artifact required.
* It does not touch ``configs/data.yaml``, acquisition gates, or downstream
  artifacts. It writes only its configured geometry/sidecar. Historical AOIs
  are opened read-only, for comparison, never for terminus selection.
* It does not fetch satellite or OSM data and does not run acquisition.

Usage
-----
    python scripts/build_corridor_aoi.py            # build and write
    python scripts/build_corridor_aoi.py --check    # rebuild, compare, write nothing
    python scripts/build_corridor_aoi.py --config aoi/trishuli-corridor-hydrorivers-v10 --check

``--check`` exits 1 if geometry differs, or if approved-AOI measured provenance,
source/implementation hashes or software versions differ. It writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio
import shapely
import yaml
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.geometry import shape as shapely_shape
from shapely.validation import explain_validity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floodmap.acquisition.aoi import geometry_from_geojson_document  # noqa: E402
from floodmap.hydrology.hydrorivers import (  # noqa: E402
    DELIVERY_CRS,
    STUDY_AREA_CRS,
    load_hydrorivers,
    reproject_geometry,
    select_coverage_bounded_corridor,
    select_hydrological_corridor,
)
from floodmap.hydrology.coverage import (  # noqa: E402
    APPROVED_COVERAGE_DISCLOSURE,
    measure_observation_coverage,
    safe_annotation_footprint,
    scene_clearance,
)
from floodmap.utils.config import load_config, project_root  # noqa: E402
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS  # noqa: E402

CONFIG_NAME = "aoi/trishuli-corridor-hydrorivers-v10-m615"

#: From HydroRIVERS Technical Documentation v1.0 §4.2 — the citation the
#: provider asks for, reproduced verbatim rather than paraphrased.
HYDRORIVERS_CITATION = (
    "Lehner, B., Grill G. (2013): Global river hydrography and network routing: "
    "baseline data and new approaches to study the world's large river systems. "
    "Hydrological Processes, 27(15): 2171-2186. Data is available at "
    "www.hydrosheds.org."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Digest a directory (an ESRI file geodatabase is a directory).

    Files are visited in sorted order and each name is mixed in with its digest,
    so a renamed or replaced part changes the result. Used only when the
    delivered archive is not retained beside the extracted geodatabase.
    """
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def code_version() -> str:
    """Best-effort repository revision, recorded so a rebuild is attributable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git absent
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def raster_footprint(path: Path) -> tuple[Any, Dict[str, Any]]:
    """Footprint of one delivered product, reprojected to EPSG:4326."""
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"{path} has no CRS; its footprint cannot be placed")
        crs = dataset.crs.to_string()
        bounds = dataset.bounds
        footprint = reproject_geometry(box(*bounds), src_crs=crs, dst_crs=DELIVERY_CRS)
        record = {
            "path": str(path.relative_to(project_root())),
            "sha256": sha256_file(path),
            "crs": crs,
            "bounds_4326": [round(v, 7) for v in footprint.bounds],
            "width": dataset.width,
            "height": dataset.height,
        }
    return footprint, record


def compare_with_replaced(new_polygon_wgs84, old_path: Path) -> Dict[str, Any]:
    """How the derived corridor compares with the artifact it replaces.

    Reported in metres on the projected CRS. Similarity here is descriptive: the
    two artifacts are not interchangeable, and the point of the comparison is to
    show the replacement is a different, better-documented extent — not a copy.
    """
    if not old_path.is_file():
        return {"path": str(old_path), "status": "absent"}
    raw = old_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    old = shapely_shape(geometry_from_geojson_document(document))
    new = reproject_geometry(new_polygon_wgs84)
    old_utm = reproject_geometry(old)
    intersection = new.intersection(old_utm).area
    union = new.union(old_utm).area
    return {
        "path": str(old_path.relative_to(project_root())),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "area_km2": round(float(old_utm.area) / 1e6, 4),
        "intersection_km2": round(float(intersection) / 1e6, 4),
        "jaccard": round(float(intersection / union), 6) if union else None,
        "centroid_distance_m": round(float(new.centroid.distance(old_utm.centroid)), 1),
    }


def build(config: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the corridor and return ``(geojson_document, provenance)``."""
    if config["construction"]["terminus_rule"] == "human_approved_hydrological_reach":
        return build_approved(config)
    root = project_root()
    params = config["construction"]
    source_cfg = params["source"]
    gdb_path = root / source_cfg["gdb_path"]
    if not gdb_path.exists():
        raise FileNotFoundError(
            f"HydroRIVERS source not found at {gdb_path}. Place the delivered "
            f"{source_cfg['tile']} geodatabase there (see "
            "docs/m6.10-hydrorivers-aoi-construction.md §3) and re-run."
        )

    network, geometries = load_hydrorivers(gdb_path, layer=source_cfg["layer"])

    footprints: List[Any] = []
    footprint_records: List[Dict[str, Any]] = []
    for relative in params["footprint"]:
        footprint, record = raster_footprint(root / relative)
        footprints.append(footprint)
        footprint_records.append(record)
    if not footprints:
        raise ValueError("no product footprints configured; the corridor would be unbounded")
    observed = footprints[0]
    for extra in footprints[1:]:
        observed = observed.intersection(extra)

    selection = select_coverage_bounded_corridor(
        network,
        geometries,
        (float(params["reference"]["lon"]), float(params["reference"]["lat"])),
        observed,
        buffer_m=float(params["buffer_m"]),
        cap_style=params["cap_style"],
        join_style=params["join_style"],
        quad_segs=int(params.get("quad_segs", 8)),
    )

    # Buffer sensitivity: the same rule at other half-widths. Recorded, not
    # adopted — the primary extent is the configured buffer.
    transformer_utm = Transformer.from_crs(STUDY_AREA_CRS, DELIVERY_CRS, always_xy=True)
    sensitivity: List[Dict[str, Any]] = []
    for candidate in params.get("buffer_sensitivity_m", []):
        try:
            alternative = select_coverage_bounded_corridor(
                network,
                geometries,
                (float(params["reference"]["lon"]), float(params["reference"]["lat"])),
                observed,
                buffer_m=float(candidate),
                cap_style=params["cap_style"],
                join_style=params["join_style"],
                quad_segs=int(params.get("quad_segs", 8)),
            )
        except ValueError as exc:
            sensitivity.append(
                {"buffer_m": float(candidate), "status": "no_fully_covered_corridor",
                 "reason": str(exc)}
            )
            continue
        sensitivity.append(
            {
                "buffer_m": float(candidate),
                "status": "constructed",
                "terminus_reach_id": alternative.terminus_reach_id,
                "centreline_length_m": round(alternative.centreline_length_m, 2),
                "area_km2": round(alternative.area_km2, 4),
            }
        )

    polygon_wgs84 = reproject_geometry(selection.polygon, src_crs=STUDY_AREA_CRS, dst_crs=DELIVERY_CRS)
    validity = explain_validity(selection.polygon)

    reach_path = selection.path
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "aoi_id": config["aoi_id_prefix"],
                    "name": "Upper Trishuli / Bhote Koshi corridor (derived from HydroRIVERS v1.0)",
                    "provenance_status": config["provenance_status"],
                    "source": f"HydroRIVERS {source_cfg['version']} tile {source_cfg['tile']}",
                    "buffer_m": params["buffer_m"],
                    "study_crs": params["study_crs"],
                    "note": (
                        "Project-defined analytical extent. Not a competition-selected "
                        "boundary and not an acquisition authorisation."
                    ),
                },
                "geometry": mapping(polygon_wgs84),
            }
        ],
    }

    gdb_parts = sorted(p for p in gdb_path.rglob("*") if p.is_file())
    provenance: Dict[str, Any] = {
        "artifact_type": "study_area_definition",
        "aoi_id": config["aoi_id_prefix"],
        "name": "Upper Trishuli / Bhote Koshi corridor",
        "created_utc": None,  # filled by the caller so output order is explicit
        "generator": "scripts/build_corridor_aoi.py",
        "code_version": code_version(),
        "conventions": {
            "crs_delivery": DELIVERY_CRS,
            "crs_construction": STUDY_AREA_CRS,
            "area_units": "km^2",
            "linear_units": "m",
        },
        "source": {
            "product": source_cfg["product"],
            "version": source_cfg["version"],
            "tile": source_cfg["tile"],
            "grid": source_cfg["grid"],
            "provider": source_cfg["provider"],
            "product_page": source_cfg["product_page"],
            "licence": source_cfg["licence"],
            "citation": HYDRORIVERS_CITATION,
            "delivery_crs": source_cfg["delivery_crs"],
            "layer": source_cfg["layer"],
            "retrieval_date": source_cfg["retrieval_date"],
            "local_path": source_cfg["gdb_path"],
            "local_sha256": sha256_tree(gdb_path),
            "local_part_count": len(gdb_parts),
            "local_bytes": int(sum(p.stat().st_size for p in gdb_parts)),
        },
        "geography": {
            "reference": {
                "name": params["reference"]["name"],
                "lon": params["reference"]["lon"],
                "lat": params["reference"]["lat"],
                "source": params["reference"]["source"],
                "uncertainty_m": params["reference"].get("uncertainty_m"),
                "note": params["reference"].get("note"),
            },
            "reach_selection": selection.to_record(),
            "terminus_rule": params["terminus_rule"],
            "buffer": {
                "buffer_m": params["buffer_m"],
                "cap_style": params["cap_style"],
                "join_style": params["join_style"],
                "quad_segs": params.get("quad_segs"),
                "sensitivity": sensitivity,
            },
        },
        "geometry": {
            "path": config["geojson_path"],
            "sha256": None,  # filled after writing
            "crs": DELIVERY_CRS,
            "format": "GeoJSON FeatureCollection with exactly one Polygon feature",
            "vertex_count": int(len(list(polygon_wgs84.exterior.coords))),
            "is_valid": bool(selection.polygon.is_valid),
            "validity_report": validity,
            "area_km2": round(selection.area_km2, 4),
            "bounds_4326": [round(v, 7) for v in polygon_wgs84.bounds],
        },
        "coverage": {
            "footprints": footprint_records,
            "observed_intersection_bounds_4326": [round(v, 7) for v in observed.bounds],
            "corridor_coverage_fraction": round(selection.coverage_fraction, 6),
            "definition": (
                "fraction of the delivered corridor polygon that lies inside the intersection "
                "of the delivered product footprints, measured in " + STUDY_AREA_CRS
            ),
        },
        "comparison_to_replaced_artifact": compare_with_replaced(
            polygon_wgs84, root / config["replaces"]
        ),
        "governance": {
            "permitted_use": (
                "Defines the analysis extent only. HydroRIVERS is not an inference input: it "
                "contributes no feature, mask or label to the flood model (AGENTS.md §3)."
            ),
            "not_a_competition_boundary": (
                "Project-defined analytical AOI; the challenge does not supply this geometry."
            ),
            "attributions": list(REQUIRED_ATTRIBUTIONS) + [HYDRORIVERS_CITATION],
        },
        "authorisation": {
            "authorises_download": False,
            "provenance_status": config["provenance_status"],
            "adoption": config["adoption"],
            "configured_acquisition_aoi": "configs/data.yaml (unchanged by this milestone)",
            "exploratory_exception": "absent",
        },
        "limitations": [
            "Main stem only: tributaries are excluded by the approved policy, so a flood that "
            "moved up a tributary valley is outside this extent.",
            "The downstream end is an instrument boundary (the delivered product coverage), not "
            "a hydrological node: the corridor's continuation beyond it is unobserved here.",
            "Corridor width is the M6.2-M6.5 1 km half-width, carried over for comparability; it "
            "is not yet an operator-ratified parameter.",
            "HydroRIVERS is a 15 arc-second (~460 m at this latitude) product: the centreline is "
            "a coarse delineation, not a surveyed river bank, so edge exposure is approximate at "
            "the buffer margin.",
            "The reference coordinate is a published site coordinate for the border crossing "
            "(~150 m site uncertainty), not a surveyed river station.",
        ],
    }
    return document, provenance


def _portable(value):
    """Normalize generated records to JSON types and repository-relative paths."""
    root = str(project_root()) + "/"
    if isinstance(value, dict):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(v) for v in value]
    if isinstance(value, str) and value.startswith(root):
        return value[len(root):]
    return value


def build_approved(config):
    """Orchestrate hydrological selection and independent local acceptance checks."""
    import pyogrio
    import pyproj

    root = project_root()
    params = config["construction"]
    source = params["source"]
    if params["study_crs"] != STUDY_AREA_CRS or params["delivery_crs"] != DELIVERY_CRS:
        raise ValueError("construction and delivery CRS must match the implemented metric workflow")
    if config["provenance_status"] != "unverified":
        raise ValueError("geometry adoption must not promote acquisition authorization")
    if config["approval"]["disclosure"] != APPROVED_COVERAGE_DISCLOSURE:
        raise ValueError("approval disclosure differs from the human decision")
    gdb = root / source["gdb_path"]
    if not gdb.is_dir():
        raise FileNotFoundError(f"HydroRIVERS source not found: {gdb}")
    tree_digest = sha256_tree(gdb)
    if tree_digest != source["expected_tree_sha256"]:
        raise ValueError("HydroRIVERS source tree hash differs from the recorded delivery")
    info = pyogrio.read_info(gdb, layer=source["layer"])
    if pyproj.CRS(info["crs"]) != pyproj.CRS(source["delivery_crs"]):
        raise ValueError("HydroRIVERS delivery CRS differs from the recorded source")
    network, geometries = load_hydrorivers(gdb, layer=source["layer"])

    def select(width):
        return select_hydrological_corridor(
            network, geometries,
            (float(params["reference"]["lon"]), float(params["reference"]["lat"])),
            terminus_reach_id=int(params["terminus_reach_id"]),
            terminus_index=int(params["terminus_index"]),
            buffer_m=float(width), cap_style=params["cap_style"],
            join_style=params["join_style"], quad_segs=int(params["quad_segs"]),
        )

    selection = select(params["buffer_m"])
    if selection.path.start_reach_id != params["start_reach_id"]:
        raise ValueError("documented reference no longer selects the approved start reach")
    end = network.get(selection.terminus_reach_id)
    following = network.get(end.next_down)
    if (end.ord_stra != params["terminus_strahler_order"]
            or end.next_down != params["next_down_reach_id"]
            or following.ord_stra != params["next_down_strahler_order"]
            or any(network.get(h).ord_stra != end.ord_stra for h in selection.path.reach_ids[1:])):
        raise ValueError("selected terminus is not the documented last homogeneous order-5 reach")

    footprints, scene_records = [], []
    for path in params["scene_footprints"]:
        footprint, record = safe_annotation_footprint(root / path)
        footprints.append(footprint)
        scene_records.append(record)
    if len(scene_records) != 2:
        raise ValueError("exactly one pre/post SAFE pair is required")
    for record in scene_records:
        if (record["relative_orbit"] != params["relative_orbit"]
                or record["orbit_direction"] != params["orbit_direction"]):
            raise ValueError("SAFE footprints are not the approved same-track pair")
    event_day = params["event_date"]
    if not (scene_records[0]["acquisition_start_utc"][:10] < event_day
            < scene_records[1]["acquisition_start_utc"][:10]):
        raise ValueError("SAFE acquisition dates do not bracket the event")
    if params["inset_role"] != "acceptance_only_not_terminus_definition":
        raise ValueError("inset must be an independent acceptance criterion")
    clearance = scene_clearance(selection.polygon, footprints, inset_m=params["inset_m"])
    if not clearance["accepted"]:
        raise ValueError("approved corridor fails independent scene-footprint inset acceptance")

    raster_paths = [root / p for p in params["observation_rasters"]]
    if len(raster_paths) != 2:
        raise ValueError("exactly one local pre/post RTC observation pair is required")
    coverage = measure_observation_coverage(selection.polygon, raster_paths)
    for raster, scene in zip(coverage["sources"], scene_records):
        if [b["description"] for b in raster["bands"]] != ["VV", "VH"]:
            raise ValueError("local observation must preserve named VV/VH bands")
        scene_prefix = Path(scene["path"]).name.removesuffix(".SAFE.zip").rsplit("_", 1)[0]
        if raster["tags"].get("source_item_id") != scene_prefix + "_rtc":
            raise ValueError("RTC observation identity differs from the independently checked SAFE")
        if raster["tags"].get("radiometric_convention") != "gamma0_linear_power":
            raise ValueError("validity measurement requires the recorded linear gamma0 radiometry")
    coverage["measured_disclosure"] = coverage["disclosure"]
    coverage["disclosure"] = config["approval"]["disclosure"]
    if f"{coverage['delivered_fraction'] * 100:.1f}" != "63.4":
        raise ValueError("measured coverage no longer supports the approved approximate disclosure")
    coverage["claims_scope"] = config["approval"]["claims"]
    coverage["partial_coverage_accepted"] = config["approval"]["partial_coverage_accepted"]
    coverage["scene_acceptance"] = {**clearance, "footprints": scene_records}

    sensitivity = []
    for width in params["buffer_sensitivity_m"]:
        alternative = selection if width == params["buffer_m"] else select(width)
        gate = scene_clearance(alternative.polygon, footprints, inset_m=params["inset_m"])
        measured = coverage if width == params["buffer_m"] else measure_observation_coverage(
            alternative.polygon, raster_paths
        )
        if not gate["accepted"]:
            raise ValueError(f"sensitivity buffer {width} fails independent inset acceptance")
        sensitivity.append({
            "buffer_m": float(width), "status": "constructed",
            "terminus_reach_id": alternative.terminus_reach_id,
            "centreline_length_m": alternative.centreline_length_m,
            "area_km2": alternative.area_km2,
            "clearance_m": gate["clearance_m"], "inset_accepted": gate["accepted"],
            "delivered_fraction": measured["delivered_fraction"],
            "delivered_area_km2": measured["delivered_area_km2"],
            "valid_area_km2": measured["valid_area_km2"],
            "valid_fraction_of_delivered_pixels": measured["valid_fraction_of_delivered_pixels"],
            "combined_valid_pixel_count": measured["combined_valid_pixel_count"],
            "exposure_and_connectivity_sensitivity": "unmeasured; authorized pre-event OSM absent",
        })

    polygon = reproject_geometry(selection.polygon, src_crs=STUDY_AREA_CRS, dst_crs=DELIVERY_CRS)
    if not polygon.is_valid or polygon.geom_type != "Polygon" or polygon.interiors:
        raise ValueError("delivered AOI must be a valid single Polygon without unexplained holes")
    document = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {
            "aoi_id_prefix": config["aoi_id_prefix"],
            "name": "Human-approved Upper Trishuli / Bhote Koshi analytical corridor",
            "provenance_status": "unverified", "source": "HydroRIVERS 1.0 tile HydroRIVERS_v10_as",
            "buffer_m": params["buffer_m"], "study_crs": STUDY_AREA_CRS,
            "terminus_reach_id": selection.terminus_reach_id,
            "terminus_index": params["terminus_index"],
            "decision_record": config["approval"]["record"],
            "coverage_disclosure": config["approval"]["disclosure"],
            "note": "Project-defined extent, not a flood map, competition boundary or acquisition authorization.",
        }, "geometry": mapping(polygon),
    }]}
    # The byte-digest-derived ID is in the sidecar/config, not embedded in the
    # bytes it hashes (which would create an impossible self-referential digest).
    return document, _approved_provenance(
        config, selection, polygon, coverage, sensitivity, tree_digest, info
    )


def _approved_provenance(config, selection, polygon, coverage, sensitivity, tree_digest, info):
    """Record exact local inputs, implementation/dependency versions and limitations."""
    import pyogrio
    import pyproj

    root = project_root()
    params, source = config["construction"], config["construction"]["source"]
    gdb = root / source["gdb_path"]
    parts = sorted(p for p in gdb.rglob("*") if p.is_file())
    source_record = {**source, "citation": HYDRORIVERS_CITATION,
                     "local_sha256": tree_digest, "local_part_count": len(parts),
                     "local_bytes": sum(p.stat().st_size for p in parts),
                     "feature_count": info["features"], "measured_crs": info["crs"]}
    archive = root / source["archive_path"]
    if archive.is_file():
        source_record["archive_sha256"] = sha256_file(archive)
        if source_record["archive_sha256"] != source["expected_archive_sha256"]:
            raise ValueError("HydroRIVERS archive hash differs from recorded delivery")
    software = {name: importlib.metadata.version(name) for name in (
        "numpy", "shapely", "pyproj", "pyogrio", "rasterio", "PyYAML", "geopandas"
    )}
    software.update(python=platform.python_version(), geos=shapely.geos_version_string,
                    rasterio_gdal=rasterio.__gdal_version__,
                    pyogrio_gdal=pyogrio.__gdal_version_string__, proj=pyproj.proj_version_str)
    implementation = {p: sha256_file(root / p) for p in (
        "scripts/build_corridor_aoi.py", "src/floodmap/hydrology/hydrorivers.py",
        "src/floodmap/hydrology/coverage.py",
    )}
    record = {
        "artifact_type": "study_area_definition", "aoi_id": config["aoi_id_prefix"],
        "name": "Human-approved Upper Trishuli / Bhote Koshi analytical corridor",
        "created_utc": None, "generator": "scripts/build_corridor_aoi.py",
        "code_version": code_version(), "implementation_sha256": implementation,
        "software": software, "configuration": config,
        "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "approval": config["approval"],
        "conventions": {"crs_delivery": DELIVERY_CRS, "crs_construction": STUDY_AREA_CRS,
                        "area_units": "km^2", "linear_units": "m"},
        "source": source_record,
        "geography": {
            "reference": params["reference"], "reach_selection": selection.to_record(),
            "terminus_rule": params["terminus_rule"],
            "downstream_context": {"next_down": params["next_down_reach_id"],
                                   "ord_stra": params["next_down_strahler_order"],
                                   "included": False},
            "buffer": {k: params[k] for k in ("buffer_m", "cap_style", "join_style", "quad_segs")},
        },
        "geometry": {"path": config["geojson_path"], "sha256": None,
                     "crs": DELIVERY_CRS, "format": "GeoJSON FeatureCollection with exactly one Polygon feature",
                     "vertex_count": len(polygon.exterior.coords), "is_valid": polygon.is_valid,
                     "validity_report": explain_validity(polygon), "area_km2": selection.area_km2,
                     "bounds_4326": [round(v, 7) for v in polygon.bounds]},
        "coverage": coverage,
        "comparison_to_replaced_artifact": compare_with_replaced(polygon, root / config["replaces"]),
        "historical_candidate": {"path": config["historical_candidate"],
                                 "sha256": sha256_file(root / config["historical_candidate"])},
        "governance": {
            "permitted_use": "Defines analysis extent only; HydroRIVERS is not an inference input (AGENTS.md §3).",
            "not_a_competition_boundary": "Project-defined analytical AOI; the challenge does not supply this geometry.",
            "attributions": list(REQUIRED_ATTRIBUTIONS) + [HYDRORIVERS_CITATION]},
        "authorisation": {"authorises_download": False, "provenance_status": "unverified",
                          "adoption": config["adoption"], "configured_acquisition_aoi": "configs/data.yaml",
                          "exploratory_exception": "absent", "training_started": False,
                          "data_downloaded": False, "falkordb_populated": False},
        "limitations": [
            "Main stem only: tributary valleys excluded; not a flood extent or damage boundary.",
            config["approval"]["disclosure"],
            "HydroRIVERS is a coarse 15 arc-second (~460 m) delineation, not a surveyed bank or named-river boundary.",
            "Reference is a published place coordinate with ~150 m uncertainty, not a surveyed station.",
            "1000 m buffer ratified for scope; exposure/connectivity sensitivity remains unmeasured without authorized pre-event OSM.",
            "GRD acquisition swath acceptance does not establish valid RTC pixels; delivered coverage and masks are independently measured.",
            "Historical RTC crop inputs remain tied to the replaced envelope; no pixels outside those local files were retrieved.",
            "No official competition endorsement is established; geometry approval does not verify acquisition provenance or authorize downloads.",
            "RTC is gamma0 linear power; the training sigma0 representation mismatch remains unresolved. No training or inference performed.",
        ],
    }
    record["geography"]["buffer"]["sensitivity"] = sensitivity
    return _portable(record)


def write_outputs(
    document: Dict[str, Any], provenance: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    root = project_root()
    geojson_path = root / config["geojson_path"]
    provenance_path = root / config["provenance_path"]
    if not geojson_path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {geojson_path.parent}")

    text = json.dumps(document, indent=1, ensure_ascii=False) + "\n"
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    provenance["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance["geometry"]["sha256"] = digest
    provenance["geometry"]["bytes"] = len(payload)
    provenance["aoi_id"] = f"{config['aoi_id_prefix']}-{digest[:8]}"

    geojson_path.write_bytes(payload)
    provenance_path.write_text(
        yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    return {"geojson_path": geojson_path, "provenance_path": provenance_path, "sha256": digest}


def check_outputs(document: Dict[str, Any], config: Dict[str, Any], provenance=None) -> bool:
    root = project_root()
    geojson_path = root / config["geojson_path"]
    if not geojson_path.is_file():
        print(f"MISSING  {geojson_path}", file=sys.stderr)
        return False
    rebuilt = (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    on_disk = geojson_path.read_bytes()
    if rebuilt != on_disk:
        print(
            f"DIFFERS  {geojson_path}\n  rebuilt sha256 {hashlib.sha256(rebuilt).hexdigest()}\n"
            f"  on-disk sha256 {hashlib.sha256(on_disk).hexdigest()}",
            file=sys.stderr,
        )
        return False
    if provenance is not None and config["construction"]["terminus_rule"] == "human_approved_hydrological_reach":
        recorded = yaml.safe_load((root / config["provenance_path"]).read_text(encoding="utf-8"))
        expected = _portable(provenance)
        expected["aoi_id"] = f"{config['aoi_id_prefix']}-{hashlib.sha256(rebuilt).hexdigest()[:8]}"
        expected["geometry"].update(sha256=hashlib.sha256(rebuilt).hexdigest(), bytes=len(rebuilt))
        # Only the wall-clock generation time and Git dirty/revision label vary.
        # Scientific values, hashes, dependencies and construction parameters do not.
        for key in ("created_utc", "code_version"):
            expected.pop(key, None)
            recorded.pop(key, None)
        if expected != recorded:
            print("DIFFERS  provenance evidence (rebuild required)", file=sys.stderr)
            return False
    print(f"MATCH    {geojson_path} sha256 {hashlib.sha256(on_disk).hexdigest()}")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default=CONFIG_NAME,
        help="named construction config under configs/; defaults to the approved M6.15 policy",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and compare with the artifacts on disk; write nothing",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    document, provenance = build(config)

    if args.check:
        return 0 if check_outputs(document, config, provenance) else 1

    written = write_outputs(document, provenance, config)
    geometry = provenance["geometry"]
    coverage = provenance["coverage"]
    selection = provenance["geography"]["reach_selection"]
    print(f"wrote geometry    {written['geojson_path'].relative_to(project_root())}")
    print(f"wrote provenance  {written['provenance_path'].relative_to(project_root())}")
    print(f"aoi_id            {provenance['aoi_id']}")
    print(f"sha256            {written['sha256']}")
    print(f"area_km2          {geometry['area_km2']}")
    print(f"bounds_4326       {geometry['bounds_4326']}")
    print(f"centreline_m      {selection['centreline_length_m']}")
    print(f"upstream_trim_m   {selection['upstream_trim_m']}")
    reaches = selection["selected_reaches"]
    print(f"reaches           {reaches['reach_count']} "
          f"({reaches['start_reach_id']} -> {reaches['end_reach_id']})")
    print(f"coverage          {coverage.get('delivered_fraction', coverage.get('corridor_coverage_fraction'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
