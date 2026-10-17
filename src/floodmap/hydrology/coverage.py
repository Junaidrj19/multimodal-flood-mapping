"""Local observation coverage, independent of study-area construction.

All AOI arguments are caller-declared EPSG:32645 Polygon/MultiPolygon geometries
(Shapely geometries carry no CRS). Invalid topology is rejected, never repaired.
SAFE footprints are EPSG:4326 longitude/latitude, derived from annotation grid
edges, not a catalogue bounding box or an AOI-dependent crop.

Raster measurements require identical, north-up, projected-metre grids: CRS,
all affine coefficients, resolution, dimensions and extent must match exactly.
Integer-offset crops are also rejected. Only local GeoTIFFs are opened; neither
rasters nor masks are resampled. A vector AOI may be projected into a different
projected-metre raster CRS; geometric areas are always measured in EPSG:32645.

Coverage schema (version 1)
---------------------------
``measure_observation_coverage`` returns JSON-compatible dictionaries. Areas
ending in ``_km2`` are square kilometres. ``delivered_*`` measure the continuous
AOI intersection with the common local raster window, irrespective of nodata.
``unobserved_fraction`` is the *geometrically undelivered* AOI fraction, matching
the approved disclosure's meaning. Invalid pixels within delivery are reported
separately. ``valid_area_km2`` is a pixel-count × pixel-area estimate, not an
exact polygon area. Its raw AOI ratio can exceed one at rasterized boundaries;
``valid_fraction_of_aoi`` is explicitly clipped to [0, 1], and the raw ratio and
clipping flag are retained. It must not replace geometric delivered coverage.

``sources`` retain input order (index 0 = pre, index 1 = post, later indices =
observation_N). Each source reports all-band validity and per-band counts on
the *same* delivered AOI pixel-center mask. Combined validity requires every
band of every source: GDAL mask != 0, not declared nodata, finite, and > 0.
This last rule assumes linear SAR power; dB imagery is not supported.

``observation_footprint`` identifies geometric delivery (AOI ∩ common window),
not the valid-pixel footprint. Geometry records contain CRS, GeoJSON geometry,
bounds and a little-endian 2-D WKB SHA-256. Hashes of source files and local
sidecars returned by GDAL retain traceability of external masks/metadata.
``disclosure`` is generated from measurements; the fixed approved wording is
available separately as ``APPROVED_COVERAGE_DISCLOSURE`` and is never substituted
for a measurement. These functions do not select dates, enforce orbit pairing,
construct/adopt AOIs, or infer flood absence from missing observations.
"""

from __future__ import annotations

import hashlib
import math
import re
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

__all__ = [
    "APPROVED_COVERAGE_DISCLOSURE",
    "safe_annotation_footprint",
    "scene_clearance",
    "measure_observation_coverage",
]

APPROVED_COVERAGE_DISCLOSURE = (
    "The study corridor extends beyond the currently available valid satellite observation "
    "coverage. Approximately 63.4% of the AOI has delivered imagery coverage; the remaining "
    "36.6% is unobserved in the current dataset. Results apply only to the observed, "
    "valid-data area. Unobserved areas must not be interpreted as unaffected, flood-free, "
    "or undamaged."
)

_METRIC_CRS = "EPSG:32645"
_GEOGRAPHIC_CRS = "EPSG:4326"


def _local_file(path: Path) -> Path:
    # Also reject Path("https://...") after pathlib collapses it to "https:/...".
    text = str(path)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", text) or text.startswith(
        ("/vsi", "vsi", "//", "\\\\")
    ):
        raise ValueError(f"local files only; URLs and GDAL virtual paths are forbidden: {text}")
    result = Path(path).resolve(strict=True)
    if not result.is_file():
        raise ValueError(f"not a local regular file: {result}")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wkb_sha256(geometry: BaseGeometry) -> str:
    return hashlib.sha256(
        shapely.to_wkb(geometry, byte_order=1, output_dimension=2, include_srid=False)
    ).hexdigest()


def _polygon(geometry: BaseGeometry, name: str) -> BaseGeometry:
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(f"{name} must be a Polygon or MultiPolygon")
    if geometry.is_empty or not np.isfinite(shapely.get_coordinates(geometry)).all():
        raise ValueError(f"{name} must be nonempty with finite coordinates")
    if not geometry.is_valid:
        raise ValueError(f"invalid {name} topology: {explain_validity(geometry)}")
    if not math.isfinite(geometry.area) or geometry.area <= 0:
        raise ValueError(f"{name} must have positive finite area")
    return geometry


def _project(geometry: BaseGeometry, source: str, target: str) -> BaseGeometry:
    if CRS.from_user_input(source) == CRS.from_user_input(target):
        return geometry
    transformer = Transformer.from_crs(source, target, always_xy=True)
    return shapely.transform(
        geometry,
        lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])),
    )


def _area_only(geometry: BaseGeometry) -> BaseGeometry:
    """Discard zero-area contacts, including lines in GeometryCollections."""
    if geometry.is_empty:
        return Polygon()
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if geometry.geom_type == "GeometryCollection":
        return shapely.union_all([_area_only(part) for part in geometry.geoms])
    return Polygon()


def _geometry_record(geometry: BaseGeometry, crs: str) -> dict:
    return {
        "crs": crs,
        "geometry": mapping(geometry),
        "bounds": None if geometry.is_empty else list(geometry.bounds),
        "wkb_sha256": _wkb_sha256(geometry),
        "wkb_encoding": "little_endian_2d_no_srid",
    }


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    matches = [(child.text or "").strip() for child in element if _tag(child) == name]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"expected one nonempty {name} in {_tag(element)}")
    return matches[0]


def _utc_timestamp(text: str) -> str:
    date = datetime.fromisoformat(text.replace("Z", "+00:00"))
    # Sentinel SAFE UTC timestamps commonly omit the suffix.
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_annotation_footprint(path: Path) -> tuple[Polygon, dict]:
    """Read one local SAFE.zip's unique VV image annotation and optional manifest.

    Complete rectangular geolocation grids (>=2 rows/columns) are required.
    Perimeter order is min-line left→right, max-pixel top→bottom, max-line
    right→left, min-pixel bottom→top, then closure. A 10×21 grid has 58 distinct
    perimeter points and 59 ring coordinates including closure. Extra VV image
    annotations are ambiguous and rejected rather than arbitrarily selected.

    Record keys: path, archive_sha256, annotation_member, annotation_sha256,
    manifest_member/sha256, footprint_crs, footprint_wkb_sha256/encoding,
    footprint_bounds, grid_rows/columns/point_count, line_values, pixel_values,
    image_lines/samples, perimeter_point_count, vertex_count, derivation,
    acquisition_start_utc/stop_utc, relative_orbit, orbit_direction and
    metadata_evidence (member plus raw/normalized value for each metadata item).
    Missing optional metadata is None; conflicting metadata raises ValueError.
    """
    path = _local_file(path)
    with ZipFile(path) as archive:
        candidates = []
        for member in sorted(archive.namelist()):
            member_path = PurePosixPath(member)
            if member_path.parent.name != "annotation" or member_path.suffix != ".xml":
                continue
            raw = archive.read(member)
            root = ET.fromstring(raw)
            polarisations = {
                (node.text or "").strip().upper()
                for node in root.iter()
                if _tag(node) == "polarisation"
            }
            if polarisations == {"VV"}:
                candidates.append((member, raw, root))
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one VV image annotation, found {len(candidates)}")
        member, annotation_bytes, annotation = candidates[0]
        points = {}
        for node in annotation.iter():
            if _tag(node) != "geolocationGridPoint":
                continue
            line, pixel = int(_child_text(node, "line")), int(_child_text(node, "pixel"))
            lon, lat = float(_child_text(node, "longitude")), float(_child_text(node, "latitude"))
            if line < 0 or pixel < 0 or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("invalid geolocation grid indices or longitude/latitude")
            if (line, pixel) in points:
                raise ValueError("duplicate geolocation grid (line, pixel)")
            points[line, pixel] = (lon, lat)
        rows = sorted({line for line, _ in points})
        columns = sorted({pixel for _, pixel in points})
        if len(rows) < 2 or len(columns) < 2 or len(points) != len(rows) * len(columns):
            raise ValueError("geolocation grid must be a complete rectangular grid >= 2 by 2")
        for node in annotation.iter():
            if _tag(node) == "geolocationGridPointList" and "count" in node.attrib:
                if int(node.attrib["count"]) != len(points):
                    raise ValueError("geolocation grid declared count disagrees with points")
        edge = (
            [(rows[0], col) for col in columns]
            + [(row, columns[-1]) for row in rows[1:]]
            + [(rows[-1], col) for col in columns[-2::-1]]
            + [(row, columns[0]) for row in rows[-2:0:-1]]
        )
        footprint = Polygon([points[index] for index in edge])
        _polygon(footprint, "SAFE footprint")
        manifests = [
            name for name in archive.namelist()
            if PurePosixPath(name) == PurePosixPath(member).parent.parent / "manifest.safe"
        ]
        if len(manifests) > 1:
            raise ValueError("duplicate SAFE manifest members")
        manifest_member = manifests[0] if manifests else None
        manifest_bytes = archive.read(manifest_member) if manifest_member else None
        trees = [(member, annotation)]
        if manifest_bytes is not None:
            trees.append((manifest_member, ET.fromstring(manifest_bytes)))

    fields = {
        "startTime": ("acquisition_start_utc", _utc_timestamp),
        "stopTime": ("acquisition_stop_utc", _utc_timestamp),
        "relativeOrbitNumber": ("relative_orbit", int),
        "pass": ("orbit_direction", str.upper),
        "orbitDirection": ("orbit_direction", str.upper),
        "numberOfLines": ("image_lines", int),
        "numberOfSamples": ("image_samples", int),
    }
    evidence: dict[str, list[dict]] = {key: [] for key, _ in fields.values()}
    for source_member, root in trees:
        for node in root.iter():
            if _tag(node) not in fields:
                continue
            key, parse = fields[_tag(node)]
            raw = (node.text or "").strip()
            evidence[key].append({"member": source_member, "raw_value": raw, "value": parse(raw)})
    metadata = {}
    for key, items in evidence.items():
        values = {item["value"] for item in items}
        if len(values) > 1:
            raise ValueError(f"conflicting SAFE metadata for {key}: {values}")
        metadata[key] = next(iter(values)) if values else None
    if metadata["orbit_direction"] not in (None, "ASCENDING", "DESCENDING"):
        raise ValueError("invalid SAFE orbit direction")
    if metadata["relative_orbit"] is not None and not 1 <= metadata["relative_orbit"] <= 175:
        raise ValueError("invalid Sentinel-1 relative orbit")
    if metadata["acquisition_start_utc"] and metadata["acquisition_stop_utc"]:
        if datetime.fromisoformat(metadata["acquisition_start_utc"]) > datetime.fromisoformat(
            metadata["acquisition_stop_utc"]
        ):
            raise ValueError("SAFE acquisition stop precedes start")
    for key, maximum in (("image_lines", rows[-1]), ("image_samples", columns[-1])):
        if metadata[key] is not None and metadata[key] <= maximum:
            raise ValueError(f"SAFE {key} does not contain the geolocation grid")
    return footprint, {
        "path": str(path),
        "archive_sha256": _file_sha256(path),
        "annotation_member": member,
        "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "manifest_member": manifest_member,
        "manifest_sha256": (
            hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None
        ),
        "footprint_crs": _GEOGRAPHIC_CRS,
        "footprint_wkb_sha256": _wkb_sha256(footprint),
        "footprint_wkb_encoding": "little_endian_2d_no_srid",
        "footprint_bounds": list(footprint.bounds),
        "grid_rows": len(rows),
        "grid_columns": len(columns),
        "grid_point_count": len(points),
        "line_values": rows,
        "pixel_values": columns,
        "perimeter_point_count": len(edge),
        "vertex_count": len(footprint.exterior.coords),
        "derivation": "sorted_geolocationGridPoint_line_pixel_perimeter_no_crop",
        "metadata_evidence": evidence,
        **metadata,
    }


def scene_clearance(
    polygon_utm: BaseGeometry,
    footprints_wgs84: Sequence[BaseGeometry],
    *,
    inset_m: float = 250,
) -> dict:
    """Measure AOI clearance from the common full-swath boundary in EPSG:32645.

    Clearance is the shortest AOI-to-common-boundary distance when fully covered,
    otherwise zero. Acceptance requires full coverage and clearance >= inset_m.
    A disjoint intersection reports zero common area/clearance and rejection.
    Invalid or empty input geometries and negative/nonfinite insets raise.
    """
    aoi = _polygon(polygon_utm, "EPSG:32645 AOI")
    if not math.isfinite(inset_m) or inset_m < 0:
        raise ValueError("inset_m must be finite and nonnegative")
    if not footprints_wgs84:
        raise ValueError("at least one full-swath footprint is required")
    projected = []
    for footprint in footprints_wgs84:
        _polygon(footprint, "EPSG:4326 footprint")
        left, bottom, right, top = footprint.bounds
        if not (-180 <= left <= right <= 180 and -90 <= bottom <= top <= 90):
            raise ValueError("footprint must use EPSG:4326 longitude/latitude")
        projected.append(_polygon(_project(footprint, _GEOGRAPHIC_CRS, _METRIC_CRS), "swath"))
    common = projected[0]
    for footprint in projected[1:]:
        common = _area_only(common.intersection(footprint))
    if not common.is_valid:
        raise ValueError("invalid common swath topology")
    covered = bool(common.covers(aoi))
    clearance = float(aoi.distance(common.boundary)) if covered else 0.0
    return {
        "metric_crs": _METRIC_CRS,
        "common_swath_area_km2": float(common.area / 1e6),
        "aoi_intersection_area_km2": float(common.intersection(aoi).area / 1e6),
        "common_swath": _geometry_record(common, _METRIC_CRS),
        "input_footprint_wkb_sha256": [_wkb_sha256(f) for f in footprints_wgs84],
        "fully_covered": covered,
        "clearance_m": clearance,
        "clearance_definition": "minimum_AOI_to_common_boundary_if_fully_covered_else_zero",
        "inset_m": float(inset_m),
        "accepted": covered and clearance >= inset_m,
    }


def _json_number(value: float | None) -> float | str | None:
    """Keep nodata metadata JSON-compatible, including NaN/Inf declarations."""
    if value is None:
        return None
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return float(value)


def _grid(dataset: rasterio.io.DatasetReader) -> dict:
    if dataset.crs is None:
        raise ValueError(f"missing raster CRS: {dataset.name}")
    crs = CRS.from_wkt(dataset.crs.to_wkt())
    if not crs.is_projected or len(crs.axis_info) < 2 or any(
        axis.unit_name.lower() not in ("metre", "meter") or axis.unit_conversion_factor != 1
        for axis in crs.axis_info[:2]
    ):
        raise ValueError(f"raster CRS must be projected in metres: {dataset.name}")
    affine = dataset.transform
    if not all(math.isfinite(value) for value in affine[:6]):
        raise ValueError(f"nonfinite raster transform: {dataset.name}")
    if affine.b != 0 or affine.d != 0 or affine.a <= 0 or affine.e >= 0:
        raise ValueError(f"only north-up, unrotated, positive-resolution grids: {dataset.name}")
    if dataset.width <= 0 or dataset.height <= 0 or dataset.count <= 0:
        raise ValueError(f"empty raster grid: {dataset.name}")
    if any(np.dtype(dtype).kind not in "fiu" for dtype in dataset.dtypes):
        raise ValueError(f"linear SAR requires real numeric bands: {dataset.name}")
    return {
        "crs": dataset.crs.to_string(),
        "crs_wkt": dataset.crs.to_wkt(),
        "transform": list(affine[:6]),
        "resolution_m": list(dataset.res),
        "width": dataset.width,
        "height": dataset.height,
        "bounds": list(dataset.bounds),
        "pixel_area_m2": float(abs(affine.a * affine.e)),
    }


def _require_same_grid(datasets: list, grids: list[dict]) -> None:
    reference = datasets[0]
    for dataset, grid in zip(datasets[1:], grids[1:]):
        if dataset.crs != reference.crs:
            raise ValueError(f"raster grids disagree on CRS: {dataset.name}")
        for field in ("resolution_m", "transform", "width", "height", "bounds"):
            if grid[field] != grids[0][field]:
                raise ValueError(
                    f"raster grids disagree on {field} (alignment/extent); "
                    "no resampling is permitted"
                )


def _source_record(path: Path, dataset: rasterio.io.DatasetReader, index: int, grid: dict) -> dict:
    dependencies = sorted({_local_file(Path(name)) for name in dataset.files} | {path})
    files = [{"path": str(file), "sha256": _file_sha256(file)} for file in dependencies]
    return {
        "index": index,
        "role": ("pre" if index == 0 else "post" if index == 1 else f"observation_{index + 1}"),
        "path": str(path),
        "sha256": next(file["sha256"] for file in files if file["path"] == str(path)),
        "files": files,
        "grid": grid,
        "tags": dataset.tags(),
        "all_bands_valid_pixel_count": 0,
        "bands": [
            {
                "index": band,
                "description": dataset.descriptions[band - 1] or f"band_{band}",
                "dtype": dataset.dtypes[band - 1],
                "nodata": _json_number(dataset.nodatavals[band - 1]),
                "tags": dataset.tags(band),
                "valid_pixel_count": 0,
                "invalid_pixel_count": 0,
            }
            for band in dataset.indexes
        ],
    }


def _count_pixels(datasets: list, sources: list[dict], delivered: BaseGeometry) -> tuple[int, int]:
    """Bounded-memory, native-grid reads over the delivered AOI bounding window."""
    if delivered.is_empty or delivered.area <= 0:
        return 0, 0
    reference = datasets[0]
    window = from_bounds(*delivered.bounds, transform=reference.transform)
    col_start = max(0, math.floor(window.col_off))
    row_start = max(0, math.floor(window.row_off))
    col_stop = min(reference.width, math.ceil(window.col_off + window.width))
    row_stop = min(reference.height, math.ceil(window.row_off + window.height))
    covered_count = combined_count = 0
    for row in range(row_start, row_stop, 512):
        for col in range(col_start, col_stop, 512):
            height, width = min(512, row_stop - row), min(512, col_stop - col)
            block = Window(col, row, width, height)
            centers = geometry_mask(
                [mapping(delivered)],
                out_shape=(height, width),
                transform=reference.window_transform(block),
                all_touched=False,
                invert=True,
            )
            count = int(np.count_nonzero(centers))
            covered_count += count
            if not count:
                continue
            combined = centers.copy()
            for dataset, source in zip(datasets, sources):
                all_bands = centers.copy()
                for band_record in source["bands"]:
                    band = band_record["index"]
                    values = dataset.read(band, window=block)
                    valid = (
                        centers
                        & (dataset.read_masks(band, window=block) != 0)
                        & np.isfinite(values)
                        & (values > 0)
                    )
                    nodata = dataset.nodatavals[band - 1]
                    # Explicit nodata remains necessary when an external/internal
                    # GDAL mask overrides GDAL's nodata-derived mask.
                    if nodata is not None:
                        valid &= values != nodata
                    band_record["valid_pixel_count"] += int(np.count_nonzero(valid))
                    all_bands &= valid
                source["all_bands_valid_pixel_count"] += int(np.count_nonzero(all_bands))
                combined &= all_bands
            combined_count += int(np.count_nonzero(combined))
    for source in sources:
        for band in source["bands"]:
            band["invalid_pixel_count"] = covered_count - band["valid_pixel_count"]
    return covered_count, combined_count


def measure_observation_coverage(
    polygon_utm: BaseGeometry, raster_paths: Sequence[Path]
) -> dict:
    """Measure geometric delivery and joint valid linear-SAR observations locally.

    Empty source lists, URLs/virtual paths, missing files, unsupported raster
    formats, missing/geographic/non-metre CRSs, grid mismatches and invalid AOIs
    raise before joint pixel reads. No overlap is a successful zero-coverage
    measurement. Zero pixel-center coverage has a None delivered-valid fraction.

    Top-level fields: schema_version, metric_crs, aoi_area_km2,
    delivered_area_km2, undelivered_area_km2, delivered_fraction,
    unobserved_fraction, delivered_pixel_count, combined_valid_pixel_count,
    valid_area_km2, valid_fraction_of_delivered_pixels, valid_fraction_of_aoi,
    raw_valid_fraction_of_aoi, valid_fraction_of_aoi_clipped,
    invalid_or_undelivered_fraction_of_aoi, aoi, common_local_window,
    observation_footprint, grid, sources, methods, disclosure.
    """
    aoi = _polygon(polygon_utm, "EPSG:32645 AOI")
    if not raster_paths:
        raise ValueError("at least one local raster is required")
    paths = [_local_file(path) for path in raster_paths]
    # Restrict the reader *before opening*, so a local VRT cannot fetch remote
    # SourceFilename dependencies. Full-resolution GeoTIFF reads need no URLs.
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="FALSE"), ExitStack() as stack:
        datasets = [
            stack.enter_context(rasterio.open(path, driver="GTiff"))
            for path in paths
        ]
        if any(dataset.driver != "GTiff" for dataset in datasets):
            raise ValueError("only local GeoTIFF rasters are supported")
        grids = [_grid(dataset) for dataset in datasets]
        _require_same_grid(datasets, grids)
        grid = grids[0]
        native_aoi = _polygon(_project(aoi, _METRIC_CRS, grid["crs"]), "AOI in raster CRS")
        common_window = box(*grid["bounds"])
        delivered_native = _area_only(native_aoi.intersection(common_window))
        # Boundary-only contact is not delivered area and has no covered pixels.
        if delivered_native.area <= 0:
            delivered_native = Polygon()
        delivered = _project(delivered_native, grid["crs"], _METRIC_CRS)
        if not delivered.is_valid:
            raise ValueError("invalid delivered observation footprint topology")
        sources = [
            _source_record(path, dataset, index, source_grid)
            for index, (path, dataset, source_grid) in enumerate(zip(paths, datasets, grids))
        ]
        covered_count, valid_count = _count_pixels(datasets, sources, delivered_native)

    aoi_area = float(aoi.area)
    delivered_area = min(aoi_area, max(0.0, float(delivered.area)))
    delivered_fraction = delivered_area / aoi_area
    valid_area = float(valid_count * grid["pixel_area_m2"])
    raw_valid_fraction = valid_area / aoi_area
    valid_fraction = min(1.0, max(0.0, raw_valid_fraction))
    valid_of_delivered = valid_count / covered_count if covered_count else None
    disclosure = (
        f"Approximately {100 * delivered_fraction:.1f}% of the AOI has delivered imagery "
        f"coverage; the remaining {100 * (1 - delivered_fraction):.1f}% is geometrically "
        "undelivered (unobserved in this local dataset). "
    )
    if valid_of_delivered is None:
        disclosure += "No delivered AOI pixel centers were covered; pixel validity is undefined. "
    else:
        disclosure += (
            f"Of {covered_count} delivered AOI pixel centers, {valid_count} "
            f"({100 * valid_of_delivered:.1f}%) are jointly valid across every source and band. "
        )
    disclosure += (
        f"Valid pixel-area estimate: {valid_area / 1e6:.6f} km2; "
        f"full geometric AOI area: {aoi_area / 1e6:.6f} km2. "
        "Pixel-center rasterization estimates and geometric delivered coverage are distinct. "
        "Results apply only to the observed, valid-data area. Unobserved areas must not be "
        "interpreted as unaffected, flood-free, or undamaged."
    )
    return {
        "schema_version": 1,
        "metric_crs": _METRIC_CRS,
        "aoi_area_km2": aoi_area / 1e6,
        "delivered_area_km2": delivered_area / 1e6,
        "undelivered_area_km2": (aoi_area - delivered_area) / 1e6,
        "delivered_fraction": delivered_fraction,
        "unobserved_fraction": max(0.0, 1 - delivered_fraction),
        "delivered_pixel_count": covered_count,
        "combined_valid_pixel_count": valid_count,
        "valid_area_km2": valid_area / 1e6,
        "valid_fraction_of_delivered_pixels": valid_of_delivered,
        "valid_fraction_of_aoi": valid_fraction,
        "raw_valid_fraction_of_aoi": raw_valid_fraction,
        "valid_fraction_of_aoi_clipped": raw_valid_fraction != valid_fraction,
        "invalid_or_undelivered_fraction_of_aoi": max(0.0, 1 - valid_fraction),
        "aoi": _geometry_record(aoi, _METRIC_CRS),
        "common_local_window": _geometry_record(common_window, grid["crs"]),
        "observation_footprint": {
            **_geometry_record(delivered, _METRIC_CRS),
            "definition": "AOI_intersect_common_local_window_geometric_delivery_not_valid_mask",
        },
        "observation_footprint_4326": _geometry_record(
            _project(delivered, _METRIC_CRS, _GEOGRAPHIC_CRS), _GEOGRAPHIC_CRS
        ),
        "unobserved_footprint_4326": {
            **_geometry_record(
                _project(_area_only(aoi.difference(delivered)), _METRIC_CRS, _GEOGRAPHIC_CRS),
                _GEOGRAPHIC_CRS,
            ),
            "definition": "AOI_outside_common_local_window_unknown_not_negative_evidence",
        },
        "grid": {
            **grid,
            "alignment_verified": True,
            "extent_verified": True,
            "comparison": "exact_CRS_affine_resolution_dimensions_extent",
            "resampling": "none",
        },
        "sources": sources,
        "methods": {
            "rasterization": "center",
            "all_touched": False,
            "resampling": "none",
            "geometric_area_crs": _METRIC_CRS,
            "valid_pixel_area_crs": grid["crs"],
            "validity": "all_sources_all_bands: GDAL_mask AND not_nodata AND finite AND >0",
            "radiometry_assumption": "linear_SAR_power_not_dB",
            "valid_area": "pixel_count_times_native_projected_pixel_area_not_polygon_clipping",
            "valid_fraction_of_aoi": "clip(valid_area_estimate/full_geometric_AOI_area,0,1)",
            "rasterization_distinction": (
                "Boundary pixels contribute their full area if their center is included; "
                "the pixel-area estimate can exceed geometric AOI/delivered area. "
                "For other projected-metre CRSs, native pixel area and EPSG:32645 area "
                "also have different projection scale; vector edges are transformed at vertices."
            ),
            "unobserved_fraction": "1-delivered_fraction_geometric_not_invalid_pixel_fraction",
            "source_roles": "input_order: pre,post,observation_3,...; no_date_or_orbit_gate",
        },
        "disclosure": disclosure,
    }
