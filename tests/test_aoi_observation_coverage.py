"""Coverage science tests: fixtures are SYNTHETIC, not flood evidence.

Optional tests additionally inspect the two local SAFE archives. They never
download data and skip when the git-ignored inputs are absent. No test adopts,
changes or crops a study AOI, or modifies an existing source raster/archive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest
import rasterio
import shapely
from pyproj import Transformer
from rasterio.transform import Affine, from_origin
from shapely.geometry import MultiPolygon, Polygon, box, shape

from floodmap.hydrology.coverage import (
    APPROVED_COVERAGE_DISCLOSURE,
    measure_observation_coverage,
    safe_annotation_footprint,
    scene_clearance,
)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_TRANSFORM = from_origin(500_000, 3_100_030, 10, 10)
SYNTHETIC_WINDOW = box(500_000, 3_100_000, 500_040, 3_100_030)


def _wkb_hash(geometry):
    return hashlib.sha256(shapely.to_wkb(geometry, byte_order=1, output_dimension=2)).hexdigest()


def _synthetic_grid(rows=10, columns=21):
    # Curved edges distinguish the actual ordered perimeter from a bbox/hull.
    return [
        (
            row * 100,
            col * 200,
            85 + col * 0.01 + row**2 * 0.00002,
            27 + row * 0.02 + (col - 10)**2 * 0.0001,
        )
        for row in range(rows)
        for col in range(columns)
    ]


def _synthetic_safe(tmp_path, *, points=None, metadata=True, manifest=True, duplicate_vv=False):
    """Write a small, explicitly synthetic SAFE-like archive in pytest temp space."""
    points = _synthetic_grid() if points is None else points
    root = ET.Element("product", xmlns="urn:synthetic-sentinel-fixture")
    header = ET.SubElement(root, "adsHeader")
    ET.SubElement(header, "polarisation").text = "VV"
    if metadata:
        ET.SubElement(header, "startTime").text = "2026-08-16T12:21:41.058298"
        ET.SubElement(header, "stopTime").text = "2026-08-16T12:22:06.056880"
        ET.SubElement(root, "pass").text = "Ascending"
        ET.SubElement(root, "numberOfLines").text = "901"
        ET.SubElement(root, "numberOfSamples").text = "4001"
    grid = ET.SubElement(root, "geolocationGridPointList", count=str(len(points)))
    # Reverse order to ensure XML iteration order is never used as ring order.
    for line, pixel, lon, lat in reversed(points):
        point = ET.SubElement(grid, "geolocationGridPoint")
        fields = (("line", line), ("pixel", pixel), ("longitude", lon), ("latitude", lat))
        for name, value in fields:
            ET.SubElement(point, name).text = str(value)
    annotation = ET.tostring(root)
    member = "SYNTHETIC.SAFE/annotation/synthetic-vv.xml"
    manifest_bytes = (
        b'<manifest xmlns="urn:synthetic-safe">'
        b'<relativeOrbitNumber type="start">85</relativeOrbitNumber>'
        b'<relativeOrbitNumber type="stop">85</relativeOrbitNumber>'
        b'<pass>ASCENDING</pass>'
        b'<startTime>2026-08-16T12:21:41.058298Z</startTime>'
        b'<stopTime>2026-08-16T12:22:06.056880Z</stopTime></manifest>'
    )
    path = tmp_path / "SYNTHETIC.SAFE.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(member, annotation)
        # Not image annotation: this is a distractor with the same polarisation.
        archive.writestr("SYNTHETIC.SAFE/annotation/rfi/rfi-vv.xml", annotation)
        if duplicate_vv:
            archive.writestr("SYNTHETIC.SAFE/annotation/another-vv.xml", annotation)
        if manifest:
            archive.writestr("SYNTHETIC.SAFE/manifest.safe", manifest_bytes)
    return path, member, annotation, manifest_bytes


def _synthetic_raster(
    tmp_path,
    name="synthetic.tif",
    *,
    values=None,
    crs="EPSG:32645",
    affine=SYNTHETIC_TRANSFORM,
    nodata=None,
    mask=None,
):
    """SYNTHETIC linear-power pixels; no values represent measured flooding."""
    values = np.ones((2, 3, 4), dtype="float32") if values is None else values
    path = tmp_path / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[2],
        height=values.shape[1],
        count=values.shape[0],
        dtype=values.dtype,
        crs=crs,
        transform=affine,
        nodata=nodata,
    ) as dataset:
        dataset.write(values)
        for band in dataset.indexes:
            dataset.set_band_description(band, ("VV", "VH")[band - 1])
        dataset.update_tags(fixture="SYNTHETIC_NOT_OBSERVATIONAL_EVIDENCE")
        if mask is not None:
            dataset.write_mask(mask)
    return path


def _wgs84(geometry):
    transformer = Transformer.from_crs(32645, 4326, always_xy=True)
    return shapely.transform(
        geometry, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1]))
    )


def test_synthetic_safe_sorted_perimeter_and_independent_source_hashes(tmp_path):
    path, member, annotation, manifest = _synthetic_safe(tmp_path)
    footprint, record = safe_annotation_footprint(path)
    points = {(line, pixel): (lon, lat) for line, pixel, lon, lat in _synthetic_grid()}
    expected = (
        [points[0, col * 200] for col in range(21)]
        + [points[row * 100, 4000] for row in range(1, 10)]
        + [points[900, col * 200] for col in range(19, -1, -1)]
        + [points[row * 100, 0] for row in range(8, 0, -1)]
    )
    assert list(footprint.exterior.coords) == expected + [expected[0]]
    assert footprint.is_valid
    assert footprint.area < footprint.envelope.area
    assert footprint.area < footprint.convex_hull.area
    dimensions = (record["grid_rows"], record["grid_columns"], record["grid_point_count"])
    assert dimensions == (10, 21, 210)
    assert record["perimeter_point_count"] == 58
    assert record["vertex_count"] == 59
    assert record["footprint_crs"] == "EPSG:4326"
    assert record["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["annotation_member"] == member
    assert record["annotation_sha256"] == hashlib.sha256(annotation).hexdigest()
    assert record["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert record["footprint_wkb_sha256"] == _wkb_hash(footprint)
    assert record["acquisition_start_utc"] == "2026-08-16T12:21:41.058298Z"
    assert record["acquisition_stop_utc"] == "2026-08-16T12:22:06.056880Z"
    assert record["relative_orbit"] == 85
    assert record["orbit_direction"] == "ASCENDING"
    assert (record["image_lines"], record["image_samples"]) == (901, 4001)
    assert {item["member"] for item in record["metadata_evidence"]["acquisition_start_utc"]} == {
        member,
        "SYNTHETIC.SAFE/manifest.safe",
    }
    json.dumps(record, allow_nan=False)


def test_synthetic_safe_manifest_fallback_and_optional_absence(tmp_path):
    path, *_ = _synthetic_safe(tmp_path, metadata=False)
    _, record = safe_annotation_footprint(path)
    assert record["relative_orbit"] == 85
    assert record["acquisition_start_utc"] == "2026-08-16T12:21:41.058298Z"
    assert record["image_lines"] is None
    path, *_ = _synthetic_safe(tmp_path, metadata=False, manifest=False)
    _, record = safe_annotation_footprint(path)
    for key in ("relative_orbit", "orbit_direction", "acquisition_start_utc", "manifest_sha256"):
        assert record[key] is None


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite", "degenerate", "ambiguous"])
def test_synthetic_safe_rejects_unverifiable_grid(tmp_path, defect):
    points = _synthetic_grid()
    if defect == "missing":
        points.pop(100)
    elif defect == "duplicate":
        points.append(points[100])
    elif defect == "nonfinite":
        points[100] = (*points[100][:2], float("nan"), 28)
    elif defect == "degenerate":
        points = [(line, pixel, 85.0, 27.0) for line, pixel, _, _ in points]
    path, *_ = _synthetic_safe(tmp_path, points=points, duplicate_vv=defect == "ambiguous")
    with pytest.raises(ValueError):
        safe_annotation_footprint(path)


def test_synthetic_safe_metadata_conflict_is_not_silently_selected(tmp_path):
    path, member, annotation, manifest = _synthetic_safe(tmp_path)
    with ZipFile(path, "w") as archive:
        archive.writestr(member, annotation)
        archive.writestr(
            "SYNTHETIC.SAFE/manifest.safe", manifest.replace(b"ASCENDING", b"DESCENDING")
        )
    with pytest.raises(ValueError, match="conflicting SAFE metadata"):
        safe_annotation_footprint(path)


def test_synthetic_scene_clearance_is_metric_and_uses_common_swath():
    first = box(500_000, 3_100_000, 502_000, 3_102_000)
    second = box(500_200, 3_100_000, 502_500, 3_102_000)
    aoi = box(500_500, 3_100_500, 501_000, 3_101_000)
    result = scene_clearance(aoi, [_wgs84(first), _wgs84(second)])
    assert result["common_swath_area_km2"] == pytest.approx(3.6, abs=1e-8)
    assert result["clearance_m"] == pytest.approx(300, abs=1e-6)
    assert result["inset_m"] == 250
    assert result["fully_covered"] and result["accepted"]
    assert not scene_clearance(aoi, [_wgs84(first), _wgs84(second)], inset_m=301)["accepted"]
    outside = box(500_000, 3_100_500, 500_600, 3_101_000)
    result = scene_clearance(outside, [_wgs84(first), _wgs84(second)], inset_m=0)
    assert not result["fully_covered"] and not result["accepted"]
    assert result["clearance_m"] == 0
    json.dumps(result, allow_nan=False)


def test_synthetic_scene_disjoint_swaths():
    first = _wgs84(box(500_000, 3_100_000, 501_000, 3_101_000))
    second = _wgs84(box(505_000, 3_100_000, 506_000, 3_101_000))
    result = scene_clearance(SYNTHETIC_WINDOW, [first, second])
    assert result["common_swath_area_km2"] == result["clearance_m"] == 0
    assert not result["accepted"]


def test_synthetic_scene_zero_area_contacts_do_not_break_common_boundary():
    # Intersection has both positive area and a detached shared edge. Its
    # GeometryCollection boundary is undefined; clearance uses its area parts.
    first = MultiPolygon([
        box(85, 27, 85.02, 27.02),
        box(85.04, 27, 85.06, 27.02),
    ])
    second = MultiPolygon([
        box(85, 27, 85.02, 27.02),
        box(85.06, 27, 85.08, 27.02),
    ])
    transformer = Transformer.from_crs(4326, 32645, always_xy=True)
    aoi = shapely.transform(
        box(85.005, 27.005, 85.01, 27.01),
        lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])),
    )
    result = scene_clearance(aoi, [first, second], inset_m=100)
    assert result["accepted"]
    assert result["clearance_m"] > 400
    assert result["common_swath"]["geometry"]["type"] == "Polygon"
    result = scene_clearance(aoi, [box(85, 27, 85.02, 27.02), box(85.02, 27, 85.04, 27.02)])
    assert result["common_swath_area_km2"] == 0
    assert not result["accepted"]


@pytest.mark.parametrize("inset", [-1, float("nan"), float("inf")])
def test_synthetic_scene_invalid_inset(inset):
    with pytest.raises(ValueError, match="inset_m"):
        scene_clearance(SYNTHETIC_WINDOW, [_wgs84(SYNTHETIC_WINDOW)], inset_m=inset)


@pytest.mark.parametrize("footprint", [
    Polygon([(85, 27), (86, 28), (85, 28), (86, 27), (85, 27)]),
    SYNTHETIC_WINDOW,
])
def test_synthetic_scene_rejects_invalid_or_mislabeled_footprints(footprint):
    with pytest.raises(ValueError, match="topology|longitude/latitude"):
        scene_clearance(SYNTHETIC_WINDOW, [footprint])


def test_synthetic_delivery_is_separate_from_full_polygon_and_valid_pixels(tmp_path):
    pre = _synthetic_raster(tmp_path, "synthetic_pre.tif")
    post = _synthetic_raster(tmp_path, "synthetic_post.tif")
    # Full AOI is twice the delivered window, without changing the raster.
    aoi = box(499_960, 3_100_000, 500_040, 3_100_030)
    result = measure_observation_coverage(aoi, [pre, post])
    assert result["aoi_area_km2"] == pytest.approx(0.0024)
    assert result["delivered_area_km2"] == pytest.approx(0.0012)
    assert result["undelivered_area_km2"] == pytest.approx(0.0012)
    assert result["delivered_fraction"] == result["unobserved_fraction"] == 0.5
    assert result["valid_area_km2"] == pytest.approx(0.0012)
    assert result["delivered_pixel_count"] == result["combined_valid_pixel_count"] == 12
    assert result["valid_fraction_of_delivered_pixels"] == 1
    assert result["valid_fraction_of_aoi"] == 0.5
    footprint = result["observation_footprint"]
    assert shape(footprint["geometry"]).equals(SYNTHETIC_WINDOW)
    assert footprint["wkb_sha256"] == _wkb_hash(shape(footprint["geometry"]))
    assert result["methods"]["rasterization"] == "center"
    assert result["methods"]["resampling"] == result["grid"]["resampling"] == "none"
    assert result["grid"]["alignment_verified"] and result["grid"]["extent_verified"]
    assert [source["role"] for source in result["sources"]] == ["pre", "post"]
    for source, path in zip(result["sources"], [pre, post]):
        assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert source["bands"][0]["valid_pixel_count"] == 12
        assert source["tags"]["fixture"] == "SYNTHETIC_NOT_OBSERVATIONAL_EVIDENCE"
    assert "50.0%" in result["disclosure"]
    assert "63.4%" not in result["disclosure"]
    json.dumps(result, allow_nan=False)


def test_synthetic_masks_nodata_zero_nonfinite_and_per_band_pre_post_counts(tmp_path):
    pre_values = np.ones((2, 3, 4), dtype="float32")
    post_values = pre_values.copy()
    pre_values[0].flat[:6] = [0, np.nan, np.inf, -np.inf, -1, 999]
    pre_values[1].flat[6] = np.nan
    post_values[0].flat[7] = 0
    post_values[1].flat[8] = 999
    mask = np.full((3, 4), 255, dtype="uint8")
    mask.flat[9] = 0
    # Force an external mask so its hash is independently checked. Positive
    # nodata must still be rejected even when this mask marks it valid.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        pre = _synthetic_raster(
            tmp_path, "synthetic_pre.tif", values=pre_values, nodata=999, mask=mask
        )
    post = _synthetic_raster(tmp_path, "synthetic_post.tif", values=post_values, nodata=999)
    result = measure_observation_coverage(SYNTHETIC_WINDOW, [pre, post])
    assert result["delivered_fraction"] == 1
    assert result["unobserved_fraction"] == 0
    assert result["combined_valid_pixel_count"] == 2
    assert result["valid_area_km2"] == pytest.approx(0.0002)
    assert result["valid_fraction_of_delivered_pixels"] == pytest.approx(2 / 12)
    assert result["valid_fraction_of_aoi"] == pytest.approx(2 / 12)
    assert [[b["valid_pixel_count"] for b in s["bands"]] for s in result["sources"]] == [
        [5, 10],
        [11, 11],
    ]
    assert [s["all_bands_valid_pixel_count"] for s in result["sources"]] == [4, 10]
    assert result["sources"][0]["bands"][0]["invalid_pixel_count"] == 7
    sidecar = pre.with_suffix(".tif.msk")
    assert sidecar.is_file()
    assert {file["path"]: file["sha256"] for file in result["sources"][0]["files"]}[
        str(sidecar)
    ] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert result["invalid_or_undelivered_fraction_of_aoi"] == pytest.approx(10 / 12)


@pytest.mark.parametrize("fill", [0, float("nan"), float("inf"), -1, 999])
def test_synthetic_delivered_but_entirely_invalid(tmp_path, fill):
    values = np.full((2, 3, 4), fill, dtype="float32")
    path = _synthetic_raster(tmp_path, values=values, nodata=999)
    result = measure_observation_coverage(SYNTHETIC_WINDOW, [path])
    assert result["delivered_fraction"] == 1
    assert result["combined_valid_pixel_count"] == result["valid_area_km2"] == 0
    assert result["valid_fraction_of_delivered_pixels"] == 0


@pytest.mark.parametrize("nodata", [float("nan"), float("inf"), -32768])
def test_synthetic_nonfinite_nodata_is_json_compatible(tmp_path, nodata):
    values = np.ones((2, 3, 4), dtype="float32")
    values[0, 0, 0] = nodata
    path = _synthetic_raster(tmp_path, values=values, nodata=nodata)
    result = measure_observation_coverage(SYNTHETIC_WINDOW, [path])
    assert result["combined_valid_pixel_count"] == 11
    json.dumps(result, allow_nan=False)


def test_synthetic_no_overlap_and_boundary_only_contact(tmp_path):
    path = _synthetic_raster(tmp_path)
    for aoi in [box(0, 0, 100, 100), box(500_040, 3_100_000, 500_050, 3_100_030)]:
        result = measure_observation_coverage(aoi, [path])
        assert result["delivered_area_km2"] == result["valid_area_km2"] == 0
        assert result["unobserved_fraction"] == 1
        assert result["delivered_pixel_count"] == result["combined_valid_pixel_count"] == 0
        assert result["valid_fraction_of_delivered_pixels"] is None
        assert shape(result["observation_footprint"]["geometry"]).is_empty
        json.dumps(result, allow_nan=False)


def test_synthetic_subpixel_delivery_and_rasterization_distinction(tmp_path):
    path = _synthetic_raster(tmp_path)
    # Geometric delivery exists, but no pixel center is covered.
    aoi = box(500_001, 3_100_001, 500_002, 3_100_002)
    result = measure_observation_coverage(aoi, [path])
    assert result["delivered_area_km2"] == pytest.approx(1e-6)
    assert result["delivered_fraction"] == 1
    assert result["delivered_pixel_count"] == 0
    assert result["valid_fraction_of_delivered_pixels"] is None
    # A 1m2 AOI covering one center contributes one whole 100m2 pixel estimate.
    aoi = box(500_004.5, 3_100_004.5, 500_005.5, 3_100_005.5)
    result = measure_observation_coverage(aoi, [path])
    assert result["combined_valid_pixel_count"] == 1
    assert result["valid_area_km2"] == pytest.approx(0.0001)
    assert result["raw_valid_fraction_of_aoi"] == 100
    assert result["valid_fraction_of_aoi"] == 1
    assert result["valid_fraction_of_aoi_clipped"]
    assert result["unobserved_fraction"] == 0


def test_synthetic_holes_and_multipart_aoi_are_honored(tmp_path):
    path = _synthetic_raster(tmp_path)
    hole = box(500_010, 3_100_010, 500_020, 3_100_020)
    aoi = Polygon(SYNTHETIC_WINDOW.exterior.coords, [hole.exterior.coords])
    result = measure_observation_coverage(aoi, [path])
    assert result["delivered_pixel_count"] == 11
    assert result["aoi_area_km2"] == pytest.approx(0.0011)
    multipart = MultiPolygon([
        box(500_000, 3_100_000, 500_010, 3_100_010),
        box(500_030, 3_100_020, 500_040, 3_100_030),
    ])
    assert measure_observation_coverage(multipart, [path])["delivered_pixel_count"] == 2


def test_synthetic_other_projected_metre_crs_reprojects_only_vector_aoi(tmp_path):
    path = _synthetic_raster(tmp_path, crs="EPSG:32644")
    transformer = Transformer.from_crs(32644, 32645, always_xy=True)
    aoi = shapely.transform(
        SYNTHETIC_WINDOW,
        lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])),
    )
    result = measure_observation_coverage(aoi, [path])
    assert result["grid"]["crs"] == "EPSG:32644"
    assert result["metric_crs"] == "EPSG:32645"
    assert result["combined_valid_pixel_count"] == 12
    # Allow sub-millimetre PROJ round-trip error at this small synthetic extent.
    assert result["delivered_fraction"] == pytest.approx(1, abs=1e-7)
    assert result["methods"]["valid_pixel_area_crs"] == "EPSG:32644"
    assert result["methods"]["resampling"] == "none"


def test_synthetic_native_grid_block_seams_are_counted_once(tmp_path):
    values = np.ones((2, 513, 513), dtype="float32")
    values[0, 511, 512] = 0
    values[1, 512, 511] = np.nan
    path = _synthetic_raster(tmp_path, values=values)
    aoi = box(500_000, 3_100_030 - 5130, 505_130, 3_100_030)
    result = measure_observation_coverage(aoi, [path])
    assert result["delivered_pixel_count"] == 513**2
    assert result["combined_valid_pixel_count"] == 513**2 - 2


@pytest.mark.parametrize(
    "defect", ["crs", "resolution", "half_pixel", "integer_offset", "extent", "rotation"]
)
def test_synthetic_joint_reads_reject_incompatible_grids(tmp_path, defect):
    first = _synthetic_raster(tmp_path, "synthetic_pre.tif")
    kwargs = {}
    if defect == "crs":
        kwargs["crs"] = "EPSG:32644"
    elif defect == "resolution":
        kwargs["affine"] = from_origin(500_000, 3_100_030, 20, 10)
    elif defect in ("half_pixel", "integer_offset"):
        easting = 500_005 if defect == "half_pixel" else 500_010
        kwargs["affine"] = from_origin(easting, 3_100_030, 10, 10)
    elif defect == "extent":
        kwargs["values"] = np.ones((2, 3, 5), dtype="float32")
    elif defect == "rotation":
        kwargs["affine"] = Affine(10, 0.1, 500_000, 0.1, -10, 3_100_030)
    second = _synthetic_raster(tmp_path, "synthetic_post.tif", **kwargs)
    with pytest.raises(ValueError, match="grids disagree|north-up"):
        measure_observation_coverage(SYNTHETIC_WINDOW, [first, second])


@pytest.mark.parametrize("crs", [None, "EPSG:4326", "EPSG:2263"])
def test_synthetic_missing_geographic_or_foot_crs_rejected(tmp_path, crs):
    path = _synthetic_raster(tmp_path, crs=crs)
    with pytest.raises(ValueError, match="CRS"):
        measure_observation_coverage(SYNTHETIC_WINDOW, [path])


@pytest.mark.parametrize("remote", [
    "https://example.invalid/remote.tif", "s3://bucket/data.tif",
    "/vsicurl/https://example.invalid/remote.tif", "//server/share/data.tif",
])
def test_network_paths_are_rejected_before_any_raster_open(monkeypatch, remote):
    def forbidden(*args, **kwargs):
        pytest.fail("rasterio.open must not be reached for a network path")
    monkeypatch.setattr(rasterio, "open", forbidden)
    with pytest.raises(ValueError, match="local files only"):
        measure_observation_coverage(SYNTHETIC_WINDOW, [Path(remote)])
    with pytest.raises(ValueError, match="local files only"):
        safe_annotation_footprint(Path(remote))


def test_local_vrt_cannot_hide_remote_sources(tmp_path):
    path = tmp_path / "SYNTHETIC_remote_source.vrt"
    path.write_text(
        '<VRTDataset rasterXSize="1" rasterYSize="1"><VRTRasterBand dataType="Float32" band="1">'
        '<SimpleSource><SourceFilename>/vsicurl/https://example.invalid/remote.tif</SourceFilename>'
        '</SimpleSource></VRTRasterBand></VRTDataset>'
    )
    with pytest.raises(rasterio.errors.RasterioIOError):
        measure_observation_coverage(SYNTHETIC_WINDOW, [path])


@pytest.mark.parametrize("aoi", [Polygon(), Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])])
def test_synthetic_invalid_aoi_is_not_silently_repaired(tmp_path, aoi):
    with pytest.raises(ValueError, match="nonempty|topology"):
        measure_observation_coverage(aoi, [_synthetic_raster(tmp_path)])
    with pytest.raises(ValueError, match="nonempty|topology"):
        scene_clearance(aoi, [_wgs84(SYNTHETIC_WINDOW)])


def test_empty_inputs_and_missing_files_fail(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        measure_observation_coverage(SYNTHETIC_WINDOW, [])
    with pytest.raises(ValueError, match="at least one"):
        scene_clearance(SYNTHETIC_WINDOW, [])
    with pytest.raises(FileNotFoundError):
        measure_observation_coverage(SYNTHETIC_WINDOW, [tmp_path / "missing.tif"])


def test_approved_disclosure_is_exact_and_separate_from_measurements():
    assert APPROVED_COVERAGE_DISCLOSURE == (
        "The study corridor extends beyond the currently available valid satellite observation "
        "coverage. Approximately 63.4% of the AOI has delivered imagery coverage; the remaining "
        "36.6% is unobserved in the current dataset. Results apply only to the observed, "
        "valid-data area. Unobserved areas must not be interpreted as unaffected, flood-free, "
        "or undamaged."
    )


@pytest.mark.parametrize("date,expected_start", [
    ("20260816", "2026-08-16T12:21:41.058298Z"),
    ("20260828", "2026-08-28T12:21:41.762029Z"),
])
def test_local_safe_full_swath_annotation_if_present(date, expected_start):
    """Actual local annotation check; not an end-to-end flood validation."""
    paths = sorted((ROOT / "data/raw/sentinel1").glob(f"*_{date}T*.SAFE.zip"))
    if not paths:
        pytest.skip("optional local Sentinel-1 SAFE archive is absent")
    assert len(paths) == 1
    footprint, record = safe_annotation_footprint(paths[0])
    assert footprint.is_valid and footprint.area > 0
    dimensions = (record["grid_rows"], record["grid_columns"], record["grid_point_count"])
    assert dimensions == (10, 21, 210)
    assert record["vertex_count"] == 59
    assert record["perimeter_point_count"] == 58
    assert record["relative_orbit"] == 85
    assert record["orbit_direction"] == "ASCENDING"
    assert record["acquisition_start_utc"] == expected_start
    assert (record["image_lines"], record["image_samples"]) == (16732, 25769)
    assert record["footprint_wkb_sha256"] == _wkb_hash(footprint)
    with ZipFile(paths[0]) as archive:
        assert record["annotation_sha256"] == hashlib.sha256(
            archive.read(record["annotation_member"])
        ).hexdigest()
    assert len(record["archive_sha256"]) == 64
