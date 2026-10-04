"""Offline M2 tests using tiny synthetic GeoTIFFs only."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from affine import Affine
from rasterio.transform import from_origin

from floodmap.acquisition.aoi import AreaOfInterest
from floodmap.acquisition.manifest import AcquisitionManifest
from floodmap.acquisition.outcomes import DiscoveryStatus
from floodmap.acquisition.providers.base import DiscoveryResult, SearchRequest
from floodmap.acquisition.scenes import Sentinel1Scene, Sentinel2Scene, SensorKind
from floodmap.acquisition.selection import select_sentinel1_pair, select_sentinel2_pair
from floodmap.acquisition.temporal import EventSpec, SearchWindow
from floodmap.preprocessing.config import PreprocessingConfig
from floodmap.preprocessing.errors import (
    AlignmentError,
    InputValidationError,
    PreprocessingConfigurationError,
)
from floodmap.preprocessing.grid import AnalysisGrid, compare_grids
from floodmap.preprocessing.inputs import PreprocessingInputs
from floodmap.preprocessing.pipeline import (
    preprocess_dem,
    preprocess_sentinel1_pair,
    preprocess_sentinel2_pair,
)
from floodmap.preprocessing.validation import inspect_raster, validate_raster
from floodmap.utils.provenance import ArtifactType, ProductionInput


AOI = AreaOfInterest.from_bbox("synthetic-m2", [0.0, 0.0, 4.0, 4.0], is_synthetic=True)
EVENT = EventSpec(
    event_date=date(2024, 1, 10),
    event_time_utc=datetime(2024, 1, 10, 12, tzinfo=timezone.utc),
)
PLAN = EVENT.plan(SearchWindow(before_days=5, after_days=5))
TRANSFORM = from_origin(0.0, 4.0, 1.0, 1.0)


def _s1(scene_id: str, acquired_at: datetime) -> Sentinel1Scene:
    return Sentinel1Scene(
        scene_id=scene_id,
        product_id=f"product-{scene_id}",
        acquired_at=acquired_at,
        product_type="IW_GRDH_1S",
        platform_short_name="SENTINEL-1",
        platform_unit="S1A",
        provider="cdse-odata",
        relative_orbit=12,
        orbit_direction="DESCENDING",
        polarisations=("VV", "VH"),
    )


def _s2(scene_id: str, acquired_at: datetime) -> Sentinel2Scene:
    return Sentinel2Scene(
        scene_id=scene_id,
        product_id=f"product-{scene_id}",
        acquired_at=acquired_at,
        product_type="S2MSI2A",
        platform_short_name="SENTINEL-2",
        platform_unit="S2A",
        provider="cdse-odata",
        scene_cloud_percent=12.0,
    )


def _manifest(sensor: SensorKind, before, after) -> AcquisitionManifest:
    selection = (
        select_sentinel1_pair([before, after], PLAN)
        if sensor is SensorKind.SENTINEL1
        else select_sentinel2_pair([before, after], PLAN, strategy="nearest_in_time")
    )
    request = SearchRequest(
        aoi=AOI,
        sensor=sensor,
        interval=PLAN.full_extent,
        product_type=before.product_type,
    )
    discovery = DiscoveryResult(
        status=DiscoveryStatus.SUCCESS,
        request=request,
        scenes=(before, after),
        provider="cdse-odata",
    )
    return AcquisitionManifest.from_run(
        aoi=AOI,
        event=EVENT,
        temporal_plans={sensor.value: PLAN},
        requests={sensor.value: request},
        discoveries={sensor.value: discovery},
        selections={sensor.value: selection},
        provider="cdse-odata",
        config_version="m2-test-config",
    )


def _grid() -> AnalysisGrid:
    return AnalysisGrid(
        crs="EPSG:4326",
        transform=TRANSFORM,
        width=4,
        height=4,
        resolution_m=1.0,
    )


def _config(**overrides) -> PreprocessingConfig:
    mapping = {
        "version": "m2-test-config",
        "pipeline_version": "m2-test-pipeline",
        "target_grid": {
            "crs": "EPSG:4326",
            "resolution_m": 1.0,
            "transform": list(TRANSFORM[:6]),
            "width": 4,
            "height": 4,
        },
        "alignment_checks": {"transform_tolerance": 0.0, "on_failure": "fail"},
        "sentinel1": {
            "required_polarizations": ["VV", "VH"],
            "radiometric_calibration": "source_calibrated",
            "terrain_correction": "source_terrain_corrected",
            "resampling": "nearest",
            "output_nodata": -9999.0,
        },
        "sentinel2": {
            "required_bands": ["B02", "B03"],
            "quality_mask": {"enabled": True, "band_name": "QA", "valid_values": [0]},
            "resampling": "nearest",
            "output_nodata": -9999.0,
        },
        "dem": {
            "source_product": "Copernicus WorldDEM-30",
            "resampling": "nearest",
            "output_nodata": -9999.0,
        },
    }
    for key, value in overrides.items():
        mapping[key] = value
    return PreprocessingConfig.model_validate(mapping)


def _write_raster(
    path: Path, arrays: list[np.ndarray], descriptions: tuple[str, ...], *, nodata=-9999.0
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=len(arrays),
        dtype="float32",
        crs="EPSG:4326",
        transform=TRANSFORM,
        nodata=nodata,
    ) as dataset:
        for index, array in enumerate(arrays, start=1):
            dataset.write(array.astype("float32"), index)
            dataset.set_band_description(index, descriptions[index - 1])


def test_raster_validation_checks_geospatial_metadata_and_named_bands(tmp_path):
    path = tmp_path / "valid.tif"
    _write_raster(path, [np.ones((4, 4), dtype="float32")], ("B02",))
    metadata = inspect_raster(path)
    assert metadata.crs == "EPSG:4326"
    assert validate_raster(path, expected_crs="EPSG:4326", required_bands=("B02",)).valid
    report = validate_raster(path, required_bands=("B03",))
    assert not report.valid
    assert "B03" in report.errors[0]


def test_raster_validation_rejects_missing_crs(tmp_path):
    path = tmp_path / "no-crs.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype="float32"), 1)
    report = validate_raster(path)
    assert not report.valid
    assert "no CRS" in report.errors[0]


def test_grid_alignment_reports_crs_transform_and_resolution_failures():
    reference = _grid()
    shifted = AnalysisGrid(
        crs="EPSG:3857",
        transform=Affine(2, 0, 0, 0, -2, 4),
        width=4,
        height=4,
        resolution_m=2.0,
    )
    report = compare_grids(reference, shifted)
    assert not report.aligned
    assert {"crs", "resolution", "transform", "extent"}.issubset(report.failures)
    with pytest.raises(AlignmentError):
        from floodmap.preprocessing.grid import require_aligned

        require_aligned(reference, shifted)


def test_m1_manifest_is_the_only_source_selection_boundary(tmp_path):
    before = _s1("s1-before", datetime(2024, 1, 9, 12, tzinfo=timezone.utc))
    after = _s1("s1-after", datetime(2024, 1, 11, 12, tzinfo=timezone.utc))
    manifest = _manifest(SensorKind.SENTINEL1, before, after)
    with pytest.raises(Exception, match="no explicit local source path"):
        PreprocessingInputs.from_manifest(manifest, {})
    before_path = tmp_path / "before.tif"
    after_path = tmp_path / "after.tif"
    _write_raster(before_path, [np.ones((4, 4)), np.ones((4, 4))], ("VV", "VH"))
    _write_raster(after_path, [np.ones((4, 4)), np.ones((4, 4))], ("VV", "VH"))
    inputs = PreprocessingInputs.from_manifest(
        manifest,
        {before.scene_id: before_path, after.scene_id: after_path},
    )
    assert inputs.sentinel1 is not None
    assert inputs.sentinel1[0].manifest_id == manifest.artifact_id


def test_sentinel1_processing_masks_invalid_pixels_and_records_provenance(tmp_path):
    before = _s1("s1-before", datetime(2024, 1, 9, 12, tzinfo=timezone.utc))
    after = _s1("s1-after", datetime(2024, 1, 11, 12, tzinfo=timezone.utc))
    manifest = _manifest(SensorKind.SENTINEL1, before, after)
    before_path = tmp_path / "before.tif"
    after_path = tmp_path / "after.tif"
    before_values = np.ones((4, 4), dtype="float32")
    before_values[0, 0] = -9999
    _write_raster(before_path, [before_values, before_values], ("VV", "VH"))
    _write_raster(after_path, [before_values, before_values], ("VV", "VH"))
    metadata = {
        scene_id: {
            "radiometric_calibrated": True,
            "terrain_corrected": True,
            "pair_geometry_verified": True,
        }
        for scene_id in (before.scene_id, after.scene_id)
    }
    artifacts = preprocess_sentinel1_pair(
        manifest,
        {before.scene_id: before_path, after.scene_id: after_path},
        tmp_path / "processed",
        config=_config(),
        grid=_grid(),
        source_metadata=metadata,
    )
    assert artifacts[0].quality["invalid_pixel_count"] == 1
    with rasterio.open(artifacts[0].artifact_path) as dataset:
        assert dataset.read(1)[0, 0] == -9999.0
    assert artifacts[0].provenance.artifact_type is ArtifactType.PREPROCESSED_RASTER
    assert artifacts[0].provenance.acquisition_manifest_id == manifest.artifact_id
    assert set(artifacts[0].provenance.source_product_ids) == {
        before.product_id,
        after.product_id,
    }
    assert (
        yaml.safe_load(artifacts[0].provenance_path.read_text())["quality"]["status"] == "success"
    )
    assert artifacts[0].artifact_path.exists()


def test_sentinel1_processing_requires_explicit_correction_metadata(tmp_path):
    before = _s1("s1-before", datetime(2024, 1, 9, 12, tzinfo=timezone.utc))
    after = _s1("s1-after", datetime(2024, 1, 11, 12, tzinfo=timezone.utc))
    manifest = _manifest(SensorKind.SENTINEL1, before, after)
    paths = {}
    for scene in (before, after):
        path = tmp_path / f"{scene.scene_id}.tif"
        _write_raster(path, [np.ones((4, 4)), np.ones((4, 4))], ("VV", "VH"))
        paths[scene.scene_id] = path
    with pytest.raises(PreprocessingConfigurationError, match="radiometric_calibrated"):
        preprocess_sentinel1_pair(
            manifest,
            paths,
            tmp_path / "processed",
            config=_config(),
            grid=_grid(),
            source_metadata={
                scene.scene_id: {"pair_geometry_verified": True} for scene in (before, after)
            },
        )


def test_sentinel1_processing_rejects_different_orbit_tracks(tmp_path):
    before = _s1("s1-before", datetime(2024, 1, 9, 12, tzinfo=timezone.utc))
    after_base = _s1("s1-after", datetime(2024, 1, 11, 12, tzinfo=timezone.utc))
    after = after_base.model_copy(update={"relative_orbit": 13})
    manifest = _manifest(SensorKind.SENTINEL1, before, after)
    paths = {}
    for scene in (before, after):
        path = tmp_path / f"{scene.scene_id}.tif"
        _write_raster(path, [np.ones((4, 4)), np.ones((4, 4))], ("VV", "VH"))
        paths[scene.scene_id] = path
    metadata = {
        scene.scene_id: {
            "pair_geometry_verified": True,
            "radiometric_calibrated": True,
            "terrain_corrected": True,
        }
        for scene in (before, after)
    }
    # M1 refuses to create a selected pair before M2 can process it.
    with pytest.raises(InputValidationError, match="no selected Sentinel-1 pair"):
        preprocess_sentinel1_pair(
            manifest,
            paths,
            tmp_path / "processed",
            config=_config(),
            grid=_grid(),
            source_metadata=metadata,
        )


def test_sentinel2_quality_mask_is_pixel_based_and_scene_cloud_is_preserved(tmp_path):
    before = _s2("s2-before", datetime(2024, 1, 9, 12, tzinfo=timezone.utc))
    after = _s2("s2-after", datetime(2024, 1, 11, 12, tzinfo=timezone.utc))
    manifest = _manifest(SensorKind.SENTINEL2, before, after)
    paths = {}
    for scene in (before, after):
        path = tmp_path / f"{scene.scene_id}.tif"
        qa = np.zeros((4, 4), dtype="float32")
        qa[0, 0] = 1
        _write_raster(path, [np.ones((4, 4)), np.ones((4, 4)), qa], ("B02", "B03", "QA"))
        paths[scene.scene_id] = path
    metadata = {scene.scene_id: {"pair_geometry_verified": True} for scene in (before, after)}
    artifacts = preprocess_sentinel2_pair(
        manifest,
        paths,
        tmp_path / "processed",
        config=_config(),
        grid=_grid(),
        source_metadata=metadata,
    )
    assert artifacts[0].quality["scene_cloud_percent"] == 12.0
    assert artifacts[0].quality["aoi_cloud_percent"]["unknown"] is True
    assert artifacts[0].quality["quality_mask"]["masked_pixel_percent"] > 0


def test_dem_source_and_metadata_are_preserved(tmp_path):
    before = _s1("s1-before", datetime(2024, 1, 9, 12, tzinfo=timezone.utc))
    after = _s1("s1-after", datetime(2024, 1, 11, 12, tzinfo=timezone.utc))
    manifest = _manifest(SensorKind.SENTINEL1, before, after)
    dem_path = tmp_path / "dem.tif"
    _write_raster(dem_path, [np.arange(16, dtype="float32").reshape(4, 4)], ("elevation",))
    artifact = preprocess_dem(
        manifest,
        dem_path,
        tmp_path / "processed",
        config=_config(),
        source_metadata={
            "product_name": "Copernicus WorldDEM-30",
            "version": "synthetic-test-version",
            "product_id": "dem-product-1",
        },
        grid=_grid(),
    )
    assert artifact.quality["source_metadata"]["product_name"] == "Copernicus WorldDEM-30"
    assert artifact.provenance.production_inputs == [ProductionInput.COPERNICUS_DEM]
    assert artifact.provenance.source_product_ids == ["dem-product-1"]
    assert artifact.provenance.dem_version == "synthetic-test-version"


def test_unresolved_production_grid_is_not_defaulted():
    config = PreprocessingConfig.model_validate({"target_grid": {}})
    with pytest.raises(PreprocessingConfigurationError, match="target_grid is unresolved"):
        config.require_grid()


def test_wrong_dem_source_is_rejected(tmp_path):
    before = _s1("s1-before", datetime(2024, 1, 9, 12, tzinfo=timezone.utc))
    after = _s1("s1-after", datetime(2024, 1, 11, 12, tzinfo=timezone.utc))
    manifest = _manifest(SensorKind.SENTINEL1, before, after)
    dem_path = tmp_path / "dem.tif"
    _write_raster(dem_path, [np.ones((4, 4))], ("elevation",))
    with pytest.raises(InputValidationError, match="Copernicus WorldDEM-30"):
        preprocess_dem(
            manifest,
            dem_path,
            tmp_path / "processed",
            config=_config(),
            source_metadata={"product_name": "arbitrary-dem", "version": "x", "product_id": "x"},
            grid=_grid(),
        )
