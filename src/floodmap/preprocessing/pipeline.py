"""M2 analysis-ready raster preprocessing.

This module consumes only selected products resolved from an M1 manifest. It
does not discover scenes, download data, run hydrology, or create predictions.
Operations that require product-level facts or external SAR tooling fail until
those facts are supplied explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..acquisition.manifest import AcquisitionManifest
from ..acquisition.scenes import SensorKind
from ..utils.provenance import ProductionInput
from .artifacts import ProcessedArtifact, write_raster_artifact
from .config import PreprocessingConfig
from .errors import (
    AlignmentError,
    InputValidationError,
    PreprocessingConfigurationError,
    UnsupportedPreprocessingError,
)
from .grid import (
    AnalysisGrid,
    compare_grids,
    require_aligned,
    require_rasterio,
    resampling_from_name,
)
from .inputs import ManifestSourceProduct, PreprocessingInputs
from .validation import (
    RasterMetadata,
    read_valid_mask,
    require_valid_raster,
    validate_required_bands,
)

try:
    import rasterio
    from rasterio.warp import reproject
except ImportError as exc:  # pragma: no cover
    rasterio = None  # type: ignore[assignment]
    reproject = None  # type: ignore[assignment]
    _RASTERIO_ERROR = exc
else:
    _RASTERIO_ERROR = None

__all__ = [
    "preprocess_dem",
    "preprocess_sentinel1_pair",
    "preprocess_sentinel2_pair",
    "validate_preprocessing_pair_alignment",
]


def _require_import() -> None:
    require_rasterio()
    if _RASTERIO_ERROR is not None:
        raise PreprocessingConfigurationError(
            "rasterio is required for M2 preprocessing"
        ) from _RASTERIO_ERROR


def _source_metadata(
    source: ManifestSourceProduct,
    metadata: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    value = metadata.get(source.scene_id)
    if value is None:
        raise PreprocessingConfigurationError(
            f"explicit product metadata is required for selected scene {source.scene_id!r}; "
            "M2 will not infer calibration or terrain-correction state"
        )
    return value


def _require_true(metadata: Mapping[str, Any], key: str, scene_id: str) -> None:
    if metadata.get(key) is not True:
        raise PreprocessingConfigurationError(
            f"{scene_id}: source metadata must explicitly establish {key}=true"
        )


def _band_indexes(metadata: RasterMetadata, names: Sequence[str]) -> Tuple[int, ...]:
    descriptions = {
        description.strip(): index + 1
        for index, description in enumerate(metadata.descriptions)
        if description
    }
    missing = [name for name in names if name not in descriptions]
    if missing:
        raise InputValidationError(
            f"{metadata.path} missing required named bands {missing}; "
            f"available descriptions are {sorted(descriptions)}"
        )
    return tuple(descriptions[name] for name in names)


def _read_bands(
    source: ManifestSourceProduct, indexes: Sequence[int]
) -> tuple[RasterMetadata, np.ndarray, np.ndarray]:
    _require_import()
    metadata = require_valid_raster(source.path, min_bands=max(indexes))
    with rasterio.open(source.path) as dataset:
        data = dataset.read(indexes).astype("float32", copy=False)
    return metadata, data, read_valid_mask(source.path, band_indexes=indexes)


def _reproject_to_grid(
    data: np.ndarray,
    valid: np.ndarray,
    source: RasterMetadata,
    target: AnalysisGrid,
    *,
    resampling_name: str,
    nodata: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _require_import()
    source_grid = source.as_grid()
    alignment = compare_grids(source_grid, target)
    method = resampling_from_name(resampling_name)
    if alignment.aligned:
        result = data.copy()
        result[:, ~valid] = nodata
        return (
            result,
            valid.copy(),
            {"source_to_target": alignment.to_dict(), "reprojected": False},
        )
    result = np.full((data.shape[0], target.height, target.width), nodata, dtype="float32")
    for index in range(data.shape[0]):
        reproject(
            source=data[index],
            destination=result[index],
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata if source.nodata is not None else nodata,
            dst_transform=target.transform,
            dst_crs=target.crs,
            dst_nodata=nodata,
            resampling=method,
        )
    target_valid = np.zeros((target.height, target.width), dtype="uint8")
    reproject(
        source=valid.astype("uint8"),
        destination=target_valid,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=0,
        dst_transform=target.transform,
        dst_crs=target.crs,
        dst_nodata=0,
        resampling=resampling_from_name("nearest", field="valid-mask resampling"),
    )
    target_valid = target_valid.astype(bool)
    result[:, ~target_valid] = nodata
    return result, target_valid, {"source_to_target": alignment.to_dict(), "reprojected": True}


def _require_pair_geometry(
    before: ManifestSourceProduct,
    after: ManifestSourceProduct,
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    before_meta = _source_metadata(before, metadata)
    after_meta = _source_metadata(after, metadata)
    if (
        before_meta.get("pair_geometry_verified") is not True
        or after_meta.get("pair_geometry_verified") is not True
    ):
        raise AlignmentError(
            "pre/post registration quality is not established; provide explicit "
            "pair_geometry_verified=true metadata for both selected products"
        )


def validate_preprocessing_pair_alignment(
    before_path: Path | str,
    after_path: Path | str,
    *,
    tolerance: Optional[float] = None,
) -> dict[str, Any]:
    """Return measurable source-grid alignment information without resampling."""
    before = require_valid_raster(before_path)
    after = require_valid_raster(after_path)
    report = compare_grids(before.as_grid(), after.as_grid(), tolerance=tolerance)
    return report.to_dict()


def preprocess_sentinel1_pair(
    manifest: AcquisitionManifest,
    source_paths: Mapping[str, Path | str],
    output_dir: Path | str,
    *,
    config: PreprocessingConfig,
    grid: Optional[AnalysisGrid] = None,
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> Tuple[ProcessedArtifact, ProcessedArtifact]:
    """Prepare a selected Sentinel-1 pair after explicit source validation."""
    _require_import()
    inputs = PreprocessingInputs.from_manifest(manifest, source_paths)
    if inputs.sentinel1 is None:
        raise InputValidationError("M1 manifest contains no selected Sentinel-1 pair")
    if grid is None:
        grid = config.require_grid()
    before, after = inputs.sentinel1
    _require_pair_geometry(before, after, source_metadata)
    settings = config.sentinel1
    if settings.same_track_only and (
        not isinstance(before.relative_orbit, int)
        or not isinstance(after.relative_orbit, int)
        or before.relative_orbit != after.relative_orbit
    ):
        raise AlignmentError(
            "Sentinel-1 same_track_only is enabled, but the selected pair does not "
            "have matching known relative-orbit metadata"
        )
    if settings.invalid_pixel_policy != "mask":
        raise PreprocessingConfigurationError("Sentinel-1 invalid_pixel_policy must be 'mask'")
    if settings.required_polarizations is None:
        raise PreprocessingConfigurationError("sentinel1.required_polarizations is unresolved")
    if settings.radiometric_calibration is None:
        raise PreprocessingConfigurationError("sentinel1.radiometric_calibration is unresolved")
    if settings.terrain_correction is None:
        raise PreprocessingConfigurationError("sentinel1.terrain_correction is unresolved")
    if settings.resampling is None:
        raise PreprocessingConfigurationError("sentinel1.resampling is unresolved")
    if settings.radiometric_calibration == "source_calibrated":
        for source in (before, after):
            _require_true(
                _source_metadata(source, source_metadata), "radiometric_calibrated", source.scene_id
            )
    elif settings.radiometric_calibration != "linear_to_db":
        raise UnsupportedPreprocessingError(
            "Sentinel-1 calibration mode must be explicitly source_calibrated or linear_to_db"
        )
    if settings.terrain_correction == "source_terrain_corrected":
        for source in (before, after):
            _require_true(
                _source_metadata(source, source_metadata), "terrain_corrected", source.scene_id
            )
    elif settings.terrain_correction == "external_tool":
        raise UnsupportedPreprocessingError(
            "terrain correction via an external SAR processor is not implemented by the local M2 engine"
        )
    else:
        raise UnsupportedPreprocessingError(
            "Sentinel-1 terrain_correction must be source_terrain_corrected or external_tool"
        )

    artifacts = []
    for label, source in (("before", before), ("after", after)):
        metadata = require_valid_raster(source.path, required_bands=settings.required_polarizations)
        indexes = _band_indexes(metadata, settings.required_polarizations)
        metadata, data, valid = _read_bands(source, indexes)
        if settings.radiometric_calibration == "linear_to_db":
            _require_true(
                _source_metadata(source, source_metadata), "linear_backscatter", source.scene_id
            )
            positive = data > 0
            valid &= positive.all(axis=0)
            data = np.where(
                positive, 10.0 * np.log10(np.maximum(data, np.finfo("float32").tiny)), 0.0
            )
        nodata = settings.output_nodata if settings.output_nodata is not None else -9999.0
        data, valid, alignment = _reproject_to_grid(
            data,
            valid,
            metadata,
            grid,
            resampling_name=settings.resampling,
            nodata=nodata,
        )
        operations = [
            "validate_selected_m1_source",
            f"validate_polarizations:{','.join(settings.required_polarizations)}",
            f"radiometric_calibration:{settings.radiometric_calibration}",
            f"terrain_correction:{settings.terrain_correction}",
            f"resampling:{settings.resampling}",
            "mask_invalid_pixels",
            "prepare_common_analysis_grid",
        ]
        artifacts.append(
            write_raster_artifact(
                data=data,
                valid_mask=valid,
                grid=grid,
                output_dir=Path(output_dir) / "sentinel1",
                stem=f"{label}_{source.scene_id}",
                manifest=manifest,
                sensor=SensorKind.SENTINEL1,
                source_product_ids=[before.product_id, after.product_id],
                source_scene_ids=[before.scene_id, after.scene_id],
                operations=operations,
                config_version=config.version,
                preprocessing_version=config.pipeline_version,
                band_names=settings.required_polarizations,
                quality_extra={
                    "sensor": SensorKind.SENTINEL1.value,
                    "temporal_role": label,
                    "source_product": source.to_dict(),
                    "alignment": alignment,
                    "source_metadata": dict(_source_metadata(source, source_metadata)),
                },
                nodata=nodata,
            )
        )
    return artifacts[0], artifacts[1]


def preprocess_sentinel2_pair(
    manifest: AcquisitionManifest,
    source_paths: Mapping[str, Path | str],
    output_dir: Path | str,
    *,
    config: PreprocessingConfig,
    grid: Optional[AnalysisGrid] = None,
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> Tuple[ProcessedArtifact, ProcessedArtifact]:
    """Prepare a selected Sentinel-2 pair with explicit quality-mask semantics."""
    _require_import()
    inputs = PreprocessingInputs.from_manifest(manifest, source_paths)
    if inputs.sentinel2 is None:
        raise InputValidationError("M1 manifest contains no selected Sentinel-2 pair")
    if grid is None:
        grid = config.require_grid()
    before, after = inputs.sentinel2
    _require_pair_geometry(before, after, source_metadata)
    settings = config.sentinel2
    if settings.invalid_pixel_policy != "mask":
        raise PreprocessingConfigurationError("Sentinel-2 invalid_pixel_policy must be 'mask'")
    if settings.required_bands is None:
        raise PreprocessingConfigurationError("sentinel2.required_bands is unresolved")
    if settings.resampling is None:
        raise PreprocessingConfigurationError("sentinel2.resampling is unresolved")
    quality_settings = settings.quality_mask
    if quality_settings.enabled is True and (
        quality_settings.band_name is None or quality_settings.valid_values is None
    ):
        raise PreprocessingConfigurationError(
            "enabled Sentinel-2 quality_mask requires band_name and valid_values"
        )
    artifacts = []
    for label, source in (("before", before), ("after", after)):
        metadata = require_valid_raster(source.path)
        indexes = _band_indexes(metadata, settings.required_bands)
        metadata, data, valid = _read_bands(source, indexes)
        operations = [
            "validate_selected_m1_source",
            f"validate_bands:{','.join(settings.required_bands)}",
        ]
        quality_extra: dict[str, Any] = {
            "sensor": SensorKind.SENTINEL2.value,
            "temporal_role": label,
            "source_product": source.to_dict(),
            "scene_cloud_percent": source.scene_cloud_percent,
            "aoi_cloud_percent": {
                "unknown": True,
                "reason": "not_available_from_source",
                "note": "AOI cloud fraction requires a pixel mask clipped to the AOI.",
            },
        }
        if quality_settings.enabled is True:
            all_descriptions = {
                description.strip(): index + 1
                for index, description in enumerate(metadata.descriptions)
                if description
            }
            quality_index = all_descriptions.get(quality_settings.band_name or "")
            if quality_index is None:
                raise InputValidationError(
                    f"quality mask band {quality_settings.band_name!r} is absent from {source.path}"
                )
            with rasterio.open(source.path) as dataset:
                quality_values = dataset.read(quality_index)
            quality_valid = np.isin(quality_values, quality_settings.valid_values)
            valid &= quality_valid
            operations.append(
                f"quality_mask:{quality_settings.band_name} valid_values={list(quality_settings.valid_values)}"
            )
            quality_extra["quality_mask"] = {
                "band_name": quality_settings.band_name,
                "valid_values": list(quality_settings.valid_values),
                "masked_pixel_percent": 100.0 * float((~quality_valid).sum()) / quality_valid.size,
            }
        else:
            operations.append("quality_mask:unavailable_not_inferred")
            quality_extra["quality_mask"] = {"status": "not_available"}
        if settings.reflectance_scale is not None or settings.reflectance_offset is not None:
            if settings.reflectance_scale is None or settings.reflectance_offset is None:
                raise PreprocessingConfigurationError(
                    "Sentinel-2 reflectance scale and offset must be configured together"
                )
            data = data * settings.reflectance_scale + settings.reflectance_offset
            operations.append(
                f"reflectance_scale_offset:{settings.reflectance_scale},{settings.reflectance_offset}"
            )
        nodata = settings.output_nodata if settings.output_nodata is not None else -9999.0
        data, valid, alignment = _reproject_to_grid(
            data,
            valid,
            metadata,
            grid,
            resampling_name=settings.resampling,
            nodata=nodata,
        )
        operations.extend(
            [
                f"resampling:{settings.resampling}",
                "mask_invalid_pixels",
                "prepare_common_analysis_grid",
            ]
        )
        quality_extra["alignment"] = alignment
        artifacts.append(
            write_raster_artifact(
                data=data,
                valid_mask=valid,
                grid=grid,
                output_dir=Path(output_dir) / "sentinel2",
                stem=f"{label}_{source.scene_id}",
                manifest=manifest,
                sensor=SensorKind.SENTINEL2,
                source_product_ids=[before.product_id, after.product_id],
                source_scene_ids=[before.scene_id, after.scene_id],
                operations=operations,
                config_version=config.version,
                preprocessing_version=config.pipeline_version,
                band_names=settings.required_bands,
                quality_extra=quality_extra,
                nodata=nodata,
            )
        )
    return artifacts[0], artifacts[1]


def preprocess_dem(
    manifest: AcquisitionManifest,
    source_path: Path | str,
    output_dir: Path | str,
    *,
    config: PreprocessingConfig,
    source_metadata: Mapping[str, Any],
    grid: Optional[AnalysisGrid] = None,
) -> ProcessedArtifact:
    """Validate and prepare the configured Copernicus WorldDEM-30 input."""
    _require_import()
    if source_metadata.get("product_name") != config.dem.source_product:
        raise InputValidationError(
            f"DEM source metadata must identify {config.dem.source_product!r}; "
            f"received {source_metadata.get('product_name')!r}"
        )
    if source_metadata.get("version") is None:
        raise PreprocessingConfigurationError(
            "DEM version is unavailable; record it before processing"
        )
    if not source_metadata.get("product_id"):
        raise PreprocessingConfigurationError(
            "DEM source metadata must include the exact source product ID"
        )
    if grid is None:
        grid = config.require_grid()
    source = require_valid_raster(source_path, min_bands=1)
    with rasterio.open(source_path) as dataset:
        data = dataset.read(1).astype("float32", copy=False)[None, ...]
    valid = read_valid_mask(source_path, band_indexes=(1,))
    if config.dem.resampling is None:
        raise PreprocessingConfigurationError("dem.resampling is unresolved")
    nodata = config.dem.output_nodata if config.dem.output_nodata is not None else -9999.0
    data, valid, alignment = _reproject_to_grid(
        data,
        valid,
        source,
        grid,
        resampling_name=config.dem.resampling,
        nodata=nodata,
    )
    quality = {
        "sensor": "copernicus-dem",
        "source_metadata": dict(source_metadata),
        "vertical_datum": source_metadata.get("vertical_datum")
        or {"unknown": True, "reason": "not_available_from_source"},
        "alignment": alignment,
    }
    return write_raster_artifact(
        data=data,
        valid_mask=valid,
        grid=grid,
        output_dir=Path(output_dir) / "dem",
        stem="elevation",
        manifest=manifest,
        sensor=None,
        production_input=ProductionInput.COPERNICUS_DEM,
        source_product_ids=[str(source_metadata.get("product_id"))],
        source_scene_ids=[],
        operations=[
            "validate_copernicus_worlddem30_source",
            f"resampling:{config.dem.resampling}",
            "mask_invalid_pixels",
            "prepare_common_analysis_grid",
        ],
        config_version=config.version,
        preprocessing_version=config.pipeline_version,
        band_names=("elevation",),
        dem_version=str(source_metadata["version"]),
        quality_extra=quality,
        nodata=nodata,
    )
