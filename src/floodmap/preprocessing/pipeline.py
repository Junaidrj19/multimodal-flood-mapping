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
    require_rasterio,
)
from .inputs import ManifestSourceProduct, PreprocessingInputs
from .reprojection import (
    linear_to_db_after_resampling,
    reproject_categorical,
    reproject_continuous,
)
from .s2_optical import (
    SCL_CLASSES,
    OpticalValidationReport,
    aoi_mask_for_grid,
    aoi_statistics,
    band_availability,
    classify_scl,
    observation_mask,
)
from .validation import (
    RasterMetadata,
    read_valid_mask,
    require_valid_raster,
    validate_required_bands,
)

try:
    import rasterio
except ImportError as exc:  # pragma: no cover
    rasterio = None  # type: ignore[assignment]
    _RASTERIO_ERROR = exc
else:
    _RASTERIO_ERROR = None

__all__ = [
    "preprocess_dem",
    "preprocess_sentinel1_pair",
    "preprocess_sentinel2_pair",
    "validate_preprocessing_pair_alignment",
]

#: Target-pixel value used by the shared categorical warp for "no observation
#: reached this pixel". Distinct from every SCL class code, which are 0-11.
_SCL_IGNORE_INDEX = -1


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
    domain: str = "unitless",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Warp continuous bands onto the analysis grid.

    Delegates to :func:`floodmap.preprocessing.reprojection.reproject_continuous`
    so M2 production preprocessing and M5 training use the SAME spatial
    transformation code. The previous local implementation masked invalid
    pixels only *after* warping, which cleaned pixels that were themselves
    invalid but left valid pixels that had absorbed an invalid neighbour
    through the bilinear kernel.
    """
    _require_import()
    source_grid = source.as_grid()
    if compare_grids(source_grid, target).aligned:
        # Already on the target grid: masking is the whole operation, and
        # warping would be a no-op that still costs an interpolation pass.
        result = data.copy()
        result[:, ~valid] = nodata
        return (
            result,
            valid.copy(),
            {
                "source_to_target": compare_grids(source_grid, target).to_dict(),
                "reprojected": False,
            },
        )
    outcome = reproject_continuous(
        data,
        valid,
        source=source_grid,
        target=target,
        resampling_name=resampling_name,
        nodata=nodata,
        domain=domain,
    )
    provenance = dict(outcome.provenance)
    # Preserved key name so existing artifact consumers keep working.
    provenance["source_to_target"] = provenance.pop("source_to_target_alignment")
    return outcome.data, outcome.valid, provenance


def _aoi_geojson_on_grid(aoi: Any, grid: AnalysisGrid, geometry: Mapping[str, Any]) -> tuple[dict, bool]:
    """Express the AOI as a GeoJSON Polygon in the analysis grid's CRS.

    Returns the geometry and whether a reprojection was applied. The grid CRS in
    production is projected while the AOI is EPSG:4326, so the reprojection is the
    normal case; it is recorded in the report rather than assumed, because a mask
    built from coordinates in the wrong CRS silently selects the wrong pixels
    (``AGENTS.md`` §7).
    """
    if aoi is not None and hasattr(aoi, "geometry"):
        geometry = aoi.geometry
        crs = getattr(aoi, "crs", None)
    else:
        crs = geometry.get("crs") if isinstance(geometry, Mapping) else None

    exterior = getattr(geometry, "exterior", None)
    holes = getattr(geometry, "holes", ())
    if exterior is None and hasattr(geometry, "to_ring"):
        exterior, holes = geometry.to_ring(), ()
    if exterior is None:
        exterior = geometry["exterior"]
        holes = geometry.get("holes", ())
    ring = [[float(x), float(y)] for x, y in exterior]
    polygons = [ring] + [[[float(x), float(y)] for x, y in hole] for hole in holes]

    reprojected = False
    source_crs = crs or "EPSG:4326"
    if str(source_crs) != str(grid.crs):
        from pyproj import Transformer

        transformer = Transformer.from_crs(source_crs, grid.crs, always_xy=True)
        rings = []
        for polygon in polygons:
            xs, ys = zip(*polygon)
            rings.append(
                [
                    [float(lon), float(lat)]
                    for lon, lat in zip(*transformer.transform(list(xs), list(ys)))
                ]
            )
        polygons = rings
        reprojected = True
    return {"type": "Polygon", "coordinates": polygons}, reprojected


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
        nodata = settings.output_nodata if settings.output_nodata is not None else -9999.0
        convert_to_db = settings.radiometric_calibration == "linear_to_db"
        if convert_to_db:
            _require_true(
                _source_metadata(source, source_metadata), "linear_backscatter", source.scene_id
            )
            # Only the domain is checked here. The conversion itself happens
            # AFTER reprojection: see below.
            positive = data > 0
            valid &= positive.all(axis=0)

        # C3 ORDERING, corrected 2026-10-09. SAR is resampled in LINEAR POWER
        # and converted to decibels afterwards. The previous order converted
        # first, so bilinear resampling averaged logarithms -- which is not the
        # mean of the underlying power and is biased low. The frozen contract
        # has always required this order
        # (configs/preprocessing.yaml -> target_grid.resampling:
        # sar_domain: linear_power, sar_resample_before_db_conversion: true,
        # pinned by tests/test_contract_freeze.py), so the implementation was
        # contradicting its own declared policy.
        data, valid, alignment = _reproject_to_grid(
            data,
            valid,
            metadata,
            grid,
            resampling_name=settings.resampling,
            nodata=nodata,
            domain="linear_power" if convert_to_db else "unitless",
        )

        if convert_to_db:
            converted = np.empty_like(data)
            db_valid = valid.copy()
            for index in range(data.shape[0]):
                band, band_valid = linear_to_db_after_resampling(data[index], valid)
                converted[index] = band
                db_valid &= band_valid
            data = np.where(db_valid, converted, nodata).astype("float32", copy=False)
            valid = db_valid

        operations = [
            "validate_selected_m1_source",
            f"validate_polarizations:{','.join(settings.required_polarizations)}",
            f"terrain_correction:{settings.terrain_correction}",
            f"resampling:{settings.resampling}",
            "mask_invalid_pixels",
            "prepare_common_analysis_grid",
        ]
        # Ordering is recorded in the artifact, not just in a comment, so a
        # consumer can verify which domain the resampling happened in.
        if convert_to_db:
            operations.insert(2, "resample_in_linear_power")
            operations.insert(3, f"radiometric_calibration:{settings.radiometric_calibration}")
            operations.insert(4, "linear_to_db_after_resampling")
        else:
            operations.insert(2, f"radiometric_calibration:{settings.radiometric_calibration}")
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
    if quality_settings.enabled is True and quality_settings.band_name is None:
        raise PreprocessingConfigurationError(
            "enabled Sentinel-2 quality_mask requires band_name"
        )
    if (
        quality_settings.enabled is True
        and settings.cloud_mask_source != "scl"
        and quality_settings.valid_values is None
    ):
        # The SCL path is class-based, so a valid_values list would be
        # meaningless there. Any other source is values-based and still needs it.
        raise PreprocessingConfigurationError(
            "enabled Sentinel-2 quality_mask requires band_name and valid_values "
            "unless cloud_mask_source is 'scl'"
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
        if quality_settings.enabled is True and settings.cloud_mask_source != "scl":
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
        optical_report: Optional[OpticalValidationReport] = None
        if settings.cloud_mask_source == "scl":
            # The product's own Scene Classification Layer: the documented way to
            # know *why* a pixel is unusable (cloud, cirrus, shadow, no-data)
            # rather than only that it is. Masking here is by class, so a cloudy
            # pixel and a pixel outside the granule stay distinguishable.
            scl_name = quality_settings.band_name or "SCL"
            scl_index = _band_indexes(metadata, (scl_name,))[0]
            with rasterio.open(source.path) as dataset:
                raw_scl = dataset.read(scl_index)
            scl_codes = np.rint(np.asarray(raw_scl, dtype="float64"))
            if not np.allclose(scl_codes, np.asarray(raw_scl, dtype="float64")):
                raise InputValidationError(
                    f"band {scl_name!r} of {source.path} holds non-integral values, so it is "
                    "not the categorical SCL band; refusing to interpret it as classes"
                )
            scl_source = scl_codes.astype("int32")
            scl_valid = read_valid_mask(source.path, band_indexes=(scl_index,))
            scl_summary = classify_scl(scl_source)
            scl_observable = observation_mask(scl_source)
            valid &= scl_observable
            operations.append(
                f"scl_mask:{scl_name} classes={len(SCL_CLASSES)} "
                f"excluded={int((~scl_observable).sum())}"
            )
            quality_extra["quality_mask"] = {
                "band_name": scl_name,
                "source": "scl",
                "masked_pixel_percent": 100.0
                * float((~scl_observable).sum())
                / float(scl_observable.size),
                "categories": scl_summary.to_dict(),
            }
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
            domain="reflectance",
        )
        operations.extend(
            [
                f"resampling:{settings.resampling}",
                "mask_invalid_pixels",
                "prepare_common_analysis_grid",
            ]
        )
        if settings.cloud_mask_source == "scl":
            # Statistics are computed on the ANALYSIS GRID over the AOI, after the
            # warp, so "valid pixel fraction" describes the pixels this project
            # will actually use, not the granule's own coverage.
            # The SCL is warped by the SHARED categorical reprojection, not by a
            # local warp: M2 and M5 must not diverge in how a class raster meets
            # the canonical grid, and that path already refuses averaging.
            scl_grid = reproject_categorical(
                scl_source,
                scl_valid,
                source=metadata.as_grid(),
                target=grid,
                ignore_index=_SCL_IGNORE_INDEX,
                resampling_name=config.require_resampling("categorical_label"),
                semantic_values=tuple(sorted(SCL_CLASSES)),
            )
            # `ignore_index` means "no observation reached this target pixel".
            # That is the same statement as SCL class 0 (no data), so it is
            # recorded as missing data rather than as an unknown class.
            scl_on_grid = np.where(scl_grid.data == _SCL_IGNORE_INDEX, 0, scl_grid.data)
            aoi_geometry: Any = None
            aoi_source = getattr(manifest, "aoi", None)
            aoi_geometry = getattr(aoi_source, "geometry", None)
            if aoi_geometry is None and isinstance(aoi_source, Mapping):
                aoi_geometry = aoi_source.get("geometry")
            if aoi_geometry is not None:
                geojson, reprojected = _aoi_geojson_on_grid(aoi_source, grid, aoi_geometry)
                aoi_mask = aoi_mask_for_grid(geojson, grid)
                # `valid` was rebound to the warped mask by _reproject_to_grid
                # above, so the statistic and the stored artifact mask are the
                # same array rather than two computations that could drift.
                statistics = aoi_statistics(
                    scl_on_grid, aoi_mask, raster_valid=valid, scl_summary=scl_summary
                )
                bands = band_availability(
                    tuple(d for d in metadata.descriptions if d), settings.required_bands
                )
                optical_report = OpticalValidationReport(
                    scene_id=source.scene_id,
                    temporal_role=label,
                    scl_summary=scl_summary,
                    statistics=statistics,
                    bands=bands,
                    reflectance_scaling={
                        "scale": settings.reflectance_scale,
                        "offset": settings.reflectance_offset,
                        "source": "configuration",
                    },
                    resampling={
                        "reflectance": settings.resampling,
                        "scl": config.require_resampling("categorical_label"),
                        "scl_method_source": "target_grid.resampling.categorical_label",
                        "scl_reprojection_provenance": dict(scl_grid.provenance),
                    },
                    masking={
                        "scl_band": quality_settings.band_name or "SCL",
                        "categories": scl_summary.to_dict(),
                        "invalid_pixel_policy": settings.invalid_pixel_policy,
                        "nodata_value": settings.nodata_value,
                    },
                    limitations=(
                        "Valid-pixel and cloud fractions are computed over the AOI on the "
                        "analysis grid; they are availability statistics, not a flood "
                        "observation, and say nothing about whether water is present.",
                        "Snow/ice is reported separately and excluded from valid "
                        "observations; in this project's monsoon-season AOI a large snow "
                        "count would indicate an acquisition-date or AOI problem.",
                        "SCL is the product's own classification. Its accuracy is not "
                        "independently verified here.",
                    ),
                )
                report_path = optical_report.write(
                    Path(output_dir)
                    / "sentinel2"
                    / f"{label}_{source.scene_id}.optical-validation.json"
                )
                quality_extra["optical_validation"] = {
                    **optical_report.to_dict(),
                    "report_path": str(report_path),
                }
                quality_extra["aoi_cloud_percent"] = {
                    "unknown": False,
                    "source": "scl",
                    "cloud_fraction": statistics.cloud_fraction,
                    "cloud_shadow_fraction": statistics.shadow_fraction,
                    "valid_fraction": statistics.valid_fraction,
                }

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
        domain="elevation",
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
