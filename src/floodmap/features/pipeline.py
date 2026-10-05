"""M3 Earth-observation feature generation.

Boundary
--------
```text
M2 analysis-ready products + provenance + QA
        -> M3 feature generation
        -> feature stack + per-feature masks + registry + QA + provenance
        -> M4 segmentation
```

What this module does **not** do, deliberately:

* no flood or debris classification, and no threshold of any kind. A feature is
  evidence for the segmentation model; the decision belongs to M4 and the
  threshold to the evaluation milestone
  (``configs/segmentation.yaml -> inference.probability_threshold``);
* no resampling, reprojection or re-gridding. M2 owns spatial normalisation, so
  a grid disagreement is an explicit failure here;
* no scene discovery, selection or download;
* no normalisation parameter fitting;
* no hydrology, routing, infrastructure or network analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..preprocessing.grid import AnalysisGrid, require_rasterio
from ..utils.provenance import BeforeAfterPair, ProductionInput
from . import numerics
from .artifacts import FeatureSetArtifact, write_feature_set
from .config import FeatureConfig
from .errors import (
    FeatureConfigurationError,
    FeatureInputError,
    UnsupportedFeatureError,
)
from .inputs import FeatureInputs
from .registry import (
    ArtifactRole,
    FeatureDefinition,
    FeatureFamily,
    FeatureRegistry,
    TransformKind,
    build_registry,
)

try:
    from rasterio.crs import CRS
except ImportError:  # pragma: no cover - require_rasterio raises with an explanation
    CRS = None  # type: ignore[assignment]

__all__ = ["generate_feature_set", "build_feature_registry"]


def build_feature_registry(config: FeatureConfig) -> FeatureRegistry:
    """Resolve the configured feature set without touching any raster.

    Exposed separately so a caller — or a configuration-validation step — can
    inspect exactly which features a configuration produces, and what each one
    means, before committing compute to a run.
    """
    return build_registry(
        enabled_template_ids=config.require_enabled_templates(),
        polarisation_roles=config.sentinel1.polarisation_roles or (),
        spectral_band_roles=config.sentinel2.spectral_band_roles or (),
        band_role_bindings=config.resolved_band_roles(),
        backscatter_representation=config.sentinel1.backscatter_representation,
        registry_version=config.registry_version,
        feature_set_version=config.require_feature_set_version(),
        config_version=config.version,
    )


def _require_projected_metre_grid(grid: AnalysisGrid, config: FeatureConfig) -> None:
    """Terrain derivatives need a planar grid whose unit matches the elevation unit.

    The slope kernel divides an elevation difference by the pixel size. On a
    geographic grid the denominator is in degrees, so the gradient is not a
    slope at all — and the resulting raster would look entirely plausible. The
    raster cannot report its elevation unit, so the operator must declare it.
    """
    if config.terrain.elevation_unit is None:
        raise FeatureConfigurationError(
            "terrain features are enabled but terrain.elevation_unit is unresolved. "
            "The slope kernel assumes elevation and grid resolution share a linear "
            "unit, and that cannot be read from the raster."
        )
    if not config.terrain.require_projected_grid:
        return
    require_rasterio()
    crs = CRS.from_user_input(grid.crs)
    if not crs.is_projected:
        raise FeatureConfigurationError(
            f"terrain derivatives require a projected analysis grid; the M2 grid CRS "
            f"is {grid.crs!r}. A gradient computed over degrees is not a slope."
        )
    linear_units = (crs.linear_units or "").lower()
    if linear_units not in {"metre", "meter", "m"}:
        raise FeatureConfigurationError(
            f"terrain derivatives require a grid in metres; CRS {grid.crs!r} reports "
            f"linear units {crs.linear_units!r}, which does not match "
            f"terrain.elevation_unit={config.terrain.elevation_unit!r}"
        )


def _read_feature_inputs(
    feature: FeatureDefinition, inputs: FeatureInputs
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    read: list[Tuple[np.ndarray, np.ndarray]] = []
    for item in feature.inputs:
        if item.band_name is None:  # pragma: no cover - build_registry guarantees this
            raise FeatureConfigurationError(
                f"{feature.name}: band role {item.band_role!r} resolved to no band name"
            )
        read.append(inputs.require(item.artifact).read_band(item.band_name))
    return tuple(read)


def _compute(
    feature: FeatureDefinition,
    inputs: FeatureInputs,
    computed: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    *,
    grid: AnalysisGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch one registry feature to its transform in :mod:`.numerics`.

    The registry decides *what* is computed; this function only routes. Every
    branch returns ``(values, valid)`` and no branch invents a value for an
    invalid pixel.
    """
    transform = feature.transform

    if transform is TransformKind.DERIVED_DIFFERENCE:
        if len(feature.derived_from) != 2:
            raise UnsupportedFeatureError(
                f"{feature.name}: a derived difference needs exactly two source features"
            )
        post_name, pre_name = feature.derived_from
        post_values, post_valid = computed[post_name]
        pre_values, pre_valid = computed[pre_name]
        return numerics.difference(post_values, pre_values, post_valid, pre_valid)

    read = _read_feature_inputs(feature, inputs)

    if transform is TransformKind.PASSTHROUGH:
        if len(read) != 1:
            raise UnsupportedFeatureError(f"{feature.name}: passthrough needs exactly one input")
        values, valid = read[0]
        return values, numerics.finite_and_valid(values, valid)

    if transform is TransformKind.SLOPE_HORN_DEGREES:
        if len(read) != 1:
            raise UnsupportedFeatureError(f"{feature.name}: slope needs exactly one input")
        values, valid = read[0]
        return numerics.slope_horn_degrees(values, valid, resolution_m=grid.resolution_m)

    if len(read) != 2:
        raise UnsupportedFeatureError(
            f"{feature.name}: transform {transform.value!r} needs exactly two inputs"
        )
    (first, first_valid), (second, second_valid) = read

    if transform is TransformKind.DIFFERENCE:
        return numerics.difference(first, second, first_valid, second_valid)
    if transform is TransformKind.LOG_RATIO_DB:
        return numerics.log_ratio_db(first, second, first_valid, second_valid)
    if transform is TransformKind.CROSS_POL_DIFFERENCE_DB:
        return numerics.cross_polarisation_difference_db(first, second, first_valid, second_valid)
    if transform is TransformKind.NORMALIZED_DIFFERENCE:
        return numerics.normalized_difference(first, second, first_valid, second_valid)

    raise UnsupportedFeatureError(  # pragma: no cover - TransformKind is closed
        f"{feature.name}: transform {transform.value!r} has no implementation"
    )


def _range_report(
    feature: FeatureDefinition,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    normalisation_enabled: bool,
) -> Optional[Dict[str, Any]]:
    """Count valid pixels outside the feature's declared range.

    Reported, not corrected. An excursion is information — for reflectance it
    usually means atmospheric correction returned a slightly negative value over
    a dark surface — and clipping it would hide that while changing the recorded
    physical value.

    Evaluated on the physical quantity, before any machine-learning scaling,
    because the declared range describes the physical feature rather than its
    rescaled representation. The flag records that, so a reader cannot mistake
    the count for a statement about the stored raster when normalisation is on.
    """
    if feature.valid_range is None:
        return None
    low, high = feature.valid_range
    outside = valid & ((values < low) | (values > high))
    count = int(outside.sum())
    return {
        "declared_range": [float(low), float(high)],
        "condition": feature.valid_range_condition,
        "valid_pixels_outside_range": count,
        "clipped": False,
        "evaluated_before_normalisation": bool(normalisation_enabled),
    }


def _load_normalisation_statistics(config: FeatureConfig) -> Dict[str, Dict[str, float]]:
    path = config.normalisation.require_statistics_path()
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FeatureConfigurationError(
            f"could not parse normalisation statistics at {path}: {exc}"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise FeatureConfigurationError(
            f"normalisation statistics at {path} must be a mapping of feature name -> parameters"
        )
    return {str(key): {str(k): float(v) for k, v in value.items()} for key, value in parsed.items()}


def _apply_normalisation(
    feature: FeatureDefinition,
    values: np.ndarray,
    valid: np.ndarray,
    statistics: Mapping[str, Mapping[str, float]],
    method: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply pre-fitted scaling to one feature.

    Parameters are looked up by feature name and must be present: silently
    leaving one feature unscaled while its neighbours are scaled would produce a
    stack whose channels are not mutually comparable.
    """
    parameters = statistics.get(feature.name)
    if parameters is None:
        raise FeatureConfigurationError(
            f"normalisation is enabled but the statistics file has no entry for "
            f"{feature.name!r}; a partially normalised feature stack is not usable"
        )
    if method == "zscore":
        mean = parameters.get("mean")
        std = parameters.get("std")
        if mean is None or std is None:
            raise FeatureConfigurationError(
                f"{feature.name}: zscore normalisation requires 'mean' and 'std'"
            )
        if std <= 0:
            raise FeatureConfigurationError(
                f"{feature.name}: zscore normalisation requires a positive 'std'"
            )
        scaled = (values - mean) / std
        applied = {"method": method, "mean": float(mean), "std": float(std)}
    else:
        minimum = parameters.get("min")
        maximum = parameters.get("max")
        if minimum is None or maximum is None:
            raise FeatureConfigurationError(
                f"{feature.name}: minmax normalisation requires 'min' and 'max'"
            )
        if maximum <= minimum:
            raise FeatureConfigurationError(
                f"{feature.name}: minmax normalisation requires max > min"
            )
        scaled = (values - minimum) / (maximum - minimum)
        applied = {"method": method, "min": float(minimum), "max": float(maximum)}
    valid = valid & np.isfinite(scaled)
    return scaled, valid, applied


def _before_after_pairs(
    inputs: FeatureInputs,
) -> Tuple[Optional[BeforeAfterPair], Optional[BeforeAfterPair], Optional[str]]:
    """Carry the M2 scene-pair records forward so the chain to scenes survives."""
    sentinel1 = None
    sentinel2 = None
    dem_version = None
    for role in (ArtifactRole.SENTINEL1_BEFORE, ArtifactRole.SENTINEL1_AFTER):
        candidate = inputs.inputs.get(role)
        if candidate is not None and candidate.provenance.sentinel1 is not None:
            sentinel1 = candidate.provenance.sentinel1
            break
    for role in (ArtifactRole.SENTINEL2_BEFORE, ArtifactRole.SENTINEL2_AFTER):
        candidate = inputs.inputs.get(role)
        if candidate is not None and candidate.provenance.sentinel2 is not None:
            sentinel2 = candidate.provenance.sentinel2
            break
    dem = inputs.inputs.get(ArtifactRole.DEM)
    if dem is not None and isinstance(dem.provenance.dem_version, str):
        dem_version = dem.provenance.dem_version
    return sentinel1, sentinel2, dem_version


def generate_feature_set(
    artifact_paths: Mapping[ArtifactRole, Path | str],
    output_dir: Path | str,
    *,
    config: FeatureConfig,
    feature_set_id: str,
) -> FeatureSetArtifact:
    """Generate one feature set from M2 analysis-ready artifacts.

    Order of operations is deliberate: configuration is fully resolved, then the
    registry is built, then inputs are loaded and grid-checked, and only then is
    any pixel read. An unresolved scientific parameter therefore fails before
    any compute or IO, and before a partial artifact can appear on disk.
    """
    require_rasterio()
    if not feature_set_id or not feature_set_id.strip():
        raise FeatureConfigurationError("feature_set_id is required to identify the artifact")

    config.numerics.require_clip_policy()
    nodata = config.numerics.require_nodata()
    allowed_sources = config.require_allowed_sources()
    registry = build_feature_registry(config)

    inputs = FeatureInputs.load(artifact_paths, allowed_sources=allowed_sources)
    inputs.require_all(registry.required_artifacts())
    alignment = inputs.require_common_grid(tolerance=config.grid.transform_tolerance)
    grid = inputs.reference_grid()

    families = {feature.family for feature in registry.features}
    if FeatureFamily.TERRAIN in families:
        _require_projected_metre_grid(grid, config)

    declared_sources = {feature.source for feature in registry.features}
    unavailable = [
        source.value for source in declared_sources if source not in set(allowed_sources)
    ]
    if unavailable:  # pragma: no cover - require_allowed_sources precedes this
        raise FeatureConfigurationError(
            f"enabled features require sources not in allowed_sources: {unavailable}"
        )

    statistics: Mapping[str, Mapping[str, float]] = {}
    if config.normalisation.enabled:
        statistics = _load_normalisation_statistics(config)

    computed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    per_feature_quality: Dict[str, Any] = {}
    normalisation_applied: Dict[str, Any] = {}

    for feature in registry.features:
        values, valid = _compute(feature, inputs, computed, grid=grid)
        if values.shape != (grid.height, grid.width):
            raise FeatureInputError(
                f"{feature.name}: computed shape {values.shape} does not match the "
                f"analysis grid {(grid.height, grid.width)}"
            )
        range_report = (
            _range_report(
                feature,
                values,
                valid,
                normalisation_enabled=config.normalisation.enabled,
            )
            if config.numerics.report_out_of_declared_range
            else None
        )
        if config.normalisation.enabled:
            values, valid, applied = _apply_normalisation(
                feature, values, valid, statistics, str(config.normalisation.method)
            )
            normalisation_applied[feature.name] = applied
        values = numerics.apply_nodata(values, valid, nodata)
        computed[feature.name] = (values, valid)

        total = int(valid.size)
        valid_count = int(valid.sum())
        per_feature_quality[feature.name] = {
            "family": feature.family.value,
            "transform": feature.transform.value,
            "units": feature.units,
            "feature_version": feature.feature_version,
            "inputs": [item.to_dict() for item in feature.inputs],
            "derived_from": list(feature.derived_from),
            "valid_pixel_count": valid_count,
            "invalid_pixel_count": total - valid_count,
            "invalid_pixel_percent": (100.0 * (total - valid_count) / total if total else None),
            "declared_range_report": range_report,
        }

    names = registry.names()
    dtype = config.numerics.output_dtype
    data = np.stack([computed[name][0] for name in names]).astype(dtype, copy=False)
    masks = np.stack([computed[name][1] for name in names]).astype("uint8", copy=False)

    all_valid = masks.all(axis=0)
    any_valid = masks.any(axis=0)
    total_pixels = int(all_valid.size)

    operations = [
        "load_m2_analysis_ready_artifacts",
        "verify_common_analysis_grid_no_resampling",
        f"resolve_feature_registry:{registry.feature_set_version}",
        f"backscatter_representation:{(config.sentinel1.backscatter_representation.value if config.sentinel1.backscatter_representation else 'not_applicable')}",
        "compute_features",
        "propagate_input_validity_masks",
        (
            f"normalisation:{config.normalisation.method} "
            f"(statistics_source={config.normalisation.statistics_source}, "
            f"version={config.normalisation.version})"
            if config.normalisation.enabled
            else "normalisation:not_applied_features_are_physical_quantities"
        ),
        "write_feature_stack_per_feature_masks_registry_and_provenance",
    ]

    quality: Dict[str, Any] = {
        "status": "success",
        "feature_set_id": feature_set_id,
        "feature_pipeline_version": config.pipeline_version,
        "feature_set_version": registry.feature_set_version,
        "registry_version": registry.registry_version,
        "config_version": config.version,
        "feature_count": len(names),
        "feature_names": list(names),
        "grid": grid.to_dict(),
        "alignment": alignment,
        "output_dtype": dtype,
        "output_nodata": nodata,
        "clipping_applied": bool(config.numerics.clip_to_valid_range),
        "mask_semantics": {
            "per_feature": "1=valid feature value, 0=not computable from valid inputs",
            "note": (
                "An invalid feature pixel means the feature could not be computed "
                "from valid observations. It is not evidence of absence."
            ),
        },
        "validity_summary": {
            "pixels": total_pixels,
            "valid_in_every_feature": int(all_valid.sum()),
            "valid_in_at_least_one_feature": int(any_valid.sum()),
            "valid_in_every_feature_percent": (
                100.0 * float(all_valid.sum()) / total_pixels if total_pixels else None
            ),
        },
        "features": per_feature_quality,
        "normalisation": (
            {
                "applied": True,
                "method": config.normalisation.method,
                "statistics_source": config.normalisation.statistics_source,
                "statistics_path": config.normalisation.statistics_path,
                "version": config.normalisation.version,
                "fitted_in_m3": False,
                "parameters": normalisation_applied,
            }
            if config.normalisation.enabled
            else {
                "applied": False,
                "reason": (
                    "M3 emits scientifically interpretable physical quantities. "
                    "Machine-learning scaling is a modelling decision and must use "
                    "training-split statistics supplied explicitly."
                ),
                "fitted_in_m3": False,
            }
        ),
        "inputs": inputs.to_dict(),
        "source_artifact_ids": list(inputs.source_product_ids()),
        "acquisition_manifest_ids": list(inputs.acquisition_manifest_ids()),
        "classification_performed": False,
    }

    sentinel1_pair, sentinel2_pair, dem_version = _before_after_pairs(inputs)
    # Deterministic choice of the artifact whose AOI/event scope is carried
    # onto the feature record: the first required artifact in ArtifactRole
    # declaration order. build_registry guarantees at least one exists, because
    # a purely derived feature set is rejected.
    required = registry.required_artifacts()
    if not required:  # pragma: no cover - build_registry rejects this
        raise FeatureConfigurationError(
            "the resolved feature set reads no M2 artifact; it cannot be generated"
        )
    reference = inputs.require(required[0])

    return write_feature_set(
        feature_set_id=feature_set_id,
        data=data,
        valid_masks=masks,
        grid=grid,
        registry=registry,
        output_dir=output_dir,
        quality=quality,
        acquisition_manifest_ids=inputs.acquisition_manifest_ids(),
        source_product_ids=inputs.source_product_ids(),
        production_inputs=inputs.production_inputs(),
        operations=operations,
        pipeline_version=config.pipeline_version,
        config_version=config.version,
        nodata=nodata,
        sentinel1_pair=sentinel1_pair,
        sentinel2_pair=sentinel2_pair,
        dem_version=dem_version,
        aoi=reference.provenance.aoi,
        aoi_crs=(
            reference.provenance.aoi_crs if isinstance(reference.provenance.aoi_crs, str) else None
        ),
        event_date=(
            reference.provenance.event_date
            if isinstance(reference.provenance.event_date, str)
            else None
        ),
    )
