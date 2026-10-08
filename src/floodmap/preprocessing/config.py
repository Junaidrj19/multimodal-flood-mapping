"""Typed, explicit configuration for M2 preprocessing.

Every model here uses ``extra="forbid"``.

That is a correction, not a style preference. Until 2026-10-08 five of these
models used ``extra="allow"`` and several real configuration blocks were not
modelled at all, so they were parsed into ``__pydantic_extra__`` and silently
discarded. Measured against the shipped ``configs/preprocessing.yaml``:

* ``sentinel1.steps`` — including ``speckle_filter`` — was absorbed. Setting a
  speckle policy was a **no-op**, and ``speckle_fliter`` was accepted without
  complaint.
* ``target_grid.resampling`` — the entire resampling policy, which
  ``AGENTS.md`` §7 requires to be explicit — was absorbed.
* ``sentinel1.nodata_value``, ``sentinel2.steps``, ``sentinel2.cloud_mask_source``,
  ``dem.steps``, ``osm`` and ``qa`` were all absorbed.
* A mistyped top-level section such as ``sentinal1:`` was accepted, and the
  real ``sentinel1`` then fell back to all-``None`` defaults — a configuration
  that looks populated and behaves as if it were empty.

A configuration that silently ignores a scientific parameter is worse than one
that has none, because it reports success. The models below therefore describe
the whole file, and an unknown key raises
:class:`~floodmap.preprocessing.errors.PreprocessingConfigurationError`.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.config import load_config
from .errors import PreprocessingConfigurationError
from .grid import AnalysisGrid

__all__ = [
    "PreprocessingConfig",
    "SpeckleConfig",
    "TargetGridConfig",
    "TargetGridResamplingConfig",
    "load_preprocessing_config",
]

#: Raster kinds that carry their own resampling method (contract decision C3).
#:
#: Keyed by kind rather than by a single ``continuous``/``categorical`` pair,
#: and with no generic fallback, because the kinds fail differently: bilinear
#: on a class label invents a class that was never in the source, and nearest
#: on continuous backscatter discards the averaging that makes a coarsened
#: pixel physically meaningful.
RESAMPLING_KINDS: Tuple[str, ...] = (
    "continuous_sar",
    "continuous_optical",
    "categorical_label",
    "binary_validity_mask",
    "dem",
)


class TargetGridResamplingConfig(BaseModel):
    """Per-raster-kind resampling methods. Contract decision C3."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuous_sar: Optional[str] = None
    continuous_optical: Optional[str] = None
    categorical_label: Optional[str] = None
    binary_validity_mask: Optional[str] = None
    dem: Optional[str] = None
    # SAR must be resampled in linear power, before any dB conversion:
    # averaging decibels averages logarithms, which is not the mean of the
    # underlying power and is biased low.
    sar_domain: Optional[str] = None
    sar_resample_before_db_conversion: Optional[bool] = None
    forbid_generic_method: bool = True
    document_reason: bool = True

    def require(self, kind: str) -> str:
        """Resolve the method for one raster kind, or fail explicitly.

        There is deliberately no fallback to a generic method. A fallback is
        what lets a newly added raster kind be resampled by whatever happened
        to be configured for something else.
        """
        if kind not in RESAMPLING_KINDS:
            raise PreprocessingConfigurationError(
                f"unknown resampling kind {kind!r}; choose one of {', '.join(RESAMPLING_KINDS)}"
            )
        method = getattr(self, kind)
        if method is None:
            raise PreprocessingConfigurationError(
                f"target_grid.resampling.{kind} is unresolved; no resampling default is used"
            )
        return str(method)


class TargetGridConfig(BaseModel):
    """The canonical analysis grid. Contract decisions C1 and C2.

    ``crs`` stays ``None`` while ``crs_rule`` is set, and the two are not in
    conflict: C1 freezes a *rule* (the UTM zone of the sample's own centroid)
    rather than one global EPSG code, because the training corpus spans roughly
    -18 to +60 degrees latitude and no single projected CRS holds metre
    fidelity across that. The concrete code is derived per run from an AOI that
    ``configs/data.yaml`` deliberately leaves undefined.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    crs: Optional[str] = None
    crs_family: Optional[str] = None
    crs_units: Optional[str] = None
    crs_rule: Optional[str] = None
    crs_is_target_not_source: Optional[bool] = None
    forbid_geographic_crs: Optional[bool] = None
    resolution_m: Optional[float] = None
    resolution_is_native_for_any_source: Optional[bool] = None
    transform: Optional[Tuple[float, ...]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    resampling: TargetGridResamplingConfig = Field(default_factory=TargetGridResamplingConfig)

    @field_validator("transform", mode="before")
    @classmethod
    def _transform_tuple(cls, value: Any) -> Any:
        return tuple(float(item) for item in value) if value is not None else None

    def require_grid(self) -> AnalysisGrid:
        return AnalysisGrid.from_mapping(self.model_dump(mode="python"))


class AlignmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verify_crs: bool = True
    verify_transform: bool = True
    verify_resolution: bool = True
    verify_extent: bool = True
    verify_pixel_alignment: bool = True
    verify_acquisition_geometry: bool = True
    transform_tolerance: Optional[float] = None
    on_failure: str = "fail"

    @field_validator("on_failure")
    @classmethod
    def _failure_policy(cls, value: str) -> str:
        if value not in {"fail", "warn"}:
            raise ValueError("alignment_checks.on_failure must be 'fail' or 'warn'")
        return value


class Sentinel1StepsConfig(BaseModel):
    """The Sentinel-1 correction chain.

    Previously unmodelled and therefore discarded. ``speckle_filter`` lived
    here, which is why a speckle decision could be written down and have no
    effect whatsoever.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    apply_orbit_file: Optional[bool] = None
    thermal_noise_removal: Optional[bool] = None
    radiometric_calibration: Optional[bool] = None
    speckle_filter: Optional[bool] = None
    speckle_filter_params: Optional[Mapping[str, Any]] = None
    terrain_correction: Optional[bool] = None
    convert_to_db: Optional[bool] = None


class SpeckleConfig(BaseModel):
    """Speckle-filter policy. Contract decision C4.

    The frozen M4 baseline applies no filter on any path, and parity with Kuro
    Siwo is **not** claimed: that corpus arrives already Lee Sigma filtered and
    cannot be un-filtered, so no setting here makes the training and inference
    paths identical in texture statistics. The mismatch is disclosed and must
    be quantified by ablation rather than engineered away.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    applied_by_our_pipeline: Optional[bool] = None
    filter_family: Optional[str] = None
    window: Optional[int] = None
    target_window: Optional[int] = None
    sigma: Optional[float] = None
    pipeline_site: Optional[str] = None
    parity_established: Optional[bool] = None
    parity_possible: Optional[bool] = None
    source_state_kuro_siwo: Optional[str] = None
    source_state_sen1floods11: Optional[str] = None
    source_state_production: Optional[str] = None
    mismatch_disclosed: Optional[bool] = None
    ablation_required_before_transfer_claim: Optional[bool] = None

    @model_validator(mode="after")
    def _enabled_policy_must_be_specified(self) -> "SpeckleConfig":
        """A filter that is on must say which filter and where.

        ``applied_by_our_pipeline: true`` with a null family or site would
        describe a filter nobody can reproduce or audit, which is the failure
        this block exists to prevent.
        """
        if self.applied_by_our_pipeline:
            missing = [
                name for name in ("filter_family", "pipeline_site") if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "speckle.applied_by_our_pipeline is true but "
                    f"{', '.join(missing)} is unresolved; an enabled filter must "
                    "name its family and its site in the pipeline"
                )
        return self


class Sentinel1ClipConfig(BaseModel):
    """Upper clip on the production Sentinel-1 path. Contract decision C10.

    The authoritative site is the training-dataset adapter; this block exists
    so the production path can apply the *same* bound at the equivalent point.
    The clip is specified in the linear domain because ``0.15`` is the exact
    published bound and ``-8.2391 dB`` is a rounded transform of it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Optional[bool] = None
    domain: Optional[str] = None
    max_linear: Optional[float] = None
    site: Optional[str] = None
    must_match_training_adapter: Optional[bool] = None


class Sentinel1Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: Sentinel1StepsConfig = Field(default_factory=Sentinel1StepsConfig)
    speckle: SpeckleConfig = Field(default_factory=SpeckleConfig)
    clip: Sentinel1ClipConfig = Field(default_factory=Sentinel1ClipConfig)
    nodata_value: Optional[float] = None
    required_polarizations: Optional[Tuple[str, ...]] = None
    radiometric_calibration: Optional[str] = None
    terrain_correction: Optional[str] = None
    resampling: Optional[str] = None
    invalid_pixel_policy: str = "mask"
    output_nodata: Optional[float] = None
    same_track_only: bool = True

    @field_validator("required_polarizations", mode="before")
    @classmethod
    def _polarization_tuple(cls, value: Any) -> Any:
        return tuple(str(item).upper() for item in value) if value is not None else None

    @model_validator(mode="after")
    def _speckle_declarations_must_agree(self) -> "Sentinel1Config":
        """``steps.speckle_filter`` and ``speckle`` must not contradict.

        Two places can express the same decision, so they can disagree. The
        dangerous direction is a policy block that says "no filter" beside a
        step flag that says "filter", because the step flag is what a pipeline
        would act on. Fail rather than pick a winner.
        """
        step_flag = self.steps.speckle_filter
        policy_flag = self.speckle.applied_by_our_pipeline
        if step_flag is not None and policy_flag is not None and step_flag != policy_flag:
            raise ValueError(
                "sentinel1.steps.speckle_filter "
                f"({step_flag}) contradicts sentinel1.speckle.applied_by_our_pipeline "
                f"({policy_flag}); the speckle decision must be stated once and consistently"
            )
        return self


class QualityMaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Optional[bool] = None
    band_name: Optional[str] = None
    valid_values: Optional[Tuple[int, ...]] = None

    @field_validator("valid_values", mode="before")
    @classmethod
    def _valid_values_tuple(cls, value: Any) -> Any:
        return tuple(int(item) for item in value) if value is not None else None


class Sentinel2StepsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    harmonise_bands: Optional[bool] = None
    cloud_masking: Optional[bool] = None
    shadow_masking: Optional[bool] = None
    scale_reflectance: Optional[bool] = None


class Sentinel2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: Sentinel2StepsConfig = Field(default_factory=Sentinel2StepsConfig)
    cloud_mask_source: Optional[str] = None
    nodata_value: Optional[float] = None
    required_bands: Optional[Tuple[str, ...]] = None
    quality_mask: QualityMaskConfig = Field(default_factory=QualityMaskConfig)
    reflectance_scale: Optional[float] = None
    reflectance_offset: Optional[float] = None
    resampling: Optional[str] = None
    invalid_pixel_policy: str = "mask"
    output_nodata: Optional[float] = None

    @field_validator("required_bands", mode="before")
    @classmethod
    def _bands_tuple(cls, value: Any) -> Any:
        return tuple(str(item) for item in value) if value is not None else None


class DemStepsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reproject_to_target_grid: Optional[bool] = None
    fill_voids: Optional[bool] = None
    void_fill_method: Optional[str] = None


class DemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_product: str = "Copernicus WorldDEM-30"
    version: Optional[str] = None
    vertical_datum: Optional[str] = None
    resampling: Optional[str] = None
    output_nodata: Optional[float] = None
    steps: DemStepsConfig = Field(default_factory=DemStepsConfig)
    derivatives: Tuple[str, ...] = ()

    @field_validator("derivatives", mode="before")
    @classmethod
    def _derivatives_tuple(cls, value: Any) -> Any:
        return tuple(str(item) for item in value) if value is not None else ()


class OsmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reproject_to_target_grid_crs: bool = True
    fix_invalid_geometries: Optional[bool] = None
    road_buffer_m: Optional[float] = None
    building_buffer_m: Optional[float] = None


class QaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_provenance: bool = True
    provenance_schema: Optional[str] = None
    preserve_geospatial_metadata: bool = True
    preserve_acquisition_timestamps: bool = True


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "0.1.0"
    pipeline_version: str = "m2-raster-0.1.0"
    target_grid: TargetGridConfig = Field(default_factory=TargetGridConfig)
    alignment_checks: AlignmentConfig = Field(default_factory=AlignmentConfig)
    sentinel1: Sentinel1Config = Field(default_factory=Sentinel1Config)
    sentinel2: Sentinel2Config = Field(default_factory=Sentinel2Config)
    dem: DemConfig = Field(default_factory=DemConfig)
    osm: OsmConfig = Field(default_factory=OsmConfig)
    qa: QaConfig = Field(default_factory=QaConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PreprocessingConfig":
        try:
            return cls.model_validate(mapping)
        except Exception as exc:
            raise PreprocessingConfigurationError(
                f"invalid preprocessing configuration: {exc}"
            ) from exc

    def require_grid(self) -> AnalysisGrid:
        return self.target_grid.require_grid()

    def require_resampling(self, kind: str) -> str:
        """Resolve the target-grid resampling method for one raster kind (C3)."""
        return self.target_grid.resampling.require(kind)

    def require_s1_resampling(self) -> Any:
        if self.sentinel1.resampling is None:
            raise PreprocessingConfigurationError(
                "sentinel1.resampling is unresolved; no resampling method is assumed"
            )
        return self.sentinel1.resampling

    def require_s2_resampling(self) -> Any:
        if self.sentinel2.resampling is None:
            raise PreprocessingConfigurationError(
                "sentinel2.resampling is unresolved; no resampling method is assumed"
            )
        return self.sentinel2.resampling

    def require_dem_resampling(self) -> Any:
        if self.dem.resampling is None:
            raise PreprocessingConfigurationError(
                "dem.resampling is unresolved; no resampling method is assumed"
            )
        return self.dem.resampling

    def require_speckle_policy(self) -> SpeckleConfig:
        """The speckle policy, or an explicit failure if it was never stated.

        Contract decision C4 is "no filter, mismatch disclosed". That is a
        decision, so ``applied_by_our_pipeline`` must be present — ``None``
        means nobody chose, which is exactly the state the old ``extra="allow"``
        schema made indistinguishable from a chosen ``false``.
        """
        if self.sentinel1.speckle.applied_by_our_pipeline is None:
            raise PreprocessingConfigurationError(
                "sentinel1.speckle.applied_by_our_pipeline is unresolved; the "
                "speckle decision must be explicit (contract decision C4)"
            )
        return self.sentinel1.speckle


def load_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig.from_mapping(load_config("preprocessing"))
