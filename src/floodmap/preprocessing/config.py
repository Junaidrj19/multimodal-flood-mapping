"""Typed, explicit configuration for M2 preprocessing."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.config import load_config
from .errors import PreprocessingConfigurationError
from .grid import AnalysisGrid

__all__ = ["PreprocessingConfig", "load_preprocessing_config"]


class TargetGridConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    crs: Optional[str] = None
    resolution_m: Optional[float] = None
    transform: Optional[Tuple[float, ...]] = None
    width: Optional[int] = None
    height: Optional[int] = None

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


class Sentinel1Config(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

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


class QualityMaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Optional[bool] = None
    band_name: Optional[str] = None
    valid_values: Optional[Tuple[int, ...]] = None

    @field_validator("valid_values", mode="before")
    @classmethod
    def _valid_values_tuple(cls, value: Any) -> Any:
        return tuple(int(item) for item in value) if value is not None else None


class Sentinel2Config(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

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


class DemConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    source_product: str = "Copernicus WorldDEM-30"
    version: Optional[str] = None
    vertical_datum: Optional[str] = None
    resampling: Optional[str] = None
    output_nodata: Optional[float] = None


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    version: str = "0.1.0"
    pipeline_version: str = "m2-raster-0.1.0"
    target_grid: TargetGridConfig = Field(default_factory=TargetGridConfig)
    alignment_checks: AlignmentConfig = Field(default_factory=AlignmentConfig)
    sentinel1: Sentinel1Config = Field(default_factory=Sentinel1Config)
    sentinel2: Sentinel2Config = Field(default_factory=Sentinel2Config)
    dem: DemConfig = Field(default_factory=DemConfig)

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


def load_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig.from_mapping(load_config("preprocessing"))
