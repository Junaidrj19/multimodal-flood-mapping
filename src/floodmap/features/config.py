"""Typed, explicit configuration for M3 feature generation.

Follows the M2 convention: a scientific parameter that has not been verified is
``None``, and a production run fails on it rather than receiving a default.
The sections that would be tempting to default — band-role bindings, the
Sentinel-1 backscatter representation, the terrain linear unit — are precisely
the ones a wrong default would corrupt silently, so none of them has one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.config import load_config
from ..utils.provenance import ProductionInput, ValidationOnlySource
from .errors import FeatureBoundaryError, FeatureConfigurationError
from .registry import ArtifactRole, BackscatterRepresentation

__all__ = ["FeatureConfig", "NormalisationConfig", "load_feature_config"]

#: Permitted output storage dtypes. float16 is excluded because decibel values
#: and elevation in metres both exceed its useful precision.
_ALLOWED_DTYPES = ("float32", "float64")


class NormalisationConfig(BaseModel):
    """Machine-learning normalisation — applied here, never fitted here.

    M3's product is scientifically interpretable physical quantities. Rescaling
    them is a modelling convenience, not a feature definition, so the default is
    no normalisation at all.

    When it is enabled, two rules are enforced rather than documented:

    1. statistics are **loaded** from an explicit path. M3 cannot compute them,
       because the only data M3 has in hand is the scene being processed — and
       fitting on it would be the textbook form of test-time leakage.
    2. ``statistics_source`` must declare a training-only origin, matching
       ``configs/segmentation.yaml -> features.normalisation.statistics_source``.
       Fitting on the unseen Himalayan evaluation scenes is prohibited by
       ``AGENTS.md`` §5 and would void the evaluation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Optional[str] = None
    statistics_source: str = "training_split_only"
    statistics_path: Optional[str] = None
    version: Optional[str] = None

    @field_validator("method")
    @classmethod
    def _known_method(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in {"zscore", "minmax"}:
            raise ValueError("normalisation.method must be null, 'zscore' or 'minmax'")
        return value

    @property
    def enabled(self) -> bool:
        return self.method is not None

    def require_statistics_path(self) -> Path:
        if self.statistics_source != "training_split_only":
            raise FeatureBoundaryError(
                "normalisation statistics must come from the training split only; "
                f"statistics_source={self.statistics_source!r} would risk fitting "
                "scaling parameters on evaluation data (AGENTS.md §5)"
            )
        if not self.statistics_path:
            raise FeatureConfigurationError(
                "normalisation.method is set but normalisation.statistics_path is "
                "unresolved. M3 never fits normalisation parameters from the scene "
                "it is processing."
            )
        if self.version is None:
            raise FeatureConfigurationError(
                "normalisation.version is unresolved; a scaling applied to model "
                "inputs must be versioned so an inference can be reproduced"
            )
        path = Path(self.statistics_path)
        if not path.is_file():
            raise FeatureConfigurationError(f"normalisation statistics file does not exist: {path}")
        return path


class Sentinel1FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    polarisation_roles: Optional[Tuple[str, ...]] = None
    backscatter_representation: Optional[BackscatterRepresentation] = None

    @field_validator("polarisation_roles", mode="before")
    @classmethod
    def _roles(cls, value: Any) -> Any:
        return tuple(str(item).strip().lower() for item in value) if value is not None else None


class Sentinel2FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spectral_band_roles: Optional[Tuple[str, ...]] = None

    @field_validator("spectral_band_roles", mode="before")
    @classmethod
    def _roles(cls, value: Any) -> Any:
        return tuple(str(item).strip().lower() for item in value) if value is not None else None


class TerrainFeatureConfig(BaseModel):
    """Terrain derivative settings.

    ``elevation_unit`` and ``require_projected_grid`` exist because the slope
    kernel divides an elevation difference by a pixel size. If the grid is
    geographic the denominator is in degrees and the result is meaningless, and
    if the elevation unit is not the grid unit the gradient is off by a constant
    factor. Neither condition is detectable from the raster itself, so both must
    be asserted by the operator.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    elevation_unit: Optional[str] = None
    require_projected_grid: bool = True

    @field_validator("elevation_unit")
    @classmethod
    def _unit(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value != "m":
            raise ValueError(
                "terrain.elevation_unit must be 'm' or null; the slope kernel "
                "assumes elevation and grid resolution share a linear unit"
            )
        return value


class NumericsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dtype: str = "float32"
    output_nodata: Optional[float] = None
    report_out_of_declared_range: bool = True
    #: Clipping is off, and turning it on requires a written scientific reason.
    clip_to_valid_range: bool = False
    clip_justification: Optional[str] = None

    @field_validator("output_dtype")
    @classmethod
    def _dtype(cls, value: str) -> str:
        if value not in _ALLOWED_DTYPES:
            raise ValueError(f"numerics.output_dtype must be one of {_ALLOWED_DTYPES}")
        return value

    def require_nodata(self) -> float:
        if self.output_nodata is None:
            raise FeatureConfigurationError(
                "numerics.output_nodata is unresolved. The sentinel must be a value "
                "no feature can legitimately take, and that depends on the enabled "
                "features' ranges, so it is not defaulted."
            )
        return float(self.output_nodata)

    def require_clip_policy(self) -> None:
        if self.clip_to_valid_range and not self.clip_justification:
            raise FeatureConfigurationError(
                "numerics.clip_to_valid_range is enabled without a justification. "
                "Clipping alters recorded physical values and must not be used "
                "merely to make a computation succeed."
            )


class GridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Tolerance passed to M2's grid comparison. ``None`` means exact equality.
    transform_tolerance: Optional[float] = None
    #: M3 never resamples; this flag exists so the intent is explicit in config.
    allow_resampling: bool = False

    @field_validator("allow_resampling")
    @classmethod
    def _no_resampling(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "grid.allow_resampling must be false. Spatial normalisation is M2's "
                "responsibility; resampling again in M3 would hide a real "
                "misalignment behind an interpolation."
            )
        return value


class FeatureConfig(BaseModel):
    """Complete M3 configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "0.1.0"
    pipeline_version: str = "m3-features-0.1.0"
    registry_version: str = "1.0.0"
    feature_set_version: Optional[str] = None

    enabled_templates: Tuple[str, ...] = ()
    allowed_sources: Tuple[str, ...] = ()

    sentinel1: Sentinel1FeatureConfig = Field(default_factory=Sentinel1FeatureConfig)
    sentinel2: Sentinel2FeatureConfig = Field(default_factory=Sentinel2FeatureConfig)
    terrain: TerrainFeatureConfig = Field(default_factory=TerrainFeatureConfig)
    band_roles: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    numerics: NumericsConfig = Field(default_factory=NumericsConfig)
    normalisation: NormalisationConfig = Field(default_factory=NormalisationConfig)
    grid: GridConfig = Field(default_factory=GridConfig)

    @field_validator("enabled_templates", "allowed_sources", mode="before")
    @classmethod
    def _tuple(cls, value: Any) -> Any:
        return tuple(str(item) for item in value) if value is not None else ()

    @field_validator("band_roles", mode="before")
    @classmethod
    def _band_roles(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("band_roles must be a mapping of artifact role -> {role: band}")
        resolved: Dict[str, Dict[str, str]] = {}
        for artifact, mapping in value.items():
            if mapping is None:
                # An explicitly unresolved artifact binding. Kept out of the
                # resolved map so any feature needing it fails by name.
                continue
            if not isinstance(mapping, Mapping):
                raise ValueError(f"band_roles.{artifact} must be a mapping or null")
            resolved[str(artifact)] = {
                str(role).strip().lower(): str(band) for role, band in mapping.items() if band
            }
        return resolved

    @field_validator("allowed_sources")
    @classmethod
    def _sources_are_permitted_production_inputs(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        """Reject anything outside the closed permitted-production-input set.

        ``allowed_sources`` is the configuration-level expression of the
        ``AGENTS.md`` §3 boundary. Checking it against both enums means a
        validation-only identifier is rejected by name, and an unrecognised
        identifier is rejected for not being on the permitted list — so neither
        a forbidden source nor a typo can quietly widen the boundary.
        """
        permitted = {item.value for item in ProductionInput}
        forbidden = {item.value for item in ValidationOnlySource}
        for raw in value:
            token = raw.strip().lower()
            if token in forbidden:
                raise FeatureBoundaryError(
                    f"{raw!r} is a validation-only source and must never be a "
                    "feature-generation input (AGENTS.md §3, architecture.md §18)"
                )
            if token not in permitted:
                raise FeatureBoundaryError(
                    f"{raw!r} is not a permitted production input; allowed values "
                    f"are {sorted(permitted)}"
                )
        return value

    # --- resolution helpers ------------------------------------------------

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FeatureConfig":
        try:
            return cls.model_validate(mapping)
        except FeatureBoundaryError:
            raise
        except Exception as exc:
            raise FeatureConfigurationError(f"invalid feature configuration: {exc}") from exc

    def require_feature_set_version(self) -> str:
        if not self.feature_set_version:
            raise FeatureConfigurationError(
                "feature_set_version is unresolved; a feature artifact must declare "
                "the feature-definition version it was produced under"
            )
        return self.feature_set_version

    def require_enabled_templates(self) -> Tuple[str, ...]:
        if not self.enabled_templates:
            raise FeatureConfigurationError(
                "enabled_templates is empty; no feature family has been enabled. "
                "M3 does not pick a default feature set."
            )
        return self.enabled_templates

    def require_allowed_sources(self) -> Tuple[ProductionInput, ...]:
        if not self.allowed_sources:
            raise FeatureConfigurationError(
                "allowed_sources is empty; the permitted production inputs for this "
                "feature set must be declared explicitly"
            )
        return tuple(ProductionInput(item.strip().lower()) for item in self.allowed_sources)

    def resolved_band_roles(self) -> Dict[ArtifactRole, Dict[str, str]]:
        resolved: Dict[ArtifactRole, Dict[str, str]] = {}
        for artifact, mapping in self.band_roles.items():
            try:
                role = ArtifactRole(artifact)
            except ValueError as exc:
                raise FeatureConfigurationError(
                    f"band_roles key {artifact!r} is not a known artifact role; "
                    f"expected one of {[item.value for item in ArtifactRole]}"
                ) from exc
            resolved[role] = dict(mapping)
        return resolved


def load_feature_config() -> FeatureConfig:
    return FeatureConfig.from_mapping(load_config("features"))
