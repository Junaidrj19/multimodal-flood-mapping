"""Typed configuration consumed by the Earth Observation acquisition stage."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.config import load_config
from .aoi import AreaOfInterest
from .errors import AoiNotConfigured, SelectionStrategyNotConfigured
from .outcomes import SelectionStrategy
from .temporal import EventSpec, SearchWindow

__all__ = [
    "AcquisitionConfig",
    "CredentialConfig",
    "DownloadConfig",
    "ProviderConfig",
    "SearchWindowConfig",
    "SensorAcquisitionConfig",
    "load_acquisition_config",
]


class CredentialConfig(BaseModel):
    """Names of environment variables, never the credential values themselves."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    username_env: str = "CDSE_USERNAME"
    password_env: str = "CDSE_PASSWORD"

    def values_from_environment(self) -> tuple[Optional[str], Optional[str]]:
        return os.getenv(self.username_env), os.getenv(self.password_env)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "cdse-odata"
    catalogue_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    download_url_template: Optional[str] = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_results: int = Field(default=200, ge=1)
    page_size: int = Field(default=100, ge=1)
    credentials: CredentialConfig = Field(default_factory=CredentialConfig)


class SearchWindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    before_days: Optional[int] = None
    after_days: Optional[int] = None


class SensorAcquisitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    search_window: SearchWindowConfig = Field(default_factory=SearchWindowConfig)
    product_types: tuple[str, ...] = ()
    selection_strategy: Optional[SelectionStrategy] = None
    max_scene_cloud_percent: Optional[float] = None
    require_same_relative_orbit: bool = True
    on_no_same_track_pair: str = "fail"

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_flat_window_keys(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        if "search_window" not in values:
            values["search_window"] = {
                "before_days": values.pop("window_before_days", None),
                "after_days": values.pop("window_after_days", None),
            }
        else:
            values.pop("window_before_days", None)
            values.pop("window_after_days", None)
        return values

    @field_validator("product_types", mode="before")
    @classmethod
    def _tuple_product_types(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

    @field_validator("on_no_same_track_pair")
    @classmethod
    def _known_no_pair_policy(cls, value: str) -> str:
        if value not in {"fail", "warn_and_continue", "record"}:
            raise ValueError(
                "on_no_same_track_pair must be 'fail', 'warn_and_continue' or 'record'"
            )
        return value

    @property
    def window_before_days(self) -> Optional[int]:
        return self.search_window.before_days

    @property
    def window_after_days(self) -> Optional[int]:
        return self.search_window.after_days

    def resolved_search_window(
        self, *, sensor: str, fallback: Optional[Mapping[str, Any]] = None
    ) -> SearchWindow:
        before = self.search_window.before_days
        after = self.search_window.after_days
        if fallback is not None:
            before = before if before is not None else fallback.get("window_before_days")
            after = after if after is not None else fallback.get("window_after_days")
        return SearchWindow.from_config(before, after, sensor=sensor)


class DownloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    destination: str = "data/raw"


class AcquisitionConfig(BaseModel):
    """The acquisition subsection of ``configs/data.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    aoi: Optional[AreaOfInterest] = None
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    sentinel1: SensorAcquisitionConfig = Field(default_factory=SensorAcquisitionConfig)
    sentinel2: SensorAcquisitionConfig = Field(default_factory=SensorAcquisitionConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)

    @classmethod
    def from_data_config(cls, data: Mapping[str, Any]) -> "AcquisitionConfig":
        raw = data.get("acquisition")
        if raw is None:
            raise ValueError("configs/data.yaml is missing the top-level acquisition block")
        if not isinstance(raw, Mapping):
            raise TypeError("configs/data.yaml acquisition block must be a mapping")
        values = dict(raw)
        raw_aoi = values.get("aoi")
        if raw_aoi is not None:
            if not isinstance(raw_aoi, Mapping):
                raise TypeError("acquisition.aoi must be null or a mapping")
            values["aoi"] = _aoi_from_mapping(raw_aoi)
        return cls.model_validate(values)

    def require_aoi(self, override: Optional[AreaOfInterest] = None) -> AreaOfInterest:
        if override is not None:
            return override
        if self.aoi is None:
            raise AoiNotConfigured(
                "no acquisition AOI is configured; supply --aoi-bbox or --aoi-geojson "
                "rather than using an implicit study-area geometry"
            )
        return self.aoi

    def event_spec(
        self, data: Mapping[str, Any], *, event_date: Optional[date] = None
    ) -> EventSpec:
        raw_event = data.get("event", {})
        if not isinstance(raw_event, Mapping) or raw_event.get("date") is None:
            raise ValueError("data config must define event.date")
        selected_date = event_date or date.fromisoformat(str(raw_event["date"]))
        return EventSpec(
            event_date=selected_date,
            label=str(raw_event.get("name", "")),
        )

    def selection_strategy(self, sensor: str) -> SelectionStrategy:
        config = self.sentinel1 if sensor == "sentinel1" else self.sentinel2
        if config.selection_strategy is None:
            raise SelectionStrategyNotConfigured(
                f"acquisition.{sensor}.selection_strategy is null; choose an explicit "
                "deterministic strategy"
            )
        return config.selection_strategy


def _aoi_from_mapping(raw: Mapping[str, Any]) -> AreaOfInterest:
    aoi_id = str(raw.get("aoi_id") or raw.get("id") or raw.get("name") or "")
    name = str(raw.get("name") or "")
    is_synthetic = bool(raw.get("is_synthetic", False))
    if raw.get("bbox") is not None:
        return AreaOfInterest.from_bbox(
            aoi_id,
            raw["bbox"],
            name=name,
            is_synthetic=is_synthetic,
            crs=str(raw.get("crs", "EPSG:4326")),
        )
    geometry = raw.get("geometry")
    if geometry is not None:
        return AreaOfInterest.from_geojson_geometry(
            aoi_id, geometry, name=name, is_synthetic=is_synthetic
        )
    raise AoiNotConfigured("configured acquisition.aoi has no bbox or Polygon geometry")


def load_acquisition_config(name: str = "data") -> tuple[AcquisitionConfig, dict[str, Any]]:
    """Load the typed acquisition block and the complete data config."""
    data = load_config(name)
    return AcquisitionConfig.from_data_config(data), data
