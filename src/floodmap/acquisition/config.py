"""Typed configuration consumed by the Earth Observation acquisition stage."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.config import load_config, project_root
from .aoi import AoiProvenanceStatus, AreaOfInterest
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
    # Minimum fraction of the AOI a candidate's footprint must cover to be
    # eligible, in (0, 1]. None means "intersection required, measured coverage
    # recorded but not thresholded" — no threshold is invented. The value is
    # configured explicitly in configs/data.yaml (0.9), matching the existing
    # min_aoi_coverage decision for the Sentinel-1 RTC path
    # (configs/baseline.yaml), so the two acquisition routes agree.
    minimum_aoi_coverage: Optional[float] = None
    # Sentinel-1 only. Channels that must be present on both products, e.g.
    # ["VV", "VH"]. Empty means the reported set is recorded but not required.
    # Set explicitly in configs/data.yaml because the served model consumes four
    # channels (pre VV/VH, post VV/VH); a single-channel product would leave an
    # input empty rather than fail.
    required_polarisations: tuple[str, ...] = ()
    # Sentinel-2 only. Require both sides of the pair to report the same MGRS
    # tile. Differing tiles mean the scenes were not processed on the same grid.
    require_same_tile: bool = True
    # Sentinel-2 only. When true, a pair whose sides are on different processing
    # levels is ranked below same-level pairs and, if selected, recorded as a
    # disclosed limitation (BOA and TOA are not interchangeable in an index).
    # When false, mixed-level pairs are ineligible.
    allow_mixed_processing_levels: bool = True
    # Sentinel-2 processing level preference: "L2A" (prefer L2A/BOA), "L1C" (prefer L1C/TOA), or None (no preference).
    # Only meaningful for Sentinel-2. Default None preserves backward compatibility.
    processing_level_preference: Optional[str] = None
    # If True and no eligible product matches processing_level_preference, allow falling back to the other level.
    # If False and no eligible product matches the preference, the selection fails. Default True for backward compatibility.
    allow_processing_level_fallback: bool = True

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

    @field_validator("minimum_aoi_coverage")
    @classmethod
    def _coverage_threshold_in_range(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(
                f"minimum_aoi_coverage must be in (0, 1], got {value}; 0 would accept a "
                "scene that does not intersect the AOI at all"
            )
        return float(value)

    @field_validator("required_polarisations", mode="before")
    @classmethod
    def _tuple_polarisations(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        return tuple(dict.fromkeys(str(item).strip().upper() for item in value if str(item).strip()))

    @field_validator("processing_level_preference", mode="before")
    @classmethod
    def _validate_processing_level_preference(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("processing_level_preference must be a string or null")
        normalized = value.strip().upper()
        if normalized not in {"L2A", "L1C"}:
            raise ValueError(
                f"processing_level_preference must be 'L2A' or 'L1C', got {value!r}"
            )
        return normalized

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


class ExploratoryDownloadAuthorisation(BaseModel):
    """A narrow, auditable operator exception permitting one download run.

    Why this exists
    ---------------
    Product bytes are normally fetched only against an AOI whose provenance is
    established (``AreaOfInterest.authorises_download``). An operator may
    nonetheless want a *bounded* exploratory fetch before that provenance work
    is finished — to see real pixels at all.

    This records that exception instead of removing the gate. It deliberately
    does **not** touch ``provenance_status``: the AOI stays UNVERIFIED, every
    downstream disclosure keeps firing, and the exception is visible in the
    manifest rather than hidden in a flipped flag.

    Why it is scoped this tightly
    -----------------------------
    An unscoped "allow downloads" switch would be indistinguishable from
    verifying the AOI, and would silently apply to every future run and every
    other product. So authorisation is valid only for:

    * ``product_ids`` — an exact allowlist. A product not named here is refused
      even inside an authorised run.
    * ``aoi_sha256`` — the AOI bytes this exception was granted against. If the
      geometry changes, the authorisation no longer applies, because it was
      granted for a specific study extent.

    ``reason`` and ``authorised_by`` are mandatory and non-empty: an exception to
    a correctness gate with no recorded justification is not auditable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    authorised_by: str = ""
    reason: str = ""
    #: Exact catalogue product UUIDs this exception covers. Empty means nothing
    #: is authorised: an exception with no scope is not a blanket permission.
    product_ids: tuple[str, ...] = ()
    #: sha256 of the AOI geometry file the exception was granted against.
    aoi_sha256: Optional[str] = None

    @model_validator(mode="after")
    def _enabled_exception_must_be_fully_specified(self) -> "ExploratoryDownloadAuthorisation":
        if not self.enabled:
            return self
        if not self.product_ids:
            raise ValueError(
                "exploratory_download_authorisation.enabled is true but product_ids is "
                "empty. An authorisation with no product scope would be a blanket "
                "download permission, which is exactly what this mechanism exists to "
                "avoid; list the exact product ids."
            )
        if not self.aoi_sha256:
            raise ValueError(
                "exploratory_download_authorisation.enabled is true but aoi_sha256 is "
                "unset. The exception is granted against a specific study extent, so it "
                "must pin the AOI bytes it applies to."
            )
        if not self.authorised_by.strip() or not self.reason.strip():
            raise ValueError(
                "exploratory_download_authorisation requires non-empty authorised_by and "
                "reason: an exception to a correctness gate without a recorded "
                "justification is not auditable."
            )
        return self

    def covers(self, *, aoi_sha256: Optional[str], product_id: Optional[str]) -> bool:
        """Whether this authorisation permits fetching ``product_id``."""
        if not self.enabled:
            return False
        if not aoi_sha256 or aoi_sha256 != self.aoi_sha256:
            return False
        return bool(product_id) and product_id in self.product_ids

    def to_record(self) -> dict[str, Any]:
        """Disclosure form for the manifest. Carries no credential material."""
        return {
            "enabled": self.enabled,
            "authorised_by": self.authorised_by,
            "reason": self.reason,
            "product_ids": list(self.product_ids),
            "aoi_sha256": self.aoi_sha256,
            "scope_note": (
                "Operator exception permitting an exploratory fetch of the listed "
                "products only, against the pinned AOI bytes only. The AOI provenance "
                "status is UNCHANGED and remains unverified; this is not a verification "
                "and confers no permission for any other product or AOI."
            ),
        }


class DownloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    destination: str = "data/raw"
    exploratory_authorisation: ExploratoryDownloadAuthorisation = Field(
        default_factory=ExploratoryDownloadAuthorisation
    )


class AcquisitionConfig(BaseModel):
    """The acquisition subsection of ``configs/data.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    aoi: Optional[AreaOfInterest] = None
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    sentinel1: SensorAcquisitionConfig = Field(default_factory=SensorAcquisitionConfig)
    sentinel2: SensorAcquisitionConfig = Field(default_factory=SensorAcquisitionConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)

    @model_validator(mode="after")
    def _polarisation_requirement_is_not_on_sentinel2(self) -> "AcquisitionConfig":
        if self.sentinel2.required_polarisations:
            raise ValueError(
                "acquisition.sentinel2.required_polarisations is set, but polarisation is a "
                "Sentinel-1 concept; an optical scene has bands, not channels. Remove it "
                "rather than leaving an unenforced requirement in the configuration."
            )
        return self

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

    # Provenance is read from configuration rather than inferred, and it
    # defaults to UNVERIFIED: an operator has to *assert* that an extent's
    # origin is established, because the alternative — treating an unannotated
    # geometry as authoritative — is how an unverified study area becomes the
    # silent basis of every downstream number (docs/dataset-registry.md §1.6).
    raw_status = raw.get("provenance_status")
    try:
        provenance_status = (
            AoiProvenanceStatus(str(raw_status).strip().lower())
            if raw_status is not None
            else AoiProvenanceStatus.UNVERIFIED
        )
    except ValueError as exc:
        permitted = ", ".join(sorted(status.value for status in AoiProvenanceStatus))
        raise ValueError(
            f"acquisition.aoi.provenance_status must be one of [{permitted}], "
            f"got {raw_status!r}"
        ) from exc

    raw_gaps = raw.get("provenance_gaps") or ()
    if isinstance(raw_gaps, str) or not isinstance(raw_gaps, (list, tuple)):
        raise TypeError(
            "acquisition.aoi.provenance_gaps must be a list of strings naming what "
            f"is unestablished about this geometry, got {type(raw_gaps).__name__}"
        )
    provenance_gaps = tuple(str(gap) for gap in raw_gaps)
    expected_sha256 = raw.get("sha256")

    geojson_path = raw.get("geojson_path")
    if geojson_path is not None:
        path = Path(str(geojson_path))
        if not path.is_absolute():
            path = project_root() / path
        if not path.is_file():
            raise AoiNotConfigured(
                f"acquisition.aoi.geojson_path points at {path}, which does not exist. "
                "The study extent cannot be defaulted; fix the path rather than "
                "falling back to an implicit geometry."
            )
        return AreaOfInterest.from_geojson_file(
            aoi_id,
            path,
            name=name,
            expected_sha256=None if expected_sha256 is None else str(expected_sha256),
            provenance_status=provenance_status,
            provenance_gaps=provenance_gaps,
        )

    if raw.get("bbox") is not None:
        return AreaOfInterest.from_bbox(
            aoi_id,
            raw["bbox"],
            name=name,
            is_synthetic=is_synthetic,
            crs=str(raw.get("crs", "EPSG:4326")),
            provenance_status=provenance_status,
            source_sha256=None if expected_sha256 is None else str(expected_sha256),
            provenance_gaps=provenance_gaps,
        )
    geometry = raw.get("geometry")
    if geometry is not None:
        return AreaOfInterest.from_geojson_geometry(
            aoi_id,
            geometry,
            name=name,
            is_synthetic=is_synthetic,
            provenance_status=provenance_status,
            source_sha256=None if expected_sha256 is None else str(expected_sha256),
            provenance_gaps=provenance_gaps,
        )
    raise AoiNotConfigured(
        "configured acquisition.aoi has no geojson_path, bbox or Polygon geometry"
    )


def load_acquisition_config(name: str = "data") -> tuple[AcquisitionConfig, dict[str, Any]]:
    """Load the typed acquisition block and the complete data config."""
    data = load_config(name)
    return AcquisitionConfig.from_data_config(data), data
