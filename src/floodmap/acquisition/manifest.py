"""Serializable acquisition run manifest and its provenance record."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.provenance import (
    ArtifactProvenance,
    ArtifactType,
    BeforeAfterPair,
    ProductionInput,
    Unknown,
    UnknownReason,
)
from .aoi import AreaOfInterest
from .outcomes import DiscoveryStatus, DownloadStatus, ManifestStatus
from .providers.base import DiscoveryResult, DownloadResult, SearchRequest
from .scenes import DiscoveredScene, SensorKind
from .selection import SelectionOutcome
from .temporal import EventSpec, TemporalPlan

__all__ = ["AcquisitionManifest"]


def _serialise(value: Any) -> Any:
    """Convert domain objects into plain JSON/YAML-compatible values."""
    if isinstance(value, Enum):
        return _serialise(value.value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialise(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def _selected_pair(outcome: Any) -> Optional[BeforeAfterPair]:
    if not isinstance(outcome, SelectionOutcome) or not outcome.is_usable:
        return None
    before = outcome.selected_before
    after = outcome.selected_after
    if before is None or after is None:
        return None
    return BeforeAfterPair(
        before=before.to_scene_reference(),
        after=after.to_scene_reference(),
        same_relative_orbit=(
            before.relative_orbit == after.relative_orbit
            if isinstance(before.relative_orbit, int) and isinstance(after.relative_orbit, int)
            else None
        ),
    )


def _status_for(discoveries: Mapping[str, Any], selections: Mapping[str, Any]) -> ManifestStatus:
    if any(
        getattr(result, "status", None) is DiscoveryStatus.FAILED for result in discoveries.values()
    ):
        return ManifestStatus.FAILED
    usable = []
    for sensor, outcome in selections.items():
        key = sensor.value if isinstance(sensor, SensorKind) else str(sensor)
        discovery = discoveries.get(key)
        if getattr(outcome, "is_usable", False) and (
            discovery is None or getattr(discovery, "status", None) is not DiscoveryStatus.SUCCESS
        ):
            # A selected scene is only a valid acquisition result when it came
            # from a successful, recorded discovery for the same sensor.
            return ManifestStatus.FAILED
        usable.append(getattr(outcome, "is_usable", False))
    if usable and all(usable):
        return ManifestStatus.COMPLETE
    if any(usable):
        return ManifestStatus.PARTIAL
    return ManifestStatus.NO_USABLE_DATA


class AcquisitionManifest(BaseModel):
    """One auditable metadata acquisition run.

    The domain objects are accepted directly while the serialized form is
    deliberately plain. This lets a manifest be written before any raster
    download and read back without importing provider-specific classes.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    manifest_version: str = "1.0"
    artifact_id: str = "acquisition-manifest"
    aoi: Any
    event: Any
    status: ManifestStatus = ManifestStatus.DRY_RUN
    provider: str = ""
    temporal_plans: dict[str, Any] = Field(default_factory=dict)
    requests: dict[str, Any] = Field(default_factory=dict)
    discoveries: dict[str, Any] = Field(default_factory=dict)
    selections: dict[str, Any] = Field(default_factory=dict)
    downloads: list[Any] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_version: Optional[str] = None
    provenance: Optional[ArtifactProvenance] = None

    @model_validator(mode="after")
    def _attach_provenance(self) -> "AcquisitionManifest":
        if self.provenance is None:
            self.provenance = self.to_artifact_provenance()
        return self

    @classmethod
    def from_run(
        cls,
        *,
        aoi: AreaOfInterest,
        event: EventSpec,
        temporal_plans: Mapping[str, TemporalPlan],
        requests: Mapping[str, SearchRequest],
        discoveries: Mapping[str, DiscoveryResult],
        selections: Mapping[str, SelectionOutcome],
        downloads: Optional[list[DownloadResult]] = None,
        provider: str = "",
        config_version: Optional[str] = None,
        dry_run: bool = False,
        artifact_id: Optional[str] = None,
    ) -> "AcquisitionManifest":
        status = ManifestStatus.DRY_RUN if dry_run else _status_for(discoveries, selections)
        limitations: list[str] = []
        for plan in temporal_plans.values():
            limitations.extend(plan.limitations())
        for outcome in selections.values():
            limitations.extend(rejection.rationale for rejection in outcome.rejected_candidates)
        manifest = cls(
            artifact_id=artifact_id or f"acquisition-{aoi.aoi_id}-{event.event_date.isoformat()}",
            aoi=aoi,
            event=event,
            status=status,
            provider=provider,
            temporal_plans=dict(temporal_plans),
            requests=dict(requests),
            discoveries=dict(discoveries),
            selections=dict(selections),
            downloads=list(downloads or []),
            limitations=list(dict.fromkeys(limitations)),
            config_version=config_version,
        )
        return manifest

    def to_artifact_provenance(self) -> ArtifactProvenance:
        """Build the linked ``ACQUISITION_MANIFEST`` provenance record."""
        sentinel1 = _selected_pair(self.selections.get(SensorKind.SENTINEL1.value))
        sentinel2 = _selected_pair(self.selections.get(SensorKind.SENTINEL2.value))
        # Also accept enum keys for callers constructing a manifest directly.
        sentinel1 = sentinel1 or _selected_pair(self.selections.get(SensorKind.SENTINEL1))
        sentinel2 = sentinel2 or _selected_pair(self.selections.get(SensorKind.SENTINEL2))
        production_inputs: list[ProductionInput] = []
        if sentinel1 is not None:
            production_inputs.append(ProductionInput.SENTINEL1)
        if sentinel2 is not None:
            production_inputs.append(ProductionInput.SENTINEL2)
        event_date = getattr(self.event, "event_date", None)
        if event_date is None and isinstance(self.event, Mapping):
            event_date = self.event.get("event_date")
        aoi_value = _serialise(self.aoi)
        aoi_crs = getattr(self.aoi, "crs", None)
        if aoi_crs is None and isinstance(self.aoi, Mapping):
            aoi_crs = self.aoi.get("crs")
        return ArtifactProvenance(
            artifact_type=ArtifactType.ACQUISITION_MANIFEST,
            artifact_id=self.artifact_id,
            aoi=aoi_value,
            aoi_crs=aoi_crs,
            event_date=event_date.isoformat() if hasattr(event_date, "isoformat") else event_date,
            sentinel1=sentinel1,
            sentinel2=sentinel2,
            production_inputs=production_inputs,
            config_version=self.config_version,
            code_version=None,
            limitations=list(self.limitations),
            notes=(
                "Metadata discovery manifest; download status is recorded per scene. "
                "A selected scene is not evidence that a raster was downloaded."
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "manifest_version": self.manifest_version,
            "artifact_id": self.artifact_id,
            "aoi": self.aoi,
            "event": self.event,
            "status": self.status,
            "provider": self.provider,
            "temporal_plans": self.temporal_plans,
            "requests": self.requests,
            "discoveries": self.discoveries,
            "selections": self.selections,
            "downloads": self.downloads,
            "limitations": self.limitations,
            "generated_at": self.generated_at,
            "config_version": self.config_version,
            "provenance": self.provenance or self.to_artifact_provenance(),
        }
        return _serialise(data)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AcquisitionManifest":
        return cls.model_validate(dict(data))

    @classmethod
    def from_json(cls, text: str) -> "AcquisitionManifest":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> "AcquisitionManifest":
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, Mapping):
            raise TypeError("acquisition manifest YAML must contain a mapping")
        return cls.from_dict(loaded)
