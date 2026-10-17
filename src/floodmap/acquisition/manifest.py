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
from .outcomes import (
    DiscoveryStatus,
    DownloadStatus,
    DownloadSummary,
    ManifestStatus,
    summarise_downloads,
)
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

    Two statuses, deliberately separate
    -----------------------------------
    ``status`` answers *"did selection find a usable pair?"* — a finding about
    the world and about our query. ``download_status`` answers *"did we obtain
    the bytes?"* — an operational fact. Collapsing them would allow a manifest to
    read ``complete`` while every download failed, which is exactly the
    misleading record this milestone must avoid. A download failure therefore
    never rewrites ``status``; it downgrades ``download_status``.

    ``status`` can also be :attr:`ManifestStatus.SELECTION_COMPLETE_DOWNLOAD_PENDING`,
    written before the download loop so an interrupted run leaves a truthful
    record of what was selected and what was attempted rather than no record at
    all. That state is distinct from ``COMPLETE`` precisely so it cannot be
    mistaken for a finished acquisition.

    The domain objects are accepted directly while the serialized form is
    deliberately plain. This lets a manifest be written before any raster
    download and read back without importing provider-specific classes.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    manifest_version: str = "1.1"
    artifact_id: str = "acquisition-manifest"
    aoi: Any
    event: Any
    status: ManifestStatus = ManifestStatus.DRY_RUN
    download_status: DownloadSummary = DownloadSummary.NOT_ATTEMPTED
    provider: str = ""
    temporal_plans: dict[str, Any] = Field(default_factory=dict)
    requests: dict[str, Any] = Field(default_factory=dict)
    discoveries: dict[str, Any] = Field(default_factory=dict)
    selections: dict[str, Any] = Field(default_factory=dict)
    downloads: list[Any] = Field(default_factory=list)
    # Set when the catalogue query used a geometry other than the AOI itself
    # (an envelope pre-filter). Recorded so a reader can see that the *query*
    # geometry and the *coverage* geometry differ, and why.
    query_prefilter: Optional[dict[str, Any]] = None
    download_authorisation: Optional[dict[str, Any]] = None
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
        download_pending: bool = False,
        artifact_id: Optional[str] = None,
        query_prefilter: Optional[Mapping[str, Any]] = None,
        download_authorisation: Optional[Mapping[str, Any]] = None,
    ) -> "AcquisitionManifest":
        if download_pending:
            # Written before any download is attempted. Never COMPLETE: the run
            # has not finished, and saying so would be the false-success record
            # this change exists to prevent.
            status = ManifestStatus.SELECTION_COMPLETE_DOWNLOAD_PENDING
        elif dry_run:
            status = ManifestStatus.DRY_RUN
        else:
            status = _status_for(discoveries, selections)
        limitations: list[str] = []
        # The AOI caveat comes first. It is the widest-scoped limitation in the
        # artifact: every selection, coverage fraction and downstream number is
        # computed against this extent, so a reader must see its provenance
        # status before any per-scene detail. Empty when provenance is verified.
        aoi_limitations = getattr(aoi, "provenance_limitations", None)
        if callable(aoi_limitations):
            limitations.extend(aoi_limitations())
        # An exploratory download exception is a limitation of the RESULT, not a
        # verification: it must read as a disclosed deviation in the artifact.
        if download_authorisation and download_authorisation.get("enabled"):
            limitations.append(
                "Products were fetched under a narrow operator exception "
                "(acquisition.download.exploratory_authorisation) while the AOI "
                "provenance was still UNVERIFIED. The exception named "
                f"{len(download_authorisation.get('product_ids') or [])} product id(s) "
                "and pinned the AOI bytes; it is not a verification of the study extent "
                "and authorises no flood-evidence claim."
            )
        if query_prefilter:
            # Second only to the AOI caveat in scope: it explains why the query
            # geometry in `requests` differs from the AOI in `selections`, so a
            # reviewer does not read the envelope as the study area.
            limitations.append(
                "The catalogue query used the AOI's bounding envelope as a spatial "
                "pre-filter, not the AOI geometry itself: "
                f"{query_prefilter.get('reason', 'reason not recorded')}. The envelope is "
                "a strict superset of the AOI, so no candidate the AOI would have "
                "matched was excluded, and coverage was still measured against the AOI "
                "(see each selection's `pair_validation.coverage`). "
                "`requests[].aoi_wkt` is the pre-filter; "
                "`selections[].selection_policy.aoi_wkt` is the AOI."
            )
        for plan in temporal_plans.values():
            limitations.extend(plan.limitations())
        for outcome in selections.values():
            limitations.extend(rejection.rationale for rejection in outcome.rejected_candidates)
            # A pair can be usable and still carry a disclosed limitation, such
            # as mixed processing levels. Those are limitations of the *result*,
            # not reasons a candidate was discarded, so they are surfaced here
            # and must reach the report.
            validation = getattr(outcome, "pair_validation", None)
            if isinstance(validation, Mapping) and validation.get("mixed_processing_levels"):
                levels = validation.get("processing_levels", {})
                limitations.append(
                    "The selected Sentinel-2 pair mixes processing levels ("
                    f"{levels.get('before')} pre-event vs {levels.get('after')} post-event). "
                    "Surface (BOA) and top-of-atmosphere (TOA) reflectance are not "
                    "interchangeable in NDVI/NDWI/MNDWI, so any spectral index computed "
                    "across this pair carries a systematic offset and this comparison is "
                    "not like-for-like."
                )
        manifest = cls(
            artifact_id=artifact_id or f"acquisition-{aoi.aoi_id}-{event.event_date.isoformat()}",
            aoi=aoi,
            event=event,
            status=status,
            download_status=summarise_downloads(downloads),
            provider=provider,
            temporal_plans=dict(temporal_plans),
            requests=dict(requests),
            discoveries=dict(discoveries),
            selections=dict(selections),
            downloads=list(downloads or []),
            query_prefilter=dict(query_prefilter) if query_prefilter else None,
            download_authorisation=(
                dict(download_authorisation) if download_authorisation else None
            ),
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
                "Metadata discovery manifest. `status` describes SELECTION only; "
                f"`download_status` describes the fetch ({self.download_status.value}). "
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
            "download_status": self.download_status,
            "provider": self.provider,
            "temporal_plans": self.temporal_plans,
            "requests": self.requests,
            "discoveries": self.discoveries,
            "selections": self.selections,
            "downloads": self.downloads,
            "query_prefilter": self.query_prefilter,
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
