"""Explicit, serializable provenance records for generated artifacts.

Why this module exists
----------------------
``AGENTS.md`` §22 states the project must be able to answer "How do you know
this?" with a chain of evidence. ``architecture.md`` §12 and §18.9 require every
major artifact to carry provenance metadata. This module provides that record in
one place so no pipeline stage invents its own ad-hoc format.

The field names deliberately mirror the YAML example in ``architecture.md`` §12
so that a serialized record is recognisable against the architecture document.

Two design decisions worth stating
----------------------------------
1. **"Unknown" is not the same as "absent".**
   A missing value could mean "this pipeline stage did not record it" or "we
   looked and the data genuinely does not exist". Collapsing both into ``None``
   would quietly manufacture certainty, so :class:`Unknown` makes the second
   case explicit and requires a reason. ``None`` means only "not recorded".

2. **The production/validation boundary is encoded in types.**
   :class:`ProductionInput` is a closed enumeration of permitted production
   sources, and :class:`ValidationOnlySource` enumerates the forbidden ones.
   :class:`ArtifactProvenance` rejects a validation-only source being recorded
   as a production input, so the ``AGENTS.md`` §3 rule fails loudly at
   construction time rather than silently producing a misleading artifact.

Status
------
This is the schema only. No pipeline stage populates it yet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Mandatory attribution (AGENTS.md §18)
#
# VERIFIED(spec). Every submitted artifact, dashboard view and situation report
# must carry these three strings. This module is the ENFORCING copy: every
# ``ArtifactProvenance`` attaches them automatically, and the validator below
# re-inserts any that a caller omits, so an artifact cannot be produced without
# them. ``configs/data.yaml`` carries a mirror copy for config consumers, and
# ``tests/test_configs.py`` asserts the two are byte-identical.
#
# DISCREPANCY NOTE (unresolved — do not "tidy" without checking the spec PDF):
# the WorldDEM-30 string retains "© DLR e.V.", matching AGENTS.md §18 and
# README.md §12. A later restatement of the specification omitted that "©". The
# form with "©" is used because AGENTS.md is this project's declared source of
# truth (§2), and dropping a copyright mark from a required attribution is a
# legal defect rather than a stylistic choice. Confirm before submission.
# =============================================================================

SENTINEL_ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026."

WORLDDEM_ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
    "the European Union and ESA; all rights reserved."
)

OSM_ATTRIBUTION = "© OpenStreetMap contributors."

#: Required in this order on every artifact. Order is part of the contract so
#: that rendered output is deterministic and diffable.
REQUIRED_ATTRIBUTIONS: tuple[str, ...] = (
    SENTINEL_ATTRIBUTION,
    WORLDDEM_ATTRIBUTION,
    OSM_ATTRIBUTION,
)


class ProductionInput(str, Enum):
    """Data sources permitted in the production pipeline.

    Closed set, from ``AGENTS.md`` §3, ``README.md`` §4 and
    ``architecture.md`` §2. Adding a member is a scientific decision that must
    be justified against the challenge specification, not a convenience edit.
    """

    SENTINEL1 = "sentinel-1"
    SENTINEL2 = "sentinel-2"
    COPERNICUS_DEM = "copernicus-dem"
    OSM_PRE_EVENT = "osm-pre-event"
    PERMITTED_TRAINING_DATASET = "permitted-training-dataset"


class ValidationOnlySource(str, Enum):
    """Sources usable ONLY for post-hoc validation and comparison.

    From ``AGENTS.md`` §3 and ``architecture.md`` §2/§18. These must never enter
    training, feature construction, preprocessing, inference or threshold
    selection. They are enumerated here precisely so that the prohibition is
    machine-checkable rather than a matter of reviewer memory.
    """

    EMSR927 = "emsr927"
    COPERNICUS_EMS = "copernicus-ems"
    UNOSAT = "unosat"
    PUBLISHED_DAMAGE_MAP = "published-damage-map"
    OSM_POST_EVENT = "osm-post-event"


class ArtifactType(str, Enum):
    """Kinds of artifact the pipeline is expected to produce.

    Derived from the stages in ``architecture.md`` §1. Extended as stages are
    implemented.
    """

    ACQUISITION_MANIFEST = "acquisition_manifest"
    PREPROCESSED_RASTER = "preprocessed_raster"
    FEATURE_STACK = "feature_stack"
    SEGMENTATION_PREDICTION = "segmentation_prediction"
    INFRASTRUCTURE_EXPOSURE = "infrastructure_exposure"
    ROAD_DISRUPTION = "road_disruption"
    SETTLEMENT_CONNECTIVITY = "settlement_connectivity"
    FLOOD_PATH_TRACE = "flood_path_trace"
    EVALUATION_RESULT = "evaluation_result"
    SITUATION_REPORT = "situation_report"


class UnknownReason(str, Enum):
    """Why a value is not available.

    Distinguishing these prevents a gap in the record from being mistaken for a
    measured absence (``architecture.md`` §17 requires failure to be explicit
    rather than silently producing a misleading result).
    """

    NOT_YET_VERIFIED = "not_yet_verified"
    NOT_AVAILABLE_FROM_SOURCE = "not_available_from_source"
    NOT_APPLICABLE = "not_applicable"
    ACQUISITION_FAILED = "acquisition_failed"


class Unknown(BaseModel):
    """An explicitly unknown value, carrying the reason it is unknown.

    Use this instead of ``None`` whenever the absence is itself a finding. For
    example, "no same-track Sentinel-1 pair exists for this date" is a
    scientific result that downstream interpretation depends on, and must not
    be indistinguishable from "nobody filled this field in".
    """

    model_config = ConfigDict(frozen=True)

    unknown: bool = True
    reason: UnknownReason
    note: str = ""


# A field that may hold a real value, be explicitly unknown, or be unrecorded.
Maybe = Union[str, Unknown, None]


class SceneReference(BaseModel):
    """Identity and acquisition metadata for a single satellite scene.

    ``AGENTS.md`` §4 requires acquisition timestamps to be preserved, and orbit
    track to be known for Sentinel-1 so that same-track comparison can be
    verified rather than assumed.
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    acquired_at: Maybe = Field(
        default=None,
        description="Acquisition timestamp, ISO 8601 with timezone. Required for temporal reasoning.",
    )
    relative_orbit: Optional[int] = Field(
        default=None,
        description="Relative orbit / track number. Needed to verify same-track S1 comparison.",
    )
    orbit_direction: Maybe = Field(
        default=None, description="ASCENDING or DESCENDING where reported by the provider."
    )
    processing_level: Maybe = Field(
        default=None, description="Provider processing level, e.g. S1 GRD, S2 L2A."
    )
    source: Optional[ProductionInput] = Field(
        default=None, description="Which permitted production source this scene came from."
    )


class BeforeAfterPair(BaseModel):
    """A pre-event / post-event scene pair.

    ``AGENTS.md`` §4 forbids silently mixing temporal windows, so both halves of
    a change-detection comparison are recorded together with an explicit flag
    for whether the same-orbit-track condition was actually met.
    """

    model_config = ConfigDict(extra="forbid")

    before: Optional[SceneReference] = None
    after: Optional[SceneReference] = None
    same_relative_orbit: Optional[bool] = Field(
        default=None,
        description=(
            "True only when before.relative_orbit == after.relative_orbit and both are known. "
            "False is not a failure, but it must be surfaced: different tracks view terrain "
            "from different angles and should not be compared pixel-by-pixel as equivalent "
            "observations (AGENTS.md §4)."
        ),
    )

    @field_validator("same_relative_orbit")
    @classmethod
    def _must_not_claim_unverifiable(cls, v: Optional[bool], info) -> Optional[bool]:
        """Do not allow a same-track claim that the recorded orbits cannot support."""
        if v is not True:
            return v
        data = info.data
        before, after = data.get("before"), data.get("after")
        b_orbit = getattr(before, "relative_orbit", None)
        a_orbit = getattr(after, "relative_orbit", None)
        if b_orbit is None or a_orbit is None:
            raise ValueError(
                "same_relative_orbit=True requires a known relative_orbit on both scenes; "
                "use None when the orbit is unrecorded rather than asserting same-track."
            )
        if b_orbit != a_orbit:
            raise ValueError(
                f"same_relative_orbit=True contradicts recorded orbits ({b_orbit} != {a_orbit})."
            )
        return v


class ArtifactProvenance(BaseModel):
    """Provenance record for one generated artifact.

    Mirrors the YAML shape in ``architecture.md`` §12. Serialize with
    :meth:`to_yaml` or :meth:`to_json` and store alongside the artifact.
    """

    model_config = ConfigDict(extra="forbid")

    # --- what this is -----------------------------------------------------
    artifact_type: ArtifactType
    artifact_id: Maybe = Field(default=None, description="Stable identifier for this artifact.")

    # --- spatial / temporal scope ----------------------------------------
    aoi: Maybe = Field(
        default=None,
        description=(
            "Area of interest. TODO(contract): the canonical representation (bbox, GeoJSON "
            "geometry, or named region) is not yet fixed; see docs/data-contract.md."
        ),
    )
    aoi_crs: Maybe = Field(default=None, description="CRS of the AOI geometry, e.g. EPSG:4326.")
    event_date: Maybe = Field(default=None, description="Event date, ISO 8601.")

    # --- inputs -----------------------------------------------------------
    sentinel1: Optional[BeforeAfterPair] = None
    sentinel2: Optional[BeforeAfterPair] = None
    dem_version: Maybe = Field(default=None, description="DEM product and version identifier.")
    osm_snapshot_date: Maybe = Field(
        default=None,
        description="Date of the OSM snapshot. MUST predate event_date (AGENTS.md §3).",
    )
    production_inputs: List[ProductionInput] = Field(
        default_factory=list,
        description="Permitted production sources that contributed to this artifact.",
    )

    # --- how it was produced ---------------------------------------------
    model_version: Maybe = Field(default=None, description="Segmentation model version.")
    preprocessing_version: Maybe = Field(
        default=None, description="Preprocessing pipeline version."
    )
    config_version: Maybe = Field(default=None, description="Configuration version or hash.")
    code_version: Maybe = Field(default=None, description="Git commit of the generating code.")
    random_seed: Optional[int] = Field(default=None, description="Seed, where behaviour is seeded.")

    # --- bookkeeping ------------------------------------------------------
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp at which this record was created.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description=(
            "Known limitations affecting interpretation of THIS artifact. "
            "AGENTS.md §17 makes limitations part of the product, not an afterthought."
        ),
    )
    notes: str = ""

    # --- mandatory attribution (AGENTS.md §18) ----------------------------
    attribution: List[str] = Field(
        default_factory=lambda: list(REQUIRED_ATTRIBUTIONS),
        description=(
            "Required challenge attributions, attached automatically. Extra "
            "entries (e.g. training-dataset citations) may be appended; the "
            "required strings cannot be removed."
        ),
    )

    @field_validator("attribution")
    @classmethod
    def _required_attributions_are_always_present(cls, v: List[str]) -> List[str]:
        """Re-insert any required attribution a caller omitted.

        This injects rather than raises on purpose. The goal is that an artifact
        is *incapable* of being produced without attribution, including when a
        caller passes a partial list in order to append a dataset citation.
        Raising would make the common case (append one extra line) require
        re-listing all three, which invites copy-paste drift.

        Required strings come first, in ``REQUIRED_ATTRIBUTIONS`` order, so
        rendered output is deterministic. Caller-supplied extras keep their
        relative order. Duplicates are collapsed.
        """
        extras = [s for s in v if s not in REQUIRED_ATTRIBUTIONS]
        deduped: List[str] = []
        for item in extras:
            if item not in deduped:
                deduped.append(item)
        return list(REQUIRED_ATTRIBUTIONS) + deduped

    @field_validator("production_inputs")
    @classmethod
    def _no_validation_only_source_as_input(cls, v: List[ProductionInput]) -> List[ProductionInput]:
        """Reject any validation-only source recorded as a production input.

        ``ProductionInput`` is already a closed enum, so a correctly typed call
        cannot reach here with a forbidden member. This guard catches the case
        where a raw string bypasses typing, which is exactly how such a defect
        would realistically be introduced.
        """
        forbidden = {s.value for s in ValidationOnlySource}
        for item in v:
            raw = item.value if isinstance(item, ProductionInput) else str(item)
            if raw.strip().lower() in forbidden:
                raise ValueError(
                    f"{raw!r} is a validation-only source and must never be recorded as a "
                    "production input (AGENTS.md §3, architecture.md §18)."
                )
        return v

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain serializable dict, enums reduced to their string values."""
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactProvenance":
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: str) -> "ArtifactProvenance":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> "ArtifactProvenance":
        return cls.from_dict(yaml.safe_load(text))


__all__ = [
    "ArtifactProvenance",
    "ArtifactType",
    "BeforeAfterPair",
    "Maybe",
    "ProductionInput",
    "SceneReference",
    "Unknown",
    "UnknownReason",
    "ValidationOnlySource",
]
