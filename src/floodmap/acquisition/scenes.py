"""Typed records for discovered satellite scenes.

Why Sentinel-1 and Sentinel-2 are separate types
------------------------------------------------
``docs/data-contract.md`` §1.1 and §2.1 specify different REQUIRED fields per
sensor. A single unified scene model would make it syntactically legal to ask
a SAR scene for its cloud percentage, or an optical scene for its polarisation
channels, and the resulting ``None`` would be indistinguishable from "the
provider did not report it". Separate types make those questions impossible to
ask instead of merely wrong to answer.

Fields the catalogue cannot supply
----------------------------------
Three fields that the data contract marks REQUIRED are **not obtainable at
discovery time**. They are recorded as :class:`~floodmap.utils.provenance.Unknown`
with a reason, never defaulted or omitted:

* ``incidence_angle`` (§1.1) — not a catalogue attribute. It must come from
  product metadata once a product is read, i.e. in preprocessing.
* ``aoi_cloud_percent`` (§2.1) — cannot be computed from metadata at all. It
  needs the cloud mask pixels clipped to the AOI, which is preprocessing work.
  The provider's scene-level figure is *not* a substitute: §2.1 notes the two
  "can differ greatly" and that the AOI figure "is the figure that actually
  matters".
* ``platform_unit`` (§1.1, e.g. S1A/S1B) — the catalogue reports
  ``platformShortName`` as ``"SENTINEL-1"``, the mission and not the
  individual satellite. The unit is derived from the product name prefix where
  that name follows the documented convention, and is ``Unknown`` otherwise.

Recording these as ``Unknown`` rather than ``None`` matters because ``None``
means only "not recorded", while these are cases where we looked and the value
is genuinely unavailable from this source.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.provenance import ProductionInput, SceneReference, Unknown, UnknownReason

__all__ = [
    "DiscoveredScene",
    "SceneFootprint",
    "Sentinel1Scene",
    "Sentinel2Scene",
    "SensorKind",
    "derive_platform_unit",
    "parse_polarisations",
]

#: Separator the catalogue uses inside ``polarisationChannels`` (e.g. "VV&VH").
#: Verified against live responses.
_POLARISATION_SEPARATOR = "&"

#: Product names begin with the satellite unit, e.g. "S1A_IW_GRDH_1SDV_...",
#: "S2B_MSIL2A_...". Deliberately strict: anything that does not match this
#: exact shape yields Unknown rather than a partial guess.
_PLATFORM_UNIT_PATTERN = re.compile(r"^(S1[A-D]|S2[A-D])_")

#: Reason used for every field the catalogue genuinely cannot provide.
_NOT_FROM_CATALOGUE = UnknownReason.NOT_AVAILABLE_FROM_SOURCE


def catalogue_unknown(field: str) -> Unknown:
    """Record an attribute that the catalogue did not represent."""
    return Unknown(reason=_NOT_FROM_CATALOGUE, note=f"Catalogue did not report {field}.")


class SensorKind(str, Enum):
    """Which permitted production sensor a scene came from."""

    SENTINEL1 = "sentinel-1"
    SENTINEL2 = "sentinel-2"

    @property
    def production_input(self) -> ProductionInput:
        """The matching closed-enum production source, for provenance."""
        return (
            ProductionInput.SENTINEL1 if self is SensorKind.SENTINEL1 else ProductionInput.SENTINEL2
        )


def parse_polarisations(raw: Optional[str]) -> Tuple[str, ...]:
    """Split a ``polarisationChannels`` value into individual channels.

    The catalogue returns ``"VV&VH"``. Returns an empty tuple when the
    attribute is absent, which the caller distinguishes from a successfully
    parsed empty value.
    """
    if not raw or not raw.strip():
        return ()
    parts = (part.strip().upper() for part in raw.split(_POLARISATION_SEPARATOR))
    return tuple(part for part in parts if part)


def derive_platform_unit(product_name: str) -> Optional[str]:
    """Extract the satellite unit (S1A, S2B, ...) from a product name.

    The catalogue's ``platformShortName`` is the mission (``"SENTINEL-1"``),
    not the unit, so the unit is read from the product name prefix, which
    follows a documented and stable convention.

    Returns ``None`` when the name does not match the expected shape. This is
    a parse, not an inference: no attempt is made to guess the unit from orbit
    numbers or dates.
    """
    match = _PLATFORM_UNIT_PATTERN.match(product_name.strip())
    return match.group(1) if match else None


class SceneFootprint(BaseModel):
    """Spatial footprint of a scene, as reported by the provider.

    Stored as a GeoJSON-style geometry. Kept because the footprint determines
    whether the AOI is fully or only partially covered, which is a limitation
    that must reach the report: a scene that clips the AOI gives a flood extent
    that is bounded by the image edge rather than by the flood.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(description="GeoJSON geometry type, e.g. Polygon or MultiPolygon.")
    coordinates: Any = Field(description="GeoJSON coordinate array, as provided.")

    @field_validator("type")
    @classmethod
    def _recognised_geometry_type(cls, v: str) -> str:
        allowed = {"Polygon", "MultiPolygon"}
        if v not in allowed:
            raise ValueError(f"footprint type must be one of {sorted(allowed)}, got {v!r}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "coordinates": self.coordinates}


class DiscoveredScene(BaseModel):
    """Fields common to every discovered scene, whichever sensor.

    Not instantiated directly; see :class:`Sentinel1Scene` and
    :class:`Sentinel2Scene`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- identity ---------------------------------------------------------
    scene_id: str = Field(
        min_length=1,
        description="Provider granule name, e.g. S1A_IW_GRDH_1SDV_20240304T122236_...",
    )
    product_id: str = Field(
        min_length=1,
        description="Provider-internal product UUID. Needed to build a download URL.",
    )

    # --- acquisition ------------------------------------------------------
    acquired_at: datetime = Field(
        description="Acquisition start, timezone-aware UTC. Required for temporal reasoning."
    )
    product_type: str | Unknown = Field(
        description="Provider product type, e.g. IW_GRDH_1S or S2MSI2A."
    )
    platform_short_name: str | Unknown = Field(
        description=(
            "Mission as reported by the provider, e.g. SENTINEL-1. Explicitly "
            "Unknown when the catalogue response does not expose it."
        )
    )
    platform_unit: str | Unknown = Field(
        description=(
            "Individual satellite (S1A, S2B, ...), derived from the product name. "
            "Unknown when the name does not match the documented convention."
        )
    )

    # --- spatial ----------------------------------------------------------
    footprint: SceneFootprint | Unknown | None = Field(
        default=None, description="Scene footprint as reported by the provider."
    )

    # --- availability -----------------------------------------------------
    online: bool | Unknown | None = Field(
        default=None,
        description=(
            "Whether the provider holds the product in immediately retrievable "
            "storage. An offline product needs a staging request before download, "
            "so this affects feasibility, not suitability."
        ),
    )
    content_length_bytes: int | Unknown | None = Field(
        default=None, description="Product size as reported by the provider."
    )

    # --- traceability -----------------------------------------------------
    provider: str = Field(description="Identifier of the provider adapter that returned this.")
    reference_url: Optional[str] = Field(
        default=None,
        description=(
            "Provider URL for this product's metadata or download endpoint. "
            "Recorded for traceability; following it is a separate, explicit step."
        ),
    )

    @field_validator("acquired_at")
    @classmethod
    def _acquired_at_is_aware(cls, v: datetime) -> datetime:
        """Reject a naive acquisition timestamp.

        ``AGENTS.md`` §4 requires timestamps be preserved. A naive value cannot
        be reliably placed relative to the event instant, and assuming UTC
        could move the scene across the event boundary.
        """
        if v.tzinfo is None:
            raise ValueError(
                "acquired_at must be timezone-aware; a naive timestamp cannot be "
                "reliably placed before or after the event (AGENTS.md §4)"
            )
        return v

    @property
    def sensor(self) -> SensorKind:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError

    def _base_dict(self) -> Dict[str, Any]:
        data = self.model_dump(mode="json")
        data["sensor"] = self.sensor.value
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form for the manifest."""
        return self._base_dict()


class Sentinel1Scene(DiscoveredScene):
    """A discovered Sentinel-1 scene.

    Carries the orbit metadata that makes the same-track requirement in
    ``AGENTS.md`` §4 *verifiable* rather than assumed. ``relative_orbit`` is
    ``None`` when the provider did not report it, and selection rejects such a
    candidate explicitly rather than treating it as compatible.
    """

    relative_orbit: int | Unknown | None = Field(
        default=None,
        description=(
            "Relative orbit / track number. None means the provider did not "
            "report it, in which case same-track compatibility cannot be verified."
        ),
    )
    orbit_direction: str | Unknown | None = Field(
        default=None, description="ASCENDING or DESCENDING, where reported."
    )
    absolute_orbit: int | Unknown | None = Field(
        default=None, description="Absolute orbit number, where reported."
    )
    operational_mode: str | Unknown | None = Field(
        default=None, description="Acquisition mode, e.g. IW."
    )
    swath_identifier: str | Unknown | None = Field(
        default=None, description="Swath, where reported."
    )
    polarisations: Tuple[str, ...] | Unknown = Field(
        default=(),
        description="Polarisation channels, parsed from the provider value (e.g. VV&VH).",
    )
    incidence_angle: float | Unknown = Field(
        default_factory=lambda: Unknown(
            reason=_NOT_FROM_CATALOGUE,
            note=(
                "Incidence angle is not a catalogue attribute. docs/data-contract.md "
                "§1.1 lists it as REQUIRED for reasoning about geometric "
                "comparability, so it must be read from product metadata during "
                "preprocessing rather than assumed here."
            ),
        ),
        description="Incidence angle. Unknown at discovery; see class docstring.",
    )

    @field_validator("orbit_direction")
    @classmethod
    def _normalise_orbit_direction(cls, v: str | Unknown | None) -> str | Unknown | None:
        """Uppercase a reported direction, and reject an unrecognised value.

        Rejecting is right here: an unrecognised direction means the record is
        not what the parser expects, and silently keeping it would let it reach
        the orbit-direction compatibility check where it could never match.
        """
        if not isinstance(v, str):
            return v
        normalised = v.strip().upper()
        if not normalised:
            return None
        if normalised not in {"ASCENDING", "DESCENDING"}:
            raise ValueError(
                f"orbit_direction must be ASCENDING or DESCENDING, got {v!r}; an "
                "unrecognised value suggests the provider response was misparsed"
            )
        return normalised

    @property
    def sensor(self) -> SensorKind:
        return SensorKind.SENTINEL1

    @property
    def orbit_is_known(self) -> bool:
        """Whether same-track compatibility can be verified for this scene."""
        return isinstance(self.relative_orbit, int)

    def to_scene_reference(self) -> SceneReference:
        """Adapt to the provenance-layer scene record.

        The provenance :class:`~floodmap.utils.provenance.SceneReference` is
        deliberately a summary mirroring ``architecture.md`` §12. Full scene
        detail stays in the acquisition manifest, so there is exactly one rich
        record and one provenance record, not two competing ones.
        """
        return SceneReference(
            scene_id=self.scene_id,
            acquired_at=self.acquired_at.isoformat(),
            relative_orbit=self.relative_orbit,
            orbit_direction=self.orbit_direction,
            processing_level=self.product_type,
            platform=self.platform_short_name,
            product_type=self.product_type,
            source=ProductionInput.SENTINEL1,
        )


class Sentinel2Scene(DiscoveredScene):
    """A discovered Sentinel-2 scene.

    ``scene_cloud_percent`` is the provider's scene-level figure. It is **not**
    evidence that the AOI is clear: ``docs/data-contract.md`` §2.1 records that
    the two can differ greatly and that the AOI figure is the one that matters.
    Per-pixel cloud masking is preprocessing work, so ``aoi_cloud_percent`` is
    ``Unknown`` here by construction.
    """

    processing_level: str | Unknown | None = Field(
        default=None, description="Provider processing level, e.g. L2A."
    )
    tile_id: str | Unknown | None = Field(default=None, description="MGRS tile, where reported.")
    relative_orbit: int | Unknown | None = Field(
        default=None, description="Relative orbit, where reported."
    )
    scene_cloud_percent: float | Unknown | None = Field(
        default=None,
        description=(
            "Provider-reported scene-level cloud fraction. Metadata only: it is "
            "not evidence that the AOI itself is cloud-free."
        ),
    )
    aoi_cloud_percent: float | Unknown = Field(
        default_factory=lambda: Unknown(
            reason=_NOT_FROM_CATALOGUE,
            note=(
                "Cloud fraction within the AOI cannot be computed from catalogue "
                "metadata; it requires the cloud mask clipped to the AOI, which is "
                "preprocessing work. docs/data-contract.md §2.1 marks this REQUIRED "
                "and notes it can differ greatly from the scene-level figure."
            ),
        ),
        description="Cloud fraction inside the AOI. Unknown at discovery.",
    )

    @field_validator("scene_cloud_percent")
    @classmethod
    def _cloud_percent_in_range(cls, v: float | Unknown | None) -> float | Unknown | None:
        if not isinstance(v, (int, float)):
            return v
        if not 0.0 <= v <= 100.0:
            raise ValueError(
                f"scene_cloud_percent must be a percentage in [0, 100], got {v}; "
                "an out-of-range value suggests the provider response was misparsed"
            )
        return float(v)

    @property
    def sensor(self) -> SensorKind:
        return SensorKind.SENTINEL2

    @property
    def cloud_is_known(self) -> bool:
        """Whether a cloud-aware ranking can be applied to this scene."""
        return isinstance(self.scene_cloud_percent, (int, float))

    def to_scene_reference(self) -> SceneReference:
        """Adapt to the provenance-layer scene record. See S1 counterpart."""
        return SceneReference(
            scene_id=self.scene_id,
            acquired_at=self.acquired_at.isoformat(),
            relative_orbit=self.relative_orbit,
            orbit_direction=None,
            processing_level=self.processing_level or self.product_type,
            platform=self.platform_short_name,
            product_type=self.product_type,
            source=ProductionInput.SENTINEL2,
        )
