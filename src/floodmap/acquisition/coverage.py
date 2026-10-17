"""Planar coverage of an area of interest by a scene footprint.

Why this module exists
----------------------
Discovery returns candidates whose footprints are *claimed* to intersect the
AOI, because the catalogue filter is an intersection test. An intersection test
is not a coverage test: it is satisfied by a scene that clips one corner of the
study area, and a flood extent read from such a scene is bounded by the image
edge rather than by the flood. Selection therefore has to measure coverage
itself, and it has to do so with geometry rather than by trusting the query.

That trust is not merely theoretically misplaced. ``tests`` exercised the
previous implementation with a footprint on the other side of the planet and it
returned a usable pair, because the check was a constant. The policy here is
the opposite: **an unverifiable footprint is a rejection, not an assumption.**

``AGENTS.md`` §7 requires CRS, transform, resolution, extent and alignment to be
verified before pixel-level comparison. This module is that verification for the
acquisition stage, at the granularity the catalogue can support (geometry, not
pixels).

Measure
-------
Area is computed by scaling the AOI and the footprint into a local plane about
the AOI centroid latitude — the equirectangular approximation
``(lon * cos(lat0), lat)``. Both geometries are scaled by the same factor, so
the *ratio* of areas is a locally correct metric ratio rather than a ratio of
degree-squared areas, which is not an area at all.

The approximation is honest and bounded, not exact: scaling by ``cos(lat0)``
ignores the change in meridional convergence across the AOI. Over the Trishuli
corridor (``Δlat`` ≈ 0.44°, centred at 28.0°N) the residual error on the ratio
is under ~0.3%; it grows with ``Δlat² · tan(lat0)``, so an AOI spanning more
than about a degree, or one at very high latitude, must be measured in a
projected CRS instead. A non-finite or near-zero scale (a polar AOI) is
reported as unverifiable rather than silently producing nonsense.

Why shapely
-----------
The previous milestone note in :mod:`floodmap.acquisition.aoi` says ring
self-intersection is not checked because "a geometry library ... this milestone
does not pull in". This milestone does need one: deciding whether a polygon
covers another polygon is not something to reimplement. ``shapely`` is used and
is a declared runtime dependency (``pyproject.toml``).

Deliberately NOT done
---------------------
* Invalid geometry is **not** repaired with ``make_valid``. A self-intersecting
  ring means the footprint is not what the provider claims it is, and repairing
  it would silently substitute our guess for their record. It is rejected with a
  reason instead.
* Holes in the AOI *are* honoured here (a hole is not part of the study area),
  even though the provider query transmits only the exterior ring. The query
  dropping a hole can only widen the candidate set; ignoring a hole when
  measuring coverage would overstate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from shapely import affinity
from shapely.errors import ShapelyError
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..utils.provenance import Unknown
from .aoi import AreaOfInterest, BoundingBox, PolygonGeometry
from .outcomes import RejectionReason
from .scenes import SceneFootprint

__all__ = [
    "AOI_CRS",
    "FootprintCoverage",
    "FootprintStatus",
    "evaluate_footprint_coverage",
]

#: The only CRS an AOI and a catalogue footprint may be expressed in, because
#: the provider spatial filter is defined in it and the AOI model pins it.
AOI_CRS = "EPSG:4326"

#: Coverage is "full" when the uncovered fraction is within this relative
#: tolerance. Not a scientific tolerance: it absorbs the last bits of
#: floating-point noise in a coordinate magnitude of ~85 degrees, where a
#: double carries roughly 1e-14 of absolute resolution.
_FULL_COVERAGE_RELATIVE_TOLERANCE = 1e-9


class FootprintStatus(str, Enum):
    """How much of the AOI a footprint covers."""

    #: The footprint contains the AOI.
    FULL = "full_coverage"

    #: The footprint intersects the AOI but does not contain it.
    PARTIAL = "partial_coverage"

    #: The footprint does not intersect the AOI at all.
    NONE = "no_coverage"

    #: Coverage could not be established: absent, unknown, malformed, invalid,
    #: or expressed in an unsupported CRS. Never treated as acceptable.
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class FootprintCoverage:
    """The coverage verdict for one candidate, with its evidence and reason.

    ``accepted`` is the gate decision and ``reason`` explains a rejection, so a
    discarded candidate is auditable without re-running the computation.
    """

    status: FootprintStatus
    accepted: bool
    coverage_fraction: Optional[float]
    detail: str
    reason: Optional[RejectionReason]
    #: The measured coverage fraction required for acceptance, when one was set.
    minimum_fraction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "accepted": self.accepted,
            "coverage_fraction": (
                None if self.coverage_fraction is None else round(self.coverage_fraction, 6)
            ),
            "minimum_fraction": self.minimum_fraction,
            "reason": None if self.reason is None else self.reason.value,
            "detail": self.detail,
        }


def _unverifiable(detail: str) -> FootprintCoverage:
    return FootprintCoverage(
        status=FootprintStatus.UNVERIFIABLE,
        accepted=False,
        coverage_fraction=None,
        detail=detail,
        reason=RejectionReason.FOOTPRINT_UNVERIFIABLE,
    )


def _aoi_polygon(aoi: AreaOfInterest) -> BaseGeometry:
    """Build the AOI geometry, honouring holes.

    Raises
    ------
    ValueError
        When the AOI itself is degenerate (zero area). A degenerate AOI is a
        caller error, not a property of any candidate, so it fails loudly
        instead of turning into "no usable scenes", which would look like a
        data-availability finding.
    """
    geometry = aoi.geometry
    if isinstance(geometry, BoundingBox):
        polygon = Polygon(geometry.to_ring())
    elif isinstance(geometry, PolygonGeometry):
        polygon = Polygon(
            shell=list(geometry.exterior),
            holes=[list(ring) for ring in geometry.holes],
        )
    else:  # pragma: no cover - AreaOfInterest constrains the union
        raise ValueError(f"unsupported AOI geometry type {type(geometry).__name__}")

    if polygon.is_empty:
        raise ValueError(
            "the AOI geometry is empty, so coverage cannot be measured; fix the AOI "
            "rather than treating every candidate as non-covering"
        )
    # Checked before validity so that a collinear ring is diagnosed as a
    # zero-area extent rather than as an unreadable topology error.
    if polygon.area <= 0.0:
        raise ValueError(
            "the AOI geometry encloses zero area (collinear ring?), so a coverage "
            "fraction is undefined"
        )
    if not polygon.is_valid:
        raise ValueError(
            "the AOI geometry is invalid (self-intersecting ring), so coverage cannot be "
            "measured; fix the geometry rather than treating every candidate as "
            "non-covering"
        )
    return polygon


def _local_plane(geometry: BaseGeometry, latitude: float) -> BaseGeometry:
    """Scale longitudes by ``cos(latitude)`` so areas are locally metric.

    ``affinity.scale`` about the origin is exactly the transform described in the
    module docstring: ``lon -> lon * cos(lat0)``, ``lat`` unchanged. Expressed
    this way rather than as a coordinate callback so the intent is readable and
    no deprecated transform helper is involved.
    """
    scale = math.cos(math.radians(latitude))
    return affinity.scale(geometry, xfact=scale, yfact=1.0, origin=(0.0, 0.0))


def _footprint_part(ring_or_rings: Any) -> BaseGeometry:
    """Parse one polygon of a provider footprint, or raise for a bad record."""
    polygon = shape({"type": "Polygon", "coordinates": ring_or_rings})
    if polygon.is_empty:
        raise ValueError("footprint part is empty")
    if polygon.area <= 0.0:
        raise ValueError("footprint part encloses zero area")
    if not polygon.is_valid:
        raise ValueError(
            "footprint geometry is invalid (self-intersecting or degenerate ring); it is "
            "not repaired here because a repair would substitute a guess for the "
            "provider's record"
        )
    return polygon


def _footprint_polygon(footprint: SceneFootprint) -> BaseGeometry:
    """Parse a provider footprint into geometry, or raise for a bad record.

    A ``MultiPolygon`` is assembled from parts that are each validated on their
    own, then combined. That is not a repair of bad geometry: GEOS reports a
    ``MultiPolygon`` whose parts meet along an edge as *invalid* under the OGC
    ring rules, even though its meaning — "the union of these parts" — is
    unambiguous, and providers do emit footprints split exactly that way. A part
    that is itself self-intersecting, empty or zero-area is still rejected,
    because there the record really is untrustworthy.
    """
    coordinates = footprint.coordinates
    if coordinates is None:
        raise ValueError("footprint has no coordinates")
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) == 0:
        raise ValueError(f"footprint coordinates must be a non-empty array, got {coordinates!r}")
    if footprint.type == "MultiPolygon":
        parts = [_footprint_part(part) for part in coordinates]
        return unary_union(parts)
    return _footprint_part(coordinates)


def evaluate_footprint_coverage(
    footprint: Any,
    aoi: AreaOfInterest,
    *,
    minimum_coverage: Optional[float] = None,
) -> FootprintCoverage:
    """Measure how much of ``aoi`` the candidate's footprint covers.

    Parameters
    ----------
    footprint
        A :class:`~floodmap.acquisition.scenes.SceneFootprint`, ``None``, or an
        :class:`~floodmap.utils.provenance.Unknown` recorded by the provider.
    aoi
        The caller-supplied area of interest. Its CRS must be
        :data:`AOI_CRS`.
    minimum_coverage
        Optional required coverage fraction in ``(0, 1]``. When ``None``, any
        positive intersection is accepted and the measured fraction is recorded
        without being thresholded — no threshold is invented.

    Returns
    -------
    FootprintCoverage
        Never raises for a bad candidate. A malformed, unknown or
        unsupported-CRS footprint yields ``UNVERIFIABLE`` with ``accepted=False``.
    """
    if minimum_coverage is not None and not 0.0 < minimum_coverage <= 1.0:
        raise ValueError(
            f"minimum_coverage must be in (0, 1], got {minimum_coverage}; a value of 0 "
            "would accept a scene that does not intersect the AOI at all"
        )

    if footprint is None or isinstance(footprint, Unknown):
        return _unverifiable(
            "The provider did not report a usable footprint for this product, so the AOI "
            "coverage it achieves cannot be established. It is rejected rather than "
            "assumed to cover the AOI."
        )
    if not isinstance(footprint, SceneFootprint):
        return _unverifiable(
            f"footprint is a {type(footprint).__name__}, not a parsed geometry record"
        )
    if str(footprint.crs).strip().upper() != AOI_CRS:
        return _unverifiable(
            f"footprint CRS is {footprint.crs!r} but the AOI is {AOI_CRS}; comparing them "
            "without an explicit, documented reprojection would misinterpret the "
            "coordinates (AGENTS.md §7)"
        )

    try:
        footprint_polygon = _footprint_polygon(footprint)
    except (ShapelyError, ValueError, TypeError, IndexError, KeyError) as exc:
        return _unverifiable(f"footprint geometry could not be parsed: {exc}")

    aoi_polygon = _aoi_polygon(aoi)

    latitude = aoi_polygon.centroid.y
    if abs(latitude) > 89.0:
        return _unverifiable(
            f"the AOI centroid is at latitude {latitude:.4f}, where a local planar area "
            "measure is not meaningful; measure coverage in a projected CRS instead"
        )

    try:
        aoi_local = _local_plane(aoi_polygon, latitude)
        footprint_local = _local_plane(footprint_polygon, latitude)
        if aoi_local.area <= 0.0:
            return _unverifiable("the AOI has zero planar area after scaling")
        intersection = aoi_local.intersection(footprint_local)
        fraction = float(intersection.area / aoi_local.area)
    except (ShapelyError, ValueError, ZeroDivisionError) as exc:
        return _unverifiable(f"coverage could not be computed: {exc}")

    # Clamp: a numerical overshoot of 1e-12 must not read as "more than all of it".
    fraction = min(1.0, max(0.0, fraction))

    if fraction <= 0.0:
        return FootprintCoverage(
            status=FootprintStatus.NONE,
            accepted=False,
            coverage_fraction=0.0,
            minimum_fraction=minimum_coverage,
            reason=RejectionReason.FOOTPRINT_DOES_NOT_INTERSECT_AOI,
            detail="The footprint does not intersect the AOI at all.",
        )

    is_full = fraction >= 1.0 - _FULL_COVERAGE_RELATIVE_TOLERANCE
    status = FootprintStatus.FULL if is_full else FootprintStatus.PARTIAL
    shape_text = "fully covers" if is_full else "only partially covers"

    if minimum_coverage is not None and fraction < minimum_coverage:
        return FootprintCoverage(
            status=status,
            accepted=False,
            coverage_fraction=fraction,
            minimum_fraction=minimum_coverage,
            reason=RejectionReason.FOOTPRINT_COVERAGE_BELOW_MINIMUM,
            detail=(
                f"The footprint {shape_text} the AOI: measured coverage is "
                f"{fraction:.4%}, below the configured minimum of {minimum_coverage:.4%}. A "
                "flood extent read from a partially covering scene is bounded by the image "
                "edge rather than by the flood."
            ),
        )

    return FootprintCoverage(
        status=status,
        accepted=True,
        coverage_fraction=fraction,
        minimum_fraction=minimum_coverage,
        reason=None,
        detail=(
            f"The footprint {shape_text} the AOI: measured coverage is {fraction:.4%}"
            + (
                ""
                if minimum_coverage is None
                else f", at or above the configured minimum of {minimum_coverage:.4%}"
            )
            + "."
        ),
    )
