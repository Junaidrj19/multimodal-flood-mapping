"""Area-of-interest geometry for acquisition queries.

The central rule of this module
-------------------------------
**There is no default AOI, and this module contains no coordinates.**

The Trishuli case-study geometry is not fixed anywhere in this repository. The
event date, OSM snapshot and DEM source are settled by the challenge
specification; the AOI extent is not. Hard-coding a plausible bounding box for
a Nepali river corridor would manufacture a project parameter that nobody
chose, and every downstream exposure count and connectivity result would
silently inherit it.

So :class:`AreaOfInterest` is always constructed by the caller, carries a
mandatory ``aoi_id`` for provenance, and the acquisition configuration ships
with ``aoi: null``. Tests use an obviously synthetic geometry defined under
``tests/``, never here, so a test fixture cannot leak into production as if it
were the real study area.

CRS scope
---------
Only ``EPSG:4326`` is accepted. The provider's spatial filter takes
``geography'SRID=4326;...'``, so passing coordinates in any other reference
system would be silently misinterpreted as degrees. Reprojection is a
preprocessing concern (``AGENTS.md`` §7 requires the method and reason to be
documented) and is deliberately not done implicitly here.

What is NOT validated
---------------------
Ring self-intersection is **not** checked. Doing so properly requires a
geometry library, which this milestone does not pull in. A self-intersecting
polygon would be rejected by the provider rather than silently accepted, but
this module does not claim to have validated topology — see
:attr:`AreaOfInterest.validation_limitations`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AreaOfInterest",
    "BoundingBox",
    "PolygonGeometry",
    "SUPPORTED_AOI_CRS",
]

#: The only CRS accepted, because the provider filter is defined in it.
SUPPORTED_AOI_CRS = "EPSG:4326"

#: Minimum coordinate count for a closed linear ring (3 distinct + repeat).
_MIN_RING_POINTS = 4

#: Degrees. Guards against a degenerate extent that would match nothing or
#: everything. Not a scientific minimum mappable area — purely a sanity bound.
_MIN_EXTENT_DEGREES = 1e-9


def _format_coordinate(value: float) -> str:
    """Render one coordinate for WKT without scientific notation or padding.

    ``repr`` of a float can emit ``1e-05``, which the provider's WKT parser
    does not accept. ``%.10f`` keeps roughly millimetre precision at the
    equator, far finer than any product this project uses.
    """
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


class BoundingBox(BaseModel):
    """An axis-aligned extent in EPSG:4326 decimal degrees."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_lon: float = Field(description="Western bound, decimal degrees.")
    min_lat: float = Field(description="Southern bound, decimal degrees.")
    max_lon: float = Field(description="Eastern bound, decimal degrees.")
    max_lat: float = Field(description="Northern bound, decimal degrees.")

    @field_validator("min_lon", "max_lon")
    @classmethod
    def _longitude_in_range(cls, v: float) -> float:
        if not -180.0 <= v <= 180.0:
            raise ValueError(f"longitude must be within [-180, 180], got {v}")
        return v

    @field_validator("min_lat", "max_lat")
    @classmethod
    def _latitude_in_range(cls, v: float) -> float:
        if not -90.0 <= v <= 90.0:
            raise ValueError(f"latitude must be within [-90, 90], got {v}")
        return v

    @model_validator(mode="after")
    def _bounds_ordered_and_non_degenerate(self) -> "BoundingBox":
        """Reject inverted or zero-area extents.

        An inverted box is the classic consequence of swapping lat/lon at a
        call site. Caught here it is a loud error; uncaught it becomes an
        empty discovery result that looks like "no imagery available".
        """
        if self.max_lon - self.min_lon < _MIN_EXTENT_DEGREES:
            raise ValueError(
                f"max_lon must exceed min_lon (got min_lon={self.min_lon}, "
                f"max_lon={self.max_lon}); a swapped or zero-width extent would "
                "silently match no scenes"
            )
        if self.max_lat - self.min_lat < _MIN_EXTENT_DEGREES:
            raise ValueError(
                f"max_lat must exceed min_lat (got min_lat={self.min_lat}, "
                f"max_lat={self.max_lat}); a swapped or zero-height extent would "
                "silently match no scenes"
            )
        return self

    def to_ring(self) -> List[Tuple[float, float]]:
        """Closed counter-clockwise ring, first point repeated last."""
        return [
            (self.min_lon, self.min_lat),
            (self.max_lon, self.min_lat),
            (self.max_lon, self.max_lat),
            (self.min_lon, self.max_lat),
            (self.min_lon, self.min_lat),
        ]

    def to_wkt(self) -> str:
        """WKT ``POLYGON`` for the provider spatial filter."""
        points = ", ".join(
            f"{_format_coordinate(lon)} {_format_coordinate(lat)}" for lon, lat in self.to_ring()
        )
        return f"POLYGON(({points}))"

    def as_list(self) -> List[float]:
        """``[min_lon, min_lat, max_lon, max_lat]`` ordering."""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


class PolygonGeometry(BaseModel):
    """A polygon in EPSG:4326, given as a GeoJSON-style coordinate ring.

    Only the exterior ring is used for the provider query. Holes are accepted
    in the input for fidelity but are **not** transmitted: the spatial filter
    is an intersection test, and ignoring a hole can only widen the candidate
    set, never hide a scene. Narrowing is left to preprocessing, which has the
    pixels to do it correctly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exterior: Tuple[Tuple[float, float], ...] = Field(
        description="Closed exterior ring as (lon, lat) pairs, first repeated last."
    )
    holes: Tuple[Tuple[Tuple[float, float], ...], ...] = Field(
        default=(),
        description=(
            "Interior rings, retained for the record but not sent to the "
            "provider. See class docstring."
        ),
    )

    @field_validator("exterior")
    @classmethod
    def _exterior_is_a_valid_ring(
        cls, v: Tuple[Tuple[float, float], ...]
    ) -> Tuple[Tuple[float, float], ...]:
        if len(v) < _MIN_RING_POINTS:
            raise ValueError(
                f"a polygon ring needs at least {_MIN_RING_POINTS} points "
                f"(3 distinct plus the closing repeat), got {len(v)}"
            )
        for lon, lat in v:
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"longitude must be within [-180, 180], got {lon}")
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"latitude must be within [-90, 90], got {lat}")
        if v[0] != v[-1]:
            raise ValueError(
                f"ring must be closed: first point {v[0]} != last point {v[-1]}. "
                "Closing it automatically would hide a malformed geometry."
            )
        if len({point for point in v}) < 3:
            raise ValueError("ring must enclose an area; got fewer than 3 distinct points")
        return v

    def to_wkt(self) -> str:
        """WKT ``POLYGON`` of the exterior ring."""
        points = ", ".join(
            f"{_format_coordinate(lon)} {_format_coordinate(lat)}" for lon, lat in self.exterior
        )
        return f"POLYGON(({points}))"

    def bounds(self) -> BoundingBox:
        """Enclosing bounding box of the exterior ring."""
        lons = [lon for lon, _ in self.exterior]
        lats = [lat for _, lat in self.exterior]
        return BoundingBox(
            min_lon=min(lons), min_lat=min(lats), max_lon=max(lons), max_lat=max(lats)
        )


#: Either supported geometry representation.
AoiGeometry = Union[BoundingBox, PolygonGeometry]


class AreaOfInterest(BaseModel):
    """A named area of interest, supplied by the caller.

    ``aoi_id`` is mandatory and carries into the manifest and provenance. Two
    runs over different extents must not be confusable after the fact, and an
    anonymous geometry in a provenance record is not auditable.

    ``is_synthetic`` marks a geometry that exists only to exercise code. It
    propagates into the manifest so a test or demonstration run can never be
    mistaken for a real case-study result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    aoi_id: str = Field(
        min_length=1,
        description="Stable identifier for this area, recorded in provenance.",
    )
    geometry: AoiGeometry = Field(description="Bounding box or polygon, in EPSG:4326.")
    crs: str = Field(
        default=SUPPORTED_AOI_CRS,
        description=f"Coordinate reference system. Only {SUPPORTED_AOI_CRS} is supported.",
    )
    name: str = Field(default="", description="Optional human-readable label.")
    is_synthetic: bool = Field(
        default=False,
        description=(
            "True for geometry that exists only to exercise code. Propagates "
            "into the manifest so a synthetic run is never mistaken for a real "
            "case-study result."
        ),
    )

    @field_validator("aoi_id")
    @classmethod
    def _aoi_id_is_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("aoi_id must not be blank; it is the provenance key for this area")
        return v

    @field_validator("crs")
    @classmethod
    def _crs_is_supported(cls, v: str) -> str:
        """Reject any CRS other than EPSG:4326.

        Accepting a projected CRS and passing its coordinates into a
        ``SRID=4326`` filter would reinterpret metres as degrees, producing a
        query that is wrong but entirely plausible-looking.
        """
        if v.strip().upper() != SUPPORTED_AOI_CRS:
            raise ValueError(
                f"only {SUPPORTED_AOI_CRS} is supported for AOI geometry, got {v!r}. "
                "The provider spatial filter is defined in EPSG:4326; reproject "
                "explicitly in preprocessing and document the method (AGENTS.md §7)."
            )
        return SUPPORTED_AOI_CRS

    #: Honest statement of what construction did and did not check.
    validation_limitations: Tuple[str, ...] = (
        "Ring self-intersection is not checked; that requires a geometry library "
        "which this milestone does not depend on.",
        "Antimeridian-crossing extents are not handled specially and may be "
        "interpreted unexpectedly by the provider.",
        "No check that the extent is a sensible analysis size; a continent-scale "
        "AOI is accepted and would return an unusable number of candidates.",
    )

    def to_wkt(self) -> str:
        """WKT polygon for the provider spatial filter."""
        return self.geometry.to_wkt()

    def bounds(self) -> BoundingBox:
        """Enclosing bounding box, whichever representation was supplied."""
        if isinstance(self.geometry, BoundingBox):
            return self.geometry
        return self.geometry.bounds()

    def describe(self) -> str:
        """Short one-line description for logs and rationale text."""
        box = self.bounds()
        synthetic = " [SYNTHETIC]" if self.is_synthetic else ""
        return (
            f"{self.aoi_id}{synthetic} bbox=("
            f"{box.min_lon}, {box.min_lat}, {box.max_lon}, {box.max_lat}) {self.crs}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form for the manifest."""
        return self.model_dump(mode="json")

    @classmethod
    def from_bbox(
        cls,
        aoi_id: str,
        bbox: Sequence[float],
        *,
        name: str = "",
        is_synthetic: bool = False,
        crs: str = SUPPORTED_AOI_CRS,
    ) -> "AreaOfInterest":
        """Build from ``[min_lon, min_lat, max_lon, max_lat]``.

        Convenience for configuration and CLI input. Requires exactly four
        values; a shorter or longer sequence is a caller error rather than
        something to pad or truncate.
        """
        if len(bbox) != 4:
            raise ValueError(
                f"bbox must have exactly 4 values [min_lon, min_lat, max_lon, max_lat], "
                f"got {len(bbox)}"
            )
        min_lon, min_lat, max_lon, max_lat = (float(value) for value in bbox)
        return cls(
            aoi_id=aoi_id,
            geometry=BoundingBox(
                min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
            ),
            crs=crs,
            name=name,
            is_synthetic=is_synthetic,
        )

    @classmethod
    def from_geojson_geometry(
        cls,
        aoi_id: str,
        geometry: Dict[str, Any],
        *,
        name: str = "",
        is_synthetic: bool = False,
    ) -> "AreaOfInterest":
        """Build from a GeoJSON ``Polygon`` geometry mapping.

        Only ``Polygon`` is accepted. ``MultiPolygon`` is rejected rather than
        silently reduced to its first part, which would quietly shrink the
        analysis area.
        """
        geom_type = geometry.get("type")
        if geom_type != "Polygon":
            raise ValueError(
                f"only GeoJSON Polygon is supported, got {geom_type!r}. A MultiPolygon "
                "is not reduced automatically because dropping parts would silently "
                "shrink the area of interest."
            )
        rings = geometry.get("coordinates") or []
        if not rings:
            raise ValueError("GeoJSON Polygon has no coordinate rings")
        exterior = tuple((float(lon), float(lat)) for lon, lat in rings[0])
        holes = tuple(tuple((float(lon), float(lat)) for lon, lat in ring) for ring in rings[1:])
        return cls(
            aoi_id=aoi_id,
            geometry=PolygonGeometry(exterior=exterior, holes=holes),
            name=name,
            is_synthetic=is_synthetic,
        )
