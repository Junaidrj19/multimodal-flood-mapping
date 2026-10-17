"""Sentinel-2 optical validity: SCL classes, AOI statistics, validation report.

Why this module exists
----------------------
Sentinel-2 L2A products ship a `SCL` band: a per-pixel classification of what the
atmosphere and surface looked like, which is the documented way to know *why* a
pixel is missing rather than only *that* it is missing. Three states must stay
distinguishable, because they support different claims:

``missing data``
    No observation exists for the pixel at all — outside the granule's imaged
    extent, or flagged ``NO_DATA``. A gap.
``masked``
    An observation exists and is deliberately rejected: cloud, thin cirrus,
    cloud shadow, saturated/defective, snow/ice. "It was cloudy" is a statement
    about the acquisition.
``valid observation``
    A measurement this project is willing to use as evidence.

Collapsing these into one "invalid" count would make a cloudy acquisition
indistinguishable from a granule that does not cover the AOI — the distinction
``docs/evaluation-protocol.md`` and ``AGENTS.md`` §17 require to be reported.

What this module does NOT do
----------------------------
* It does not decide the flood. No threshold, no index, no classification. It
  reports observation availability, which is a *precondition* for any later
  flood claim, not a flood claim.
* It does not interpolate. An invalid pixel stays invalid; filling it from
  another sensor or another date would invent an observation.
* It does not invent the class table. :data:`SCL_CLASSES` is the Sentinel-2
  Level-2A SCL definition; class codes outside it are counted as
  ``unexpected`` and never treated as valid observations, so a change in the
  provider's encoding shows up as an anomaly instead of silently widening the
  valid mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SCL_CLASSES",
    "SCL_REFERENCE",
    "AoiStatistics",
    "BandAvailability",
    "OpticalValidationReport",
    "SclCategory",
    "SclClass",
    "aoi_mask_for_grid",
    "band_availability",
    "classify_scl",
    "observation_mask",
]

#: Where the class table comes from. The mapping is copied from the published
#: Sentinel-2 MSI Level-2A product specification, not inferred from data, and a
#: reviewer is expected to check it against the delivered product's own metadata
#: (`MTD_MSIL2A.xml`, SCL band description) rather than take it on trust
#: (`AGENTS.md` §11: an AI-generated statement is not scientific validation).
SCL_REFERENCE = (
    "Sentinel-2 MSI Level-2A product specification, Scene Classification Layer "
    "(SCL) class table, classes 0-11"
)


class SclCategory(str, Enum):
    """What an SCL class means for whether a pixel is usable evidence."""

    #: Vegetation, bare ground, water, unclassified. Usable observations.
    VALID = "valid"

    #: Opaque cloud (medium or high probability). Masked.
    CLOUD = "cloud"

    #: Thin cirrus in the product's own classification. Masked.
    CIRRUS = "cirrus"

    #: Shadow cast by cloud. Masked: reflectance is depressed by an unknown
    #: amount, which is worse than a gap because it looks like an observation.
    CLOUD_SHADOW = "cloud_shadow"

    #: No observation: unclassified/no-data and saturated-or-defective pixels.
    NO_DATA = "no_data"

    #: Snow or ice. Reported separately rather than silently accepted or
    #: silently rejected: in a monsoon-season Himalayan flood it should be rare,
    #: so a large snow count is a signal about the acquisition date or the AOI.
    SNOW_ICE = "snow_ice"


@dataclass(frozen=True)
class SclClass:
    """One SCL class with its published meaning."""

    code: int
    name: str
    category: SclCategory
    #: Whether a pixel of this class is a usable surface observation. Water is
    #: usable: the flood question is *which* water, not whether water is visible.
    is_observation: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "category": self.category.value,
            "is_observation": self.is_observation,
        }


#: The Sentinel-2 L2A SCL class table (see :data:`SCL_REFERENCE`).
SCL_CLASSES: Mapping[int, SclClass] = {
    0: SclClass(0, "no_data", SclCategory.NO_DATA, False),
    1: SclClass(1, "saturated_or_defective", SclCategory.NO_DATA, False),
    2: SclClass(2, "cast_shadow", SclCategory.CLOUD_SHADOW, False),
    3: SclClass(3, "cloud_shadow", SclCategory.CLOUD_SHADOW, False),
    4: SclClass(4, "vegetation", SclCategory.VALID, True),
    5: SclClass(5, "not_vegetated", SclCategory.VALID, True),
    6: SclClass(6, "water", SclCategory.VALID, True),
    7: SclClass(7, "unclassified", SclCategory.VALID, True),
    8: SclClass(8, "cloud_medium_probability", SclCategory.CLOUD, False),
    9: SclClass(9, "cloud_high_probability", SclCategory.CLOUD, False),
    10: SclClass(10, "thin_cirrus", SclCategory.CIRRUS, False),
    11: SclClass(11, "snow_or_ice", SclCategory.SNOW_ICE, False),
}

#: Categories whose pixels count as masked rather than missing. Kept explicit so
#: "masked" cannot drift into meaning "not valid" by accident.
MASKED_CATEGORIES: Tuple[SclCategory, ...] = (
    SclCategory.CLOUD,
    SclCategory.CIRRUS,
    SclCategory.CLOUD_SHADOW,
    SclCategory.SNOW_ICE,
)


@dataclass(frozen=True)
class SclSummary:
    """Per-category pixel counts for one SCL array."""

    total_pixels: int
    counts: Mapping[str, int]
    unexpected_codes: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pixels": self.total_pixels,
            "counts": dict(self.counts),
            "unexpected_codes": list(self.unexpected_codes),
            "reference": SCL_REFERENCE,
        }


def classify_scl(scl: np.ndarray) -> SclSummary:
    """Count SCL pixels by category.

    Codes outside :data:`SCL_CLASSES` are counted under ``unexpected`` rather
    than being folded into a neighbouring class. An unrecognised code is a
    statement about our class table or about the product, and either way it must
    not silently become a valid observation.
    """
    array = np.asarray(scl)
    if array.ndim != 2:
        raise ValueError(f"SCL must be a 2-D array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"SCL is a categorical band and must be integer-typed, got {array.dtype}; "
            "a float SCL suggests the wrong band was read"
        )
    counts: Dict[str, int] = {category.value: 0 for category in SclCategory}
    counts["unexpected"] = 0
    flat = array.ravel()
    unexpected: set[int] = set()
    for code, value in zip(*np.unique(flat, return_counts=True)):
        entry = SCL_CLASSES.get(int(code))
        if entry is None:
            counts["unexpected"] += int(value)
            unexpected.add(int(code))
            continue
        counts[entry.category.value] += int(value)
    return SclSummary(
        total_pixels=int(flat.size),
        counts=counts,
        unexpected_codes=tuple(sorted(unexpected)),
    )


def observation_mask(scl: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels usable as surface observations.

    Only classes the product specification marks as land/water surfaces qualify;
    everything else — cloud, cirrus, shadow, saturated, no-data, snow — is
    excluded. The caller is free to treat snow/ice differently, but must do so
    explicitly by inspecting :func:`classify_scl` output rather than by getting
    it for free here.
    """
    array = np.asarray(scl)
    if array.ndim != 2:
        raise ValueError(f"SCL must be a 2-D array, got shape {array.shape}")
    mask = np.zeros(array.shape, dtype=bool)
    for code, entry in SCL_CLASSES.items():
        if entry.is_observation:
            mask |= array == code
    return mask


@dataclass(frozen=True)
class BandAvailability:
    """Which required bands the delivered product actually carries."""

    required: Tuple[str, ...]
    present: Tuple[str, ...]
    missing: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": list(self.required),
            "present": list(self.present),
            "missing": list(self.missing),
            "complete": self.complete,
        }


def band_availability(
    present: Sequence[str], required: Sequence[str]
) -> BandAvailability:
    """Compare the band names a product carries against the required set.

    Matching is exact after stripping whitespace: band naming differs between
    providers (`B04` vs `B4` vs `band_4`), and a loose match would hide exactly
    the mismatch this check exists to detect.
    """
    present_clean = tuple(name.strip() for name in present if str(name).strip())
    required_clean = tuple(name.strip() for name in required)
    present_set = set(present_clean)
    return BandAvailability(
        required=required_clean,
        present=present_clean,
        missing=tuple(name for name in required_clean if name not in present_set),
    )


@dataclass(frozen=True)
class AoiStatistics:
    """Pixel accounting over the study area, in the three required states.

    Every fraction is relative to ``aoi_pixels`` — the pixels inside the AOI —
    never to the raster. A statistic over the whole granule would be dominated by
    land nobody asked about, which is the error ``docs/dataset-registry.md``
    §1.6.2 quantifies for the bounding-box case.
    """

    aoi_pixels: int
    valid_observations: int
    masked_cloud: int
    masked_cirrus: int
    masked_cloud_shadow: int
    masked_snow_ice: int
    missing_data: int
    unexpected_class: int
    #: Pixels excluded by a *separate* validity mask (nodata, GDAL mask) even
    #: though the SCL called them observable. Counted as missing, not masked.
    missing_by_raster_mask: int = 0

    @property
    def masked(self) -> int:
        return (
            self.masked_cloud
            + self.masked_cirrus
            + self.masked_cloud_shadow
            + self.masked_snow_ice
        )

    @property
    def missing(self) -> int:
        return self.missing_data + self.missing_by_raster_mask

    def _fraction(self, count: int) -> float:
        if self.aoi_pixels <= 0:
            return 0.0
        return count / self.aoi_pixels

    @property
    def valid_fraction(self) -> float:
        return self._fraction(self.valid_observations)

    @property
    def cloud_fraction(self) -> float:
        return self._fraction(self.masked_cloud)

    @property
    def shadow_fraction(self) -> float:
        return self._fraction(self.masked_cloud_shadow)

    @property
    def masked_fraction(self) -> float:
        return self._fraction(self.masked)

    @property
    def missing_fraction(self) -> float:
        return self._fraction(self.missing)

    @property
    def is_accounted_for(self) -> bool:
        """Every AOI pixel is in exactly one state. A False means a bug."""
        return self.aoi_pixels == (
            self.valid_observations + self.masked + self.missing + self.unexpected_class
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aoi_pixels": self.aoi_pixels,
            "counts": {
                "valid_observations": self.valid_observations,
                "masked_total": self.masked,
                "masked_cloud": self.masked_cloud,
                "masked_cirrus": self.masked_cirrus,
                "masked_cloud_shadow": self.masked_cloud_shadow,
                "masked_snow_ice": self.masked_snow_ice,
                "missing_total": self.missing,
                "missing_no_data_class": self.missing_data,
                "missing_by_raster_mask": self.missing_by_raster_mask,
                "unexpected_class": self.unexpected_class,
            },
            "fractions": {
                "valid_observations": self.valid_fraction,
                "masked": self.masked_fraction,
                "cloud": self.cloud_fraction,
                "cloud_shadow": self.shadow_fraction,
                "missing": self.missing_fraction,
            },
            "fully_accounted_for": self.is_accounted_for,
            "denominator": "pixels inside the AOI",
        }


def aoi_statistics(
    scl: np.ndarray,
    aoi_mask: np.ndarray,
    *,
    raster_valid: Optional[np.ndarray] = None,
    scl_summary: Optional[SclSummary] = None,
) -> AoiStatistics:
    """Reduce an SCL array and a raster validity mask to AOI-level statistics.

    ``aoi_mask`` is the pixel mask of the study area (True inside it) on the same
    grid as ``scl``. ``raster_valid``, when supplied, is the validity mask the
    reader already computed from nodata/GDAL masks; pixels the SCL called
    observable but the raster could not deliver are counted as *missing*.
    """
    classes = np.asarray(scl)
    mask = np.asarray(aoi_mask, dtype=bool)
    if classes.shape != mask.shape:
        raise ValueError(
            f"SCL shape {classes.shape} and AOI mask shape {mask.shape} disagree; "
            "statistics must be computed on one grid"
        )
    if not mask.any():
        raise ValueError(
            "the AOI mask selects no pixels, so no optical statistic can be computed; "
            "an empty AOI is a geometry/grid error, not a data-availability finding"
        )
    if raster_valid is not None:
        valid_raster = np.asarray(raster_valid, dtype=bool)
        if valid_raster.shape != classes.shape:
            raise ValueError(
                f"validity mask shape {valid_raster.shape} does not match SCL shape "
                f"{classes.shape}"
            )
    else:
        valid_raster = np.ones(classes.shape, dtype=bool)

    summary = scl_summary or classify_scl(classes)
    in_aoi = mask
    categories = {category.value: 0 for category in SclCategory}
    unexpected = 0
    for code in np.unique(classes[in_aoi]):
        count = int((in_aoi & (classes == code)).sum())
        entry = SCL_CLASSES.get(int(code))
        if entry is None:
            unexpected += count
            continue
        categories[entry.category.value] += count

    # A pixel counts as an observation only if the SCL says it is observable AND
    # the raster actually carries a value for it. The two failure modes are
    # reported separately: available band vs available observation.
    observable = in_aoi & observation_mask(classes)
    deliverable = observable & valid_raster
    missing_by_raster = int((observable & ~valid_raster).sum())

    return AoiStatistics(
        aoi_pixels=int(in_aoi.sum()),
        valid_observations=int(deliverable.sum()),
        masked_cloud=categories[SclCategory.CLOUD.value],
        masked_cirrus=categories[SclCategory.CIRRUS.value],
        masked_cloud_shadow=categories[SclCategory.CLOUD_SHADOW.value],
        masked_snow_ice=categories[SclCategory.SNOW_ICE.value],
        missing_data=categories[SclCategory.NO_DATA.value],
        unexpected_class=unexpected,
        missing_by_raster_mask=missing_by_raster,
    )


def aoi_mask_for_grid(geometry: Mapping[str, Any], grid: Any) -> np.ndarray:
    """Rasterise a GeoJSON geometry onto the analysis grid.

    ``geometry`` is a GeoJSON geometry mapping (``manifest.aoi["geometry"]``).
    Uses ``rasterio.features.geometry_mask`` so the rasterisation is GDAL's, not
    a hand-rolled point-in-polygon on a projected grid — the same library that
    warps the rasters, which removes a class of half-pixel disagreements between
    the mask and the data it masks.
    """
    from rasterio.features import geometry_mask  # local: rasterio is optional at import time

    shape = {"type": geometry["type"], "coordinates": geometry["coordinates"]}
    return geometry_mask(
        [shape],
        out_shape=(int(grid.height), int(grid.width)),
        transform=grid.transform,
        invert=True,
        all_touched=False,
    )


@dataclass
class OpticalValidationReport:
    """Machine-readable optical validation record for one Sentinel-2 scene."""

    scene_id: str
    temporal_role: str
    scl_summary: Optional[SclSummary]
    statistics: Optional[AoiStatistics]
    bands: Optional[BandAvailability]
    reflectance_scaling: Mapping[str, Any]
    resampling: Mapping[str, Any]
    masking: Mapping[str, Any]
    limitations: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact": "sentinel2_optical_validation",
            "scene_id": self.scene_id,
            "temporal_role": self.temporal_role,
            "observation_states": (
                None if self.statistics is None else self.statistics.to_dict()
            ),
            "scl": None if self.scl_summary is None else self.scl_summary.to_dict(),
            "bands": None if self.bands is None else self.bands.to_dict(),
            "reflectance_scaling": dict(self.reflectance_scaling),
            "resampling": dict(self.resampling),
            "masking": dict(self.masking),
            "limitations": list(self.limitations),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target
