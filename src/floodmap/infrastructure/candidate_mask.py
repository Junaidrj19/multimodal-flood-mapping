"""M6.3 change-candidate mask loading and validation.

Scientific responsibility
-------------------------
Read the Sentinel-1 RTC change-candidate raster produced by
``floodmap.baseline.rtc_change_candidates`` and refuse anything that is not
that artifact, so no downstream infrastructure number can be computed from a
raster whose class meanings are different from the ones assumed here.

Why this is a separate module from :mod:`floodmap.infrastructure.mask`
----------------------------------------------------------------------
:mod:`floodmap.infrastructure.mask` loads the frozen M4/M5 segmentation output
contract: ``no water`` / ``permanent waters`` / ``floods`` / nodata, with class
ids read from ``configs/segmentation.yaml``.

The M6.3 candidate raster is a **different contract**:

``0`` not_candidate, ``1`` darkening_candidate, ``255`` not_observed

It carries no permanent-water class, and ``1`` does **not** mean "flood".
Loading it through :func:`floodmap.infrastructure.mask.load_flood_mask` would
succeed numerically — ``0`` and ``1`` are both legal segmentation values — and
silently reinterpret "darkening candidate" as "permanent water". This module
exists so that cannot happen: it validates the M6.3 stage tag and the declared
``semantic_values`` string, and rejects any value outside ``{0, 1, 255}``.

Nothing here converts the candidate mask into the three-class contract, and no
permanent-water label is created.

Three pixel populations are kept distinct throughout
----------------------------------------------------
* **candidate** (``1``) — both polarisations darkened past the M6.3 threshold;
* **not_candidate** (``0``) — observed, and the rule did not fire;
* **not_observed** (``255``) — no valid pre/post VV+VH pair, so the rule could
  not be evaluated at all.

``not_observed`` is never treated as ``not_candidate``. An unobserved road is
reported as unobserved, not as unaffected (``docs/scientific-assumptions.md``
§8 draws the same distinction for the segmentation mask).

Interoperability with the existing exposure kernel
--------------------------------------------------
:class:`CandidateMask` deliberately exposes the same attribute surface the
measurement kernel in :mod:`floodmap.infrastructure.exposure` reads
(``classes``, ``transform``, ``crs``, ``shape``, ``bounds``, ``semantics``), so
that the tested line-sampling and polygon-rasterising code is reused rather
than duplicated (``AGENTS.md`` §12). The mapping onto that kernel's vocabulary
is stated once, in :func:`candidate_semantics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from floodmap.baseline.rtc_change_candidates import (
    CHANGE_CANDIDATE_NODATA,
    RTC_CHANGE_VERSION,
)
from floodmap.infrastructure.mask import MaskSemantics
from floodmap.preprocessing.grid import AnalysisGrid, require_aligned

#: Raster values of the M6.3 candidate mask. ``CANDIDATE`` is change evidence,
#: not flood.
NOT_CANDIDATE = 0
CANDIDATE = 1
NOT_OBSERVED_VALUE = CHANGE_CANDIDATE_NODATA

#: The exact ``semantic_values`` tag M6.3 writes. Compared literally: a changed
#: string means the upstream contract moved and this stage must be revisited
#: rather than quietly carry on.
CANDIDATE_SEMANTIC_VALUES_TAG = "0=not_candidate,1=darkening_candidate,255=not_observed"

#: The M6.3 valid-mask tag. ``1`` marks a pixel inside the configured study
#: extent where a pre/post VV+VH change could be computed at all. The extent is
#: part of the support from stage 0.2.0 onward: the delivered RTC window is a
#: bounding-box crop that reaches well outside the AOI, so a validity layer that
#: was not AOI-scoped would hand this stage out-of-extent pixels to analyse.
VALID_MASK_SEMANTICS_TAG = (
    "1=inside configured AOI with valid pre/post VV+VH change,0=outside AOI or not observed"
)

#: Placeholder occupying the kernel's ``permanent_water`` slot.
#:
#: The candidate contract has no permanent-water class. The shared kernel's
#: :class:`MaskSemantics` requires four distinct ids, so this value is supplied
#: purely to keep that slot occupied by something the candidate mask can never
#: contain — :func:`load_candidate_mask` rejects every value outside
#: ``{0, 1, 255}``. Permanent-water counts are consequently always zero and are
#: not reported by M6.4. No permanent-water label is created anywhere.
_NO_PERMANENT_WATER_CLASS = 254


class CandidateMaskError(ValueError):
    """The raster is not a usable M6.3 change-candidate mask."""


def candidate_semantics() -> MaskSemantics:
    """Candidate class ids expressed in the shared exposure kernel's slots.

    The kernel in :mod:`floodmap.infrastructure.exposure` asks a mask which
    value is "the class being tested", which is nodata, and which is permanent
    water. For a candidate mask that maps to:

    ===========================  ==============================================
    kernel slot                  candidate meaning
    ===========================  ==============================================
    ``flood``                    ``1`` darkening candidate (the tested class)
    ``no_water``                 ``0`` observed, rule did not fire
    ``nodata``                   ``255`` not observed
    ``permanent_water``          absent — see ``_NO_PERMANENT_WATER_CLASS``
    ===========================  ==============================================

    The kernel's column names are translated to candidate vocabulary by
    :mod:`floodmap.infrastructure.candidate_exposure`, so no M6.4 output ever
    says "flooded".
    """
    return MaskSemantics(
        no_water=NOT_CANDIDATE,
        permanent_water=_NO_PERMANENT_WATER_CLASS,
        flood=CANDIDATE,
        nodata=NOT_OBSERVED_VALUE,
    )


@dataclass
class CandidateMask:
    """A validated M6.3 change-candidate mask held in memory.

    ``valid`` is the independently loaded M6.3 valid-data mask when one was
    supplied, otherwise it is derived from the class raster. Keeping it
    separate preserves the distinction between "observed and not a candidate"
    and "never observed".
    """

    path: Path
    classes: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: CRS
    semantics: MaskSemantics
    stage: str
    tags: Dict[str, str]
    valid_mask_path: Optional[Path] = None
    alignment: Optional[Dict[str, Any]] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.classes.shape

    @property
    def pixel_size_m(self) -> Tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        height, width = self.classes.shape
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (width, height)
        return min(left, right), min(top, bottom), max(left, right), max(top, bottom)

    def pixel_counts(self) -> Dict[str, int]:
        counts = np.bincount(self.classes.ravel(), minlength=256)
        return {
            "candidate": int(counts[CANDIDATE]),
            "not_candidate": int(counts[NOT_CANDIDATE]),
            "not_observed": int(counts[NOT_OBSERVED_VALUE]),
        }

    def describe(self) -> Dict[str, Any]:
        """Machine-readable description of the mask, for summaries/provenance."""
        counts = self.pixel_counts()
        px_w, px_h = self.pixel_size_m
        total = int(self.classes.size)
        return {
            "path": str(self.path),
            "valid_mask_path": str(self.valid_mask_path) if self.valid_mask_path else None,
            "stage": self.stage,
            # Which study extent the upstream mask was scoped to, read from the
            # raster's own tags. Carried so a downstream report can be traced to
            # an extent by digest instead of by assumption.
            "aoi_id": self.tags.get("aoi_id"),
            "aoi_source_sha256": self.tags.get("aoi_source_sha256"),
            "analysis_scope": self.tags.get("analysis_scope"),
            "crs": self.crs.to_string(),
            "pixel_size_m": [px_w, px_h],
            "shape": list(self.shape),
            "bounds": list(self.bounds),
            "class_values": {
                "not_candidate": NOT_CANDIDATE,
                "candidate": CANDIDATE,
                "not_observed": NOT_OBSERVED_VALUE,
            },
            "pixel_counts": counts,
            "total_pixels": total,
            "observed_pixels": total - counts["not_observed"],
            "candidate_fraction_of_observed": (
                round(counts["candidate"] / (total - counts["not_observed"]), 6)
                if total - counts["not_observed"]
                else 0.0
            ),
            "not_observed_fraction": round(counts["not_observed"] / max(1, total), 6),
            # Raster-grid area of candidate pixels. NOT a flood area.
            "candidate_grid_area_km2": round(counts["candidate"] * px_w * px_h / 1e6, 4),
            "interpretation": (
                "Candidate pixels are Sentinel-1 RTC gamma0 dual-polarisation darkening "
                "observations. They are not confirmed flood, damage or ground truth."
            ),
            "alignment": self.alignment,
        }

    def as_grid(self) -> AnalysisGrid:
        height, width = self.shape
        px_w, px_h = self.pixel_size_m
        return AnalysisGrid(
            crs=self.crs.to_string(),
            transform=self.transform,
            width=width,
            height=height,
            resolution_m=float((px_w + px_h) / 2),
        )


def _open_single_band_uint8(path: Path, *, label: str):
    """Shared structural checks for both M6.3 single-band uint8 rasters."""
    if not path.is_file():
        raise CandidateMaskError(f"{label} not found: {path}")
    with rasterio.open(path) as src:
        if src.count != 1:
            raise CandidateMaskError(f"{label} must have exactly one band, got {src.count}")
        if src.dtypes[0] != "uint8":
            raise CandidateMaskError(f"{label} must be uint8, got {src.dtypes[0]}")
        if src.crs is None:
            raise CandidateMaskError(f"{label} has no CRS")
        if not src.crs.is_projected:
            raise CandidateMaskError(
                f"{label} CRS {src.crs} is geographic; metre-based length and area "
                "measurement requires a projected CRS"
            )
        units = (src.crs.linear_units or "").lower()
        if units not in ("metre", "meter", "metres", "meters", "m"):
            raise CandidateMaskError(f"{label} CRS units must be metres, got {units!r}")
        if src.transform.b != 0 or src.transform.d != 0:
            raise CandidateMaskError(f"{label} is rotated or sheared, which is not supported")
        return (
            src.read(1),
            src.transform,
            src.crs,
            {str(k): str(v) for k, v in src.tags().items()},
            src.nodata,
            AnalysisGrid.from_dataset(src),
        )


def load_candidate_mask(
    mask_path: Path | str,
    valid_mask_path: Optional[Path | str] = None,
    *,
    expected_stage: str = RTC_CHANGE_VERSION,
    alignment_tolerance: Optional[float] = 0.0,
    require_stage_tag: bool = True,
) -> CandidateMask:
    """Open and validate an M6.3 change-candidate mask.

    Parameters
    ----------
    mask_path
        ``rtc_change_candidates.tif`` from the M6.3 stage.
    valid_mask_path
        ``rtc_valid_mask.tif``. Optional, but strongly preferred: it is the
        upstream stage's own statement of which pixels were observable, and
        loading it lets this function cross-check the two rasters agree
        instead of trusting a single source.
    expected_stage
        Required value of the ``artifact_stage`` GeoTIFF tag.
    require_stage_tag
        Leave ``True``. ``False`` exists only for tests that build a minimal
        candidate raster without the full M6.3 tag set.

    Raises
    ------
    CandidateMaskError
        On any contract violation. Nothing is coerced: a mask whose class
        meanings cannot be confirmed would silently corrupt every road,
        building and settlement verdict downstream.
    """
    mask_path = Path(mask_path)
    classes, transform, crs, tags, nodata, grid = _open_single_band_uint8(
        mask_path, label="candidate mask"
    )

    if nodata is not None and int(nodata) != NOT_OBSERVED_VALUE:
        raise CandidateMaskError(
            f"candidate mask declares nodata={nodata}; the M6.3 contract uses "
            f"{NOT_OBSERVED_VALUE} for not_observed"
        )

    stage = tags.get("artifact_stage", "")
    if require_stage_tag and stage != expected_stage:
        raise CandidateMaskError(
            f"candidate mask {mask_path} declares artifact_stage={stage!r}, expected "
            f"{expected_stage!r}. This stage refuses to interpret a raster from an "
            "unknown producer as change candidates."
        )
    declared = tags.get("semantic_values")
    if declared is not None and declared != CANDIDATE_SEMANTIC_VALUES_TAG:
        raise CandidateMaskError(
            f"candidate mask {mask_path} declares semantic_values={declared!r}, expected "
            f"{CANDIDATE_SEMANTIC_VALUES_TAG!r}; the class meanings cannot be assumed"
        )

    present = set(np.unique(classes).tolist())
    allowed = {NOT_CANDIDATE, CANDIDATE, NOT_OBSERVED_VALUE}
    unexpected = present - allowed
    if unexpected:
        raise CandidateMaskError(
            f"candidate mask contains values {sorted(unexpected)} outside the M6.3 "
            f"contract {sorted(allowed)}. This stage consumes a two-class candidate "
            "mask plus not_observed; it does not accept the three-class segmentation "
            "mask (no water / permanent water / floods) and does not reinterpret one "
            "as the other."
        )

    valid = classes != NOT_OBSERVED_VALUE
    alignment: Optional[Dict[str, Any]] = None
    resolved_valid_path: Optional[Path] = None
    if valid_mask_path is not None:
        resolved_valid_path = Path(valid_mask_path)
        valid_values, _, _, valid_tags, _, valid_grid = _open_single_band_uint8(
            resolved_valid_path, label="valid mask"
        )
        alignment = require_aligned(grid, valid_grid, tolerance=alignment_tolerance).to_dict()
        declared_valid = valid_tags.get("mask_semantics")
        if declared_valid is not None and declared_valid != VALID_MASK_SEMANTICS_TAG:
            raise CandidateMaskError(
                f"valid mask {resolved_valid_path} declares mask_semantics="
                f"{declared_valid!r}, expected {VALID_MASK_SEMANTICS_TAG!r}"
            )
        valid_present = set(np.unique(valid_values).tolist())
        if not valid_present <= {0, 1}:
            raise CandidateMaskError(
                f"valid mask contains values {sorted(valid_present - {0, 1})}; only 0 and 1 "
                "are meaningful"
            )
        from_valid = valid_values.astype(bool)
        disagreeing = int(np.count_nonzero(from_valid != valid))
        if disagreeing:
            raise CandidateMaskError(
                f"candidate mask and valid mask disagree on {disagreeing} pixels: "
                "not_observed in the class raster must coincide exactly with 0 in the "
                "valid mask, otherwise the unobserved region is ambiguous"
            )
        valid = from_valid

    return CandidateMask(
        path=mask_path,
        classes=classes,
        valid=valid,
        transform=transform,
        crs=crs,
        semantics=candidate_semantics(),
        stage=stage,
        tags=tags,
        valid_mask_path=resolved_valid_path,
        alignment=alignment,
    )


__all__ = [
    "CANDIDATE",
    "CANDIDATE_SEMANTIC_VALUES_TAG",
    "CandidateMask",
    "CandidateMaskError",
    "NOT_CANDIDATE",
    "NOT_OBSERVED_VALUE",
    "VALID_MASK_SEMANTICS_TAG",
    "candidate_semantics",
    "load_candidate_mask",
]
