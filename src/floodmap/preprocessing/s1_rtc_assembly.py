"""Assemble Sentinel-1 RTC VV/VH single-band COGs into the M2 input contract.

Why this module exists
----------------------
Microsoft Planetary Computer's ``sentinel-1-rtc`` collection publishes each
polarisation as its own single-band Cloud-Optimised GeoTIFF, with a descriptive
band name (``'Sentinel-1 Calibrated and Terrain Corrected VV'``). The M2 input
contract wants the opposite shape: **one raster, ≥2 bands, band descriptions
exactly ``VV`` and ``VH``** — see
:func:`floodmap.preprocessing.validation.validate_required_bands`, whose
docstring is explicit that *"band order is never guessed"*.

That is the whole gap between the RTC products and M2. This module closes it by
**satisfying** the named-band gate, not by relaxing it: it writes real band
descriptions so the downstream index lookup resolves honestly.

Scope
-----
Deliberately narrow. This is the RTC-to-M2 adapter for the M6.2 validation
workflow, not a new acquisition framework. It does not search catalogues, choose
pairs, or decide scientific policy; it is handed two signed asset URLs and a
window, and it writes one file.

What it refuses to do
---------------------
* **No resampling.** The window is snapped to whole source pixels and the output
  inherits the source transform exactly. Resampling here would silently change
  the radiometry of a product whose entire value is that it is already
  radiometrically terrain corrected.
* **No reconciling incompatible assets.** If VV and VH disagree on CRS,
  resolution, transform, dimensions or nodata, that is raised. Two polarisations
  of one acquisition are produced by one processor run on one grid; if they
  disagree, the inputs are not what they claim to be, and stacking them would
  place VV and VH pixels from different ground locations in the same cell.
* **No radiometric interpretation.** The values are written through unchanged.
  The convention (γ⁰ linear power) is *recorded* in the provenance record and
  never converted, because γ⁰ and σ⁰ differ by ``1/cos θ`` and this module has
  no business inventing that conversion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

__all__ = [
    "RTC_RADIOMETRIC_CONVENTION",
    "AssembledBandPair",
    "RtcAssemblyError",
    "assemble_rtc_window",
]

#: The radiometric convention these products carry, recorded verbatim so no
#: downstream consumer has to infer it. Established from the provider's
#: documented methodology (ESA-EOPG-CSCOP-TN-0002 calibration; terrain
#: correction after D. Small, "Flattening Gamma", IEEE TGRS 49(8) 2011, whose
#: gamma adjustment divides by the cosine of the incidence angle).
#: It is NOT sigma-nought.
RTC_RADIOMETRIC_CONVENTION = "gamma0_linear_power"

#: Band descriptions M2 requires, in the order they are written.
BAND_NAMES: Tuple[str, str] = ("VV", "VH")


class RtcAssemblyError(RuntimeError):
    """The two polarisation assets are not a compatible pair, or are unreadable."""


@dataclass(frozen=True)
class AssembledBandPair:
    """The written raster and everything needed to audit how it was made."""

    path: Path
    crs: str
    transform: Tuple[float, ...]
    width: int
    height: int
    dtype: str
    nodata: float
    band_descriptions: Tuple[str, ...]
    radiometric_convention: str
    window_bounds: Tuple[float, float, float, float]
    window_offsets: Tuple[int, int]
    sources: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "crs": self.crs,
            "transform": list(self.transform),
            "width": self.width,
            "height": self.height,
            "dtype": self.dtype,
            "nodata": self.nodata,
            "band_descriptions": list(self.band_descriptions),
            "radiometric_convention": self.radiometric_convention,
            "radiometric_note": (
                "gamma-nought (gamma0) linear power, NOT sigma-nought. No conversion "
                "was applied. gamma0 = sigma0 / cos(incidence angle), so the two are "
                "not interchangeable."
            ),
            "window_bounds": list(self.window_bounds),
            "window_offsets_col_row": list(self.window_offsets),
            "sources": self.sources,
            "statistics": self.statistics,
        }


def _profile_of(dataset: Any) -> Dict[str, Any]:
    return {
        "crs": None if dataset.crs is None else dataset.crs.to_string(),
        "res": tuple(round(value, 9) for value in dataset.res),
        "transform": tuple(round(value, 6) for value in dataset.transform[:6]),
        "shape": (dataset.height, dataset.width),
        "dtype": dataset.dtypes[0],
        "nodata": dataset.nodata,
        "count": dataset.count,
    }


def _require_compatible(vv: Dict[str, Any], vh: Dict[str, Any]) -> None:
    """Reject a VV/VH pair that is not on one grid.

    Checked explicitly rather than trusted: both polarisations of one
    acquisition come from a single processor run, so a disagreement means the
    assets are not the pair they are presented as. Stacking them anyway would
    put VV and VH samples from different ground positions in the same cell —
    a defect invisible in the output raster.
    """
    for field_name in ("crs", "res", "transform", "shape", "dtype", "nodata"):
        if vv[field_name] != vh[field_name]:
            raise RtcAssemblyError(
                f"VV and VH disagree on {field_name}: {vv[field_name]!r} vs "
                f"{vh[field_name]!r}. They are not a single-grid pair and are not "
                "reconciled here, because resampling or relabelling one of them would "
                "hide a real inconsistency in the source products."
            )
    if vv["crs"] is None:
        raise RtcAssemblyError(
            "the RTC assets report no CRS; an analysis-ready product must be "
            "georeferenced, so this is rejected rather than assumed"
        )
    if vv["count"] != 1 or vh["count"] != 1:
        raise RtcAssemblyError(
            f"expected one band per polarisation asset, got VV={vv['count']} "
            f"VH={vh['count']}"
        )
    if vv["nodata"] is None:
        raise RtcAssemblyError(
            "the RTC assets declare no nodata value; M2 masks invalid pixels by "
            "nodata, so an undeclared fill value would be read as a real observation"
        )


def _aoi_window(dataset: Any, aoi_bounds_4326: Sequence[float]) -> Window:
    """Whole-pixel window covering the AOI, clipped to the raster.

    Snapped outward to integer pixel edges so the read needs no resampling and
    the output transform stays exactly on the source grid.
    """
    bounds = transform_bounds("EPSG:4326", dataset.crs, *aoi_bounds_4326)
    if not all(np.isfinite(value) for value in bounds):
        # A lon/lat box far outside the projection's valid domain transforms to
        # non-finite coordinates. Reported geographically rather than surfacing
        # as "cannot convert float NaN to integer" from the windowing helper.
        raise RtcAssemblyError(
            f"the AOI {tuple(aoi_bounds_4326)} does not project into "
            f"{dataset.crs}: transformed bounds are {bounds}. The AOI is outside "
            "this product's projection domain, so no window exists."
        )
    window = from_bounds(*bounds, transform=dataset.transform)
    window = window.round_offsets(op="floor").round_lengths(op="ceil")
    full = Window(0, 0, dataset.width, dataset.height)
    try:
        clipped = window.intersection(full)
    except rasterio.errors.WindowError as exc:
        # rasterio raises rather than returning a zero-size window. Translate it
        # into this module's error so the caller sees a geographic explanation
        # rather than a windowing internal.
        raise RtcAssemblyError(
            f"the AOI does not intersect {dataset.name}: AOI bounds in raster CRS are "
            f"{bounds}, raster covers {tuple(dataset.bounds)}"
        ) from exc
    if clipped.width <= 0 or clipped.height <= 0:
        raise RtcAssemblyError(
            f"the AOI does not intersect {dataset.name}: AOI bounds in raster CRS are "
            f"{bounds}, raster covers {tuple(dataset.bounds)}"
        )
    return clipped


def assemble_rtc_window(
    vv_href: str,
    vh_href: str,
    aoi_bounds_4326: Sequence[float],
    output_path: Path | str,
    *,
    source_item_id: str,
    source_assets: Optional[Mapping[str, Any]] = None,
    compress: str = "deflate",
) -> AssembledBandPair:
    """Read the AOI window from VV and VH and write one 2-band GeoTIFF.

    Parameters
    ----------
    vv_href, vh_href
        Readable (signed) URLs or paths to the single-band RTC assets.
    aoi_bounds_4326
        ``(min_lon, min_lat, max_lon, max_lat)``. Transformed into the raster CRS
        before windowing; the AOI is never assumed to be in the raster's CRS.
    output_path
        Destination GeoTIFF. Parent directories are created.
    source_item_id
        STAC item id, recorded in provenance so the output is traceable to the
        exact acquisition.
    source_assets
        Optional per-polarisation detail (href, ETag/checksum) to record.

    Returns
    -------
    AssembledBandPair
        Output metadata plus the provenance record.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(vv_href) as vv, rasterio.open(vh_href) as vh:
        vv_profile, vh_profile = _profile_of(vv), _profile_of(vh)
        _require_compatible(vv_profile, vh_profile)

        window = _aoi_window(vv, aoi_bounds_4326)
        # The same window is valid for VH only because the grids were just
        # proven identical above.
        window_transform = vv.window_transform(window)
        window_bounds = rasterio.windows.bounds(window, vv.transform)
        nodata = float(vv.nodata)

        vv_data = vv.read(1, window=window)
        vh_data = vh.read(1, window=window)

        profile = {
            "driver": "GTiff",
            "height": int(window.height),
            "width": int(window.width),
            "count": 2,
            "dtype": vv_profile["dtype"],
            "crs": vv.crs,
            "transform": window_transform,
            "nodata": nodata,
            "compress": compress,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

    stack = np.stack([vv_data, vh_data], axis=0)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)
        # The entire point of this module: real band names, so M2's named-band
        # lookup resolves instead of guessing or failing.
        for index, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(index, name)
        dst.update_tags(
            radiometric_convention=RTC_RADIOMETRIC_CONVENTION,
            radiometric_note=(
                "gamma0 linear power; NOT sigma0. gamma0 = sigma0 / cos(theta)."
            ),
            source_item_id=source_item_id,
            terrain_correction="source_terrain_corrected (Catalyst RTC, Flattening Gamma)",
            radiometric_calibration="source_calibrated (ESA-EOPG-CSCOP-TN-0002)",
        )

    statistics: Dict[str, Any] = {}
    for name, data in zip(BAND_NAMES, (vv_data, vh_data)):
        finite = np.isfinite(data)
        valid = finite & (data != nodata)
        values = data[valid]
        statistics[name] = {
            "pixels": int(data.size),
            "valid_pixels": int(valid.sum()),
            "invalid_pixels": int(data.size - int(valid.sum())),
            "valid_fraction": round(float(valid.sum()) / float(data.size), 6)
            if data.size
            else 0.0,
            "all_finite": bool(finite.all()),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
        }

    with rasterio.open(output_path) as check:
        written = _profile_of(check)
        descriptions = tuple(check.descriptions)

    if descriptions != BAND_NAMES:
        raise RtcAssemblyError(
            f"band descriptions were not written as expected: got {descriptions!r}, "
            f"wanted {BAND_NAMES!r}. M2 resolves bands by name, so an unnamed output "
            "would be unusable downstream."
        )

    return AssembledBandPair(
        path=output_path,
        crs=written["crs"],
        transform=written["transform"],
        width=written["shape"][1],
        height=written["shape"][0],
        dtype=written["dtype"],
        nodata=nodata,
        band_descriptions=descriptions,
        radiometric_convention=RTC_RADIOMETRIC_CONVENTION,
        window_bounds=tuple(float(value) for value in window_bounds),
        window_offsets=(int(window.col_off), int(window.row_off)),
        sources={
            "item_id": source_item_id,
            "vv": dict(source_assets.get("vv", {})) if source_assets else {},
            "vh": dict(source_assets.get("vh", {})) if source_assets else {},
            "vv_profile": vv_profile,
            "vh_profile": vh_profile,
        },
        statistics=statistics,
    )
