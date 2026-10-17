"""Presentation-only raster-to-RGBA helpers, with explicit nodata handling.

These functions turn a geotiff band into a display array. They are pure numpy +
rasterio with no ``streamlit``/``folium`` dependency, so the nodata semantics can
be unit-tested without the dashboard extra installed.

The one scientific rule encoded here is ``AGENTS.md`` §16 / the M6.6 requirement:
**an invalid or not-observed pixel must never be painted as an observation.**
Not-observed (``255`` in the candidate mask), feature nodata (``-9999.0``),
non-finite values and any pixel outside the declared class set all render fully
transparent, so a hole in the data reads as a hole rather than as "no flood".

None of this computes a result: it only chooses colours and a decimated resolution
for on-screen display. Metrics are always taken from the full-resolution rasters
by :mod:`floodmap.dashboard.realdata`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling

__all__ = [
    "DEFAULT_MAX_DIM",
    "FEATURE_NODATA_DEFAULT",
    "candidate_mask_rgba",
    "change_rgba",
    "decimated_shape",
    "gray_rgba",
    "read_display",
    "valid_mask_rgba",
]

#: Longest side of a display raster. Keeps a 3313x4961 scene to ~1200 px so the
#: browser payload and memory stay bounded; full resolution is never used here.
DEFAULT_MAX_DIM = 1200

#: Feature nodata used by M6.3 (documented in the provenance / summary).
FEATURE_NODATA_DEFAULT = -9999.0

_RED = np.array([214, 45, 32], dtype="float32")
_GRAY = np.array([130, 130, 130], dtype="float32")
_BLUE = np.array([49, 84, 200], dtype="float32")
_WHITE = np.array([245, 245, 245], dtype="float32")


def decimated_shape(width: int, height: int, max_dim: int = DEFAULT_MAX_DIM) -> Tuple[int, int]:
    """Return ``(height, width)`` scaled so the longer side is at most ``max_dim``."""
    longest = max(width, height)
    if longest <= max_dim or longest == 0:
        return (height, width)
    scale = max_dim / float(longest)
    return (max(1, int(round(height * scale))), max(1, int(round(width * scale))))


def read_display(
    path: Path,
    *,
    band: int = 1,
    max_dim: int = DEFAULT_MAX_DIM,
    resampling: Resampling = Resampling.nearest,
) -> Tuple[np.ndarray, Optional[float]]:
    """Read one band decimated for display. Returns ``(array, nodata)``.

    Nearest-neighbour is the default so integer class values (0/1/255) survive the
    decimation unchanged.
    """
    with rasterio.open(path) as src:
        out_shape = decimated_shape(src.width, src.height, max_dim)
        data = src.read(band, out_shape=out_shape, resampling=resampling)
        nodata = float(src.nodata) if src.nodata is not None else None
    return data, nodata


def _invalid_mask(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    invalid = ~np.isfinite(data)
    if nodata is not None:
        invalid |= data == nodata
    return invalid


def candidate_mask_rgba(
    data: np.ndarray, *, nodata: Optional[float] = 255.0
) -> np.ndarray:
    """Candidate mask -> RGBA.

    ``1`` (candidate) is solid red; ``0`` (observed, not a candidate) is a faint
    grey so the observed footprint is visible; ``255``/nodata and anything else is
    fully transparent.
    """
    data = np.asarray(data)
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")

    candidate = data == 1
    observed = data == 0
    rgba[observed, :3] = _GRAY.astype("uint8")
    rgba[observed, 3] = 40
    rgba[candidate, :3] = _RED.astype("uint8")
    rgba[candidate, 3] = 215
    return rgba


def valid_mask_rgba(data: np.ndarray) -> np.ndarray:
    """Valid-data mask -> translucent grey where observed, transparent elsewhere."""
    data = np.asarray(data)
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")
    observed = data == 1
    rgba[observed, :3] = _GRAY.astype("uint8")
    rgba[observed, 3] = 45
    return rgba


def change_rgba(
    data: np.ndarray,
    *,
    nodata: Optional[float] = FEATURE_NODATA_DEFAULT,
    limit_db: float = 8.0,
) -> np.ndarray:
    """DiB log-ratio change -> diverging blue (darkening) / red (brightening) RGBA.

    A near-white centre marks ~0 dB (no change). Nodata and non-finite pixels are
    transparent.
    """
    data = np.asarray(data, dtype="float32")
    invalid = _invalid_mask(data, nodata)
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")

    t = np.clip(data / float(limit_db), -1.0, 1.0)
    t = np.where(invalid, 0.0, t)
    neg = t < 0.0
    # darkening: white -> blue ; brightening: white -> red
    colour = np.where(
        neg[..., None],
        _WHITE + (_BLUE - _WHITE) * np.abs(t)[..., None],
        _WHITE + (_RED - _WHITE) * t[..., None],
    )
    rgba[..., :3] = np.clip(colour, 0, 255).astype("uint8")
    rgba[..., 3] = np.where(invalid, 0, 200).astype("uint8")
    return rgba


def gray_rgba(
    data: np.ndarray,
    *,
    nodata: Optional[float] = None,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    """Percentile-normalized grey, with nodata transparent."""
    data = np.asarray(data, dtype="float32")
    invalid = _invalid_mask(data, nodata)
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return rgba
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return rgba
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    scaled = np.where(invalid, 0.0, scaled)
    gray = (scaled * 255.0).astype("uint8")
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(invalid, 0, 235).astype("uint8")
    return rgba
