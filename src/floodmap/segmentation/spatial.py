"""Canonical-grid reprojection for the M5 training path.

This module adds no spatial logic of its own. It adapts a
:class:`~floodmap.datasets.contracts.CanonicalSample` to the arguments of
``floodmap.preprocessing.reprojection`` and calls the SAME functions M2
production preprocessing calls. That is deliberate and is the whole point:
training on a differently-resampled grid from the one inference serves would
make every measured metric describe a grid the deployed model never sees.

Why this is needed at all. The M4 adapter records *target-grid metadata* on
each sample — a derived UTM EPSG code and a 10 m target resolution — but
performs no reprojection, and says so. Kuro tiles are delivered on an EPSG:3857
grid whose pixels are 10 PROJECTED UNITS, which is 8.736-9.798 m of true ground
across the five delivered activations and roughly 5 m near 60N. Training
directly on those tiles while serving genuinely 10 m Sentinel-1 imagery is a
train/serve mismatch in the spatial domain. Metadata alone is not reprojection.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from floodmap.datasets.contracts import CanonicalSample
from floodmap.preprocessing.grid import AnalysisGrid
from floodmap.preprocessing.reprojection import (
    CANONICAL_RESOLUTION_M,
    NodataPolicy,
    reproject_categorical,
    reproject_continuous,
    target_grid_for_source,
)

from .channels import CHANNEL_ORDER, REQUIRED_SAR_REPRESENTATION, SEMANTIC_VALUES
from .tensors import TensorContractError, stack_channels

try:
    from affine import Affine
except ImportError as exc:  # pragma: no cover
    Affine = None  # type: ignore[assignment]
    _AFFINE_ERROR = exc
else:
    _AFFINE_ERROR = None

__all__ = [
    "ReprojectedSample",
    "reproject_canonical_sample",
    "source_grid_for_sample",
]

#: Sentinel written into SAR channels where no valid measurement exists after
#: warping. Matches the frozen continuous output nodata (C9) rather than 0.0,
#: because 0.0 linear sigma0 is indistinguishable from genuine near-zero
#: backscatter, which is itself the open-water signal.
SAR_NODATA = -9999.0


class ReprojectedSample(dict):
    """Reprojected image, target, mask and the provenance to reproduce them."""

    @property
    def image(self) -> np.ndarray:
        return self["image"]

    @property
    def target(self) -> np.ndarray:
        return self["target"]

    @property
    def valid_mask(self) -> np.ndarray:
        return self["valid_mask"]

    @property
    def grid(self) -> AnalysisGrid:
        return self["grid"]

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self["provenance"]


def source_grid_for_sample(sample: CanonicalSample) -> AnalysisGrid:
    """The sample's own source grid as an :class:`AnalysisGrid`.

    ``resolution_m`` here carries the source pixel size in CRS UNITS, not true
    ground metres. For Kuro that is 10 projected units in EPSG:3857, which is
    exactly the discrepancy this module exists to remove; the field name is
    inherited from M2 and is not a claim about ground distance.
    """

    if _AFFINE_ERROR is not None:  # pragma: no cover
        raise TensorContractError("affine is required for reprojection") from _AFFINE_ERROR

    transform = Affine(*[float(value) for value in sample.source_transform[:6]])
    height, width = int(sample.source_shape[0]), int(sample.source_shape[1])
    spacing_x = abs(float(transform.a))
    spacing_y = abs(float(transform.e))
    if abs(spacing_x - spacing_y) > 1e-9:
        raise TensorContractError(
            f"sample {sample.sample_id!r} has non-square source pixels "
            f"({spacing_x}, {spacing_y}); the grid helper assumes square pixels"
        )
    return AnalysisGrid(
        crs=sample.source_crs,
        transform=transform,
        width=width,
        height=height,
        resolution_m=spacing_x,
    )


def reproject_canonical_sample(
    sample: CanonicalSample,
    *,
    ignore_index: Optional[int] = None,
    resolution_m: float = CANONICAL_RESOLUTION_M,
    tile_size: Optional[int] = None,
    nodata_policy: Optional[NodataPolicy] = None,
    sar_resampling: str = "bilinear",
    label_resampling: str = "nearest",
) -> ReprojectedSample:
    """Warp one canonical sample onto its per-sample canonical UTM grid.

    The four SAR channels, the label raster and the validity mask are all
    warped onto the SAME concrete target grid, so they remain pixel-aligned by
    construction rather than by assumption.

    SAR uses bilinear in LINEAR POWER; labels and validity use nearest
    neighbour (C3). No dB conversion happens here: the M5 baseline is frozen on
    linear sigma0, and ``reproject_continuous`` would refuse an averaging
    method on decibel input anyway.

    Returns
    -------
    ReprojectedSample
        ``image`` ``float32[4, H, W]``, ``target`` ``int64[H, W]``,
        ``valid_mask`` ``bool[H, W]``, plus the realised ``grid`` and the
        ``provenance`` needed to reproduce the transformation.
    """

    if sample.sar_representation != REQUIRED_SAR_REPRESENTATION:
        raise TensorContractError(
            f"reprojection expects {REQUIRED_SAR_REPRESENTATION!r} SAR input, got "
            f"{sample.sar_representation!r} for sample {sample.sample_id!r}"
        )

    resolved_ignore = (
        int(ignore_index) if ignore_index is not None else int(sample.nodata.ignore_index)
    )
    if resolved_ignore in SEMANTIC_VALUES:
        raise TensorContractError(
            f"ignore_index {resolved_ignore} collides with a semantic class {SEMANTIC_VALUES}"
        )

    policy = nodata_policy or NodataPolicy()
    source = source_grid_for_sample(sample)
    target = target_grid_for_source(source, resolution_m=resolution_m, tile_size=tile_size)

    stack = stack_channels(sample)
    source_valid = np.asarray(sample.validity, dtype=bool)

    sar = reproject_continuous(
        stack,
        source_valid,
        source=source,
        target=target,
        resampling_name=sar_resampling,
        nodata=SAR_NODATA,
        nodata_policy=policy,
        domain="linear_power",
    )
    labels = reproject_categorical(
        np.asarray(sample.labels),
        source_valid,
        source=source,
        target=target,
        ignore_index=resolved_ignore,
        resampling_name=label_resampling,
        semantic_values=SEMANTIC_VALUES,
    )

    # One mask governs everything. A pixel is usable only where the SAR stack
    # and the label raster BOTH have valid coverage: the two warps use
    # different methods, so their valid footprints are not identical at the
    # fringe, and taking the intersection is what keeps image, target and mask
    # describing the same pixels.
    valid = sar.valid & labels.valid

    image = np.array(sar.data, dtype=np.float32, copy=True)
    image[:, ~valid] = SAR_NODATA
    target_labels = np.where(valid, labels.data, resolved_ignore).astype(np.int64, copy=False)

    provenance: Dict[str, Any] = {
        "operation": "reproject_canonical_sample",
        "sample_id": sample.sample_id,
        "dataset_id": sample.dataset_id,
        "split": sample.split,
        "channels": list(CHANNEL_ORDER),
        "representation": sample.sar_representation,
        "db_conversion_applied": False,
        "ignore_index": resolved_ignore,
        "source_grid": source.to_dict(),
        "target_grid": target.to_dict(),
        "target_resolution_m": float(resolution_m),
        "target_resolution_is_true_ground": True,
        "source_resolution_is_true_ground": False,
        "source_spacing_units": sample.source_spacing_units,
        "tile_size": None if tile_size is None else int(tile_size),
        "nodata_policy": policy.to_dict(),
        "sar_nodata": SAR_NODATA,
        "sar": sar.provenance,
        "labels": labels.provenance,
        "shared_implementation": "floodmap.preprocessing.reprojection",
        "source_valid_pixels": int(source_valid.sum()),
        "target_valid_pixels": int(valid.sum()),
    }

    return ReprojectedSample(
        image=image,
        target=target_labels,
        valid_mask=valid,
        grid=target,
        provenance=provenance,
    )
