"""The frozen M5 baseline input definition.

These constants are the whole content of the "what goes into the model"
decision, kept in one module so no other file can disagree with it.

THE BASELINE IS DELIBERATELY NARROW. It is SAR-only, pre/post, four channels,
in linear sigma0. Sentinel-2, DEM, slope, OSM, roads, buildings and handcrafted
features are all excluded, not because they are useless but because a clean SAR
pre/post reference has to exist before anything can be shown to improve on it
(``AGENTS.md`` §5, ``configs/data.yaml`` -> ``m4_contract.baseline``).

The numbers here are not free parameters:

* ``CHANNEL_ORDER`` is the ordering the fitted-statistics artifact and every
  checkpoint are written against. Reordering it silently re-maps a trained
  model's input weights, which is why it is a module constant and is recorded
  in both artifacts rather than passed around.
* ``IGNORE_INDEX`` is 3 because Kuro Siwo stores nodata as an in-band value 3
  in the label raster. It is an INPUT-side concern and is deliberately NOT the
  same number as the output nodata 255
  (``configs/segmentation.yaml`` -> ``output``).
* ``NUM_CLASSES`` is 3 because there are three SEMANTIC classes. The nodata
  sentinel gets no logit.
"""

from __future__ import annotations

from typing import Mapping, Tuple

__all__ = [
    "CHANNEL_ORDER",
    "CHANNEL_SOURCES",
    "CLASS_NAMES",
    "FLOOD_CLASS_ID",
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "PERMANENT_WATER_CLASS_ID",
    "REQUIRED_SAR_REPRESENTATION",
    "SEMANTIC_VALUES",
]

#: Channel order of the 4-band SAR pre/post input stack. FROZEN.
#:
#: Index 0 and 1 are the pre-event epoch, 2 and 3 the post-event epoch, each
#: as (VV, VH). A model trained under this order cannot be fed a different one.
CHANNEL_ORDER: Tuple[str, ...] = ("pre_vv", "pre_vh", "post_vv", "post_vh")

#: Where each channel comes from in a ``CanonicalSample``.
#:
#: The keys are the M4 canonical temporal roles, NOT Kuro's source names: the
#: adapter already resolved ``SL1 -> pre`` and ``MS1 -> post`` from
#: ``info.json`` ``master``/``crank`` fields, and ``SL2`` is recorded as a
#: dropped source. Reading ``sample.sar["pre"]`` therefore consumes that
#: resolution rather than re-deriving it here.
CHANNEL_SOURCES: Mapping[str, Tuple[str, str]] = {
    "pre_vv": ("pre", "vv"),
    "pre_vh": ("pre", "vh"),
    "post_vv": ("post", "vv"),
    "post_vh": ("post", "vh"),
}

#: The only SAR representation this baseline accepts.
#:
#: Enforced rather than assumed. Kuro Siwo is linear sigma0 clipped at 0.15 and
#: Sen1Floods11 is decibel; feeding the latter through a model fitted on the
#: former would be a silent 20-decibel offset. ``tensors.py`` rejects any sample
#: whose ``sar_representation`` differs.
REQUIRED_SAR_REPRESENTATION = "linear_sigma0"

#: Semantic class values. 3 is NOT among them.
SEMANTIC_VALUES: Tuple[int, ...] = (0, 1, 2)

#: Human-readable class names, in ``SEMANTIC_VALUES`` order.
CLASS_NAMES: Tuple[str, ...] = ("No water", "Permanent Waters", "Floods")

#: Number of output logits. Three SEMANTIC classes; nodata gets no logit.
NUM_CLASSES = len(SEMANTIC_VALUES)

#: Kuro Siwo's in-band label nodata sentinel, used as the loss/metric ignore
#: index. Input-side only; the output nodata is 255 and is a different concept.
IGNORE_INDEX = 3

#: The class the project is actually judged on (C6).
FLOOD_CLASS_ID = 2

#: Kept distinct from the flood class on purpose: separating flood water from
#: permanent water is the capability Sen1Floods11 does not have.
PERMANENT_WATER_CLASS_ID = 1
