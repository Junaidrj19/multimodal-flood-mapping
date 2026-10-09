"""Loss for multiclass segmentation with ignored pixels.

The baseline is UNWEIGHTED ``CrossEntropyLoss(ignore_index=3)``. That is a
stated choice, not an oversight:

* Flood pixels are a small minority, so class weighting is an obvious thing to
  try. It is therefore SUPPORTED by configuration — but it is never applied by
  default, and no weight vector is computed or guessed here. Inverse-frequency
  weights look principled and are not: on this corpus the frequencies depend on
  which activations happen to be delivered locally (1111004 alone is 71% of the
  available training tiles), so a weight fitted from them would encode the
  delivery accident rather than the phenomenon.
* No composite loss. Dice, Tversky, focal and Lovasz all change what the
  optimiser considers an improvement, and adding them before a plain
  cross-entropy reference exists would make it impossible to attribute any
  later gain.

``ignore_index`` is handled by PyTorch itself, which excludes those positions
from both the loss value and the backward pass. ``tests/test_segmentation_m5.py``
verifies that rather than assuming it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .channels import IGNORE_INDEX, NUM_CLASSES

__all__ = ["build_loss"]


def build_loss(
    *,
    ignore_index: int = IGNORE_INDEX,
    class_weights: Optional[Sequence[float]] = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Build the segmentation criterion.

    Parameters
    ----------
    ignore_index
        Target value excluded from the loss. Defaults to Kuro's stored label
        nodata sentinel 3.
    class_weights
        Optional per-class weights. ``None`` means unweighted, which is the
        baseline. When provided it must have exactly ``num_classes`` entries;
        a shorter or longer vector is rejected rather than broadcast, because a
        silently misaligned weight vector would reweight the wrong classes.
    num_classes
        Number of semantic classes, used only to validate ``class_weights``.
    """

    weight_tensor = None
    if class_weights is not None:
        weights = [float(value) for value in class_weights]
        if len(weights) != int(num_classes):
            raise ValueError(
                f"class_weights has {len(weights)} entries but there are "
                f"{num_classes} semantic classes; a misaligned weight vector would "
                "reweight the wrong classes"
            )
        if any(value < 0.0 for value in weights):
            raise ValueError(f"class weights must be non-negative, got {weights}")
        if not any(value > 0.0 for value in weights):
            raise ValueError("at least one class weight must be positive")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

    return nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=int(ignore_index))
