"""Segmentation metrics, accumulated over batches from a confusion matrix.

Everything is derived from one 3x3 confusion matrix rather than averaged
per batch. That is not a style preference: per-batch IoU averaged over batches
is not the dataset IoU, and the difference is largest exactly where it matters
here — a batch containing few or no flood pixels produces an unstable or
undefined flood IoU that then contaminates the average.

IGNORED PIXELS ARE EXCLUDED BEFORE ACCUMULATION. Nodata is 6.48% of Kuro label
pixels and the model will emit a confident prediction for every one of them;
counting those would inflate "No water" performance with pixels that were never
observed. The exclusion is on the TARGET, which is where validity is known.

The primary reported number is flood-class IoU (class 2), per C6. Per-class
IoU, precision, recall, F1, macro/micro aggregates and the full matrix are all
computed too, because a single headline score hides the error pattern
(``AGENTS.md`` §6).

Zero-denominator policy is explicit. Per-class IoU, precision and recall are
``nan`` when their denominator is zero, because the corresponding quantity is
undefined. F1 is calculated directly from the confusion counts: a class with
any support or prediction has a defined F1, including ``0.0`` when true
positives are zero; a class absent from both truth and prediction remains
``nan`` and is omitted from macro F1. Micro IoU is pooled over class unions,
while pixel accuracy is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .channels import CLASS_NAMES, FLOOD_CLASS_ID, IGNORE_INDEX, NUM_CLASSES

__all__ = ["SegmentationMetrics", "MetricResult"]


@dataclass(frozen=True)
class MetricResult:
    """Computed metrics for one evaluation pass."""

    confusion_matrix: torch.Tensor
    per_class_iou: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    iou_flood_class: float
    mean_iou: float
    macro_f1: float
    micro_iou: float
    pixel_accuracy: float
    n_valid_pixels: int
    n_ignored_pixels: int

    def to_dict(self) -> Dict[str, object]:
        """A JSON-serialisable view for run records and checkpoints."""

        return {
            "iou_flood_class": self.iou_flood_class,
            "mean_iou": self.mean_iou,
            "macro_f1": self.macro_f1,
            "micro_iou": self.micro_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "per_class_iou": dict(self.per_class_iou),
            "per_class_precision": dict(self.per_class_precision),
            "per_class_recall": dict(self.per_class_recall),
            "per_class_f1": dict(self.per_class_f1),
            "per_class_support": dict(self.per_class_support),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_valid_pixels": self.n_valid_pixels,
            "n_ignored_pixels": self.n_ignored_pixels,
        }


class SegmentationMetrics:
    """Accumulate a confusion matrix across batches, then derive metrics.

    A ratio whose denominator is zero is ``nan`` because that quantity is
    undefined. Such genuinely absent classes are dropped from macro
    aggregates rather than counted as zero. A class that has support or false
    positives but no true positives is not absent: its F1 is therefore 0.0.
    Micro IoU is the pooled intersection over the pooled union, not pixel
    accuracy; both values are reported separately.
    """

    def __init__(
        self,
        *,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        class_names=CLASS_NAMES,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = tuple(class_names)[: self.num_classes]
        self.reset()

    def reset(self) -> None:
        self._matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )
        self._ignored = 0

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """Rows are ground truth, columns are predictions."""

        return self._matrix.clone()

    def update(
        self,
        logits_or_predictions: torch.Tensor,
        target: torch.Tensor,
        *,
        is_prediction: bool = False,
    ) -> None:
        """Accumulate one batch.

        Parameters
        ----------
        logits_or_predictions
            Either ``[N, C, H, W]`` logits or ``[N, H, W]`` hard predictions.
        target
            ``[N, H, W]`` int64 target containing ``ignore_index`` where no
            label is available.
        is_prediction
            Set when passing hard predictions rather than logits.
        """

        if is_prediction or logits_or_predictions.dim() == target.dim():
            predictions = logits_or_predictions
        else:
            predictions = logits_or_predictions.argmax(dim=1)

        predictions = predictions.detach().reshape(-1).to(torch.int64)
        truth = target.detach().reshape(-1).to(torch.int64)
        if predictions.shape != truth.shape:
            raise ValueError(
                f"prediction shape {tuple(predictions.shape)} does not match target "
                f"shape {tuple(truth.shape)}"
            )

        keep = truth != self.ignore_index
        self._ignored += int((~keep).sum().item())
        truth = truth[keep]
        predictions = predictions[keep]
        if truth.numel() == 0:
            return

        if int(truth.max().item()) >= self.num_classes or int(truth.min().item()) < 0:
            raise ValueError(
                "target contains values outside the semantic class range after "
                f"removing ignore_index {self.ignore_index}; got "
                f"[{int(truth.min().item())}, {int(truth.max().item())}]"
            )
        predictions = predictions.clamp(0, self.num_classes - 1)

        # Single bincount over the flattened pair index: O(pixels) and exact,
        # with no per-class Python loop.
        indices = truth * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self._matrix += counts.reshape(self.num_classes, self.num_classes).to(
            torch.int64
        )

    def compute(self) -> MetricResult:
        matrix = self._matrix.to(torch.float64)
        true_positive = matrix.diagonal()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        union = support + predicted - true_positive

        def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
            out = torch.full_like(numerator, float("nan"))
            positive = denominator > 0
            out[positive] = numerator[positive] / denominator[positive]
            return out

        iou = _ratio(true_positive, union)
        precision = _ratio(true_positive, predicted)
        recall = _ratio(true_positive, support)
        # Calculate F1 directly from counts rather than from precision and
        # recall. A supported class with no predictions has undefined
        # precision but a defined F1 of 0.0 when it has no true positives.
        f1_denominator = 2.0 * true_positive + (predicted - true_positive) + (
            support - true_positive
        )
        f1 = _ratio(2.0 * true_positive, f1_denominator)

        def _named(values: torch.Tensor) -> Dict[str, float]:
            return {
                name: float(values[index].item())
                for index, name in enumerate(self.class_names)
            }

        total = float(matrix.sum().item())
        true_positive_total = float(true_positive.sum().item())
        union_total = float(union.sum().item())
        flood_iou = float(iou[FLOOD_CLASS_ID].item()) if FLOOD_CLASS_ID < self.num_classes else float("nan")

        return MetricResult(
            confusion_matrix=self._matrix.clone(),
            per_class_iou=_named(iou),
            per_class_precision=_named(precision),
            per_class_recall=_named(recall),
            per_class_f1=_named(f1),
            per_class_support={
                name: int(support[index].item())
                for index, name in enumerate(self.class_names)
            },
            iou_flood_class=flood_iou,
            mean_iou=_nanmean(iou),
            macro_f1=_nanmean(f1),
            micro_iou=(true_positive_total / union_total) if union_total else float("nan"),
            pixel_accuracy=(true_positive_total / total) if total else float("nan"),
            n_valid_pixels=int(matrix.sum().item()),
            n_ignored_pixels=int(self._ignored),
        )


def _nanmean(values: torch.Tensor) -> float:
    """Mean over classes that have support, ``nan`` if none do."""

    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().item())
