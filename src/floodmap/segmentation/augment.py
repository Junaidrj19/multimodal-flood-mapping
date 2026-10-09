"""Geometric augmentation applied jointly to image, target and validity mask.

Scope is deliberately minimal for the first baseline:

* ONLY flips and 90-degree rotations. These are exact lattice operations: they
  permute pixels without resampling, so they introduce no interpolation and
  cannot manufacture a backscatter value the sensor never measured.
* NO radiometric transforms. Brightness, contrast, gamma and additive noise all
  alter sigma0, which IS the measurement. Scaling it would train the model on a
  physically meaningless distribution, and the flood signal here is partly a
  pre/post amplitude relationship.
* NO SAR-specific augmentation (synthetic speckle, despeckling jitter). Kuro
  Siwo arrives already Lee Sigma filtered (``configs/data.yaml`` -> C4), so
  injecting speckle would push training data further from both the corpus and
  the unfiltered production path rather than closer.
* DISABLED by default. An augmentation that helps has to be shown to help.

The single most important property is that one decision is drawn per sample and
applied to ALL FOUR channels, the target and the mask together. Augmenting
channels independently would destroy the pre/post pixel correspondence that the
change signal depends on, and augmenting the image without the target would
silently mis-align supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .tensors import TensorSample

__all__ = ["GeometricAugmentation"]


@dataclass(frozen=True)
class GeometricAugmentation:
    """Flip/rotate an entire sample as one unit.

    Parameters
    ----------
    enabled
        Master switch. ``False`` makes ``__call__`` an identity.
    horizontal_flip, vertical_flip
        Whether each flip is eligible, each drawn independently at p=0.5.
    rotate_90
        Whether a k*90-degree rotation is eligible, k drawn from {0, 1, 2, 3}.
    """

    enabled: bool = False
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotate_90: bool = False

    def __call__(
        self, sample: TensorSample, generator: Optional[torch.Generator] = None
    ) -> TensorSample:
        if not self.enabled:
            return sample

        image = sample["image"]
        target = sample["target"]
        valid = sample["valid_mask"]

        # One decision per sample, shared by every payload. The image is
        # [C, H, W] and the target/mask are [H, W], so spatial dims are the
        # last two in all three and the same negative axes apply to each.
        if self.horizontal_flip and self._coin(generator):
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
            valid = torch.flip(valid, dims=(-1,))

        if self.vertical_flip and self._coin(generator):
            image = torch.flip(image, dims=(-2,))
            target = torch.flip(target, dims=(-2,))
            valid = torch.flip(valid, dims=(-2,))

        if self.rotate_90:
            turns = int(torch.randint(0, 4, (1,), generator=generator).item())
            if turns:
                image = torch.rot90(image, turns, dims=(-2, -1))
                target = torch.rot90(target, turns, dims=(-2, -1))
                valid = torch.rot90(valid, turns, dims=(-2, -1))

        updated = TensorSample(sample)
        # contiguous() because flip/rot90 return views with negative or
        # permuted strides, which some backends reject downstream.
        updated["image"] = image.contiguous()
        updated["target"] = target.contiguous()
        updated["valid_mask"] = valid.contiguous()
        return updated

    @staticmethod
    def _coin(generator: Optional[torch.Generator]) -> bool:
        return bool(torch.randint(0, 2, (1,), generator=generator).item())

    @classmethod
    def from_config(cls, config) -> "GeometricAugmentation":
        return cls(
            enabled=config.enabled,
            horizontal_flip=config.horizontal_flip,
            vertical_flip=config.vertical_flip,
            rotate_90=config.rotate_90,
        )
