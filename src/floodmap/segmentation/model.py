"""A small, conventional U-Net baseline.

This is a REFERENCE MODEL, not a claim that U-Net is the right architecture.
``configs/segmentation.yaml`` keeps ``model.architecture: null``
because that decision must follow measured performance (``AGENTS.md`` §5); this
module provides the first thing to measure, and the experiment config names it
as one candidate rather than as the project's selected architecture.

Deliberate properties:

* no pretrained weights, so there is no undeclared external training corpus in
  the provenance chain and no ImageNet 3-channel adapter hack for 4-channel SAR;
* no transformer and no attention, so the parameter count stays small enough to
  train on 7,000 224x224 tiles without a large GPU;
* configurable width and depth, because those are the two knobs worth sweeping
  before changing architecture family;
* deterministic initialisation from an explicit seed.

Encoder/decoder structure is the standard one: double 3x3 convolutions with
batch norm and ReLU, 2x2 max pooling down, transposed convolution up, skip
concatenation at each level, 1x1 convolution to logits.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .channels import NUM_CLASSES

__all__ = ["UNet", "DoubleConv"]


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU.

    ``bias=False`` on the convolutions because the following batch norm has its
    own learnable shift, making the convolution bias redundant.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net for 4-channel SAR pre/post input and 3 semantic classes.

    Parameters
    ----------
    in_channels
        Input channel count. 4 for the frozen baseline.
    num_classes
        Output logit count. 3 semantic classes; the nodata sentinel gets no
        logit and is excluded via the loss ignore index instead.
    base_channels
        Width of the first encoder stage; each stage doubles it.
    depth
        Number of downsampling stages. The spatial size must remain divisible
        by ``2 ** depth``, which is checked in ``forward`` rather than failing
        with a shape-mismatch error deep inside the decoder.
    seed
        Optional seed for deterministic initialisation. When given, the
        global RNG state is also restored after construction, so building a
        model does not shift the caller's RNG stream — otherwise dataloader
        shuffling would depend on whether a model happened to be constructed
        first, and two runs with the same seed would see their batches in
        different orders.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        depth: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be at least 1, got {base_channels}")

        # ``nn.Conv2d.__init__`` runs ``kaiming_uniform_`` against the GLOBAL
        # RNG, so simply seeding ``reset_parameters`` is not enough to leave the
        # stream untouched. The pre-construction state is captured here and
        # restored at the end.
        rng_state = torch.get_rng_state() if seed is not None else None

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.depth = int(depth)

        widths: Sequence[int] = [base_channels * (2**level) for level in range(depth)]

        self.encoders = nn.ModuleList()
        channels = self.in_channels
        for width in widths:
            self.encoders.append(DoubleConv(channels, width))
            channels = width

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(channels, channels * 2)
        channels = channels * 2

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.upsamples.append(
                nn.ConvTranspose2d(channels, width, kernel_size=2, stride=2)
            )
            # width (from the upsample) + width (from the skip connection).
            self.decoders.append(DoubleConv(width * 2, width))
            channels = width

        self.head = nn.Conv2d(channels, self.num_classes, kernel_size=1)

        self.reset_parameters(seed=seed)

        if rng_state is not None:
            torch.set_rng_state(rng_state)

    def reset_parameters(self, *, seed: int | None = None) -> None:
        """Kaiming-initialise convolutions, deterministically when seeded.

        A local generator is used rather than a global ``manual_seed`` so
        constructing a model does not perturb the caller's RNG stream — which
        would make dataloader shuffling depend on whether a model was built
        first.
        """

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_out = (
                    module.out_channels
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
                std = (2.0 / float(fan_out)) ** 0.5
                with torch.no_grad():
                    if generator is None:
                        module.weight.normal_(0.0, std)
                    else:
                        module.weight.copy_(
                            torch.normal(
                                mean=0.0,
                                std=std,
                                size=tuple(module.weight.shape),
                                generator=generator,
                            )
                        )
                    if module.bias is not None:
                        module.bias.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                with torch.no_grad():
                    module.weight.fill_(1.0)
                    module.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected [N, C, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        divisor = 2**self.depth
        height, width = int(x.shape[-2]), int(x.shape[-1])
        if height % divisor or width % divisor:
            raise ValueError(
                f"input spatial size {(height, width)} must be divisible by "
                f"{divisor} for depth {self.depth}; 224x224 Kuro tiles satisfy this "
                "up to depth 5"
            )

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips)
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.head(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config, *, seed: int | None = None) -> "UNet":
        return cls(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            depth=config.depth,
            seed=seed,
        )
