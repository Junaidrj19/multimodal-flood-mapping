"""``CanonicalSample`` to PyTorch tensors.

This is the only place the M4 canonical boundary is turned into model input, so
every assumption about channel order, representation, label semantics and
validity is checked here rather than trusted.

What this module refuses to do, and why:

* It does not convert to decibel. Kuro Siwo is delivered and clipped in linear
  sigma0 and the frozen M4 contract records that representation on the sample;
  converting here would make the input silently disagree with both the fitted
  statistics artifact and ``configs/data.yaml``.
* It does not turn nodata into a fourth class. Stored label 3 becomes the loss
  ignore index, not a trainable category.
* It does not remap valid semantic classes. Kuro's native {0, 1, 2} already are
  the class indices; the identity mapping is asserted rather than applied, so a
  future corpus with a different encoding fails loudly instead of being
  silently mis-labelled.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from floodmap.datasets.contracts import CanonicalSample

from .channels import (
    CHANNEL_ORDER,
    CHANNEL_SOURCES,
    IGNORE_INDEX,
    REQUIRED_SAR_REPRESENTATION,
    SEMANTIC_VALUES,
)

__all__ = ["TensorSample", "sample_to_tensors", "stack_channels"]


class TensorContractError(ValueError):
    """A canonical sample does not satisfy the M5 tensor contract."""


class TensorSample(dict):
    """A plain dict of tensors, with named access for readability.

    A dict rather than a dataclass because this is what a ``DataLoader``
    collates; the named properties exist so call sites read clearly.
    """

    @property
    def image(self) -> torch.Tensor:
        return self["image"]

    @property
    def target(self) -> torch.Tensor:
        return self["target"]

    @property
    def valid_mask(self) -> torch.Tensor:
        return self["valid_mask"]


def stack_channels(sample: CanonicalSample) -> np.ndarray:
    """Build the frozen 4-channel stack from a canonical sample.

    Returns ``float32`` of shape ``[4, H, W]`` in ``CHANNEL_ORDER``.
    """

    if sample.sar_representation != REQUIRED_SAR_REPRESENTATION:
        raise TensorContractError(
            f"the M5 SAR baseline requires {REQUIRED_SAR_REPRESENTATION!r} input, got "
            f"{sample.sar_representation!r} for sample {sample.sample_id!r}; a decibel "
            "corpus must not be fed through statistics fitted on linear sigma0"
        )

    planes = []
    for name in CHANNEL_ORDER:
        role, polarisation = CHANNEL_SOURCES[name]
        epoch = sample.sar.get(role)
        if epoch is None:
            raise TensorContractError(
                f"sample {sample.sample_id!r} has no {role!r} SAR epoch; the pre/post "
                "baseline cannot be built from a single epoch, and substituting the "
                "post-event image for the missing pre-event one would make the change "
                "signal identically zero"
            )
        plane = epoch.get(polarisation)
        if plane is None:
            raise TensorContractError(
                f"sample {sample.sample_id!r} is missing {role}/{polarisation}"
            )
        if plane.shape != sample.source_shape:
            raise TensorContractError(
                f"{role}/{polarisation} shape {plane.shape} does not match the sample "
                f"shape {sample.source_shape}"
            )
        planes.append(np.asarray(plane, dtype=np.float32))

    return np.stack(planes, axis=0)


def _target_from_labels(sample: CanonicalSample, ignore_index: int) -> np.ndarray:
    """Labels to a CrossEntropyLoss target, with invalid pixels ignored.

    Validity comes from the M4 canonical mask (Kuro ``MK0_MNA == 1``) rather
    than from the label raster, so the two redundant mechanisms are still
    cross-checkable instead of being collapsed into one.
    """

    labels = np.asarray(sample.labels)
    valid = np.asarray(sample.validity, dtype=bool)

    declared = sample.label_scheme.semantic_values
    if tuple(declared) != SEMANTIC_VALUES:
        raise TensorContractError(
            f"sample {sample.sample_id!r} declares semantic values {tuple(declared)}, but "
            f"the M5 baseline is frozen on {SEMANTIC_VALUES}; a corpus with a different "
            "encoding needs an explicit label mapping, not an implicit one"
        )

    observed = set(int(value) for value in np.unique(labels[valid])) if valid.any() else set()
    unexpected = observed - set(SEMANTIC_VALUES)
    if unexpected:
        raise TensorContractError(
            f"sample {sample.sample_id!r} has valid pixels with non-semantic label values "
            f"{sorted(unexpected)}; nodata must be marked invalid, not trained on"
        )

    target = labels.astype(np.int64, copy=True)
    target[~valid] = ignore_index
    return target


def sample_to_tensors(
    sample: CanonicalSample,
    *,
    ignore_index: Optional[int] = None,
    reprojection: Optional["ReprojectionSettings"] = None,
) -> TensorSample:
    """Convert one canonical sample into model input, target and validity mask.

    Parameters
    ----------
    sample
        A ``CanonicalSample`` from an M4 adapter.
    ignore_index
        Loss ignore index. Defaults to the sample's own
        ``nodata.ignore_index``, which the Kuro adapter sets to 3. Passed
        explicitly only by tests that need to prove the plumbing is honoured.
    reprojection
        When given, the sample is first warped onto its per-sample canonical
        UTM grid using the SHARED spatial component in
        ``floodmap.preprocessing.reprojection`` — the same code M2 production
        preprocessing uses. When ``None`` the sample is used on its delivered
        source grid, which for Kuro is 10 PROJECTED units in EPSG:3857 rather
        than 10 m of true ground.

    Returns
    -------
    TensorSample
        ``image`` ``float32[4, H, W]``, ``target`` ``int64[H, W]``,
        ``valid_mask`` ``bool[H, W]``.
    """

    resolved_ignore = (
        int(ignore_index) if ignore_index is not None else int(sample.nodata.ignore_index)
    )
    if resolved_ignore in SEMANTIC_VALUES:
        raise TensorContractError(
            f"ignore_index {resolved_ignore} collides with a semantic class "
            f"{SEMANTIC_VALUES}; ignored pixels would be indistinguishable from "
            "observed ones"
        )

    if reprojection is not None:
        from .spatial import reproject_canonical_sample

        warped = reproject_canonical_sample(
            sample,
            ignore_index=resolved_ignore,
            resolution_m=reprojection.resolution_m,
            tile_size=reprojection.tile_size,
            nodata_policy=reprojection.nodata_policy(),
            sar_resampling=reprojection.sar_resampling,
            label_resampling=reprojection.label_resampling,
        )
        return TensorSample(
            image=torch.from_numpy(np.ascontiguousarray(warped.image)),
            target=torch.from_numpy(np.ascontiguousarray(warped.target)),
            valid_mask=torch.from_numpy(np.ascontiguousarray(warped.valid_mask)),
            sample_id=sample.sample_id,
            split=sample.split,
            event_id="" if sample.event_id is None else str(sample.event_id),
            spatial_provenance=warped.provenance,
        )

    image = stack_channels(sample)
    target = _target_from_labels(sample, resolved_ignore)
    valid = np.asarray(sample.validity, dtype=bool)

    return TensorSample(
        image=torch.from_numpy(image),
        target=torch.from_numpy(target),
        valid_mask=torch.from_numpy(valid.copy()),
        sample_id=sample.sample_id,
        split=sample.split,
        event_id="" if sample.event_id is None else str(sample.event_id),
    )


def tensor_shapes(sample: TensorSample) -> Tuple[Tuple[int, ...], ...]:
    """Shapes of the three payload tensors, for diagnostics and tests."""

    return (
        tuple(sample.image.shape),
        tuple(sample.target.shape),
        tuple(sample.valid_mask.shape),
    )
