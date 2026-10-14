"""The PyTorch dataset layer over the M4 Kuro Siwo adapter.

This sits ON TOP of ``KuroSiwoAdapter`` and does not re-scan the Kuro
directory tree. That matters beyond tidiness: the adapter is where the
catalogue/export consistency check, the partition-01 filter, the temporal-role
resolution and the three redundant validity cross-checks live. A dataset that
globbed the corpus itself would skip all of them and would happily ingest
label-free partition-00 tiles.

Memory behaviour is lazy by construction. ``__init__`` holds only the adapter's
index of lightweight references; pixels are read in ``__getitem__``, one sample
at a time. The 7,052 local training tiles are never materialised together.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from floodmap.datasets import KuroSiwoAdapter
from floodmap.datasets.contracts import CanonicalSample
from floodmap.datasets.quarantine import QuarantinePolicy

from .augment import GeometricAugmentation
from .channels import CHANNEL_ORDER
from .normalization import StatisticsArtifact
from .tensors import TensorSample, sample_to_tensors

__all__ = ["KuroSiwoSarDataset", "collate_samples"]


class KuroSiwoSarDataset(Dataset):
    """Kuro Siwo SAR pre/post tiles as 4-channel tensors for one split.

    Parameters
    ----------
    adapter
        An ``M4 KuroSiwoAdapter``. Shared between splits so the ~30 s index
        build is paid once per process rather than once per split.
    split
        ``train``, ``validation`` or ``test``, matching the official Kuro
        activation split the adapter assigns.
    statistics
        Optional normalisation artifact. When ``None`` the channels are left in
        raw clipped linear sigma0 — stated rather than silently assumed.
    augmentation
        Optional joint geometric augmentation. Should be left disabled for
        validation and test.
    sample_ids
        Optional explicit allowlist, used by the smoke and overfit tests to pin
        a tiny fixed subset. Order is normalised to the dataset's own
        deterministic ordering so a shuffled allowlist cannot change results.
    limit
        Optional cap applied AFTER deterministic sorting, so the same first-N
        tiles are taken on every run and every machine.
    reprojection
        Optional canonical-grid reprojection settings. When supplied, each
        sample is warped onto its per-sample 10 m UTM grid by the SHARED
        component in ``floodmap.preprocessing.reprojection`` — the same code
        M2 production preprocessing uses, which is what makes training and
        inference spatially comparable.
    quarantine
        Optional eligibility policy. Quarantined samples are removed from this
        split BEFORE any load, by name and with a reason, and the exclusion is
        recorded in :meth:`identity`. Passing ``None`` applies no policy and
        says so in the audit record; it does not mean "no exclusions exist".
    """

    def __init__(
        self,
        adapter: KuroSiwoAdapter,
        split: str,
        *,
        statistics: Optional[StatisticsArtifact] = None,
        augmentation: Optional[GeometricAugmentation] = None,
        sample_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        ignore_index: Optional[int] = None,
        reprojection: Optional[object] = None,
        quarantine: Optional[QuarantinePolicy] = None,
    ) -> None:
        self.adapter = adapter
        self.split = split
        self.statistics = statistics
        self.augmentation = augmentation
        self.ignore_index = ignore_index
        self.reprojection = reprojection
        self.quarantine = quarantine

        available = {
            ref.sample_id: ref for ref in adapter.discover() if ref.split == split
        }
        if sample_ids is not None:
            unknown = [sid for sid in sample_ids if sid not in available]
            if unknown:
                raise KeyError(
                    f"sample ids are not present in the {split!r} split: {unknown[:5]}"
                )
            # A caller that names samples gets an error rather than a shorter
            # list: silently returning fewer than were asked for is exactly the
            # silent skip the policy exists to prevent.
            if quarantine is not None:
                quarantine.require_eligible(list(sample_ids))
            selected = [sid for sid in sorted(available) if sid in set(sample_ids)]
        else:
            # Sorted, not directory order: os.listdir ordering is filesystem
            # dependent, and an unstable index would make a seeded run
            # irreproducible across machines.
            selected = sorted(available)

        policy = quarantine if quarantine is not None else QuarantinePolicy.empty(
            adapter.dataset_id
        )
        eligibility = policy.partition(selected)
        self.eligibility = eligibility
        selected = list(eligibility.eligible)

        if limit is not None:
            selected = selected[: int(limit)]

        self.sample_ids: Tuple[str, ...] = tuple(selected)
        self._activations = tuple(
            sorted({str(available[sid].activation_id) for sid in self.sample_ids})
        )

        if statistics is not None:
            self._mean = torch.from_numpy(statistics.mean_vector()).view(-1, 1, 1)
            self._std = torch.from_numpy(statistics.std_vector()).view(-1, 1, 1)
        else:
            self._mean = None
            self._std = None

    # -- identity ----------------------------------------------------------

    @property
    def activations(self) -> Tuple[str, ...]:
        """Activation ids contributing to this split, for the run record."""

        return self._activations

    def identity(self) -> Dict[str, object]:
        """Dataset identity recorded in checkpoints and run records."""

        return {
            "dataset_id": self.adapter.dataset_id,
            "split": self.split,
            "n_samples": len(self.sample_ids),
            "activations": list(self._activations),
            "channels": list(CHANNEL_ORDER),
            "statistics_kind": None if self.statistics is None else self.statistics.kind,
            "reprojected_to_canonical_grid": self.reprojection is not None,
            "spatial_transform": (
                None
                if self.reprojection is None
                else {
                    "shared_implementation": "floodmap.preprocessing.reprojection",
                    "resolution_m": getattr(self.reprojection, "resolution_m", None),
                    "tile_size": getattr(self.reprojection, "tile_size", None),
                    "sar_resampling": getattr(self.reprojection, "sar_resampling", None),
                    "label_resampling": getattr(self.reprojection, "label_resampling", None),
                    "nodata_mode": getattr(self.reprojection, "nodata_mode", None),
                }
            ),
            # Always present, even when nothing was excluded: an absent key is
            # indistinguishable from an unapplied policy.
            "quarantine": self.eligibility.audit_record(),
        }

    # -- Dataset protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_ids)

    def load_canonical(self, index: int) -> CanonicalSample:
        """The underlying canonical sample, for statistics fitting and tests."""

        return self.adapter.load_sample(self.sample_ids[index])

    def __getitem__(self, index: int) -> TensorSample:
        sample = self.adapter.load_sample(self.sample_ids[index])
        tensors = sample_to_tensors(
            sample, ignore_index=self.ignore_index, reprojection=self.reprojection
        )

        if self._mean is not None and self._std is not None:
            tensors["image"] = (tensors["image"] - self._mean) / self._std

        if self.augmentation is not None:
            tensors = self.augmentation(tensors)

        return tensors


def collate_samples(batch: Iterable[TensorSample]) -> Dict[str, object]:
    """Collate tensor samples, keeping the string metadata as lists.

    The default collate would try to stack ``sample_id`` strings; keeping them
    as a list preserves per-sample traceability through a batch.
    """

    items = list(batch)
    return {
        "image": torch.stack([item["image"] for item in items]),
        "target": torch.stack([item["target"] for item in items]),
        "valid_mask": torch.stack([item["valid_mask"] for item in items]),
        "sample_id": [item["sample_id"] for item in items],
        "split": [item["split"] for item in items],
        "event_id": [item["event_id"] for item in items],
    }
