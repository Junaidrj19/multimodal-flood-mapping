"""Deterministic training-sample selection for bounded M5 experiments.

A bounded run trains on a SUBSET of the 7,051 eligible local training tiles.
Which subset is a decision, not an implementation detail, so it is made once,
recorded as an artifact, and replayed from that artifact rather than recomputed
inside a training run.

Why not ``limit_train``
-----------------------
``dataset.limit_train`` takes the first N sample IDs after a deterministic sort.
It is reproducible, but Kuro sample IDs are prefixed with their activation
(``1111004:<grid uuid>``), so a sorted prefix is drawn almost entirely from the
lowest-numbered activation. A 256-tile cap over three training activations would
have come from 1111004 alone: one flood event, one acquisition geometry, one
region. That is a worse-than-random sample of the corpus, and a metric measured
on it would not be comparable to one measured on the whole split.

What this module does instead
-----------------------------
* allocate the budget across eligible training activations in proportion to how
  many eligible tiles each one has, using **largest remainder** (see
  :func:`allocate_largest_remainder`);
* guarantee every eligible training activation is represented, so no event
  silently drops out of a bounded run;
* sort candidates **before** any seeded draw, because ``adapter.discover()``
  order follows the filesystem and an unstable pool would make a seeded
  selection irreproducible across machines;
* draw from one seeded stream in sorted activation order, so the same seed and
  the same corpus always give the same 256 IDs;
* re-check the result against the quarantine policy. The selection pool is
  already filtered, so this is a post-condition rather than a filter: if a
  quarantined ID ever reaches the selection, that is a defect and it raises.

The manifest carries sample IDs and provenance only — no pixel data — in the
same spirit as ``configs/quarantine/kuro_siwo.yaml``. It is written under
``artifacts/`` because it is machine-generated and fully regenerable from its
own recorded seed, budget and policy version, matching how fitted normalisation
statistics are handled.

This module selects from the TRAINING split only. It has no code path that
reads, selects from, or counts the test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from floodmap.datasets.quarantine import QuarantinePolicy

__all__ = [
    "ALLOCATION_METHOD",
    "SELECTOR_VERSION",
    "ActivationAllocation",
    "SampleManifest",
    "SampleSelection",
    "SampleSelectionError",
    "allocate_largest_remainder",
    "read_manifest",
    "select_training_samples",
    "write_manifest",
]

#: Bumped when the selection algorithm changes in a way that would produce a
#: different subset from the same seed. A manifest records the version that
#: produced it, so a stale artifact is identifiable rather than silently reused.
SELECTOR_VERSION = "1.0.0"

ALLOCATION_METHOD = "largest_remainder"

#: Provenance keys without which a manifest cannot be reproduced or audited.
#: A manifest missing any of them is rejected rather than used.
REQUIRED_PROVENANCE_KEYS: Tuple[str, ...] = (
    "selector_version",
    "allocation_method",
    "seed",
    "budget",
    "dataset_id",
    "split",
    "sample_id_sha256",
    "activations",
)


class SampleSelectionError(RuntimeError):
    """A selection is impossible, or a manifest is malformed or inconsistent."""


def _digest(sample_ids: Sequence[str]) -> str:
    """Order-independent digest of a selection.

    Computed over the SORTED IDs so a manifest that was rewritten in a
    different order still verifies, while a changed membership does not.
    """

    joined = "\n".join(sorted(sample_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActivationAllocation:
    """How much of the budget one activation received, and why."""

    activation_id: str
    n_eligible: int
    #: The unrounded proportional share, kept so an auditor can see that the
    #: integer allocation really is the rounding of a proportional quota.
    exact_quota: float
    allocated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_id": self.activation_id,
            "n_eligible": self.n_eligible,
            "exact_quota": round(self.exact_quota, 6),
            "allocated": self.allocated,
        }


def allocate_largest_remainder(
    counts: Mapping[str, int],
    budget: int,
    *,
    minimum_per_group: int = 1,
) -> Dict[str, int]:
    """Split ``budget`` across groups in proportion to ``counts``.

    Largest remainder, stated plainly: each group's exact share is
    ``budget * n_group / n_total``. Every group first takes the floor of its
    share; the seats left over by flooring go to the groups with the largest
    fractional parts, one each. Ties are broken by group ID so the result never
    depends on mapping iteration order.

    ``minimum_per_group`` is then enforced as a REPAIR, not as a reservation
    taken off the top. Reserving first would distort every group's proportional
    share even when the minimum was never going to bind; repairing afterwards
    leaves the proportional result exactly intact in that case, which is the
    case for the three Kuro training activations and a 256-tile budget. When the
    minimum does bind, the shortfall is taken from the largest allocation,
    deterministically.

    Raises rather than silently returning a smaller allocation: a caller that
    asked for a budget must not be handed fewer samples without being told.
    """

    groups = sorted(counts)
    if not groups:
        raise SampleSelectionError("no groups to allocate across")
    if budget <= 0:
        raise SampleSelectionError(f"budget must be positive, got {budget}")
    if minimum_per_group < 0:
        raise SampleSelectionError(
            f"minimum_per_group must not be negative, got {minimum_per_group}"
        )

    total = sum(int(counts[g]) for g in groups)
    if total <= 0:
        raise SampleSelectionError("every group is empty; nothing can be selected")
    if budget > total:
        raise SampleSelectionError(
            f"budget {budget} exceeds the {total} available samples across "
            f"{len(groups)} group(s); a bounded run must not ask for more than exists"
        )
    if minimum_per_group * len(groups) > budget:
        raise SampleSelectionError(
            f"budget {budget} cannot give every one of {len(groups)} group(s) at least "
            f"{minimum_per_group}; raise the budget or lower the minimum"
        )
    for group in groups:
        if int(counts[group]) < minimum_per_group:
            raise SampleSelectionError(
                f"group {group!r} has only {counts[group]} sample(s) and cannot supply "
                f"the required minimum of {minimum_per_group}"
            )

    exact = {g: budget * int(counts[g]) / total for g in groups}
    allocation = {g: int(math.floor(exact[g])) for g in groups}

    # Flooring a share that is itself bounded by the group size can never
    # exceed the group size, so the only capacity pressure is on the remainder
    # seats below.
    leftover = budget - sum(allocation.values())
    remainder_order = sorted(
        groups, key=lambda g: (-(exact[g] - math.floor(exact[g])), g)
    )
    while leftover > 0:
        progressed = False
        for group in remainder_order:
            if leftover == 0:
                break
            if allocation[group] < int(counts[group]):
                allocation[group] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            raise SampleSelectionError(
                f"{leftover} sample(s) of the budget could not be placed: every group is "
                "at its available capacity"
            )

    for group in groups:
        while allocation[group] < minimum_per_group:
            donors = [
                d
                for d in sorted(groups, key=lambda x: (-allocation[x], x))
                if d != group and allocation[d] > minimum_per_group
            ]
            if not donors:
                raise SampleSelectionError(
                    f"cannot raise group {group!r} to the minimum of {minimum_per_group} "
                    "without pushing another group below it"
                )
            allocation[donors[0]] -= 1
            allocation[group] += 1

    if sum(allocation.values()) != budget:
        raise SampleSelectionError(
            f"allocation totals {sum(allocation.values())}, expected {budget}"
        )
    return allocation


@dataclass(frozen=True)
class SampleSelection:
    """A completed selection, ready to persist."""

    sample_ids: Tuple[str, ...]
    allocations: Tuple[ActivationAllocation, ...]
    provenance: Mapping[str, Any]

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "sample_ids": list(self.sample_ids),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class SampleManifest:
    """A loaded, validated manifest."""

    sample_ids: Tuple[str, ...]
    provenance: Mapping[str, Any]
    path: Optional[Path] = None

    def identity(self) -> Dict[str, Any]:
        """Compact record for the run record and checkpoint.

        Deliberately does not inline 256 sample IDs: the digest identifies the
        selection, and the manifest itself holds the membership.
        """

        return {
            "applied": True,
            "path": str(self.path) if self.path else None,
            "n_samples": len(self.sample_ids),
            "selector_version": self.provenance.get("selector_version"),
            "allocation_method": self.provenance.get("allocation_method"),
            "seed": self.provenance.get("seed"),
            "budget": self.provenance.get("budget"),
            "dataset_id": self.provenance.get("dataset_id"),
            "split": self.provenance.get("split"),
            "sample_id_sha256": self.provenance.get("sample_id_sha256"),
            "activations": self.provenance.get("activations"),
            "quarantine": self.provenance.get("quarantine"),
        }


def select_training_samples(
    adapter,
    *,
    budget: int,
    seed: int,
    split: str = "train",
    quarantine: Optional[QuarantinePolicy] = None,
    minimum_per_activation: int = 1,
    commit: Optional[str] = None,
) -> SampleSelection:
    """Select ``budget`` eligible sample IDs from ``split``, deterministically.

    Parameters
    ----------
    adapter
        An M4 ``KuroSiwoAdapter``. Its index is the only source of sample IDs
        and activation assignment, so this does not re-scan the corpus and
        cannot disagree with what the dataset layer will later load.
    budget
        Exact number of unique sample IDs to return.
    seed
        Seeds the single draw stream. The same seed, corpus and policy version
        reproduce the same selection.
    split
        Must not be ``test``; refused outright rather than trusted to a caller.
    quarantine
        Eligibility policy. ``None`` applies no policy, and the manifest says
        so rather than implying none was needed.
    """

    if split == "test":
        raise SampleSelectionError(
            "the test split must never be sampled: it is not used for training, "
            "selection, tuning or any intermediate measurement"
        )

    refs = [ref for ref in adapter.discover() if ref.split == split]
    if not refs:
        raise SampleSelectionError(
            f"the {split!r} split resolved to 0 samples; check the external dataset "
            "root and the official activation split"
        )

    # Sorted before anything seeded touches it. os.listdir order is filesystem
    # dependent, and a pool in filesystem order would make the same seed select
    # a different subset on another machine.
    discovered_ids = sorted(ref.sample_id for ref in refs)
    activation_of = {ref.sample_id: str(ref.activation_id) for ref in refs}

    policy = quarantine if quarantine is not None else QuarantinePolicy.empty(
        adapter.dataset_id
    )
    eligibility = policy.partition(discovered_ids)

    pools: Dict[str, list] = {}
    for sample_id in eligibility.eligible:
        pools.setdefault(activation_of[sample_id], []).append(sample_id)
    counts = {activation: len(ids) for activation, ids in pools.items()}

    allocation = allocate_largest_remainder(
        counts, budget, minimum_per_group=minimum_per_activation
    )

    rng = random.Random(seed)
    selected: list = []
    allocations: list = []
    total_eligible = sum(counts.values())
    for activation in sorted(pools):
        pool = sorted(pools[activation])
        take = allocation[activation]
        picked = rng.sample(pool, take)
        selected.extend(sorted(picked))
        allocations.append(
            ActivationAllocation(
                activation_id=activation,
                n_eligible=counts[activation],
                exact_quota=budget * counts[activation] / total_eligible,
                allocated=take,
            )
        )

    sample_ids = tuple(sorted(selected))

    # Post-conditions. Each of these would otherwise be a silent, plausible
    # wrong answer rather than a failure.
    if len(set(sample_ids)) != len(sample_ids):
        raise SampleSelectionError("the selection contains duplicate sample IDs")
    if len(sample_ids) != budget:
        raise SampleSelectionError(
            f"selected {len(sample_ids)} sample(s), expected exactly {budget}"
        )
    eligible_set = set(eligibility.eligible)
    stray = [sid for sid in sample_ids if sid not in eligible_set]
    if stray:
        raise SampleSelectionError(
            f"{len(stray)} selected sample(s) are not in the eligible pool: {stray[:5]}"
        )
    # Belt and braces: the pool was already filtered, so a hit here is a defect
    # in this module rather than a policy outcome.
    policy.require_eligible(list(sample_ids))
    if any(alloc.allocated < minimum_per_activation for alloc in allocations):
        raise SampleSelectionError(
            "an eligible training activation received fewer samples than the required "
            "minimum; a bounded run must represent every activation"
        )

    provenance: Dict[str, Any] = {
        "selector_version": SELECTOR_VERSION,
        "allocation_method": ALLOCATION_METHOD,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(seed),
        "budget": int(budget),
        "dataset_id": adapter.dataset_id,
        "split": split,
        "minimum_per_activation": int(minimum_per_activation),
        "n_discovered_in_split": len(discovered_ids),
        "n_eligible": total_eligible,
        "n_excluded_by_quarantine": eligibility.n_excluded,
        "quarantine": eligibility.audit_record(),
        "activations": [alloc.to_dict() for alloc in allocations],
        "sample_id_sha256": _digest(sample_ids),
        "commit": commit,
        "note": (
            "Proportional largest-remainder allocation over eligible training "
            "activations, then a seeded draw within each activation from a "
            "lexicographically sorted pool. The test split is not touched."
        ),
    }

    return SampleSelection(
        sample_ids=sample_ids,
        allocations=tuple(allocations),
        provenance=provenance,
    )


def write_manifest(
    path: Path, selection: SampleSelection, *, overwrite: bool = False
) -> Path:
    """Write a selection manifest as JSON.

    Refuses to clobber an existing manifest unless told to: a run recorded
    against a manifest path must keep meaning what it meant.
    """

    target = Path(path)
    if target.exists() and not overwrite:
        raise SampleSelectionError(
            f"{target} already exists; pass overwrite=True to replace it. A run record "
            "that names this path would otherwise silently refer to a different "
            "selection than the one it trained on"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(selection.to_manifest(), indent=2) + "\n", encoding="utf-8"
    )
    return target


def read_manifest(
    path: Path,
    *,
    dataset_id: Optional[str] = None,
    split: Optional[str] = None,
) -> SampleManifest:
    """Load and validate a selection manifest.

    Validation is deliberately strict. A manifest is an input to a training
    run, and every check here corresponds to a way a run could otherwise train
    on something other than what its config appears to say.
    """

    resolved = Path(path)
    if not resolved.is_file():
        raise SampleSelectionError(f"sample manifest not found: {resolved}")

    try:
        parsed = json.loads(resolved.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SampleSelectionError(f"{resolved} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise SampleSelectionError(f"{resolved} must parse to a mapping")

    raw_ids = parsed.get("sample_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise SampleSelectionError(
            f"{resolved}: sample_ids must be a non-empty list"
        )
    if not all(isinstance(sid, str) and sid for sid in raw_ids):
        raise SampleSelectionError(f"{resolved}: every sample_id must be a non-empty string")
    if len(set(raw_ids)) != len(raw_ids):
        duplicates = sorted({sid for sid in raw_ids if raw_ids.count(sid) > 1})
        raise SampleSelectionError(
            f"{resolved}: sample_ids contains {len(duplicates)} duplicate id(s): "
            f"{duplicates[:5]}; a duplicated tile would be over-weighted in training"
        )

    provenance = parsed.get("provenance")
    if not isinstance(provenance, Mapping):
        raise SampleSelectionError(
            f"{resolved}: provenance must be a mapping; an unprovenanced selection is "
            "unauditable and must not drive a run"
        )
    missing = [key for key in REQUIRED_PROVENANCE_KEYS if provenance.get(key) is None]
    if missing:
        raise SampleSelectionError(
            f"{resolved}: provenance is missing required key(s) {missing}"
        )

    sample_ids = tuple(raw_ids)
    declared_digest = str(provenance["sample_id_sha256"])
    actual_digest = _digest(sample_ids)
    if declared_digest != actual_digest:
        raise SampleSelectionError(
            f"{resolved}: sample_id_sha256 {declared_digest[:12]}... does not match the "
            f"listed ids ({actual_digest[:12]}...); the manifest was edited after it was "
            "generated and no longer describes a reproducible selection"
        )

    declared_budget = int(provenance["budget"])
    if declared_budget != len(sample_ids):
        raise SampleSelectionError(
            f"{resolved}: provenance declares a budget of {declared_budget} but lists "
            f"{len(sample_ids)} sample id(s)"
        )

    if dataset_id is not None and provenance.get("dataset_id") != dataset_id:
        raise SampleSelectionError(
            f"{resolved} was generated for dataset_id "
            f"{provenance.get('dataset_id')!r} but is being loaded for {dataset_id!r}; "
            "a selection must not be applied to another corpus"
        )
    if split is not None and provenance.get("split") != split:
        raise SampleSelectionError(
            f"{resolved} was generated for the {provenance.get('split')!r} split but is "
            f"being loaded for {split!r}"
        )
    if provenance.get("split") == "test":
        raise SampleSelectionError(
            f"{resolved} names the test split; it must never be loaded"
        )

    return SampleManifest(
        sample_ids=sample_ids, provenance=dict(provenance), path=resolved
    )
