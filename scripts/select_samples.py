#!/usr/bin/env python3
"""Generate a deterministic training-sample selection manifest.

A bounded M5 run trains on a subset of the eligible training tiles. WHICH
subset is a decision, so it is made once here, written to an auditable
manifest, and replayed by every run that names that manifest — rather than
recomputed inside training, where it would be invisible.

The budget is spread across the eligible training activations in proportion to
their eligible tile counts (largest remainder), and every eligible activation
is guaranteed at least one tile. See
``src/floodmap/segmentation/selection.py`` for why a plain ``limit_train``
prefix is not an acceptable substitute.

This script reads the TRAINING split only. It does not read, count or select
from the test split.

Examples
--------
Select the 256-tile M5 MPS probe set::

    FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo python3 scripts/select_samples.py \
        --budget 256 --seed 20261008 \
        --output artifacts/selections/m5_mps_probe2.json

Inspect the allocation without writing anything::

    FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo python3 scripts/select_samples.py \
        --budget 256 --seed 20261008 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from floodmap.datasets import KuroSiwoAdapter
from floodmap.datasets.quarantine import load_quarantine
from floodmap.segmentation.checkpoint import git_commit
from floodmap.segmentation.selection import (
    SampleSelectionError,
    read_manifest,
    select_training_samples,
    write_manifest,
)
from floodmap.utils.config import project_root


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Exact number of unique training sample IDs to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed for the draw. The same seed and corpus reproduce the selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the manifest. Relative paths resolve against the "
        "project root. Required unless --dry-run.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Kuro Siwo root. Defaults to $FLOODMAP_KURO_SIWO_ROOT.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to select from. Default 'train'. 'test' is refused.",
    )
    parser.add_argument(
        "--minimum-per-activation",
        type=int,
        default=1,
        help="Smallest number of tiles any eligible activation may receive. Default 1.",
    )
    parser.add_argument(
        "--no-quarantine",
        action="store_true",
        help="Apply NO eligibility policy. Recorded as such in the manifest. "
        "Off by default: the policy is normally required.",
    )
    parser.add_argument(
        "--quarantine-path",
        type=Path,
        default=None,
        help="Explicit quarantine policy path. Defaults to the conventional location.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the allocation and provenance, then exit without writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest at --output. Off by default so a run "
        "record naming that path cannot silently come to mean a different selection.",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Python logging level. Default INFO."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.output is None and not args.dry_run:
        print(
            "--output is required unless --dry-run is passed", file=sys.stderr
        )
        return 2

    adapter = KuroSiwoAdapter(args.dataset_root)
    # Required by default: selecting without the policy would quietly admit a
    # sample the project has already decided is ineligible.
    policy = (
        None
        if args.no_quarantine
        else load_quarantine(
            adapter.dataset_id, path=args.quarantine_path, required=True
        )
    )

    try:
        selection = select_training_samples(
            adapter,
            budget=args.budget,
            seed=args.seed,
            split=args.split,
            quarantine=policy,
            minimum_per_activation=args.minimum_per_activation,
            commit=git_commit(),
        )
    except SampleSelectionError as exc:
        print(f"selection failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "n_selected": len(selection.sample_ids),
        "seed": selection.provenance["seed"],
        "split": selection.provenance["split"],
        "n_discovered_in_split": selection.provenance["n_discovered_in_split"],
        "n_eligible": selection.provenance["n_eligible"],
        "n_excluded_by_quarantine": selection.provenance["n_excluded_by_quarantine"],
        "allocation_method": selection.provenance["allocation_method"],
        "activations": selection.provenance["activations"],
        "sample_id_sha256": selection.provenance["sample_id_sha256"],
    }

    if args.dry_run:
        print(json.dumps({"dry_run": True, **summary}, indent=2))
        return 0

    target = Path(args.output)
    if not target.is_absolute():
        target = project_root() / target
    try:
        written = write_manifest(target, selection, overwrite=args.overwrite)
    except SampleSelectionError as exc:
        print(f"manifest not written: {exc}", file=sys.stderr)
        return 2

    # Read it straight back through the same validation a training run uses, so
    # a manifest that would be rejected at train time is rejected here instead.
    verified = read_manifest(
        written, dataset_id=adapter.dataset_id, split=args.split
    )
    print(
        json.dumps(
            {
                "manifest": str(written),
                "verified_on_read": len(verified.sample_ids),
                **summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
