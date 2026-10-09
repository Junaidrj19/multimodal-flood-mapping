#!/usr/bin/env python3
"""Fit M5 normalisation statistics from the Kuro Siwo TRAINING split only.

Produces the C5 sidecar pair — ``statistics.json`` and
``statistics_meta.json`` — with ``kind: fitted``.

Examples
--------
Fit over the whole locally available training split::

    FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo python3 scripts/fit_statistics.py \
        --output artifacts/statistics/kuro_sar_fitted

Fit over a bounded subset while checking the mechanism::

    FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo python3 scripts/fit_statistics.py \
        --output /tmp/stats_probe --limit 50

Notes
-----
Only the training split is read. :func:`fit_training_statistics` raises on any
sample from another split, so a mistaken ``--split`` cannot leak validation or
test data into the fitted parameters.

The result is OUR parameter set. Kuro's published ``data_mean``/``data_std`` are
a tolerance-banded cross-check only and are never substituted: their pre/post
basis, split basis and aggregation basis are all unstated upstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from floodmap.datasets import KuroSiwoAdapter
from floodmap.segmentation.checkpoint import git_commit
from floodmap.segmentation.normalization import StatisticsError, fit_training_statistics


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write the statistics.json / statistics_meta.json pair into.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="External Kuro Siwo root. Defaults to FLOODMAP_KURO_SIWO_ROOT.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to fit from. Only 'train' is accepted; anything else is rejected "
        "downstream as test-time leakage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of tiles, applied after deterministic sorting.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N tiles. 0 disables.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    adapter = KuroSiwoAdapter(args.root)
    refs = sorted(
        (ref for ref in adapter.discover() if ref.split == args.split),
        key=lambda ref: ref.sample_id,
    )
    if args.limit is not None:
        refs = refs[: args.limit]
    if not refs:
        print(f"no samples found in split {args.split!r}", file=sys.stderr)
        return 1

    print(f"fitting over {len(refs)} tiles from split {args.split!r}")

    def _stream():
        for index, ref in enumerate(refs, start=1):
            if args.progress_every and index % args.progress_every == 0:
                print(f"  {index}/{len(refs)}")
            yield adapter.load_sample(ref.sample_id)

    try:
        artifact = fit_training_statistics(
            _stream(), dataset_id=adapter.dataset_id, code_version=git_commit(_REPOSITORY_ROOT)
        )
    except StatisticsError as exc:
        print(f"statistics fit refused: {exc}", file=sys.stderr)
        return 2

    values_path, provenance_path = artifact.write(args.output)
    print(f"wrote {values_path}")
    print(f"wrote {provenance_path}")
    print(json.dumps({name: vars(s) for name, s in artifact.values.items()}, indent=2))
    if args.limit is not None:
        print(
            "\nNOTE: --limit was used, so this artifact was fitted on a BOUNDED SUBSET "
            "of the training split. Its provenance records n_tiles; do not treat it as "
            "a full-split fit."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
