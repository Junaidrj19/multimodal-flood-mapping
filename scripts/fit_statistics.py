#!/usr/bin/env python3
"""Fit M5 normalisation statistics from the Kuro Siwo TRAINING split only.

Produces the C5 sidecar pair — ``statistics.json`` and
``statistics_meta.json`` — with ``kind: fitted``.

The spatial and channel pipeline is taken from an EXPERIMENT CONFIG rather than
from flags, so the fit cannot silently differ from the run that will consume
it. Bilinear resampling changes the variance of the field it is applied to, so
a std fitted on the delivered source grid is systematically wrong for a
reprojected training input.

Examples
--------
Fit for the real baseline, using its own reprojection and quarantine settings::

    FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo python3 scripts/fit_statistics.py \
        --config m5_kuro_sar_unet \
        --output artifacts/statistics/kuro_sar_fitted

Bounded mechanism check (records n_tiles; NOT a full-split fit)::

    FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo python3 scripts/fit_statistics.py \
        --config m5_kuro_sar_unet --output /tmp/probe --limit 50

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
import time
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from floodmap.datasets import KuroSiwoAdapter
from floodmap.datasets.quarantine import QuarantinePolicy
from floodmap.segmentation.checkpoint import git_commit
from floodmap.segmentation.config import load_experiment
from floodmap.segmentation.normalization import StatisticsError, fit_training_statistics
from floodmap.segmentation.train import resolve_quarantine


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment config whose reprojection/quarantine/channel pipeline to use. "
        "Taking these from the config rather than from flags is what keeps the fit "
        "and the training run identical.",
    )
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
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of tiles, applied after deterministic sorting. "
        "Produces a BOUNDED-SUBSET artifact, not a full-split fit.",
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
    config = load_experiment(args.config)

    split = config.dataset.train_split
    if split != "train":
        print(
            f"refusing to fit from split {split!r}: fitted statistics come from the "
            "training split only",
            file=sys.stderr,
        )
        return 2

    adapter = KuroSiwoAdapter(args.root if args.root else (
        Path(config.dataset.root) if config.dataset.root else None
    ))
    quarantine = resolve_quarantine(config)
    reprojection = config.reprojection if config.reprojection.enabled else None

    discovered = sorted(
        (ref for ref in adapter.discover() if ref.split == split),
        key=lambda ref: ref.sample_id,
    )
    policy = quarantine or QuarantinePolicy.empty(adapter.dataset_id)
    eligibility = policy.partition([ref.sample_id for ref in discovered])

    eligible = list(eligibility.eligible)
    if args.limit is not None:
        eligible = eligible[: args.limit]
    if not eligible:
        print(f"no eligible samples in split {split!r}", file=sys.stderr)
        return 1

    print(f"experiment:        {config.name} (v{config.version})")
    print(f"split:             {split}")
    print(f"discovered:        {len(discovered)}")
    print(f"quarantined:       {eligibility.n_excluded}")
    for entry in eligibility.excluded:
        print(f"  - EXCLUDED {entry.sample_id}  [{entry.reason_code}]")
    print(f"eligible:          {eligibility.n_eligible}")
    print(f"fitting over:      {len(eligible)}")
    print(
        "reprojection:      "
        + (
            "disabled (delivered source grid)"
            if reprojection is None
            else f"enabled, {reprojection.resolution_m} m, tile {reprojection.tile_size}, "
            f"{reprojection.sar_resampling}/{reprojection.label_resampling}, "
            f"nodata={reprojection.nodata_mode}"
        )
    )
    print()

    by_id = {ref.sample_id: ref for ref in discovered}
    started = time.time()

    def _stream():
        for index, sample_id in enumerate(eligible, start=1):
            if args.progress_every and index % args.progress_every == 0:
                rate = index / max(time.time() - started, 1e-9)
                remaining = (len(eligible) - index) / max(rate, 1e-9)
                print(
                    f"  {index}/{len(eligible)}  ({rate:.1f} tiles/s, "
                    f"~{remaining / 60:.1f} min remaining)",
                    flush=True,
                )
            yield adapter.load_sample(by_id[sample_id].sample_id)

    try:
        artifact = fit_training_statistics(
            _stream(),
            dataset_id=adapter.dataset_id,
            code_version=git_commit(_REPOSITORY_ROOT),
            reprojection=reprojection,
            quarantine=quarantine,
            # The stream was pre-filtered above, so the fitter cannot count
            # the exclusions itself; hand it the record so the ARTIFACT, not
            # just this console output, carries the audit trail.
            eligibility=eligibility,
        )
    except StatisticsError as exc:
        print(f"statistics fit refused: {exc}", file=sys.stderr)
        return 2

    values_path, provenance_path = artifact.write(args.output)
    elapsed = time.time() - started

    print()
    print(f"fitted in {elapsed / 60:.1f} min")
    print(f"wrote {values_path}")
    print(f"wrote {provenance_path}")
    print()
    print("channel statistics (linear sigma0, valid pixels only):")
    for name, stat in artifact.values.items():
        print(f"  {name:9s} mean={stat.mean:.8f}  std={stat.std:.8f}")
    print()
    print(f"n_tiles:           {artifact.provenance['n_tiles']}")
    print(f"n_valid_pixels:    {artifact.provenance['n_valid_pixels']:,}")
    print(f"activations:       {artifact.provenance['activations']}")
    print(f"kind:              {artifact.kind}")
    print(f"usable for a reported run: {artifact.usable_for_reported_results}")

    if args.limit is not None:
        print(
            "\nNOTE: --limit was used, so this artifact was fitted on a BOUNDED SUBSET "
            "of the eligible training split. Its provenance records n_tiles; do not "
            "treat it as a full-split fit."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
