#!/usr/bin/env python3
"""Run one M5 segmentation experiment from a configuration file.

No hyperparameter is defaulted here. Everything comes from the experiment
config (``AGENTS.md`` §14); the flags below only select a config and bound a
development run.

Examples
--------
Inspect what a config would do, without training::

    python3 scripts/train_segmentation.py --config m5_kuro_sar_unet --describe

Tiny pipeline-verification run on real tiles::

    FLOODMAP_KURO_SIWO_ROOT=/path/to/kuro_siwo python3 scripts/train_segmentation.py \
        --config m5_kuro_sar_unet_dev --run-dir /tmp/m5_dev_run

The real baseline config is blocked until fitted statistics exist; it reports
the remedy rather than silently normalising with placeholder values.
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

from floodmap.segmentation.config import load_experiment
from floodmap.segmentation.device import available_devices, select_device
from floodmap.segmentation.train import run_training


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment name under configs/experiments/, or a path to a YAML file.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Where checkpoints and the run record go. Defaults to "
        "<checkpoint.directory>/<experiment name>.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the resolved configuration and device, then exit without training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs. Intended for bounded development runs.",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Override dataset.limit_train.",
    )
    parser.add_argument(
        "--limit-validation",
        type=int,
        default=None,
        help="Override dataset.limit_validation.",
    )
    parser.add_argument("--device", default=None, help="Override training.device.")
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

    config = load_experiment(args.config)

    # Overrides are applied by rebuilding the validated model, so an override
    # cannot bypass the config's own invariants.
    overrides: dict = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.limit_train is not None:
        overrides.setdefault("dataset", {})["limit_train"] = args.limit_train
    if args.limit_validation is not None:
        overrides.setdefault("dataset", {})["limit_validation"] = args.limit_validation
    if overrides:
        payload = config.model_dump()
        for section, values in overrides.items():
            payload[section].update(values)
        config = type(config).model_validate(payload)

    if args.describe:
        print(json.dumps(config.to_record(), indent=2))
        print(json.dumps({"devices": available_devices()}, indent=2))
        blocked = config.normalisation.blocked_reason()
        if blocked:
            print(f"\nNOT RUNNABLE: {blocked}")
        else:
            print(f"\nresolved device: {select_device(config.training.device)}")
        return 0

    if config.status == "development_only":
        print(
            "NOTE: this is a DEVELOPMENT-ONLY configuration. It verifies that the "
            "pipeline executes; its metrics are not a performance measurement.\n"
        )

    try:
        result = run_training(config, run_dir=args.run_dir)
    except RuntimeError as exc:
        print(f"training did not start: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_val_iou_flood_class": result.best_val_iou_flood_class,
                "best_checkpoint": str(result.best_checkpoint)
                if result.best_checkpoint
                else None,
                "last_checkpoint": str(result.last_checkpoint)
                if result.last_checkpoint
                else None,
                "run_record": str(result.run_record_path)
                if result.run_record_path
                else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
