"""Checkpoint artifacts that carry their own provenance.

A checkpoint that holds only weights is not reproducible: nothing in it says
which corpus it saw, which channel order its first convolution expects, which
normalisation parameters its inputs were scaled by, or which seed produced it.
Each of those omissions is silently recoverable-looking and actually fatal —
feeding a model normalised with different statistics produces plausible,
wrong output.

So every checkpoint records model state, optimizer state, epoch, the validation
flood IoU that selected it, the full experiment config, the statistics identity,
the dataset identity, the seed and the repository commit.

Checkpoints are never committed to Git; ``.gitignore`` excludes ``runs/``,
``models/`` and ``*.pt``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .channels import CHANNEL_ORDER

__all__ = ["CHECKPOINT_FORMAT", "git_commit", "load_checkpoint", "save_checkpoint"]

#: Bumped if the stored keys change incompatibly.
CHECKPOINT_FORMAT = "floodmap.segmentation.checkpoint/1"


def git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Current commit hash, or ``None`` if it cannot be determined.

    ``None`` rather than a placeholder string: a checkpoint that claims a commit
    it was not built from is worse than one that admits it does not know.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_iou_flood_class: Optional[float],
    config: Mapping[str, Any],
    statistics_identity: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    seed: int,
    metrics: Optional[Mapping[str, Any]] = None,
    commit: Optional[str] = None,
) -> Path:
    """Write one checkpoint and return its path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch),
        "val_iou_flood_class": (
            None if val_iou_flood_class is None else float(val_iou_flood_class)
        ),
        "metrics": dict(metrics) if metrics else {},
        "config": dict(config),
        "statistics_identity": dict(statistics_identity),
        "dataset_identity": dict(dataset_identity),
        "seed": int(seed),
        "channel_order": list(CHANNEL_ORDER),
        "commit": commit if commit is not None else git_commit(),
        "model_class": type(model).__name__,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    """Read a checkpoint and verify the parts that must not drift.

    ``weights_only=False`` is required because the payload intentionally holds
    the config and provenance dicts alongside tensors. These are first-party
    artifacts written by :func:`save_checkpoint`.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(payload, dict):
        raise ValueError(f"malformed checkpoint: {path}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint format {payload.get('format')!r} is not {CHECKPOINT_FORMAT!r}"
        )
    stored_channels = list(payload.get("channel_order", []))
    if stored_channels != list(CHANNEL_ORDER):
        raise ValueError(
            f"checkpoint channel order {stored_channels} does not match the current "
            f"frozen order {list(CHANNEL_ORDER)}; its input weights would be re-mapped"
        )
    return payload
