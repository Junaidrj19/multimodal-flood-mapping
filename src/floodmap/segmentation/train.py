"""The M5 training loop.

Reproducibility is the point of the structure here. A run writes a record
containing its experiment config, resolved seed, dataset identity (corpus,
split, activation ids, sample count), statistics identity, device, commit and
per-epoch metrics, so the run can be reconstructed from the artifact rather
than from memory of which flags were passed.

Selection discipline, from C6 and ``docs/evaluation-protocol.md``:

* the best checkpoint is the one with the highest VALIDATION flood-class IoU;
* the test split is never loaded by this module at all — not read, not scored,
  not used for early stopping. The strongest way to avoid optimising the test
  split is to have no code path that touches it;
* validation runs under ``torch.no_grad`` with augmentation disabled.

This module does not decide how long to train. ``epochs`` and
``max_steps_per_epoch`` come from the experiment config, which is how a
pipeline-verification run and a real baseline run differ.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from floodmap.datasets import KuroSiwoAdapter
from floodmap.datasets.quarantine import load_quarantine
from floodmap.utils.config import project_root

from .augment import GeometricAugmentation
from .channels import CHANNEL_ORDER, FLOOD_CLASS_ID, NUM_CLASSES
from .checkpoint import git_commit, save_checkpoint
from .config import ExperimentConfig
from .dataset import KuroSiwoSarDataset, collate_samples
from .device import available_devices, describe_device, seed_everything, select_device
from .losses import build_loss
from .metrics import MetricResult, SegmentationMetrics
from .model import UNet
from .normalization import StatisticsArtifact, development_statistics

__all__ = ["TrainingResult", "build_optimizer", "evaluate", "run_training", "train_one_epoch"]

LOGGER = logging.getLogger("floodmap.segmentation.train")


@dataclass
class TrainingResult:
    """What a completed run produced."""

    best_epoch: Optional[int]
    best_val_iou_flood_class: Optional[float]
    best_checkpoint: Optional[Path]
    last_checkpoint: Optional[Path]
    run_record_path: Optional[Path]
    history: List[Dict[str, Any]] = field(default_factory=list)
    run_dir: Optional[Path] = None


def build_optimizer(
    model: torch.nn.Module, config: ExperimentConfig
) -> torch.optim.Optimizer:
    section = config.optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    if section.name == "adam":
        return torch.optim.Adam(
            parameters, lr=section.learning_rate, weight_decay=section.weight_decay
        )
    if section.name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=section.learning_rate, weight_decay=section.weight_decay
        )
    return torch.optim.SGD(
        parameters,
        lr=section.learning_rate,
        momentum=float(section.momentum or 0.0),
        weight_decay=section.weight_decay,
    )


def resolve_statistics(config: ExperimentConfig) -> Optional[StatisticsArtifact]:
    """Load the normalisation artifact the config asks for.

    A development artifact is only ever produced when the config explicitly
    opts in, so a run cannot silently normalise with placeholder parameters.
    An experiment that requires fitted statistics it does not have refuses to
    start HERE rather than at config load time, so the blocked config stays
    loadable and auditable.
    """

    section = config.normalisation
    blocked = section.blocked_reason()
    if blocked is not None:
        raise RuntimeError(blocked)
    if not section.enabled:
        return None
    if section.statistics_path:
        # Resolved against the project root, not the process CWD, so a run
        # behaves the same from a script, a test and a notebook.
        candidate = Path(section.statistics_path)
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        artifact = StatisticsArtifact.read(candidate)
        if section.required_kind == "fitted":
            artifact.require_fitted()
        return artifact
    return development_statistics(
        dataset_id=config.dataset.id,
        reason=f"experiment {config.name!r} is a pipeline-verification run",
    )


def resolve_quarantine(config: ExperimentConfig):
    """Load the eligibility policy the config asks for.

    Returns ``None`` when disabled, which the run record states explicitly so
    "no policy applied" is never confused with "no exclusions exist".
    """

    section = config.quarantine
    if not section.enabled:
        return None
    return load_quarantine(
        config.dataset.id,
        path=Path(section.path) if section.path else None,
        required=section.require_policy_file,
    )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every_n_steps: int = 10,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """One training epoch. Returns mean loss and the step count."""

    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"loss is not finite at step {steps}; a tile whose every pixel is "
                "ignored yields an undefined mean and must be excluded upstream"
            )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        steps += 1
        if log_every_n_steps and steps % log_every_n_steps == 0:
            LOGGER.info("step %d loss %.5f", steps, total_loss / steps)

    return {
        "loss": (total_loss / steps) if steps else float("nan"),
        "steps": float(steps),
    }


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    ignore_index: int,
    max_steps: Optional[int] = None,
) -> tuple[MetricResult, float]:
    """Score a split. Returns the metric result and the mean loss."""

    model.eval()
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=ignore_index)
    total_loss = 0.0
    steps = 0

    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, targets)
            if torch.isfinite(loss):
                total_loss += float(loss.detach().item())
        # Metrics accumulate on CPU so the confusion matrix is backend-agnostic.
        metrics.update(logits.detach().cpu(), targets.detach().cpu())
        steps += 1

    mean_loss = (total_loss / steps) if steps else float("nan")
    return metrics.compute(), mean_loss


def _build_loader(
    dataset: KuroSiwoSarDataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
    generator: Optional[torch.Generator],
) -> DataLoader:
    section = config.dataloader
    kwargs: Dict[str, Any] = {
        "batch_size": section.batch_size,
        "shuffle": shuffle,
        "num_workers": section.num_workers,
        "collate_fn": collate_samples,
        "drop_last": section.drop_last and shuffle,
        "pin_memory": section.pin_memory,
    }
    if section.num_workers > 0:
        kwargs["persistent_workers"] = section.persistent_workers
        if section.prefetch_factor is not None:
            kwargs["prefetch_factor"] = section.prefetch_factor
    if shuffle and generator is not None:
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def run_training(
    config: ExperimentConfig,
    *,
    adapter: Optional[KuroSiwoAdapter] = None,
    run_dir: Optional[Path] = None,
    write_artifacts: bool = True,
) -> TrainingResult:
    """Run one experiment end to end.

    Parameters
    ----------
    config
        A validated experiment config.
    adapter
        Optional pre-built adapter, so callers that already paid the index cost
        (or that are pointing at a synthetic fixture) can reuse it.
    run_dir
        Where checkpoints and the run record go. Defaults to
        ``<checkpoint.directory>/<experiment name>``.
    write_artifacts
        When ``False`` nothing is written to disk. Used by tests that only care
        about the loop.
    """

    seed = seed_everything(config.training.seed, deterministic=config.training.deterministic)
    device = select_device(config.training.device)
    LOGGER.info("device=%s seed=%d", device, seed)

    if adapter is None:
        root = Path(config.dataset.root) if config.dataset.root else None
        adapter = KuroSiwoAdapter(root)

    statistics = resolve_statistics(config)
    augmentation = GeometricAugmentation.from_config(config.augmentation)
    # One settings object, shared by both splits, so train and validation are
    # warped by identical spatial parameters.
    reprojection = config.reprojection if config.reprojection.enabled else None
    quarantine = resolve_quarantine(config)

    train_dataset = KuroSiwoSarDataset(
        adapter,
        config.dataset.train_split,
        statistics=statistics,
        augmentation=augmentation,
        limit=config.dataset.limit_train,
        ignore_index=config.dataset.ignore_index,
        reprojection=reprojection,
        quarantine=quarantine,
    )
    # Augmentation is deliberately NOT passed to validation: a flipped
    # validation tile scores a different, unrepeatable number each epoch.
    val_dataset = KuroSiwoSarDataset(
        adapter,
        config.dataset.validation_split,
        statistics=statistics,
        augmentation=None,
        limit=config.dataset.limit_validation,
        ignore_index=config.dataset.ignore_index,
        reprojection=reprojection,
        quarantine=quarantine,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"the {config.dataset.train_split!r} split resolved to 0 samples; check the "
            "external dataset root and the official activation split"
        )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = _build_loader(
        train_dataset, config, shuffle=config.dataloader.shuffle_train, generator=loader_generator
    )
    val_loader = (
        _build_loader(val_dataset, config, shuffle=False, generator=None)
        if len(val_dataset)
        else None
    )

    model = UNet.from_config(config.model, seed=seed).to(device)
    criterion = build_loss(
        ignore_index=config.loss.ignore_index,
        class_weights=config.loss.class_weights,
        num_classes=config.model.num_classes,
    ).to(device)
    optimizer = build_optimizer(model, config)

    resolved_run_dir = None
    if write_artifacts:
        resolved_run_dir = Path(
            run_dir if run_dir is not None else Path(config.checkpoint.directory) / config.name
        )
        resolved_run_dir.mkdir(parents=True, exist_ok=True)

    statistics_identity = (
        statistics.identity()
        if statistics is not None
        else {"kind": None, "note": "normalisation disabled; raw clipped linear sigma0"}
    )
    commit = git_commit()
    config_record = config.to_record()

    history: List[Dict[str, Any]] = []
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    best_checkpoint: Optional[Path] = None
    last_checkpoint: Optional[Path] = None

    for epoch in range(1, config.training.epochs + 1):
        started = time.time()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_every_n_steps=config.training.log_every_n_steps,
            max_steps=config.training.max_steps_per_epoch,
        )

        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_steps": int(train_stats["steps"]),
            "seconds": round(time.time() - started, 3),
        }

        val_metrics: Optional[MetricResult] = None
        if val_loader is not None:
            val_metrics, val_loss = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                ignore_index=config.dataset.ignore_index,
            )
            epoch_record["val_loss"] = val_loss
            epoch_record["val_metrics"] = val_metrics.to_dict()
            epoch_record["val_iou_flood_class"] = val_metrics.iou_flood_class
            LOGGER.info(
                "epoch %d train_loss=%.5f val_flood_iou=%.5f",
                epoch,
                train_stats["loss"],
                val_metrics.iou_flood_class,
            )
        else:
            # Recorded explicitly rather than silently omitted: a run with no
            # validation data cannot perform checkpoint selection at all.
            epoch_record["val_metrics"] = None
            epoch_record["val_iou_flood_class"] = None
            epoch_record["validation_absent_reason"] = (
                f"the {config.dataset.validation_split!r} split resolved to 0 samples"
            )
            LOGGER.info("epoch %d train_loss=%.5f (no validation data)", epoch, train_stats["loss"])

        history.append(epoch_record)

        if write_artifacts and resolved_run_dir is not None:
            candidate = epoch_record.get("val_iou_flood_class")
            # Selection on validation flood IoU only. NaN (no flood pixels in
            # the split) never wins, so a degenerate epoch cannot be selected.
            is_better = (
                candidate is not None
                and candidate == candidate  # not NaN
                and (best_metric is None or candidate > best_metric)
            )
            if is_better:
                best_metric = float(candidate)
                best_epoch = epoch
                best_checkpoint = save_checkpoint(
                    resolved_run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_iou_flood_class=best_metric,
                    config=config_record,
                    statistics_identity=statistics_identity,
                    dataset_identity=train_dataset.identity(),
                    seed=seed,
                    metrics=val_metrics.to_dict() if val_metrics else None,
                    commit=commit,
                )
            if config.checkpoint.save_last:
                last_checkpoint = save_checkpoint(
                    resolved_run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_iou_flood_class=epoch_record.get("val_iou_flood_class"),
                    config=config_record,
                    statistics_identity=statistics_identity,
                    dataset_identity=train_dataset.identity(),
                    seed=seed,
                    metrics=val_metrics.to_dict() if val_metrics else None,
                    commit=commit,
                )

    run_record_path = None
    if write_artifacts and resolved_run_dir is not None:
        record = {
            "experiment": config_record,
            "status": config.status,
            "primary_metric": config.primary_metric,
            "seed": seed,
            "deterministic": config.training.deterministic,
            "commit": commit,
            "device": describe_device(device),
            "device_availability": available_devices(),
            "channel_order": list(CHANNEL_ORDER),
            "flood_class_id": FLOOD_CLASS_ID,
            "spatial_preprocessing": {
                "reprojected_to_canonical_grid": reprojection is not None,
                "shared_implementation": "floodmap.preprocessing.reprojection",
                "settings": None if reprojection is None else reprojection.model_dump(mode="json"),
            },
            "statistics_identity": statistics_identity,
            "quarantine_policy": (
                {"applied": False, "reason": "disabled in experiment configuration"}
                if quarantine is None
                else quarantine.to_dict()
            ),
            "train_dataset": train_dataset.identity(),
            "validation_dataset": val_dataset.identity(),
            "model": {
                "class": type(model).__name__,
                "parameters": model.parameter_count(),
                "base_channels": config.model.base_channels,
                "depth": config.model.depth,
            },
            "loss": {
                "name": config.loss.name,
                "ignore_index": config.loss.ignore_index,
                "class_weights": config.loss.class_weights,
                "weighting": "unweighted" if config.loss.class_weights is None else "weighted",
            },
            "best_epoch": best_epoch,
            "best_val_iou_flood_class": best_metric,
            "history": history,
        }
        run_record_path = resolved_run_dir / "run_record.json"
        run_record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    return TrainingResult(
        best_epoch=best_epoch,
        best_val_iou_flood_class=best_metric,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        run_record_path=run_record_path,
        history=history,
        run_dir=resolved_run_dir,
    )
