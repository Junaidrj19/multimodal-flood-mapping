"""Flood and debris segmentation.

Scientific responsibility
-------------------------
Map an analysis-ready multimodal raster to a class probability raster, a
segmentation mask, and model metadata sufficient to reproduce the inference
(architecture.md §6).

Key constraints
---------------
* The architecture is deliberately NOT pre-selected. Model choice must follow
  measured performance, data compatibility, compute feasibility and
  reproducibility rather than popularity (AGENTS.md §5, PRD.md §8).
  ``configs/segmentation.yaml`` therefore still carries
  ``model.architecture: null``; the U-Net here is the first REFERENCE BASELINE
  to measure, declared in ``configs/experiments/``, not the project's choice.
* Before committing to a model, establish label compatibility, input modality
  compatibility, spatial resolution, splits, compute cost, baseline
  performance and unseen-Himalaya performance (AGENTS.md §5).
* Thresholds are selected on validation data, never on the unseen-Himalaya
  evaluation set (docs/evaluation-protocol.md).

Status: M5 TRAINING INFRASTRUCTURE.

The M5 baseline is SAR-only, Kuro Siwo, pre/post, 4 channels
(pre-VV, pre-VH, post-VV, post-VH) in linear sigma0, predicting 3 semantic
classes with stored label 3 as the loss/metric ignore index. No fitted
normalisation statistics exist yet; see ``normalization.py``. The primary
validation metric is flood-class IoU (C6).

See ``docs/m5-training-baseline.md``.
"""

from .channels import (
    CHANNEL_ORDER,
    CHANNEL_SOURCES,
    CLASS_NAMES,
    FLOOD_CLASS_ID,
    IGNORE_INDEX,
    NUM_CLASSES,
    REQUIRED_SAR_REPRESENTATION,
    SEMANTIC_VALUES,
)
from .augment import GeometricAugmentation
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, ReprojectionSection, load_experiment
from .dataset import KuroSiwoSarDataset, collate_samples
from .device import seed_everything, select_device
from .losses import build_loss
from .metrics import MetricResult, SegmentationMetrics
from .model import UNet
from .normalization import (
    ChannelStatistics,
    StatisticsArtifact,
    StatisticsError,
    development_statistics,
    fit_training_statistics,
)
from .tensors import TensorSample, sample_to_tensors, stack_channels
from .spatial import (
    ReprojectedSample,
    reproject_canonical_sample,
    source_grid_for_sample,
)
from .train import TrainingResult, evaluate, run_training, train_one_epoch

__all__ = [
    "CHANNEL_ORDER",
    "CHANNEL_SOURCES",
    "CLASS_NAMES",
    "ChannelStatistics",
    "ExperimentConfig",
    "FLOOD_CLASS_ID",
    "GeometricAugmentation",
    "IGNORE_INDEX",
    "KuroSiwoSarDataset",
    "MetricResult",
    "NUM_CLASSES",
    "REQUIRED_SAR_REPRESENTATION",
    "ReprojectedSample",
    "ReprojectionSection",
    "SEMANTIC_VALUES",
    "SegmentationMetrics",
    "StatisticsArtifact",
    "StatisticsError",
    "TensorSample",
    "TrainingResult",
    "UNet",
    "build_loss",
    "collate_samples",
    "development_statistics",
    "evaluate",
    "fit_training_statistics",
    "load_checkpoint",
    "load_experiment",
    "reproject_canonical_sample",
    "run_training",
    "sample_to_tensors",
    "save_checkpoint",
    "seed_everything",
    "select_device",
    "source_grid_for_sample",
    "stack_channels",
    "train_one_epoch",
]
