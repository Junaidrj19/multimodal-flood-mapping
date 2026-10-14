"""Typed M5 experiment configuration.

Experiment settings live in ``configs/experiments/*.yaml`` and are loaded here,
never defaulted inside the training script (``AGENTS.md`` §14).

Why a separate file rather than filling in ``configs/segmentation.yaml``: that
file is part of the frozen contract and deliberately keeps
``model.architecture: null`` and ``training.{seed,epochs,batch_size,optimizer,
learning_rate,loss}: null``, because the project has not SELECTED an
architecture or a hyperparameter set — it has only proposed a first baseline to
measure. Writing this experiment's values into those fields would convert a
candidate into a project-level decision. The two files answer different
questions: ``segmentation.yaml`` says what is frozen, an experiment config says
what one run did.

Every model is ``extra="forbid"``. A mistyped key in an experiment config is
otherwise silently ignored, which is how a run ends up not doing what its
config appears to say.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from floodmap.utils.config import project_root

from .channels import CHANNEL_ORDER, IGNORE_INDEX, NUM_CLASSES, REQUIRED_SAR_REPRESENTATION

__all__ = [
    "ExperimentConfig",
    "QuarantineSection",
    "ReprojectionSection",
    "load_experiment",
    "experiments_dir",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSection(_Strict):
    """Which corpus, which splits, and which channels."""

    id: str = "kuro_siwo"
    root_env: str = "FLOODMAP_KURO_SIWO_ROOT"
    root: Optional[str] = None
    train_split: str = "train"
    validation_split: str = "validation"
    channels: List[str]
    representation: str
    ignore_index: int = IGNORE_INDEX
    #: Development caps. ``None`` means "use the whole split".
    limit_train: Optional[int] = None
    limit_validation: Optional[int] = None

    @model_validator(mode="after")
    def _channels_and_representation_match_the_freeze(self) -> "DatasetSection":
        if tuple(self.channels) != CHANNEL_ORDER:
            raise ValueError(
                f"dataset.channels must be exactly {list(CHANNEL_ORDER)} for the frozen "
                f"M5 SAR baseline, got {self.channels}"
            )
        if self.representation != REQUIRED_SAR_REPRESENTATION:
            raise ValueError(
                f"dataset.representation must be {REQUIRED_SAR_REPRESENTATION!r}; the "
                "baseline is explicitly not converted to decibel"
            )
        if self.validation_split == "test" or self.train_split == "test":
            raise ValueError(
                "the test split must not be used for training or selection"
            )
        return self


class ReprojectionSection(_Strict):
    """Canonical-grid reprojection, shared with M2 production preprocessing.

    Disabled by default so enabling it is a visible, recorded decision rather
    than a silent change to what every existing run meant. The real baseline
    config enables it, because training on Kuro's delivered EPSG:3857 grid
    (10 PROJECTED units = 8.736-9.798 m of true ground across the five
    delivered activations) while serving genuinely 10 m Sentinel-1 imagery is a
    spatial train/serve mismatch.
    """

    enabled: bool = False
    resolution_m: float = Field(gt=0.0, default=10.0)
    #: Fixed output pixel count. Required when enabled: resampling a 224 px
    #: Kuro tile from its measured spacing to 10 m gives a footprint-sized
    #: output that VARIES per activation, and a dataset returning variable
    #: shapes cannot be collated into a batch.
    tile_size: Optional[int] = Field(default=None, gt=0)
    sar_resampling: str = "bilinear"
    label_resampling: str = "nearest"
    #: ``strict`` means a target pixel is valid only if its whole
    #: interpolation kernel came from valid source pixels.
    nodata_mode: str = "strict"

    @model_validator(mode="after")
    def _methods_follow_c3(self) -> "ReprojectionSection":
        if self.sar_resampling.lower() in {"nearest", "mode"}:
            raise ValueError(
                "SAR must be resampled with an averaging method; nearest discards the "
                "averaging that makes a coarsened backscatter pixel physically meaningful"
            )
        if self.label_resampling.lower() not in {"nearest", "mode"}:
            raise ValueError(
                f"labels must use nearest neighbour, got {self.label_resampling!r}: an "
                "averaging method invents classes that were never in the source "
                "(a 0/2 boundary interpolates to 1, which is 'Permanent Waters')"
            )
        if self.nodata_mode not in {"strict", "majority"}:
            raise ValueError(
                f"nodata_mode must be 'strict' or 'majority', got {self.nodata_mode!r}"
            )
        if self.enabled and self.tile_size is None:
            raise ValueError(
                "reprojection.tile_size is required when reprojection is enabled: the "
                "footprint-sized output varies per activation and could not be batched"
            )
        return self

    def nodata_policy(self):
        from floodmap.preprocessing.reprojection import NodataPolicy

        return NodataPolicy(mode=self.nodata_mode)


class QuarantineSection(_Strict):
    """Sample-level eligibility policy.

    ``enabled: true`` with no ``path`` resolves the conventional location
    ``configs/quarantine/<dataset id>.yaml``. The policy excludes whole samples
    by name; it does not touch frozen label or validity semantics.

    Disabled means NO policy is applied, which is recorded as such in the run
    record. It does not mean no exclusions exist.
    """

    enabled: bool = True
    path: Optional[str] = None
    #: Fail if the policy file is absent, rather than silently applying none.
    require_policy_file: bool = False


class DataLoaderSection(_Strict):
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0, default=0)
    shuffle_train: bool = True
    drop_last: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None


class ModelSection(_Strict):
    architecture: str = "unet"
    in_channels: int = 4
    num_classes: int = NUM_CLASSES
    base_channels: int = Field(gt=0, default=32)
    depth: int = Field(gt=0, default=4)
    pretrained_weights: Optional[str] = None

    @model_validator(mode="after")
    def _baseline_shape_is_fixed(self) -> "ModelSection":
        if self.architecture != "unet":
            raise ValueError(
                f"only the 'unet' reference baseline is implemented, got "
                f"{self.architecture!r}"
            )
        if self.in_channels != len(CHANNEL_ORDER):
            raise ValueError(
                f"model.in_channels must be {len(CHANNEL_ORDER)} to match the frozen "
                f"channel stack, got {self.in_channels}"
            )
        if self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"model.num_classes must be {NUM_CLASSES} semantic classes; the nodata "
                "sentinel must not get a logit"
            )
        if self.pretrained_weights is not None:
            raise ValueError(
                "the baseline uses no pretrained weights; an external pretraining "
                "corpus would have to be declared in provenance first"
            )
        return self


class LossSection(_Strict):
    name: str = "cross_entropy"
    ignore_index: int = IGNORE_INDEX
    #: ``None`` means unweighted, which is the documented baseline.
    class_weights: Optional[List[float]] = None

    @model_validator(mode="after")
    def _only_plain_cross_entropy(self) -> "LossSection":
        if self.name != "cross_entropy":
            raise ValueError(
                f"the baseline loss is plain cross entropy, got {self.name!r}; a "
                "composite loss must not be introduced before the reference exists"
            )
        return self


class OptimizerSection(_Strict):
    name: str = "adam"
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0, default=0.0)
    momentum: Optional[float] = None

    @model_validator(mode="after")
    def _supported(self) -> "OptimizerSection":
        if self.name not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"unsupported optimizer {self.name!r}")
        if self.name == "sgd" and self.momentum is None:
            raise ValueError("sgd requires an explicit momentum value")
        return self


class AugmentationSection(_Strict):
    enabled: bool = False
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotate_90: bool = False
    radiometric: bool = False

    @model_validator(mode="after")
    def _no_radiometric_augmentation(self) -> "AugmentationSection":
        if self.radiometric:
            raise ValueError(
                "radiometric augmentation is not permitted in this baseline: sigma0 is "
                "the measurement, and scaling it trains on a physically meaningless "
                "distribution"
            )
        return self


class NormalisationSection(_Strict):
    enabled: bool = True
    #: Directory holding the ``statistics.json`` / ``statistics_meta.json`` pair.
    #:
    #: Stays ``None`` while no fitted artifact exists. C5 is explicit that a
    #: declared artifact which does not exist is worse than none, so this is
    #: NOT pre-filled with the path the artifact will eventually occupy. The
    #: resulting "enabled, fitted, no path" state is therefore legal to LOAD
    #: and illegal to RUN: the config stays auditable and
    #: ``train.resolve_statistics`` refuses to start the run.
    statistics_path: Optional[str] = None
    #: ``fitted`` for a real run; ``development`` only for pipeline tests.
    required_kind: str = "fitted"
    allow_development_statistics: bool = False

    @model_validator(mode="after")
    def _development_statistics_are_opt_in(self) -> "NormalisationSection":
        if self.required_kind not in {"fitted", "development"}:
            raise ValueError(
                f"normalisation.required_kind must be 'fitted' or 'development', got "
                f"{self.required_kind!r}"
            )
        if self.required_kind == "development" and not self.allow_development_statistics:
            raise ValueError(
                "required_kind='development' must be paired with "
                "allow_development_statistics=true, so a development run cannot be "
                "mistaken for a fitted one"
            )
        if self.required_kind == "fitted" and self.allow_development_statistics:
            raise ValueError(
                "allow_development_statistics=true contradicts required_kind='fitted'"
            )
        return self

    @property
    def is_runnable(self) -> bool:
        """Whether a run can actually start from this normalisation setting."""

        if not self.enabled:
            return True
        if self.required_kind == "development":
            return True
        return bool(self.statistics_path)

    def blocked_reason(self) -> Optional[str]:
        """Why a run cannot start, or ``None`` if it can."""

        if self.is_runnable:
            return None
        return (
            "normalisation is enabled and requires FITTED statistics, but no "
            "statistics_path is set and NO FITTED STATISTICS EXIST YET. Fit them from "
            "the training split with scripts/fit_statistics.py and set "
            "statistics_path, or use an experiment config with "
            "required_kind='development' for pipeline verification only. Kuro's "
            "published reference values must not be substituted: their pre/post, "
            "split and aggregation bases are all unstated upstream."
        )


class CheckpointSection(_Strict):
    directory: str = "runs"
    monitor: str = "val_iou_flood_class"
    mode: str = "max"
    selection_split: str = "validation"
    save_last: bool = True

    @model_validator(mode="after")
    def _selection_follows_c6(self) -> "CheckpointSection":
        if self.monitor != "val_iou_flood_class":
            raise ValueError(
                "C6 freezes the checkpoint criterion as val_iou_flood_class: a macro "
                "average over the two easy classes can rise while flood IoU falls"
            )
        if self.mode != "max":
            raise ValueError("flood IoU is maximised")
        if self.selection_split != "validation":
            raise ValueError(
                f"checkpoint selection must use the validation split, got "
                f"{self.selection_split!r}; selecting on test or external validation "
                "would void the evaluation"
            )
        return self


class TrainingSection(_Strict):
    seed: int
    deterministic: bool = True
    epochs: int = Field(gt=0)
    device: Optional[str] = None
    log_every_n_steps: int = Field(gt=0, default=10)
    max_steps_per_epoch: Optional[int] = None


class ExperimentConfig(_Strict):
    """One M5 experiment, fully specified."""

    name: str
    version: str
    description: str = ""
    status: str = "development"
    primary_metric: str = "iou_flood_class"
    dataset: DatasetSection
    reprojection: ReprojectionSection = Field(default_factory=ReprojectionSection)
    quarantine: QuarantineSection = Field(default_factory=QuarantineSection)
    dataloader: DataLoaderSection
    model: ModelSection
    loss: LossSection
    optimizer: OptimizerSection
    augmentation: AugmentationSection
    normalisation: NormalisationSection
    checkpoint: CheckpointSection
    training: TrainingSection

    @model_validator(mode="after")
    def _primary_metric_is_flood_iou(self) -> "ExperimentConfig":
        if self.primary_metric != "iou_flood_class":
            raise ValueError(
                "the primary validation metric is frozen as iou_flood_class (C6)"
            )
        if self.loss.ignore_index != self.dataset.ignore_index:
            raise ValueError(
                f"loss.ignore_index {self.loss.ignore_index} and dataset.ignore_index "
                f"{self.dataset.ignore_index} disagree; ignored pixels would be "
                "excluded from one and not the other"
            )
        return self

    def to_record(self) -> Dict[str, Any]:
        """Serialisable copy for the run record and checkpoint."""

        return self.model_dump(mode="json")


def experiments_dir() -> Path:
    """``configs/experiments/``."""

    return project_root() / "configs" / "experiments"


def load_experiment(path_or_name: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config.

    Accepts a bare name (resolved under ``configs/experiments/``) or a path.
    """

    candidate = Path(path_or_name)
    if not candidate.is_file():
        stem = candidate.name
        if not stem.endswith(".yaml"):
            stem = f"{stem}.yaml"
        candidate = experiments_dir() / stem
    if not candidate.is_file():
        raise FileNotFoundError(f"experiment config not found: {path_or_name}")

    parsed = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise TypeError(f"experiment config must parse to a mapping: {candidate}")
    return ExperimentConfig.model_validate(dict(parsed))
