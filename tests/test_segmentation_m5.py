"""M5 segmentation training-infrastructure tests.

Synthetic fixtures throughout, reproducing the Kuro Siwo partition-01 layout
with tiny rasters so the whole file runs without the external corpus. The
real-data smoke and overfit tests live in ``test_segmentation_m5_real.py`` and
are opt-in.

The fixture is built so the interesting cases are actually exercised: it
contains invalid pixels, all three semantic classes, and a train/validation
split drawn from two different official activations.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from floodmap.datasets import KuroSiwoAdapter
from floodmap.segmentation import (
    CHANNEL_ORDER,
    FLOOD_CLASS_ID,
    IGNORE_INDEX,
    NUM_CLASSES,
    GeometricAugmentation,
    KuroSiwoSarDataset,
    SegmentationMetrics,
    StatisticsArtifact,
    StatisticsError,
    UNet,
    build_loss,
    collate_samples,
    development_statistics,
    fit_training_statistics,
    load_checkpoint,
    load_experiment,
    sample_to_tensors,
    save_checkpoint,
    seed_everything,
    select_device,
    stack_channels,
)
from floodmap.segmentation.config import ExperimentConfig
from floodmap.segmentation.device import available_devices
from floodmap.segmentation.normalization import PROVENANCE_FILE, VALUES_FILE
from floodmap.segmentation.tensors import TensorContractError
from floodmap.segmentation.train import evaluate, run_training, train_one_epoch

#: Two official activations: 1111004 is a TRAIN activation and 1111003 is a
#: VALIDATION one, per the upstream split baked into the adapter.
TRAIN_ACTIVATION = 1111004
VALIDATION_ACTIVATION = 1111003

#: Divisible by 2**2, so a depth-2 U-Net runs on it.
TILE = 16


def _write_raster(path: Path, array: np.ndarray, *, transform, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[None, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def _write_catalogue(root: Path, activation: int, grids):
    path = root / str(activation) / "catalogue.gkpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE IF EXISTS catalogue")
    connection.execute(
        "CREATE TABLE catalogue (grid_id TEXT, actid INTEGER, aoiid INTEGER, "
        "exported INTEGER, master INTEGER, crank INTEGER, source_date TEXT, s1_ids TEXT)"
    )
    for grid_id in grids:
        for master, crank in ((0, 1), (0, 2), (1, 1)):
            connection.execute(
                "INSERT INTO catalogue VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (grid_id, activation, 1, 1, master, crank, "2020-01-01", "[]"),
            )
    connection.commit()
    connection.close()


def _write_sample(root: Path, activation: int, index: int, *, seed: int) -> str:
    """One synthetic labelled tile. Returns its grid id."""

    grid_id = f"{activation}-grid-{index:04d}"
    directory = root / str(activation) / "01" / grid_id
    directory.mkdir(parents=True, exist_ok=True)
    transform = from_origin(4801325, 1292245, 10, 10)
    rng = np.random.default_rng(seed)

    # Invalid pixels: the top-left 3x3 corner. Kuro's three redundant validity
    # mechanisms must agree, so the label is 3, MNA is 0, and every SAR plane
    # is 0.0 on exactly those pixels.
    valid = np.ones((TILE, TILE), dtype=np.uint8)
    valid[:3, :3] = 0

    labels = np.zeros((TILE, TILE), dtype=np.uint8)
    # All three semantic classes present, so per-class metrics have support.
    labels[8:, :6] = 1          # permanent water
    labels[4:10, 8:14] = 2      # flood
    labels[:3, :3] = 3          # nodata

    for role, date in (("SL1", "20191115"), ("SL2", "20191109"), ("MS1", "20191127")):
        for pol in ("IVV", "IVH"):
            plane = rng.uniform(0.01, 0.2, size=(TILE, TILE)).astype(np.float32)
            # Make the post-event flood region darker, so there is a learnable
            # pre/post signal rather than pure noise.
            if role == "MS1":
                plane[labels == 2] *= 0.1
            plane[:3, :3] = 0.0
            _write_raster(
                directory / f"{role}_{pol}_{activation}_01_{date}.tif",
                plane,
                transform=transform,
                nodata=0.0,
            )

    _write_raster(
        directory / f"MK0_MLU_{activation}_01_20191127.tif",
        labels,
        transform=transform,
        nodata=3,
    )
    _write_raster(
        directory / f"MK0_MNA_{activation}_01_20191127.tif",
        valid,
        transform=transform,
        nodata=0,
    )

    sources = {
        "SL1": {"source_date": "2019-11-15", "master": False, "crank": 1},
        "SL2": {"source_date": "2019-11-09", "master": False, "crank": 2},
        "MS1": {"source_date": "2019-11-27", "master": True, "crank": 1},
    }
    (directory / "info.json").write_text(
        json.dumps(
            {"grid_id": grid_id, "actid": activation, "aoiid": 1, "sources": sources}
        ),
        encoding="utf-8",
    )
    return grid_id


@pytest.fixture(scope="module")
def kuro_root(tmp_path_factory) -> Path:
    """Four train tiles and two validation tiles, from two activations."""

    root = tmp_path_factory.mktemp("kuro_m5")
    train_grids = [
        _write_sample(root, TRAIN_ACTIVATION, i, seed=100 + i) for i in range(4)
    ]
    _write_catalogue(root, TRAIN_ACTIVATION, train_grids)
    val_grids = [
        _write_sample(root, VALIDATION_ACTIVATION, i, seed=200 + i) for i in range(2)
    ]
    _write_catalogue(root, VALIDATION_ACTIVATION, val_grids)
    return root


@pytest.fixture(scope="module")
def adapter(kuro_root) -> KuroSiwoAdapter:
    return KuroSiwoAdapter(kuro_root)


@pytest.fixture(scope="module")
def canonical_sample(adapter):
    train = [ref for ref in adapter.discover() if ref.split == "train"]
    return adapter.load_sample(sorted(r.sample_id for r in train)[0])


# ---------------------------------------------------------------------------
# Fixture self-check
# ---------------------------------------------------------------------------


class TestFixtureIsRepresentative:
    def test_splits_come_from_the_official_activation_assignment(self, adapter):
        by_split = {}
        for ref in adapter.discover():
            by_split.setdefault(ref.split, []).append(ref)
        assert len(by_split["train"]) == 4
        assert len(by_split["validation"]) == 2
        assert "test" not in by_split

    def test_the_fixture_contains_invalid_pixels_and_all_three_classes(
        self, canonical_sample
    ):
        assert set(np.unique(canonical_sample.labels)) == {0, 1, 2, 3}
        assert not canonical_sample.validity.all(), "fixture must exercise masking"
        assert canonical_sample.validity.any()


# ---------------------------------------------------------------------------
# Dataset / tensor contract
# ---------------------------------------------------------------------------


class TestTensorContract:
    def test_four_channel_shape_and_dtype(self, canonical_sample):
        tensors = sample_to_tensors(canonical_sample)
        assert tensors.image.shape == (4, TILE, TILE)
        assert tensors.image.dtype == torch.float32
        assert tensors.target.shape == (TILE, TILE)
        assert tensors.target.dtype == torch.int64
        assert tensors.valid_mask.shape == (TILE, TILE)
        assert tensors.valid_mask.dtype == torch.bool

    def test_channel_ordering_is_pre_vv_pre_vh_post_vv_post_vh(self, canonical_sample):
        """Each plane must equal the exact source array it claims to be."""
        stack = stack_channels(canonical_sample)
        assert CHANNEL_ORDER == ("pre_vv", "pre_vh", "post_vv", "post_vh")
        np.testing.assert_array_equal(stack[0], canonical_sample.sar["pre"]["vv"])
        np.testing.assert_array_equal(stack[1], canonical_sample.sar["pre"]["vh"])
        np.testing.assert_array_equal(stack[2], canonical_sample.sar["post"]["vv"])
        np.testing.assert_array_equal(stack[3], canonical_sample.sar["post"]["vh"])

    def test_pre_and_post_channels_are_not_the_same_epoch(self, canonical_sample):
        """Guards against a collapsed temporal pair."""
        stack = stack_channels(canonical_sample)
        assert not np.array_equal(stack[0], stack[2])
        assert not np.array_equal(stack[1], stack[3])

    def test_ignore_handling_marks_invalid_pixels_and_keeps_classes(
        self, canonical_sample
    ):
        tensors = sample_to_tensors(canonical_sample)
        target = tensors.target.numpy()
        valid = tensors.valid_mask.numpy()

        assert set(np.unique(target[valid])).issubset({0, 1, 2})
        assert (target[~valid] == IGNORE_INDEX).all()
        # Nodata must not become a trainable class.
        assert IGNORE_INDEX not in set(np.unique(target[valid]))

    def test_valid_semantic_classes_are_not_remapped(self, canonical_sample):
        """Kuro's native values already ARE the class indices."""
        tensors = sample_to_tensors(canonical_sample)
        target = tensors.target.numpy()
        valid = tensors.valid_mask.numpy()
        np.testing.assert_array_equal(
            target[valid], canonical_sample.labels[valid].astype(np.int64)
        )

    def test_validity_comes_from_the_canonical_mask(self, canonical_sample):
        tensors = sample_to_tensors(canonical_sample)
        np.testing.assert_array_equal(
            tensors.valid_mask.numpy(), canonical_sample.validity
        )

    def test_a_decibel_sample_is_rejected(self, canonical_sample):
        """A dB corpus must not be fed through a linear-sigma0 baseline."""
        import dataclasses

        decibel = dataclasses.replace(canonical_sample, sar_representation="decibel")
        with pytest.raises(TensorContractError, match="linear_sigma0"):
            sample_to_tensors(decibel)

    def test_a_single_epoch_sample_is_rejected(self, canonical_sample):
        """Substituting post for a missing pre would zero the change signal."""
        import dataclasses

        post_only = dataclasses.replace(
            canonical_sample, sar={"post": canonical_sample.sar["post"]}
        )
        with pytest.raises(TensorContractError, match="pre"):
            sample_to_tensors(post_only)

    def test_an_ignore_index_colliding_with_a_class_is_rejected(self, canonical_sample):
        with pytest.raises(TensorContractError, match="collides"):
            sample_to_tensors(canonical_sample, ignore_index=2)


class TestDatasetLayer:
    def test_split_filtering_and_deterministic_order(self, adapter):
        first = KuroSiwoSarDataset(adapter, "train")
        second = KuroSiwoSarDataset(adapter, "train")
        assert len(first) == 4
        assert first.sample_ids == second.sample_ids
        assert list(first.sample_ids) == sorted(first.sample_ids)

    def test_limit_is_applied_after_sorting(self, adapter):
        limited = KuroSiwoSarDataset(adapter, "train", limit=2)
        full = KuroSiwoSarDataset(adapter, "train")
        assert limited.sample_ids == full.sample_ids[:2]

    def test_sample_id_allowlist_is_order_independent(self, adapter):
        full = KuroSiwoSarDataset(adapter, "train")
        chosen = [full.sample_ids[2], full.sample_ids[0]]
        pinned = KuroSiwoSarDataset(adapter, "train", sample_ids=chosen)
        assert pinned.sample_ids == (full.sample_ids[0], full.sample_ids[2])

    def test_an_unknown_sample_id_is_rejected(self, adapter):
        with pytest.raises(KeyError):
            KuroSiwoSarDataset(adapter, "train", sample_ids=["nope"])

    def test_dataset_construction_reads_no_pixels(self, adapter, monkeypatch):
        """Memory discipline: nothing is loaded until __getitem__ is called.

        The real guarantee this protects is that the 7,052-tile training split
        is never materialised as one in-memory tensor.
        """
        calls = []
        original = adapter.load_sample
        monkeypatch.setattr(
            adapter,
            "load_sample",
            lambda sample_id: (calls.append(sample_id), original(sample_id))[1],
        )

        dataset = KuroSiwoSarDataset(adapter, "train")
        assert calls == [], f"construction loaded {len(calls)} samples"
        assert len(dataset) == 4

        dataset[0]
        assert calls == [dataset.sample_ids[0]]
        dataset[2]
        assert calls == [dataset.sample_ids[0], dataset.sample_ids[2]]

    def test_dataset_caches_no_arrays(self, adapter):
        """No decoded raster is retained on the dataset object itself."""
        dataset = KuroSiwoSarDataset(adapter, "train")
        dataset[0]
        held = [
            name
            for name, value in vars(dataset).items()
            if isinstance(value, np.ndarray)
            or (isinstance(value, torch.Tensor) and value.numel() > len(CHANNEL_ORDER))
        ]
        assert held == [], f"dataset retained decoded arrays: {held}"

    def test_normalisation_is_applied_when_statistics_are_supplied(self, adapter):
        stats = StatisticsArtifact(
            kind="development",
            values={
                name: type(development_statistics().values[name])(mean=0.1, std=0.05)
                for name in CHANNEL_ORDER
            },
            provenance=development_statistics().provenance,
        )
        plain = KuroSiwoSarDataset(adapter, "train")[0]["image"]
        scaled = KuroSiwoSarDataset(adapter, "train", statistics=stats)[0]["image"]
        torch.testing.assert_close(scaled, (plain - 0.1) / 0.05)

    def test_collate_preserves_sample_ids(self, adapter):
        dataset = KuroSiwoSarDataset(adapter, "train")
        batch = collate_samples([dataset[0], dataset[1]])
        assert batch["image"].shape == (2, 4, TILE, TILE)
        assert batch["target"].shape == (2, TILE, TILE)
        assert batch["valid_mask"].shape == (2, TILE, TILE)
        assert batch["sample_id"] == [dataset.sample_ids[0], dataset.sample_ids[1]]

    def test_identity_records_dataset_provenance(self, adapter):
        identity = KuroSiwoSarDataset(adapter, "train").identity()
        assert identity["dataset_id"] == "kuro_siwo"
        assert identity["split"] == "train"
        assert identity["activations"] == [str(TRAIN_ACTIVATION)]
        assert identity["channels"] == list(CHANNEL_ORDER)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    def test_disabled_is_an_identity(self, adapter):
        sample = KuroSiwoSarDataset(adapter, "train")[0]
        out = GeometricAugmentation(enabled=False, horizontal_flip=True)(sample)
        torch.testing.assert_close(out["image"], sample["image"])

    def test_geometry_is_applied_identically_to_channels_target_and_mask(self, adapter):
        sample = KuroSiwoSarDataset(adapter, "train")[0]
        augment = GeometricAugmentation(
            enabled=True, horizontal_flip=True, vertical_flip=True, rotate_90=True
        )
        generator = torch.Generator().manual_seed(0)
        out = augment(sample, generator)

        # Whatever transform was drawn, applying the SAME flips/rotations to
        # the original by brute-force search must reproduce all three payloads
        # under one single transform, not three different ones.
        matched = None
        for turns in range(4):
            for h in (False, True):
                for v in (False, True):
                    image = sample["image"]
                    target = sample["target"]
                    mask = sample["valid_mask"]
                    if h:
                        image, target, mask = (
                            torch.flip(image, (-1,)),
                            torch.flip(target, (-1,)),
                            torch.flip(mask, (-1,)),
                        )
                    if v:
                        image, target, mask = (
                            torch.flip(image, (-2,)),
                            torch.flip(target, (-2,)),
                            torch.flip(mask, (-2,)),
                        )
                    if turns:
                        image = torch.rot90(image, turns, (-2, -1))
                        target = torch.rot90(target, turns, (-2, -1))
                        mask = torch.rot90(mask, turns, (-2, -1))
                    if (
                        torch.equal(image, out["image"])
                        and torch.equal(target, out["target"])
                        and torch.equal(mask, out["valid_mask"])
                    ):
                        matched = (turns, h, v)
        assert matched is not None, (
            "image, target and mask were not transformed by one shared geometric "
            "operation"
        )

    def test_channels_are_not_augmented_independently(self, adapter):
        """Every channel must keep its pixel correspondence with the others."""
        sample = KuroSiwoSarDataset(adapter, "train")[0]
        augment = GeometricAugmentation(enabled=True, horizontal_flip=True, rotate_90=True)
        out = augment(sample, torch.Generator().manual_seed(3))
        # The invalid corner is 0.0 in every source plane, so wherever it
        # landed it must still be 0.0 in all four channels simultaneously.
        zero_per_channel = [(out["image"][c] == 0.0) for c in range(4)]
        for channel in zero_per_channel[1:]:
            torch.testing.assert_close(
                channel.to(torch.uint8), zero_per_channel[0].to(torch.uint8)
            )

    def test_augmentation_preserves_label_values(self, adapter):
        sample = KuroSiwoSarDataset(adapter, "train")[0]
        augment = GeometricAugmentation(enabled=True, horizontal_flip=True, rotate_90=True)
        out = augment(sample, torch.Generator().manual_seed(7))
        assert set(torch.unique(out["target"]).tolist()) == set(
            torch.unique(sample["target"]).tolist()
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestModel:
    def test_input_and_output_shape(self):
        model = UNet(in_channels=4, num_classes=3, base_channels=8, depth=2, seed=0)
        output = model(torch.zeros(2, 4, TILE, TILE))
        assert output.shape == (2, NUM_CLASSES, TILE, TILE)

    def test_gradient_flows_to_every_parameter(self):
        model = UNet(base_channels=8, depth=2, seed=0)
        logits = model(torch.randn(1, 4, TILE, TILE))
        logits.sum().backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
        ]
        assert not missing, f"no finite gradient reached: {missing}"

    def test_configurable_width_and_depth_change_capacity(self):
        small = UNet(base_channels=8, depth=2, seed=0).parameter_count()
        wide = UNet(base_channels=16, depth=2, seed=0).parameter_count()
        deep = UNet(base_channels=8, depth=3, seed=0).parameter_count()
        assert small < wide
        assert small < deep

    def test_deterministic_initialisation(self):
        first = UNet(base_channels=8, depth=2, seed=42).state_dict()
        second = UNet(base_channels=8, depth=2, seed=42).state_dict()
        third = UNet(base_channels=8, depth=2, seed=43).state_dict()
        for key, value in first.items():
            torch.testing.assert_close(value, second[key])
        assert any(
            not torch.equal(first[key], third[key])
            for key in first
            if first[key].dtype.is_floating_point
        )

    def test_construction_does_not_disturb_the_global_rng(self):
        """Building a model must not shift dataloader shuffling."""
        torch.manual_seed(5)
        expected = torch.randn(3)
        torch.manual_seed(5)
        UNet(base_channels=8, depth=2, seed=99)
        torch.testing.assert_close(torch.randn(3), expected)

    def test_wrong_channel_count_is_rejected(self):
        model = UNet(base_channels=8, depth=2, seed=0)
        with pytest.raises(ValueError, match="input channels"):
            model(torch.zeros(1, 3, TILE, TILE))

    def test_indivisible_spatial_size_is_rejected(self):
        model = UNet(base_channels=8, depth=2, seed=0)
        with pytest.raises(ValueError, match="divisible"):
            model(torch.zeros(1, 4, 15, 15))

    def test_no_pretrained_weights_are_used(self):
        assert UNet(base_channels=8, depth=2, seed=0) is not None
        config = load_experiment("m5_kuro_sar_unet")
        assert config.model.pretrained_weights is None


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class TestLoss:
    def test_default_is_unweighted_cross_entropy_ignoring_three(self):
        criterion = build_loss()
        assert isinstance(criterion, torch.nn.CrossEntropyLoss)
        assert criterion.ignore_index == IGNORE_INDEX
        assert criterion.weight is None

    def test_ignored_pixels_do_not_change_the_loss_value(self):
        """Changing a prediction only on ignored pixels must be a no-op."""
        criterion = build_loss()
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 0, :] = IGNORE_INDEX

        logits = torch.zeros(1, NUM_CLASSES, 4, 4)
        baseline = criterion(logits, target)

        perturbed = logits.clone()
        perturbed[0, :, 0, :] = torch.tensor([50.0, -50.0, 20.0]).view(3, 1)
        assert torch.allclose(criterion(perturbed, target), baseline)

    def test_ignored_pixels_receive_no_gradient(self):
        criterion = build_loss()
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 0, :] = IGNORE_INDEX
        logits = torch.zeros(1, NUM_CLASSES, 4, 4, requires_grad=True)
        criterion(logits, target).backward()
        assert torch.count_nonzero(logits.grad[0, :, 0, :]) == 0
        assert torch.count_nonzero(logits.grad[0, :, 1:, :]) > 0

    def test_an_all_ignored_target_yields_a_non_finite_loss(self):
        """Documents the degenerate case the training loop checks for."""
        criterion = build_loss()
        target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
        loss = criterion(torch.zeros(1, NUM_CLASSES, 4, 4), target)
        assert torch.isnan(loss)

    def test_class_weights_are_supported_but_validated(self):
        weighted = build_loss(class_weights=[1.0, 1.0, 4.0])
        assert weighted.weight is not None
        with pytest.raises(ValueError, match="misaligned"):
            build_loss(class_weights=[1.0, 2.0])
        with pytest.raises(ValueError, match="non-negative"):
            build_loss(class_weights=[1.0, -1.0, 1.0])

    def test_a_composite_loss_is_not_silently_accepted(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["loss"]["name"] = "dice_cross_entropy"
        with pytest.raises(ValueError, match="plain cross entropy"):
            ExperimentConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_prediction_gives_unit_iou(self):
        metrics = SegmentationMetrics()
        target = torch.tensor([[[0, 1], [2, 0]]])
        metrics.update(target.clone(), target, is_prediction=True)
        result = metrics.compute()
        assert result.iou_flood_class == pytest.approx(1.0)
        assert result.mean_iou == pytest.approx(1.0)

    def test_flood_iou_is_computed_by_hand_correctly(self):
        """2 true positives, 1 false positive, 1 false negative -> 2/4."""
        metrics = SegmentationMetrics()
        target = torch.tensor([[2, 2, 2, 0]])
        prediction = torch.tensor([[2, 2, 0, 2]])
        metrics.update(prediction, target, is_prediction=True)
        result = metrics.compute()
        assert result.iou_flood_class == pytest.approx(2.0 / 4.0)
        assert result.per_class_precision["Floods"] == pytest.approx(2.0 / 3.0)
        assert result.per_class_recall["Floods"] == pytest.approx(2.0 / 3.0)
        assert result.per_class_f1["Floods"] == pytest.approx(2.0 / 3.0)

    def test_all_metrics_match_a_hand_calculated_confusion_matrix(self):
        """Derived metrics use truth rows and prediction columns exactly."""
        metrics = SegmentationMetrics()
        target = torch.tensor([[0, 0, 1, 1, 2, 2, 2]])
        prediction = torch.tensor([[0, 1, 0, 0, 2, 0, 1]])
        metrics.update(prediction, target, is_prediction=True)

        result = metrics.compute()
        expected_matrix = torch.tensor(
            [[1, 1, 0], [2, 0, 0], [1, 1, 1]], dtype=torch.int64
        )
        torch.testing.assert_close(result.confusion_matrix, expected_matrix)

        # Class 0: TP=1, FP=3, FN=1. Class 1: TP=0, FP=2, FN=2.
        # Class 2: TP=1, FP=0, FN=2.
        assert result.per_class_iou["No water"] == pytest.approx(1.0 / 5.0)
        assert result.per_class_iou["Permanent Waters"] == pytest.approx(0.0)
        assert result.per_class_iou["Floods"] == pytest.approx(1.0 / 3.0)
        assert result.per_class_precision["No water"] == pytest.approx(1.0 / 4.0)
        assert result.per_class_precision["Permanent Waters"] == pytest.approx(0.0)
        assert result.per_class_precision["Floods"] == pytest.approx(1.0)
        assert result.per_class_recall["No water"] == pytest.approx(1.0 / 2.0)
        assert result.per_class_recall["Permanent Waters"] == pytest.approx(0.0)
        assert result.per_class_recall["Floods"] == pytest.approx(1.0 / 3.0)
        assert result.per_class_f1["No water"] == pytest.approx(1.0 / 3.0)
        assert result.per_class_f1["Permanent Waters"] == pytest.approx(0.0)
        assert result.per_class_f1["Floods"] == pytest.approx(1.0 / 2.0)
        assert result.iou_flood_class == pytest.approx(1.0 / 3.0)
        assert result.mean_iou == pytest.approx((1.0 / 5.0 + 0.0 + 1.0 / 3.0) / 3.0)
        assert result.macro_f1 == pytest.approx((1.0 / 3.0 + 0.0 + 1.0 / 2.0) / 3.0)
        # Pooled TP=2; pooled union=5+4+3=12.
        assert result.micro_iou == pytest.approx(1.0 / 6.0)
        assert result.pixel_accuracy == pytest.approx(2.0 / 7.0)

    def test_supported_class_with_zero_true_positives_has_zero_f1(self):
        """Undefined component ratios do not make a present class F1 NaN."""
        metrics = SegmentationMetrics()
        metrics.update(
            torch.tensor([[0, 0, 2]]),
            torch.tensor([[0, 1, 2]]),
            is_prediction=True,
        )
        result = metrics.compute()

        # Permanent Waters has support but no prediction: precision is undefined,
        # recall is zero, and count-based F1 is zero.
        assert np.isnan(result.per_class_precision["Permanent Waters"])
        assert result.per_class_recall["Permanent Waters"] == pytest.approx(0.0)
        assert result.per_class_f1["Permanent Waters"] == pytest.approx(0.0)
        assert result.per_class_f1["Floods"] == pytest.approx(1.0)

    def test_genuinely_absent_class_stays_nan_and_is_dropped_from_macros(self):
        """A class absent from truth and prediction remains undefined."""
        metrics = SegmentationMetrics()
        metrics.update(
            torch.tensor([[0, 2]]),
            torch.tensor([[0, 2]]),
            is_prediction=True,
        )
        result = metrics.compute()

        assert np.isnan(result.per_class_iou["Permanent Waters"])
        assert np.isnan(result.per_class_precision["Permanent Waters"])
        assert np.isnan(result.per_class_recall["Permanent Waters"])
        assert np.isnan(result.per_class_f1["Permanent Waters"])
        assert result.mean_iou == pytest.approx(1.0)
        assert result.macro_f1 == pytest.approx(1.0)

    def test_ignored_pixels_are_excluded_from_every_metric(self):
        """Adding ignored pixels with arbitrary predictions changes nothing."""
        target = torch.tensor([[2, 2, 0, 1]])
        prediction = torch.tensor([[2, 0, 0, 1]])

        clean = SegmentationMetrics()
        clean.update(prediction, target, is_prediction=True)
        clean_result = clean.compute()

        padded_target = torch.cat([target, torch.full((1, 5), IGNORE_INDEX)], dim=1)
        padded_prediction = torch.cat([prediction, torch.tensor([[2, 2, 1, 0, 2]])], dim=1)
        padded = SegmentationMetrics()
        padded.update(padded_prediction, padded_target, is_prediction=True)
        padded_result = padded.compute()

        assert padded_result.iou_flood_class == pytest.approx(clean_result.iou_flood_class)
        assert padded_result.micro_iou == pytest.approx(clean_result.micro_iou)
        assert padded_result.pixel_accuracy == pytest.approx(clean_result.pixel_accuracy)
        torch.testing.assert_close(
            padded_result.confusion_matrix, clean_result.confusion_matrix
        )
        assert padded_result.n_ignored_pixels == 5
        assert padded_result.n_valid_pixels == 4

    def test_confusion_matrix_orientation_is_truth_by_prediction(self):
        metrics = SegmentationMetrics()
        # One pixel of truth 2 predicted as 0.
        metrics.update(torch.tensor([[0]]), torch.tensor([[2]]), is_prediction=True)
        matrix = metrics.confusion_matrix
        assert matrix[2, 0] == 1
        assert matrix[0, 2] == 0
        assert matrix.sum() == 1

    def test_accepts_logits_and_argmaxes_them(self):
        metrics = SegmentationMetrics()
        logits = torch.zeros(1, NUM_CLASSES, 1, 2)
        logits[0, 2, 0, 0] = 10.0
        logits[0, 1, 0, 1] = 10.0
        metrics.update(logits, torch.tensor([[[2, 1]]]))
        assert metrics.compute().iou_flood_class == pytest.approx(1.0)

    def test_batches_accumulate_rather_than_average(self):
        pooled = SegmentationMetrics()
        pooled.update(torch.tensor([[2, 0]]), torch.tensor([[2, 2]]), is_prediction=True)
        pooled.update(torch.tensor([[2, 2]]), torch.tensor([[2, 2]]), is_prediction=True)
        # 3 TP, 0 FP, 1 FN -> 3/4, which is not the mean of 1/2 and 1.
        assert pooled.compute().iou_flood_class == pytest.approx(3.0 / 4.0)

    def test_metrics_are_serialisable(self):
        metrics = SegmentationMetrics()
        metrics.update(torch.tensor([[2, 0]]), torch.tensor([[2, 0]]), is_prediction=True)
        payload = json.loads(json.dumps(metrics.compute().to_dict()))
        assert payload["iou_flood_class"] == pytest.approx(1.0)
        assert len(payload["confusion_matrix"]) == NUM_CLASSES

    def test_an_out_of_range_target_is_rejected(self):
        metrics = SegmentationMetrics()
        with pytest.raises(ValueError, match="outside the semantic class range"):
            metrics.update(torch.tensor([[0]]), torch.tensor([[7]]), is_prediction=True)


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_development_statistics_are_identity_and_clearly_marked(self):
        artifact = development_statistics()
        assert artifact.kind == "development"
        assert artifact.is_development_only is True
        assert artifact.usable_for_reported_results is False
        assert artifact.provenance["development_only"] is True
        assert artifact.provenance["is_part_of_frozen_contract"] is False
        np.testing.assert_allclose(artifact.mean_vector(), np.zeros(4))
        np.testing.assert_allclose(artifact.std_vector(), np.ones(4))

    def test_development_statistics_are_refused_where_fitted_are_required(self):
        with pytest.raises(StatisticsError, match="DEVELOPMENT"):
            development_statistics().require_fitted()

    def test_fitting_uses_only_valid_pixels(self, adapter):
        """Invalid SAR pixels are 0.0 and must not drag the mean down."""
        dataset = KuroSiwoSarDataset(adapter, "train")
        samples = [dataset.load_canonical(i) for i in range(len(dataset))]
        artifact = fit_training_statistics(samples)

        stacks = [stack_channels(s) for s in samples]
        masks = [np.asarray(s.validity, dtype=bool) for s in samples]
        expected = [
            float(np.concatenate([st[c][m] for st, m in zip(stacks, masks)]).mean())
            for c in range(4)
        ]
        np.testing.assert_allclose(artifact.mean_vector(), expected, rtol=1e-5)

        naive = [
            float(np.concatenate([st[c].ravel() for st in stacks]).mean())
            for c in range(4)
        ]
        assert not np.allclose(expected, naive), (
            "fixture must make masked and unmasked means differ, or this test "
            "proves nothing"
        )

    def test_fitting_refuses_non_training_samples(self, adapter):
        validation = KuroSiwoSarDataset(adapter, "validation")
        with pytest.raises(StatisticsError, match="training split only"):
            fit_training_statistics([validation.load_canonical(0)])

    def test_fitted_provenance_answers_every_required_field(self, adapter):
        dataset = KuroSiwoSarDataset(adapter, "train")
        artifact = fit_training_statistics(
            [dataset.load_canonical(i) for i in range(len(dataset))]
        )
        provenance = artifact.provenance
        assert artifact.kind == "fitted"
        assert provenance["schema_id"] == "floodmap.statistics/1"
        assert provenance["split"] == "train"
        assert provenance["channels"] == list(CHANNEL_ORDER)
        assert provenance["representation"] == "linear_sigma0"
        assert provenance["clip_state"] == "applied_0.15_linear_sigma0"
        assert provenance["activations"] == [str(TRAIN_ACTIVATION)]
        assert provenance["n_tiles"] == 4
        assert provenance["n_valid_pixels"] > 0
        assert provenance["fitted_from_validation_or_test"] is False
        assert "aggregation_basis" in provenance
        assert "temporal_role_basis" in provenance
        assert artifact.require_fitted() is artifact

    def test_artifact_round_trips_as_a_sidecar_pair(self, adapter, tmp_path):
        dataset = KuroSiwoSarDataset(adapter, "train")
        artifact = fit_training_statistics(
            [dataset.load_canonical(i) for i in range(len(dataset))]
        )
        values_path, provenance_path = artifact.write(tmp_path / "stats")
        assert values_path.name == VALUES_FILE
        assert provenance_path.name == PROVENANCE_FILE

        # The values file must be flat and float-only: the features reader
        # coerces float() over every value of every top-level key.
        raw = json.loads(values_path.read_text(encoding="utf-8"))
        for entry in raw.values():
            assert set(entry) == {"mean", "std"}
            assert all(isinstance(value, float) for value in entry.values())

        reloaded = StatisticsArtifact.read(tmp_path / "stats")
        assert reloaded.kind == "fitted"
        np.testing.assert_allclose(reloaded.mean_vector(), artifact.mean_vector())

    def test_an_incomplete_artifact_is_rejected(self, adapter, tmp_path):
        artifact = development_statistics()
        artifact.write(tmp_path / "partial")
        (tmp_path / "partial" / PROVENANCE_FILE).unlink()
        with pytest.raises(StatisticsError, match="incomplete"):
            StatisticsArtifact.read(tmp_path / "partial")

    def test_reference_statistics_can_never_normalise(self):
        reference = StatisticsArtifact(
            kind="reference",
            values=development_statistics().values,
            provenance={**development_statistics().provenance, "kind": "reference"},
        )
        with pytest.raises(StatisticsError, match="never be used"):
            reference.require_fitted()

    def test_a_reordered_artifact_is_rejected(self):
        values = dict(development_statistics().values)
        reordered = {name: values[name] for name in reversed(CHANNEL_ORDER)}
        with pytest.raises(StatisticsError, match="channel order"):
            StatisticsArtifact(
                kind="development",
                values=reordered,
                provenance=development_statistics().provenance,
            )


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_the_baseline_config_loads_and_declares_the_frozen_input(self):
        config = load_experiment("m5_kuro_sar_unet")
        assert config.dataset.channels == list(CHANNEL_ORDER)
        assert config.dataset.representation == "linear_sigma0"
        assert config.dataset.ignore_index == IGNORE_INDEX
        assert config.model.in_channels == 4
        assert config.model.num_classes == NUM_CLASSES
        assert config.primary_metric == "iou_flood_class"
        assert config.loss.class_weights is None

    def test_a_config_without_a_fitted_artifact_is_loadable_but_blocked(self):
        """The blocking MECHANISM, tested independently of current state.

        C5 forbids both a fabricated value and a path to an artifact that does
        not exist, so "enabled + fitted + no path" must stay legal to LOAD and
        illegal to RUN. The production config has since been pointed at a real
        artifact (see TestProductionStatisticsWiring); this pins the behaviour
        that protected it beforehand and would protect any future config.
        """
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["normalisation"]["statistics_path"] = None
        config = ExperimentConfig.model_validate(payload)  # loads fine

        assert config.normalisation.required_kind == "fitted"
        assert config.normalisation.is_runnable is False
        assert "NO FITTED STATISTICS EXIST" in config.normalisation.blocked_reason()

    def test_the_development_config_is_marked_development_only(self):
        config = load_experiment("m5_kuro_sar_unet_dev")
        assert config.status == "development_only"
        assert config.normalisation.required_kind == "development"
        assert config.normalisation.allow_development_statistics is True
        assert config.normalisation.is_runnable is True

    def test_checkpoint_selection_follows_c6(self):
        config = load_experiment("m5_kuro_sar_unet")
        assert config.checkpoint.monitor == "val_iou_flood_class"
        assert config.checkpoint.mode == "max"
        assert config.checkpoint.selection_split == "validation"

    def test_selecting_on_test_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["checkpoint"]["selection_split"] = "test"
        with pytest.raises(ValueError, match="validation split"):
            ExperimentConfig.model_validate(payload)

    def test_training_on_the_test_split_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["dataset"]["validation_split"] = "test"
        with pytest.raises(ValueError, match="test split must not be used"):
            ExperimentConfig.model_validate(payload)

    def test_a_reordered_channel_list_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["dataset"]["channels"] = ["post_vv", "post_vh", "pre_vv", "pre_vh"]
        with pytest.raises(ValueError, match="channels must be exactly"):
            ExperimentConfig.model_validate(payload)

    def test_a_decibel_representation_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["dataset"]["representation"] = "decibel"
        with pytest.raises(ValueError, match="not converted to decibel"):
            ExperimentConfig.model_validate(payload)

    def test_extra_channels_are_rejected(self):
        """No DEM, slope, optical or handcrafted feature in this baseline."""
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["dataset"]["channels"] = list(CHANNEL_ORDER) + ["dem"]
        with pytest.raises(ValueError, match="channels must be exactly"):
            ExperimentConfig.model_validate(payload)
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["model"]["in_channels"] = 5
        with pytest.raises(ValueError, match="in_channels must be 4"):
            ExperimentConfig.model_validate(payload)

    def test_nodata_must_not_become_a_fourth_class(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["model"]["num_classes"] = 4
        with pytest.raises(ValueError, match="must not get a logit"):
            ExperimentConfig.model_validate(payload)

    def test_radiometric_augmentation_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["augmentation"]["radiometric"] = True
        with pytest.raises(ValueError, match="radiometric augmentation is not permitted"):
            ExperimentConfig.model_validate(payload)

    def test_pretrained_weights_are_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["model"]["pretrained_weights"] = "imagenet"
        with pytest.raises(ValueError, match="no pretrained weights"):
            ExperimentConfig.model_validate(payload)

    def test_a_mistyped_key_is_not_silently_ignored(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["training"]["learnign_rate"] = 0.1
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(payload)

    def test_mismatched_ignore_indices_are_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["loss"]["ignore_index"] = 255
        with pytest.raises(ValueError, match="disagree"):
            ExperimentConfig.model_validate(payload)

    def test_development_statistics_require_explicit_opt_in(self):
        payload = load_experiment("m5_kuro_sar_unet_dev").model_dump()
        payload["normalisation"]["allow_development_statistics"] = False
        with pytest.raises(ValueError, match="must be paired with"):
            ExperimentConfig.model_validate(payload)

    def test_experiment_configs_do_not_touch_the_frozen_segmentation_config(self):
        """The frozen file still defers every decision M5 made locally."""
        from floodmap.utils.config import load_config

        frozen = load_config("segmentation")
        assert frozen["model"]["architecture"] is None
        assert frozen["training"]["seed"] is None
        assert frozen["features"]["normalisation"]["statistics_path"] is None


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


class TestDevice:
    def test_cpu_path_always_works(self):
        assert select_device("cpu").type == "cpu"
        model = UNet(base_channels=8, depth=2, seed=0).to(torch.device("cpu"))
        assert model(torch.zeros(1, 4, TILE, TILE)).device.type == "cpu"

    def test_automatic_selection_returns_a_usable_device(self):
        device = select_device(None)
        assert device.type in {"cuda", "mps", "cpu"}
        model = UNet(base_channels=8, depth=2, seed=0).to(device)
        output = model(torch.zeros(1, 4, TILE, TILE, device=device))
        assert output.shape == (1, NUM_CLASSES, TILE, TILE)

    def test_automatic_priority_is_cuda_then_mps_then_cpu(self):
        available = available_devices()
        device = select_device("auto")
        if available["cuda_available"]:
            assert device.type == "cuda"
        elif available["mps_available"]:
            assert device.type == "mps"
        else:
            assert device.type == "cpu"

    def test_an_unavailable_explicit_device_raises_rather_than_falling_back(self):
        """A run meant for a GPU must not quietly spend hours on CPU."""
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                select_device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            with pytest.raises(RuntimeError, match="MPS is not available"):
                select_device("mps")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_cuda_forward_and_backward(self):
        device = torch.device("cuda")
        model = UNet(base_channels=8, depth=2, seed=0).to(device)
        logits = model(torch.randn(1, 4, TILE, TILE, device=device))
        logits.sum().backward()
        assert logits.device.type == "cuda"

    @pytest.mark.skipif(
        not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        reason="no MPS device",
    )
    def test_mps_forward_and_backward(self):
        device = torch.device("mps")
        model = UNet(base_channels=8, depth=2, seed=0).to(device)
        logits = model(torch.randn(1, 4, TILE, TILE, device=device))
        logits.sum().backward()
        assert logits.device.type == "mps"
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_seed_everything_makes_torch_numpy_and_random_deterministic(self):
        import random

        seed_everything(1234)
        first = (torch.randn(4).tolist(), np.random.rand(4).tolist(), random.random())
        seed_everything(1234)
        second = (torch.randn(4).tolist(), np.random.rand(4).tolist(), random.random())
        assert first == second

    def test_seed_everything_returns_the_seed_for_recording(self):
        assert seed_everything(77) == 77

    def test_identical_seeds_give_identical_training_trajectories(self, adapter):
        config = load_experiment("m5_kuro_sar_unet_dev")

        def _run():
            result = run_training(config, adapter=adapter, write_artifacts=False)
            return [record["train_loss"] for record in result.history]

        assert _run() == _run()

    def test_a_different_seed_changes_the_trajectory(self, adapter):
        payload = load_experiment("m5_kuro_sar_unet_dev").model_dump()
        payload["training"]["epochs"] = 2
        first = run_training(
            ExperimentConfig.model_validate(payload), adapter=adapter, write_artifacts=False
        )
        payload["training"]["seed"] = 999
        second = run_training(
            ExperimentConfig.model_validate(payload), adapter=adapter, write_artifacts=False
        )
        assert [r["train_loss"] for r in first.history] != [
            r["train_loss"] for r in second.history
        ]


# ---------------------------------------------------------------------------
# Training loop and checkpoints
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_one_step_updates_parameters(self, adapter):
        dataset = KuroSiwoSarDataset(
            adapter, "train", statistics=development_statistics()
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_samples
        )
        model = UNet(base_channels=8, depth=2, seed=0)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        stats = train_one_epoch(
            model=model,
            loader=loader,
            criterion=build_loss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_steps=1,
        )
        assert stats["steps"] == 1
        assert np.isfinite(stats["loss"])
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before, model.parameters())
        )

    def test_gradients_are_non_zero_on_real_tensors(self, adapter):
        dataset = KuroSiwoSarDataset(
            adapter, "train", statistics=development_statistics()
        )
        batch = collate_samples([dataset[0], dataset[1]])
        model = UNet(base_channels=8, depth=2, seed=0)
        loss = build_loss()(model(batch["image"]), batch["target"])
        loss.backward()
        total = sum(
            float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
        )
        assert total > 0.0

    def test_invalid_pixels_do_not_contribute_to_the_loss(self, adapter):
        """Perturbing logits only where the mask is invalid must not move the loss.

        This is the end-to-end version of the loss-level test: it goes through
        the real adapter, the real validity mask and the real target.
        """
        dataset = KuroSiwoSarDataset(adapter, "train")
        batch = collate_samples([dataset[0]])
        criterion = build_loss()
        invalid = ~batch["valid_mask"]
        assert invalid.any(), "fixture must contain invalid pixels"

        logits = torch.zeros(1, NUM_CLASSES, TILE, TILE)
        baseline = criterion(logits, batch["target"])

        perturbed = logits.clone()
        perturbed[:, 0][invalid] = 80.0
        perturbed[:, 1][invalid] = -80.0
        assert torch.allclose(criterion(perturbed, batch["target"]), baseline)

        # And no gradient reaches those positions.
        grad_logits = torch.zeros(1, NUM_CLASSES, TILE, TILE, requires_grad=True)
        criterion(grad_logits, batch["target"]).backward()
        assert float(grad_logits.grad.permute(1, 0, 2, 3)[:, invalid].abs().sum()) == 0.0

    def test_evaluate_excludes_nodata_from_metrics(self, adapter):
        dataset = KuroSiwoSarDataset(adapter, "validation")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_samples
        )
        result, loss = evaluate(
            model=UNet(base_channels=8, depth=2, seed=0),
            loader=loader,
            criterion=build_loss(),
            device=torch.device("cpu"),
            ignore_index=IGNORE_INDEX,
        )
        assert result.n_ignored_pixels > 0
        assert result.n_valid_pixels == int(
            sum(int(dataset[i]["valid_mask"].sum()) for i in range(len(dataset)))
        )
        assert np.isfinite(loss)


class TestCheckpoints:
    def test_run_writes_best_and_last_checkpoints_and_a_run_record(
        self, adapter, tmp_path
    ):
        payload = load_experiment("m5_kuro_sar_unet_dev").model_dump()
        payload["training"]["epochs"] = 2
        config = ExperimentConfig.model_validate(payload)

        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        assert result.best_checkpoint is not None and result.best_checkpoint.is_file()
        assert result.last_checkpoint is not None and result.last_checkpoint.is_file()
        assert result.run_record_path is not None and result.run_record_path.is_file()
        assert result.best_epoch in (1, 2)
        assert len(result.history) == 2

    def test_checkpoint_carries_everything_needed_to_reproduce(self, adapter, tmp_path):
        config = load_experiment("m5_kuro_sar_unet_dev")
        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        payload = load_checkpoint(result.best_checkpoint)

        assert payload["epoch"] >= 1
        assert payload["seed"] == config.training.seed
        assert payload["val_iou_flood_class"] is not None
        assert payload["config"]["name"] == "m5_kuro_sar_unet_dev"
        assert payload["channel_order"] == list(CHANNEL_ORDER)
        assert payload["dataset_identity"]["dataset_id"] == "kuro_siwo"
        assert payload["dataset_identity"]["split"] == "train"
        assert payload["statistics_identity"]["kind"] == "development"
        assert payload["statistics_identity"]["usable_for_reported_results"] is False
        assert payload["optimizer_state"] is not None
        assert "model_state" in payload

    def test_checkpoint_round_trips_into_a_fresh_model(self, adapter, tmp_path):
        config = load_experiment("m5_kuro_sar_unet_dev")
        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        payload = load_checkpoint(result.best_checkpoint)

        restored = UNet.from_config(config.model, seed=config.training.seed)
        restored.load_state_dict(payload["model_state"])
        restored.eval()

        dataset = KuroSiwoSarDataset(
            adapter, "validation", statistics=development_statistics()
        )
        batch = collate_samples([dataset[0]])
        with torch.no_grad():
            assert restored(batch["image"]).shape == (1, NUM_CLASSES, TILE, TILE)

    def test_a_checkpoint_with_a_different_channel_order_is_rejected(
        self, adapter, tmp_path
    ):
        model = UNet(base_channels=8, depth=2, seed=0)
        path = save_checkpoint(
            tmp_path / "bad.pt",
            model=model,
            optimizer=None,
            epoch=1,
            val_iou_flood_class=0.5,
            config={},
            statistics_identity={},
            dataset_identity={},
            seed=0,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["channel_order"] = ["post_vv", "post_vh", "pre_vv", "pre_vh"]
        torch.save(payload, path)
        with pytest.raises(ValueError, match="channel order"):
            load_checkpoint(path)

    def test_the_run_record_names_the_unweighted_loss_and_dev_statistics(
        self, adapter, tmp_path
    ):
        config = load_experiment("m5_kuro_sar_unet_dev")
        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))

        assert record["status"] == "development_only"
        assert record["primary_metric"] == "iou_flood_class"
        assert record["loss"]["weighting"] == "unweighted"
        assert record["loss"]["ignore_index"] == IGNORE_INDEX
        assert record["statistics_identity"]["kind"] == "development"
        assert record["channel_order"] == list(CHANNEL_ORDER)
        assert record["flood_class_id"] == FLOOD_CLASS_ID
        assert record["train_dataset"]["activations"] == [str(TRAIN_ACTIVATION)]
        assert record["validation_dataset"]["activations"] == [
            str(VALIDATION_ACTIVATION)
        ]
        assert record["seed"] == config.training.seed
        assert record["model"]["parameters"] > 0

    def test_training_never_reads_the_test_split(self, adapter, tmp_path):
        """Selection discipline: no code path loads test data."""
        config = load_experiment("m5_kuro_sar_unet_dev")
        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
        assert record["train_dataset"]["split"] == "train"
        assert record["validation_dataset"]["split"] == "validation"
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "floodmap"
            / "segmentation"
            / "train.py"
        ).read_text(encoding="utf-8")
        assert '"test"' not in source and "'test'" not in source

    def test_a_blocked_config_refuses_to_run(self, adapter, tmp_path):
        """A config requiring fitted statistics it does not have must not start."""
        payload = load_experiment("m5_kuro_sar_unet_dev").model_dump()
        payload["normalisation"]["required_kind"] = "fitted"
        payload["normalisation"]["allow_development_statistics"] = False
        payload["normalisation"]["statistics_path"] = None
        config = ExperimentConfig.model_validate(payload)

        with pytest.raises(RuntimeError, match="NO FITTED STATISTICS EXIST"):
            run_training(config, adapter=adapter, run_dir=tmp_path / "blocked")
        assert not (tmp_path / "blocked").exists() or not list(
            (tmp_path / "blocked").glob("*.pt")
        )


# ---------------------------------------------------------------------------
# Canonical-grid reprojection in the training path
# ---------------------------------------------------------------------------


class TestCanonicalGridReprojection:
    """The M5 side of the shared spatial component.

    These exercise the real ``KuroSiwoSarDataset`` path, so they cover the
    adaptation from a ``CanonicalSample`` as well as the warp itself.
    """

    def _settings(self, **overrides):
        from floodmap.segmentation.config import ReprojectionSection

        payload = {
            "enabled": True,
            "resolution_m": 10.0,
            "tile_size": 32,
            "sar_resampling": "bilinear",
            "label_resampling": "nearest",
            "nodata_mode": "strict",
        }
        payload.update(overrides)
        return ReprojectionSection.model_validate(payload)

    def test_source_grid_is_web_mercator_with_projected_unit_pixels(self, canonical_sample):
        """The premise: the delivered grid is NOT 10 m true ground."""
        from floodmap.segmentation import source_grid_for_sample

        grid = source_grid_for_sample(canonical_sample)
        assert grid.crs == "EPSG:3857"
        assert grid.resolution_m == 10.0  # projected units, not metres
        assert (grid.height, grid.width) == canonical_sample.source_shape

    def test_reprojection_produces_a_real_utm_grid(self, canonical_sample):
        from floodmap.segmentation import reproject_canonical_sample

        out = reproject_canonical_sample(canonical_sample, resolution_m=10.0, tile_size=32)
        assert out.grid.crs.startswith("EPSG:326") or out.grid.crs.startswith("EPSG:327")
        assert out.grid.crs != "EPSG:3857"
        assert out.grid.resolution_m == 10.0
        assert (out.grid.height, out.grid.width) == (32, 32)
        assert out.provenance["target_resolution_is_true_ground"] is True
        assert out.provenance["source_resolution_is_true_ground"] is False

    def test_payload_shapes_dtypes_and_alignment(self, canonical_sample):
        from floodmap.segmentation import reproject_canonical_sample

        out = reproject_canonical_sample(canonical_sample, tile_size=32)
        assert out.image.shape == (4, 32, 32)
        assert out.image.dtype == np.float32
        assert out.target.shape == (32, 32)
        assert out.target.dtype == np.int64
        assert out.valid_mask.shape == (32, 32)
        assert out.valid_mask.dtype == np.bool_

    def test_classes_and_ignore_index_survive_reprojection(self, canonical_sample):
        from floodmap.segmentation import reproject_canonical_sample

        out = reproject_canonical_sample(canonical_sample, tile_size=32)
        assert set(np.unique(out.target[out.valid_mask])).issubset({0, 1, 2})
        assert set(np.unique(out.target[~out.valid_mask])).issubset({IGNORE_INDEX})
        assert IGNORE_INDEX not in set(np.unique(out.target[out.valid_mask]))

    def test_no_nodata_sentinel_survives_on_a_valid_pixel(self, canonical_sample):
        from floodmap.segmentation import reproject_canonical_sample
        from floodmap.segmentation.spatial import SAR_NODATA

        out = reproject_canonical_sample(canonical_sample, tile_size=32)
        on_valid = out.image[:, out.valid_mask]
        assert on_valid.size > 0
        assert not np.any(on_valid == SAR_NODATA)
        assert bool(np.isfinite(on_valid).all())
        assert np.all(out.image[:, ~out.valid_mask] == SAR_NODATA)

    def test_one_mask_governs_all_four_channels_and_the_target(self, canonical_sample):
        from floodmap.segmentation import reproject_canonical_sample
        from floodmap.segmentation.spatial import SAR_NODATA

        out = reproject_canonical_sample(canonical_sample, tile_size=32)
        per_channel = [(out.image[c] == SAR_NODATA) for c in range(4)]
        for channel in per_channel[1:]:
            np.testing.assert_array_equal(channel, per_channel[0])
        np.testing.assert_array_equal(per_channel[0], ~out.valid_mask)
        np.testing.assert_array_equal(out.target == IGNORE_INDEX, ~out.valid_mask)

    def test_reprojection_does_not_convert_to_db(self, canonical_sample):
        """The baseline stays in linear sigma0 through the warp."""
        from floodmap.segmentation import reproject_canonical_sample

        out = reproject_canonical_sample(canonical_sample, tile_size=32)
        assert out.provenance["db_conversion_applied"] is False
        assert out.provenance["representation"] == "linear_sigma0"
        on_valid = out.image[:, out.valid_mask]
        # Linear sigma0 clipped at 0.15 is non-negative and small; decibel
        # backscatter would be strongly negative.
        assert float(on_valid.min()) >= 0.0
        assert float(on_valid.max()) <= 0.15 + 1e-6

    def test_sar_is_bilinear_and_labels_are_nearest(self, canonical_sample):
        from floodmap.segmentation import reproject_canonical_sample

        out = reproject_canonical_sample(canonical_sample, tile_size=32)
        assert out.provenance["sar"]["resampling"] == "bilinear"
        assert out.provenance["labels"]["resampling"] == "nearest"
        assert out.provenance["sar"]["domain"] == "linear_power"
        assert out.provenance["sar"]["db_conversion_applied_before_resampling"] is False

    def test_a_decibel_sample_is_refused(self, canonical_sample):
        import dataclasses

        from floodmap.segmentation import reproject_canonical_sample
        from floodmap.segmentation.tensors import TensorContractError

        decibel = dataclasses.replace(canonical_sample, sar_representation="decibel")
        with pytest.raises(TensorContractError, match="linear_sigma0"):
            reproject_canonical_sample(decibel, tile_size=32)

    def test_fixed_tile_size_makes_samples_batchable(self, adapter):
        """Without a fixed size the per-activation output shape varies."""
        settings = self._settings(tile_size=32)
        train = KuroSiwoSarDataset(adapter, "train", reprojection=settings)
        validation = KuroSiwoSarDataset(adapter, "validation", reprojection=settings)

        batch = collate_samples([train[0], train[1], validation[0]])
        assert batch["image"].shape == (3, 4, 32, 32)
        assert batch["target"].shape == (3, 32, 32)
        assert batch["valid_mask"].shape == (3, 32, 32)

    def test_dataset_reprojection_changes_the_grid_not_the_contract(self, adapter):
        settings = self._settings(tile_size=32)
        plain = KuroSiwoSarDataset(adapter, "train")[0]
        warped = KuroSiwoSarDataset(adapter, "train", reprojection=settings)[0]

        assert plain["image"].shape == (4, TILE, TILE)
        assert warped["image"].shape == (4, 32, 32)
        for item in (plain, warped):
            assert item["image"].dtype == torch.float32
            assert item["target"].dtype == torch.int64
            assert item["valid_mask"].dtype == torch.bool
            valid = item["valid_mask"].numpy()
            target = item["target"].numpy()
            assert set(np.unique(target[valid])).issubset({0, 1, 2})
            assert np.all(target[~valid] == IGNORE_INDEX)

    def test_reprojection_is_recorded_in_dataset_identity(self, adapter):
        settings = self._settings(tile_size=32)
        off = KuroSiwoSarDataset(adapter, "train").identity()
        on = KuroSiwoSarDataset(adapter, "train", reprojection=settings).identity()

        assert off["reprojected_to_canonical_grid"] is False
        assert off["spatial_transform"] is None
        assert on["reprojected_to_canonical_grid"] is True
        assert on["spatial_transform"]["resolution_m"] == 10.0
        assert on["spatial_transform"]["tile_size"] == 32
        assert on["spatial_transform"]["nodata_mode"] == "strict"
        assert (
            on["spatial_transform"]["shared_implementation"]
            == "floodmap.preprocessing.reprojection"
        )

    def test_a_tiny_training_step_runs_on_reprojected_tiles(self, adapter):
        """End to end: the warped grid feeds the model, loss and optimizer."""
        settings = self._settings(tile_size=32)
        dataset = KuroSiwoSarDataset(
            adapter, "train", statistics=development_statistics(), reprojection=settings
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_samples
        )
        model = UNet(base_channels=8, depth=2, seed=0)
        before = [p.detach().clone() for p in model.parameters()]
        stats = train_one_epoch(
            model=model,
            loader=loader,
            criterion=build_loss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            device=torch.device("cpu"),
            max_steps=1,
        )
        assert stats["steps"] == 1
        assert np.isfinite(stats["loss"])
        assert any(
            not torch.equal(old, new) for old, new in zip(before, model.parameters())
        )


class TestTrainingAndInferenceShareOnePath:
    """Phase 4 item 8: identical preprocessing on both paths."""

    def test_the_training_path_calls_the_m2_shared_component(self, canonical_sample):
        """Byte-identical output from the training path and a direct M2-style call.

        The training path goes CanonicalSample -> segmentation.spatial ->
        preprocessing.reprojection. The inference path goes raster ->
        preprocessing.pipeline._reproject_to_grid -> the same function. Calling
        the shared function directly with the same grids must reproduce what
        training produced, bit for bit.
        """
        from floodmap.preprocessing.reprojection import (
            NodataPolicy,
            reproject_continuous,
            target_grid_for_source,
        )
        from floodmap.segmentation import (
            reproject_canonical_sample,
            source_grid_for_sample,
            stack_channels,
        )
        from floodmap.segmentation.spatial import SAR_NODATA

        training = reproject_canonical_sample(canonical_sample, tile_size=32)

        source = source_grid_for_sample(canonical_sample)
        target = target_grid_for_source(source, resolution_m=10.0, tile_size=32)
        direct = reproject_continuous(
            stack_channels(canonical_sample),
            np.asarray(canonical_sample.validity, dtype=bool),
            source=source,
            target=target,
            resampling_name="bilinear",
            nodata=SAR_NODATA,
            nodata_policy=NodataPolicy("strict"),
            domain="linear_power",
        )

        assert training.grid.crs == direct.grid.crs
        assert tuple(training.grid.transform[:6]) == tuple(direct.grid.transform[:6])
        # Training additionally intersects with the label footprint, so its
        # valid set is a subset; where both are valid the values must match
        # exactly, not approximately.
        shared = training.valid_mask & direct.valid
        assert int(shared.sum()) > 0
        np.testing.assert_array_equal(
            training.image[:, shared], direct.data[:, shared]
        )

    def test_m2_pipeline_delegates_rather_than_reimplementing_the_warp(self):
        """Structural guard against the two paths drifting apart.

        A second local ``reproject(...)`` loop inside the M2 pipeline is exactly
        how training and inference would diverge while both looked correct, so
        its absence is asserted rather than assumed.
        """
        pipeline = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "floodmap"
            / "preprocessing"
            / "pipeline.py"
        ).read_text(encoding="utf-8")
        assert "from .reprojection import" in pipeline
        assert "reproject_continuous(" in pipeline
        # The old hand-rolled warp called rasterio's reproject directly.
        assert "reproject(\n" not in pipeline
        assert "dst_crs=" not in pipeline

    def test_both_paths_agree_on_the_target_grid_for_one_source(self, canonical_sample):
        from floodmap.preprocessing.reprojection import target_grid_for_bounds
        from floodmap.segmentation import source_grid_for_sample

        source = source_grid_for_sample(canonical_sample)
        from_source = target_grid_for_bounds(
            bounds=source.bounds, source_crs=source.crs, tile_size=32
        )
        from_sample = reproject_canonical_sample_grid(canonical_sample)
        assert from_source.crs == from_sample.crs
        assert tuple(from_source.transform[:6]) == tuple(from_sample.transform[:6])

    def test_statistics_fitting_uses_the_same_spatial_transform(self, adapter):
        """Phase 5 precondition: statistics must describe the training arrays.

        A std fitted before resampling does not describe resampled input --
        bilinear changes the variance of the field it is applied to -- so the
        fitter must accept and record the same reprojection settings.
        """
        from floodmap.segmentation.config import ReprojectionSection

        settings = ReprojectionSection.model_validate(
            {"enabled": True, "tile_size": 32, "nodata_mode": "strict"}
        )
        dataset = KuroSiwoSarDataset(adapter, "train")
        samples = [dataset.load_canonical(i) for i in range(len(dataset))]

        plain = fit_training_statistics(samples)
        warped = fit_training_statistics(samples, reprojection=settings)

        assert plain.provenance["spatial_transform"]["reprojected_to_canonical_grid"] is False
        assert warped.provenance["spatial_transform"]["reprojected_to_canonical_grid"] is True
        assert warped.provenance["spatial_transform"]["tile_size"] == 32
        assert (
            warped.provenance["spatial_transform"]["shared_implementation"]
            == "floodmap.preprocessing.reprojection"
        )
        # Resampling changes the statistics, which is the whole reason the
        # setting has to be recorded rather than assumed.
        assert not np.allclose(plain.std_vector(), warped.std_vector())


def reproject_canonical_sample_grid(sample):
    """The realised target grid for a sample, for cross-path comparison."""
    from floodmap.segmentation import reproject_canonical_sample

    return reproject_canonical_sample(sample, tile_size=32).grid


class TestReprojectionConfiguration:
    def test_the_baseline_config_enables_reprojection_with_a_fixed_tile(self):
        config = load_experiment("m5_kuro_sar_unet")
        assert config.reprojection.enabled is True
        assert config.reprojection.resolution_m == 10.0
        assert config.reprojection.tile_size == 224
        assert config.reprojection.sar_resampling == "bilinear"
        assert config.reprojection.label_resampling == "nearest"
        assert config.reprojection.nodata_mode == "strict"

    def test_enabling_reprojection_without_a_tile_size_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["reprojection"]["tile_size"] = None
        with pytest.raises(ValueError, match="tile_size is required"):
            ExperimentConfig.model_validate(payload)

    def test_nearest_sar_resampling_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["reprojection"]["sar_resampling"] = "nearest"
        with pytest.raises(ValueError, match="averaging method"):
            ExperimentConfig.model_validate(payload)

    def test_bilinear_label_resampling_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["reprojection"]["label_resampling"] = "bilinear"
        with pytest.raises(ValueError, match="invents classes"):
            ExperimentConfig.model_validate(payload)

    def test_an_unknown_nodata_mode_is_rejected(self):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["reprojection"]["nodata_mode"] = "permissive"
        with pytest.raises(ValueError, match="strict.*majority"):
            ExperimentConfig.model_validate(payload)

    def test_the_run_record_states_whether_reprojection_happened(self, adapter, tmp_path):
        config = load_experiment("m5_kuro_sar_unet_dev")
        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
        spatial = record["spatial_preprocessing"]
        assert spatial["reprojected_to_canonical_grid"] is False
        assert spatial["shared_implementation"] == "floodmap.preprocessing.reprojection"

    def test_a_reprojected_run_records_its_settings(self, adapter, tmp_path):
        payload = load_experiment("m5_kuro_sar_unet_dev").model_dump()
        payload["reprojection"]["enabled"] = True
        payload["reprojection"]["tile_size"] = 32
        config = ExperimentConfig.model_validate(payload)

        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
        spatial = record["spatial_preprocessing"]
        assert spatial["reprojected_to_canonical_grid"] is True
        assert spatial["settings"]["tile_size"] == 32
        assert spatial["settings"]["nodata_mode"] == "strict"
        assert record["train_dataset"]["reprojected_to_canonical_grid"] is True
        assert record["validation_dataset"]["reprojected_to_canonical_grid"] is True


# ---------------------------------------------------------------------------
# Quarantine applied at the two eligibility points
# ---------------------------------------------------------------------------


class TestQuarantineEligibility:
    """The same policy must govern training and statistics fitting.

    If a sample were excluded from training but still fed to the fitter, the
    normalisation parameters would describe a corpus the model never sees --
    and nothing downstream could detect it.
    """

    def _policy(self, tmp_path, sample_id):
        import yaml

        from floodmap.datasets.quarantine import load_quarantine

        path = tmp_path / "q.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "version": "test-1",
                    "dataset_id": "kuro_siwo",
                    "reason_codes": {"synthetic": "excluded by a test"},
                    "samples": [{"sample_id": sample_id, "reason_code": "synthetic"}],
                }
            ),
            encoding="utf-8",
        )
        return load_quarantine("kuro_siwo", path=path, required=True)

    def test_a_quarantined_sample_is_removed_from_the_split(self, adapter, tmp_path):
        full = KuroSiwoSarDataset(adapter, "train")
        victim = full.sample_ids[1]
        policy = self._policy(tmp_path, victim)

        filtered = KuroSiwoSarDataset(adapter, "train", quarantine=policy)
        assert len(filtered) == len(full) - 1
        assert victim not in filtered.sample_ids
        assert set(filtered.sample_ids) == set(full.sample_ids) - {victim}

    def test_the_exclusion_is_recorded_in_dataset_identity(self, adapter, tmp_path):
        full = KuroSiwoSarDataset(adapter, "train")
        victim = full.sample_ids[0]
        identity = KuroSiwoSarDataset(
            adapter, "train", quarantine=self._policy(tmp_path, victim)
        ).identity()

        audit = identity["quarantine"]
        assert audit["policy_applied"] is True
        assert audit["policy_version"] == "test-1"
        assert audit["n_excluded"] == 1
        assert audit["excluded_by_reason_code"] == {"synthetic": 1}
        assert audit["excluded_samples"][0]["sample_id"] == victim
        assert audit["excluded_samples"][0]["reason_code"] == "synthetic"

    def test_no_policy_is_reported_as_no_policy_not_as_no_exclusions(self, adapter):
        audit = KuroSiwoSarDataset(adapter, "train").identity()["quarantine"]
        assert audit["policy_applied"] is False
        assert audit["policy_version"] is None
        assert audit["n_excluded"] == 0

    def test_explicitly_requesting_a_quarantined_sample_raises(self, adapter, tmp_path):
        """Never return a shorter list than was asked for."""
        from floodmap.datasets.quarantine import QuarantineError

        full = KuroSiwoSarDataset(adapter, "train")
        victim = full.sample_ids[2]
        policy = self._policy(tmp_path, victim)

        with pytest.raises(QuarantineError, match="explicitly requested"):
            KuroSiwoSarDataset(
                adapter, "train", sample_ids=[full.sample_ids[0], victim], quarantine=policy
            )

    def test_statistics_fitting_honours_the_same_policy(self, adapter, tmp_path):
        dataset = KuroSiwoSarDataset(adapter, "train")
        samples = [dataset.load_canonical(i) for i in range(len(dataset))]
        victim = dataset.sample_ids[1]
        policy = self._policy(tmp_path, victim)

        unfiltered = fit_training_statistics(samples)
        filtered = fit_training_statistics(samples, quarantine=policy)

        assert unfiltered.provenance["n_tiles"] == len(samples)
        assert filtered.provenance["n_tiles"] == len(samples) - 1
        # Excluding a tile must actually change the parameters, or the test
        # would pass without the exclusion having any effect.
        assert not np.allclose(unfiltered.mean_vector(), filtered.mean_vector())

    def test_statistics_provenance_records_the_exclusion(self, adapter, tmp_path):
        dataset = KuroSiwoSarDataset(adapter, "train")
        samples = [dataset.load_canonical(i) for i in range(len(dataset))]
        victim = dataset.sample_ids[0]

        artifact = fit_training_statistics(
            samples, quarantine=self._policy(tmp_path, victim)
        )
        audit = artifact.provenance["quarantine"]
        assert audit["policy_applied"] is True
        assert audit["policy_version"] == "test-1"
        assert audit["n_excluded"] == 1
        assert audit["excluded_samples"][0]["sample_id"] == victim

    def test_statistics_provenance_records_when_no_policy_applied(self, adapter):
        dataset = KuroSiwoSarDataset(adapter, "train")
        samples = [dataset.load_canonical(i) for i in range(len(dataset))]
        audit = fit_training_statistics(samples).provenance["quarantine"]
        assert audit["policy_applied"] is False
        assert audit["n_excluded"] == 0

    def test_both_configs_enable_the_policy_and_require_the_file(self):
        for name in ("m5_kuro_sar_unet", "m5_kuro_sar_unet_dev"):
            config = load_experiment(name)
            assert config.quarantine.enabled is True, name
            assert config.quarantine.require_policy_file is True, name
            assert config.quarantine.path is None, name

    def test_the_run_record_names_the_policy(self, adapter, tmp_path):
        config = load_experiment("m5_kuro_sar_unet_dev")
        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))

        policy = record["quarantine_policy"]
        assert policy["dataset_id"] == "kuro_siwo"
        assert policy["version"]
        assert policy["evidence"] == "docs/m5-source-data-quarantine.md"
        assert policy["n_entries"] == 1
        # And both splits carry their own audit record.
        for split in ("train_dataset", "validation_dataset"):
            assert record[split]["quarantine"]["policy_applied"] is True

    def test_a_disabled_policy_is_recorded_as_disabled(self, adapter, tmp_path):
        payload = load_experiment("m5_kuro_sar_unet_dev").model_dump()
        payload["quarantine"]["enabled"] = False
        payload["quarantine"]["require_policy_file"] = False
        config = ExperimentConfig.model_validate(payload)

        result = run_training(config, adapter=adapter, run_dir=tmp_path / "run")
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
        assert record["quarantine_policy"]["applied"] is False
        assert "disabled" in record["quarantine_policy"]["reason"]

    def test_a_missing_required_policy_file_fails_the_run(self, adapter, tmp_path):
        from floodmap.datasets.quarantine import QuarantineError

        payload = load_experiment("m5_kuro_sar_unet_dev").model_dump()
        payload["quarantine"]["path"] = str(tmp_path / "absent.yaml")
        config = ExperimentConfig.model_validate(payload)

        with pytest.raises(QuarantineError, match="not found"):
            run_training(config, adapter=adapter, run_dir=tmp_path / "run")


# ---------------------------------------------------------------------------
# The production fitted-statistics artifact
# ---------------------------------------------------------------------------

#: The artifact is git-ignored (derived from a corpus that must not be
#: redistributed, C11), so these tests are gated on its presence rather than
#: failing on a fresh checkout.
_FITTED_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "statistics" / "kuro_sar_fitted"
)
_HAS_FITTED = (_FITTED_DIR / "statistics.json").is_file()


class TestProductionStatisticsWiring:
    """The production config must reference the artifact it was fitted with."""

    def test_the_config_declares_a_fitted_artifact_path(self):
        normalisation = load_experiment("m5_kuro_sar_unet").normalisation
        assert normalisation.enabled is True
        assert normalisation.required_kind == "fitted"
        assert normalisation.allow_development_statistics is False
        assert normalisation.statistics_path, "the fitted artifact path must be declared"

    def test_the_config_is_now_runnable(self):
        normalisation = load_experiment("m5_kuro_sar_unet").normalisation
        assert normalisation.is_runnable is True
        assert normalisation.blocked_reason() is None

    def test_development_statistics_are_still_refused_for_this_config(self):
        """A fitted run must not fall back to placeholder parameters."""
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["normalisation"]["allow_development_statistics"] = True
        with pytest.raises(ValueError, match="contradicts required_kind"):
            ExperimentConfig.model_validate(payload)

    @pytest.mark.parametrize("kind", ["development", "reference", "validation"])
    def test_every_non_fitted_kind_is_refused(self, kind):
        artifact = StatisticsArtifact(
            kind=kind,
            values=development_statistics().values,
            provenance={**development_statistics().provenance, "kind": kind},
        )
        with pytest.raises(StatisticsError):
            artifact.require_fitted()

    def test_a_missing_artifact_fails_loudly_rather_than_silently(self, tmp_path):
        payload = load_experiment("m5_kuro_sar_unet").model_dump()
        payload["normalisation"]["statistics_path"] = str(tmp_path / "absent")
        config = ExperimentConfig.model_validate(payload)
        from floodmap.segmentation.train import resolve_statistics

        with pytest.raises(StatisticsError, match="incomplete statistics artifact"):
            resolve_statistics(config)

    @pytest.mark.skipif(not _HAS_FITTED, reason="fitted artifact not present (git-ignored)")
    def test_the_declared_path_resolves_to_a_fitted_artifact(self):
        from floodmap.segmentation.train import resolve_statistics

        artifact = resolve_statistics(load_experiment("m5_kuro_sar_unet"))
        assert artifact.kind == "fitted"
        assert artifact.usable_for_reported_results is True
        assert artifact.require_fitted() is artifact

    @pytest.mark.skipif(not _HAS_FITTED, reason="fitted artifact not present (git-ignored)")
    def test_the_artifact_was_fitted_on_the_training_split_only(self):
        artifact = StatisticsArtifact.read(_FITTED_DIR)
        assert artifact.provenance["split"] == "train"
        assert artifact.provenance["fitted_from_validation_or_test"] is False
        # Only official TRAIN activations may appear.
        from floodmap.datasets.kuro_siwo import OFFICIAL_SPLITS

        official_train = {str(a) for a in OFFICIAL_SPLITS["train"]}
        assert set(artifact.provenance["activations"]).issubset(official_train)

    @pytest.mark.skipif(not _HAS_FITTED, reason="fitted artifact not present (git-ignored)")
    def test_the_artifact_matches_the_configs_spatial_pipeline(self):
        """A std fitted under different spatial settings is the wrong std."""
        config = load_experiment("m5_kuro_sar_unet")
        spatial = StatisticsArtifact.read(_FITTED_DIR).provenance["spatial_transform"]

        assert spatial["reprojected_to_canonical_grid"] is config.reprojection.enabled
        assert spatial["resolution_m"] == config.reprojection.resolution_m
        assert spatial["tile_size"] == config.reprojection.tile_size
        assert spatial["sar_resampling"] == config.reprojection.sar_resampling
        assert spatial["label_resampling"] == config.reprojection.label_resampling
        assert spatial["nodata_mode"] == config.reprojection.nodata_mode

    @pytest.mark.skipif(not _HAS_FITTED, reason="fitted artifact not present (git-ignored)")
    def test_the_artifact_records_the_quarantine_exclusion(self):
        audit = StatisticsArtifact.read(_FITTED_DIR).provenance["quarantine"]
        assert audit["policy_applied"] is True
        assert audit["n_excluded"] == 1
        assert audit["excluded_by_reason_code"] == {
            "nan_on_declared_valid_consumed_sar": 1
        }

    @pytest.mark.skipif(not _HAS_FITTED, reason="fitted artifact not present (git-ignored)")
    def test_the_artifact_declares_that_it_was_resampled(self):
        state = StatisticsArtifact.read(_FITTED_DIR).provenance["preprocessing_state"]
        assert state["resampled"] is True
        assert state["sar_clip_applied"] is True
        assert state["sar_conversion_to_db_applied"] is False

    @pytest.mark.skipif(not _HAS_FITTED, reason="fitted artifact not present (git-ignored)")
    def test_the_values_are_plausible_linear_sigma0_not_decibels(self):
        """An independent check that the representation did not drift."""
        artifact = StatisticsArtifact.read(_FITTED_DIR)
        means, stds = artifact.mean_vector(), artifact.std_vector()
        assert ((means > 0.0) & (means <= 0.15)).all(), means
        assert (stds > 0.0).all() and (stds <= 0.15).all(), stds
        # Decibel backscatter would be strongly negative.
        assert not (means < 0).any()

    @pytest.mark.skipif(not _HAS_FITTED, reason="fitted artifact not present (git-ignored)")
    def test_reference_values_were_not_adopted_as_our_parameters(self):
        """C5: the published reference is a cross-check, never our parameters."""
        artifact = StatisticsArtifact.read(_FITTED_DIR)
        values = {name: stat.mean for name, stat in artifact.values.items()}
        # Kuro's published data_mean. Equality would mean they were copied.
        assert values["pre_vv"] != pytest.approx(0.0953, abs=1e-6)
        assert values["pre_vh"] != pytest.approx(0.0264, abs=1e-6)
        assert artifact.provenance["aggregation_basis"].startswith("per valid pixel")
