"""Real-data M5 smoke and overfit tests.

OPT-IN. Skipped unless ``FLOODMAP_KURO_SIWO_ROOT`` is set, matching the M4
convention in ``test_dataset_adapters.py``. The external corpus never enters
this repository and no absolute dataset path is committed here.

These two tests answer different questions:

smoke
    Does the whole chain execute on real data — adapter, canonical sample,
    tensors, model, loss, ``backward()``, ``optimizer.step()``, validation,
    checkpoint? Correctness of plumbing, nothing else.

overfit
    Can the pipeline LEARN AT ALL? A pipeline can execute cleanly and still be
    incapable of learning: a detached graph, a target that is all ignore index,
    a mask applied to the wrong axis, or inputs that have been normalised to
    constants would each produce a tidy, flat loss curve. Driving a tiny fixed
    subset until the loss drops is the cheapest test that rules those out.

NEITHER IS A PERFORMANCE MEASUREMENT. Both use a handful of tiles and a toy
model with development-only identity normalisation. A high flood IoU here would
mean the model memorised four tiles.

Both are deliberately bounded: a few tiles, a few dozen iterations, CPU. No
full training run is launched from the test suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from floodmap.datasets import KuroSiwoAdapter
from floodmap.segmentation import (
    CHANNEL_ORDER,
    IGNORE_INDEX,
    NUM_CLASSES,
    KuroSiwoSarDataset,
    SegmentationMetrics,
    UNet,
    build_loss,
    collate_samples,
    development_statistics,
    fit_training_statistics,
    load_checkpoint,
    load_experiment,
    sample_to_tensors,
    seed_everything,
)
from floodmap.segmentation.config import ExperimentConfig
from floodmap.segmentation.train import run_training

pytestmark = pytest.mark.skipif(
    not os.environ.get("FLOODMAP_KURO_SIWO_ROOT"),
    reason="external Kuro Siwo root was not configured",
)

#: Deliberately tiny. The point is correctness, not model quality.
N_TRAIN_TILES = 4
N_VAL_TILES = 4
OVERFIT_ITERATIONS = 60


@pytest.fixture(scope="module")
def real_adapter() -> KuroSiwoAdapter:
    """One adapter for the module: indexing the real corpus costs ~30 s."""

    adapter = KuroSiwoAdapter()
    adapter.discover()
    return adapter


@pytest.fixture(scope="module")
def tiny_dev_config() -> ExperimentConfig:
    config = load_experiment("m5_kuro_sar_unet_dev")
    assert config.status == "development_only"
    return config


# ---------------------------------------------------------------------------
# The corpus is what the contract says it is
# ---------------------------------------------------------------------------


def test_the_real_corpus_matches_the_declared_split(real_adapter):
    """Sanity-check the delivery before drawing any conclusion from it."""

    refs = real_adapter.discover()
    counts: dict = {}
    for ref in refs:
        counts[ref.split] = counts.get(ref.split, 0) + 1
    assert refs, "no labelled Kuro samples were discovered"
    assert counts.get("train", 0) > 0, f"no training tiles available: {counts}"
    # Recorded rather than asserted exactly: the local delivery is permitted to
    # be incomplete, and the adapter reports which official activations are
    # missing instead of inventing them.
    print(f"\nreal Kuro split counts: {counts}")
    print(f"discovered activations: {real_adapter.index().discovered_activations}")


def test_a_real_sample_satisfies_the_tensor_contract(real_adapter):
    refs = sorted(
        (r for r in real_adapter.discover() if r.split == "train"),
        key=lambda r: r.sample_id,
    )
    sample = real_adapter.load_sample(refs[0].sample_id)
    tensors = sample_to_tensors(sample)

    height, width = sample.source_shape
    assert tensors.image.shape == (4, height, width)
    assert tensors.image.dtype == torch.float32
    assert tensors.target.shape == (height, width)
    assert tensors.target.dtype == torch.int64
    assert tensors.valid_mask.shape == (height, width)
    assert tensors.valid_mask.dtype == torch.bool

    assert sample.sar_representation == "linear_sigma0"
    # The adapter's 0.15 clip must already be in force on real data.
    assert float(tensors.image.max()) <= 0.15 + 1e-6

    target = tensors.target.numpy()
    valid = tensors.valid_mask.numpy()
    assert set(np.unique(target[valid])).issubset({0, 1, 2})
    assert (target[~valid] == IGNORE_INDEX).all()


def test_fitting_statistics_on_real_training_tiles(real_adapter):
    """The fitter works on real data and refuses non-training splits.

    Bounded to a few tiles: this verifies the MECHANISM. The resulting numbers
    are not the project's fitted statistics, and the artifact records n_tiles
    so they cannot be mistaken for a full-split fit.
    """

    train = KuroSiwoSarDataset(real_adapter, "train", limit=N_TRAIN_TILES)
    artifact = fit_training_statistics(
        [train.load_canonical(i) for i in range(len(train))]
    )
    assert artifact.kind == "fitted"
    assert artifact.provenance["split"] == "train"
    assert artifact.provenance["n_tiles"] == len(train)
    assert artifact.provenance["n_valid_pixels"] > 0
    assert tuple(artifact.values) == CHANNEL_ORDER

    means = artifact.mean_vector()
    stds = artifact.std_vector()
    assert np.isfinite(means).all() and np.isfinite(stds).all()
    assert (stds > 0).all()
    # Linear sigma0 clipped at 0.15 must land inside that range, not near a
    # decibel value of -10 to -15.
    assert ((means > 0.0) & (means <= 0.15)).all(), f"implausible means {means}"
    print(f"\nbounded real fit (n_tiles={len(train)}): mean={means}, std={stds}")

    validation = KuroSiwoSarDataset(real_adapter, "validation", limit=1)
    if len(validation):
        from floodmap.segmentation import StatisticsError

        with pytest.raises(StatisticsError, match="training split only"):
            fit_training_statistics([validation.load_canonical(0)])


# ---------------------------------------------------------------------------
# SMOKE: does the whole chain execute on real tiles?
# ---------------------------------------------------------------------------


def test_smoke_full_chain_on_real_tiles(real_adapter, tiny_dev_config, tmp_path):
    """adapter -> sample -> tensor -> model -> loss -> backward -> step ->
    validation -> checkpoint, on a handful of real tiles, for one epoch."""

    seed_everything(tiny_dev_config.training.seed)
    statistics = development_statistics()

    train = KuroSiwoSarDataset(
        real_adapter, "train", statistics=statistics, limit=N_TRAIN_TILES
    )
    assert len(train) == N_TRAIN_TILES

    loader = torch.utils.data.DataLoader(
        train, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_samples
    )
    batch = next(iter(loader))
    assert batch["image"].shape[1] == 4
    assert batch["image"].shape[0] == 2

    model = UNet(in_channels=4, num_classes=NUM_CLASSES, base_channels=8, depth=2, seed=0)
    criterion = build_loss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["image"])
    assert logits.shape == (2, NUM_CLASSES, *batch["image"].shape[-2:])

    loss = criterion(logits, batch["target"])
    assert torch.isfinite(loss), "loss on real tiles is not finite"
    loss.backward()

    grad_total = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert grad_total > 0.0, "backward produced no gradient on real data"

    optimizer.step()
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    ), "optimizer.step() did not change any parameter"

    # Validation pass with metrics, ignoring nodata.
    validation = KuroSiwoSarDataset(
        real_adapter, "validation", statistics=statistics, limit=N_VAL_TILES
    )
    if len(validation) == 0:
        pytest.skip("no validation activation is present in the local delivery")

    val_loader = torch.utils.data.DataLoader(
        validation, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_samples
    )
    metrics = SegmentationMetrics(ignore_index=IGNORE_INDEX)
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            metrics.update(model(val_batch["image"]), val_batch["target"])
    result = metrics.compute()
    assert result.n_valid_pixels > 0
    assert result.n_ignored_pixels >= 0
    assert result.confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    print(
        f"\nsmoke validation (NOT a performance measurement): "
        f"flood_iou={result.iou_flood_class:.4f}, support={result.per_class_support}"
    )

    # Checkpoint round-trip.
    from floodmap.segmentation import save_checkpoint

    path = save_checkpoint(
        tmp_path / "smoke.pt",
        model=model,
        optimizer=optimizer,
        epoch=1,
        val_iou_flood_class=result.iou_flood_class,
        config=tiny_dev_config.to_record(),
        statistics_identity=statistics.identity(),
        dataset_identity=train.identity(),
        seed=tiny_dev_config.training.seed,
        metrics=result.to_dict(),
    )
    payload = load_checkpoint(path)
    assert payload["channel_order"] == list(CHANNEL_ORDER)
    assert payload["statistics_identity"]["kind"] == "development"
    assert payload["dataset_identity"]["dataset_id"] == "kuro_siwo"

    restored = UNet(in_channels=4, num_classes=NUM_CLASSES, base_channels=8, depth=2, seed=0)
    restored.load_state_dict(payload["model_state"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch["image"]), model(batch["image"]))


def test_smoke_one_epoch_through_the_training_entrypoint(
    real_adapter, tiny_dev_config, tmp_path
):
    """The same chain, but driven by ``run_training`` rather than by hand.

    One epoch over 4 real tiles, so the real entrypoint, checkpoint selection
    and run record are all exercised rather than only the pieces.
    """

    result = run_training(
        tiny_dev_config, adapter=real_adapter, run_dir=tmp_path / "smoke_run"
    )

    assert len(result.history) == 1
    assert np.isfinite(result.history[0]["train_loss"])
    assert result.history[0]["train_steps"] >= 1
    assert result.last_checkpoint is not None and result.last_checkpoint.is_file()
    assert result.run_record_path is not None

    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert record["status"] == "development_only"
    assert record["statistics_identity"]["usable_for_reported_results"] is False
    assert record["train_dataset"]["split"] == "train"
    assert record["train_dataset"]["n_samples"] == N_TRAIN_TILES
    assert record["validation_dataset"]["split"] == "validation"
    assert record["loss"]["weighting"] == "unweighted"
    assert record["commit"] is None or isinstance(record["commit"], str)

    if record["validation_dataset"]["n_samples"] > 0:
        assert result.best_checkpoint is not None
        payload = load_checkpoint(result.best_checkpoint)
        assert payload["val_iou_flood_class"] is not None
        print(
            f"\nsmoke run best val flood IoU (pipeline evidence only): "
            f"{payload['val_iou_flood_class']:.4f}"
        )


# ---------------------------------------------------------------------------
# OVERFIT: can the pipeline learn at all?
# ---------------------------------------------------------------------------


def test_overfit_a_tiny_fixed_subset(real_adapter, tmp_path):
    """Drive a fixed 4-tile subset until the loss drops substantially.

    This is a PIPELINE CORRECTNESS test. It asserts the pipeline is capable of
    learning, which is a different and much weaker claim than the model being
    any good. The subset is fixed and sorted, so the test is reproducible.
    """

    seed_everything(4242)
    statistics = development_statistics()

    dataset = KuroSiwoSarDataset(
        real_adapter, "train", statistics=statistics, limit=N_TRAIN_TILES
    )
    assert len(dataset) == N_TRAIN_TILES
    batch = collate_samples([dataset[i] for i in range(len(dataset))])

    # The target must actually contain learnable classes, or a flat loss curve
    # would prove nothing about the pipeline.
    valid_targets = batch["target"][batch["valid_mask"]]
    present = set(int(v) for v in torch.unique(valid_targets).tolist())
    assert present, "the fixed subset has no valid labelled pixels"
    print(f"\noverfit subset classes present: {sorted(present)}")

    model = UNet(in_channels=4, num_classes=NUM_CLASSES, base_channels=8, depth=2, seed=1)
    criterion = build_loss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_parameters = [p.detach().clone() for p in model.parameters()]

    model.eval()
    with torch.no_grad():
        initial_predictions = model(batch["image"]).argmax(dim=1).clone()

    model.train()
    losses = []
    first_grad_total = None
    for step in range(OVERFIT_ITERATIONS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch["image"]), batch["target"])
        assert torch.isfinite(loss), f"loss became non-finite at step {step}"
        loss.backward()
        if step == 0:
            first_grad_total = sum(
                float(p.grad.abs().sum())
                for p in model.parameters()
                if p.grad is not None
            )
        optimizer.step()
        losses.append(float(loss.detach().item()))

    first_loss = losses[0]
    final_loss = losses[-1]
    print(
        f"overfit loss: {first_loss:.5f} -> {final_loss:.5f} "
        f"over {OVERFIT_ITERATIONS} iterations"
    )

    # 1. gradients are non-zero
    assert first_grad_total is not None and first_grad_total > 0.0

    # 2. loss decreases substantially
    assert final_loss < first_loss, (
        f"loss did not decrease at all ({first_loss:.5f} -> {final_loss:.5f}); the "
        "pipeline cannot learn from its own data"
    )
    assert final_loss < 0.5 * first_loss, (
        f"loss only fell from {first_loss:.5f} to {final_loss:.5f}. A pipeline that "
        "executes but barely learns usually means a detached graph, a mask on the "
        "wrong axis, or constant inputs"
    )

    # 3. model parameters change
    assert any(
        not torch.equal(old, new)
        for old, new in zip(initial_parameters, model.parameters())
    )

    # 4. predictions change
    model.eval()
    with torch.no_grad():
        final_predictions = model(batch["image"]).argmax(dim=1)
    assert not torch.equal(initial_predictions, final_predictions), (
        "predictions are identical before and after training"
    )

    # 5. training performance improves on the subset it was fitted to
    def _flood_iou(predictions: torch.Tensor) -> float:
        metrics = SegmentationMetrics(ignore_index=IGNORE_INDEX)
        metrics.update(predictions, batch["target"], is_prediction=True)
        return metrics.compute().iou_flood_class

    before_iou = _flood_iou(initial_predictions)
    after_iou = _flood_iou(final_predictions)
    print(f"overfit train flood IoU: {before_iou} -> {after_iou}")
    # Only assert when the subset actually contains flood pixels; otherwise
    # flood IoU is nan by design and carries no information.
    if 2 in present:
        assert np.isfinite(after_iou)
        assert after_iou >= before_iou or np.isnan(before_iou), (
            f"flood IoU did not improve on the fitted subset "
            f"({before_iou} -> {after_iou})"
        )

    # 6. the checkpoint can be loaded and reproduces the trained predictions
    from floodmap.segmentation import save_checkpoint

    path = save_checkpoint(
        tmp_path / "overfit.pt",
        model=model,
        optimizer=optimizer,
        epoch=OVERFIT_ITERATIONS,
        val_iou_flood_class=None,
        config={"note": "development-only overfit test; not a real experiment"},
        statistics_identity=statistics.identity(),
        dataset_identity=dataset.identity(),
        seed=4242,
    )
    payload = load_checkpoint(path)
    restored = UNet(in_channels=4, num_classes=NUM_CLASSES, base_channels=8, depth=2, seed=1)
    restored.load_state_dict(payload["model_state"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch["image"]), model(batch["image"]))


def test_overfit_does_not_learn_from_ignored_pixels(real_adapter):
    """Corrupting ONLY ignored pixels must not change the loss on real data.

    The strongest available evidence that nodata is genuinely excluded: it uses
    the real validity mask, which on Kuro covers a non-trivial fraction of
    every tile.
    """

    dataset = KuroSiwoSarDataset(real_adapter, "train", limit=2)
    batch = collate_samples([dataset[i] for i in range(len(dataset))])
    invalid = ~batch["valid_mask"]
    if not bool(invalid.any()):
        pytest.skip("the selected real tiles contain no invalid pixels")

    criterion = build_loss(ignore_index=IGNORE_INDEX)
    shape = (batch["image"].shape[0], NUM_CLASSES, *batch["image"].shape[-2:])
    logits = torch.zeros(shape)
    baseline = criterion(logits, batch["target"])

    perturbed = logits.clone()
    for channel in range(NUM_CLASSES):
        perturbed[:, channel][invalid] = 100.0 if channel == 0 else -100.0
    assert torch.allclose(criterion(perturbed, batch["target"]), baseline), (
        "changing predictions on ignored pixels changed the loss"
    )

    graded = torch.zeros(shape, requires_grad=True)
    criterion(graded, batch["target"]).backward()
    ignored_gradient = graded.grad.permute(1, 0, 2, 3)[:, invalid].abs().sum()
    assert float(ignored_gradient) == 0.0, (
        f"ignored pixels received gradient {float(ignored_gradient)}"
    )
