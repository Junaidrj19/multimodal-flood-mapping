"""Focused tests for deterministic bounded training-sample selection (M5).

Synthetic fixtures throughout, in the same style as ``test_segmentation_m5.py``:
a tiny Kuro-shaped corpus with three TRAIN activations of deliberately unequal
size, so proportional allocation has something to be proportional about.

What these tests are defending, in order of how badly each would hurt:

* a bounded run that trains on a different subset than its manifest names;
* a bounded run whose subset comes from one activation because sample IDs sort
  by activation prefix;
* a quarantined sample reaching a tensor;
* an edited or stale manifest being used without complaint;
* an existing config changing meaning because a new field was added.

The test split is not referenced by any test here, and
:class:`TestTestSplitIsUntouched` asserts the selector refuses it outright.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from floodmap.datasets import KuroSiwoAdapter
from floodmap.datasets.quarantine import (
    QuarantineEntry,
    QuarantineError,
    QuarantinePolicy,
)
from floodmap.segmentation import KuroSiwoSarDataset
from floodmap.segmentation.config import ExperimentConfig, load_experiment
from floodmap.segmentation.selection import (
    SampleSelectionError,
    allocate_largest_remainder,
    read_manifest,
    select_training_samples,
    write_manifest,
)
from floodmap.segmentation.train import resolve_train_sample_manifest
from tests.conftest import REPO_ROOT

#: Three TRAIN activations with unequal counts, plus one VALIDATION activation.
#: 1111004/1111005/1111006 are train and 1111003 is validation in the upstream
#: split baked into the adapter, which is the same shape as the real delivery.
TRAIN_ACTIVATIONS = {1111004: 10, 1111005: 6, 1111006: 4}
VALIDATION_ACTIVATION = 1111003
N_VALIDATION = 5

TILE = 16

PROBE_CONFIG = "m5_kuro_sar_unet_mps_probe2"
BOUNDED_CONFIG = "m5_kuro_sar_unet_mps_bounded"


# ---------------------------------------------------------------------------
# Fixture: a Kuro-shaped corpus with unequal training activations
# ---------------------------------------------------------------------------


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
    grid_id = f"{activation}-grid-{index:04d}"
    directory = root / str(activation) / "01" / grid_id
    directory.mkdir(parents=True, exist_ok=True)
    transform = from_origin(4801325, 1292245, 10, 10)
    rng = np.random.default_rng(seed)

    valid = np.ones((TILE, TILE), dtype=np.uint8)
    valid[:3, :3] = 0

    labels = np.zeros((TILE, TILE), dtype=np.uint8)
    labels[8:, :6] = 1
    labels[4:10, 8:14] = 2
    labels[:3, :3] = 3

    for role, date in (("SL1", "20191115"), ("SL2", "20191109"), ("MS1", "20191127")):
        for pol in ("IVV", "IVH"):
            plane = rng.uniform(0.01, 0.2, size=(TILE, TILE)).astype(np.float32)
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
    root = tmp_path_factory.mktemp("kuro_m5_selection")
    for activation, count in TRAIN_ACTIVATIONS.items():
        grids = [
            _write_sample(root, activation, i, seed=activation + i) for i in range(count)
        ]
        _write_catalogue(root, activation, grids)
    val_grids = [
        _write_sample(root, VALIDATION_ACTIVATION, i, seed=900 + i)
        for i in range(N_VALIDATION)
    ]
    _write_catalogue(root, VALIDATION_ACTIVATION, val_grids)
    return root


@pytest.fixture(scope="module")
def adapter(kuro_root) -> KuroSiwoAdapter:
    return KuroSiwoAdapter(kuro_root)


@pytest.fixture(scope="module")
def train_ids(adapter):
    return sorted(ref.sample_id for ref in adapter.discover() if ref.split == "train")


@pytest.fixture
def one_quarantined(adapter, train_ids) -> QuarantinePolicy:
    """A policy excluding exactly one real training sample."""

    victim = train_ids[0]
    return QuarantinePolicy(
        dataset_id=adapter.dataset_id,
        version="9.9.9",
        entries={
            victim: QuarantineEntry(
                sample_id=victim,
                reason_code="nan_on_declared_valid_consumed_sar",
                reason="synthetic fixture exclusion",
            )
        },
        reason_codes={
            "nan_on_declared_valid_consumed_sar": "synthetic fixture exclusion"
        },
        path=Path("/synthetic/quarantine.yaml"),
    )


# ---------------------------------------------------------------------------
# Fixture self-check
# ---------------------------------------------------------------------------


class TestFixtureIsRepresentative:
    def test_three_unequal_training_activations(self, adapter):
        counts = {}
        for ref in adapter.discover():
            if ref.split == "train":
                counts[ref.activation_id] = counts.get(ref.activation_id, 0) + 1
        assert counts == TRAIN_ACTIVATIONS
        # Unequal, so a proportional allocation is distinguishable from an
        # equal split.
        assert len(set(counts.values())) == len(counts)

    def test_sample_ids_sort_by_activation_prefix(self, train_ids):
        """The premise of the whole module: a sorted prefix is activation-biased."""

        prefix = train_ids[: sum(TRAIN_ACTIVATIONS.values()) // 2]
        activations = {sid.split(":")[0] for sid in prefix}
        assert activations == {"1111004"}, (
            "if a sorted prefix were already activation-balanced, limit_train "
            "would be sufficient and this module would not be needed"
        )


# ---------------------------------------------------------------------------
# Proportional allocation
# ---------------------------------------------------------------------------


class TestProportionalAllocation:
    def test_allocation_is_proportional_and_sums_to_the_budget(self):
        counts = {"a": 5000, "b": 1369, "c": 682}
        allocation = allocate_largest_remainder(counts, 256)
        assert sum(allocation.values()) == 256
        # Matches the real corpus allocation: 181/50/25.
        assert allocation == {"a": 181, "b": 50, "c": 25}

    def test_each_group_is_within_one_of_its_exact_quota(self):
        counts = {"a": 5000, "b": 1369, "c": 682}
        total = sum(counts.values())
        allocation = allocate_largest_remainder(counts, 256)
        for group, n in counts.items():
            exact = 256 * n / total
            assert abs(allocation[group] - exact) < 1.0

    def test_larger_activation_never_gets_less_than_a_smaller_one(self):
        counts = {"a": 5000, "b": 1369, "c": 682}
        allocation = allocate_largest_remainder(counts, 256)
        assert allocation["a"] > allocation["b"] > allocation["c"]

    def test_result_does_not_depend_on_mapping_order(self):
        forward = allocate_largest_remainder({"z": 10, "a": 20, "m": 30}, 17)
        reversed_ = allocate_largest_remainder({"m": 30, "z": 10, "a": 20}, 17)
        assert forward == reversed_

    def test_every_group_is_represented_even_when_flooring_would_zero_it(self):
        """A tiny activation must not silently vanish from a bounded run."""

        allocation = allocate_largest_remainder({"tiny": 1, "huge": 1000}, 10)
        assert allocation["tiny"] >= 1
        assert sum(allocation.values()) == 10

    def test_a_budget_larger_than_the_corpus_is_refused(self):
        with pytest.raises(SampleSelectionError, match="exceeds"):
            allocate_largest_remainder({"a": 5, "b": 5}, 11)

    def test_a_budget_too_small_to_represent_every_group_is_refused(self):
        with pytest.raises(SampleSelectionError, match="at least"):
            allocate_largest_remainder({"a": 5, "b": 5, "c": 5}, 2)

    def test_an_empty_group_set_is_refused(self):
        with pytest.raises(SampleSelectionError):
            allocate_largest_remainder({}, 4)

    def test_allocation_never_exceeds_availability(self):
        allocation = allocate_largest_remainder({"a": 2, "b": 100}, 50)
        assert allocation["a"] <= 2
        assert sum(allocation.values()) == 50


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------


class TestDeterministicSelection:
    def test_the_same_seed_selects_the_same_samples(self, adapter):
        first = select_training_samples(adapter, budget=12, seed=20261008)
        second = select_training_samples(adapter, budget=12, seed=20261008)
        assert first.sample_ids == second.sample_ids
        assert (
            first.provenance["sample_id_sha256"]
            == second.provenance["sample_id_sha256"]
        )

    def test_a_different_seed_selects_a_different_subset(self, adapter):
        a = select_training_samples(adapter, budget=12, seed=20261008)
        b = select_training_samples(adapter, budget=12, seed=1)
        assert a.sample_ids != b.sample_ids

    def test_selection_is_unaffected_by_adapter_discovery_order(
        self, adapter, monkeypatch
    ):
        """Pool order must be normalised before the seeded draw."""

        baseline = select_training_samples(adapter, budget=12, seed=20261008)

        original = adapter.discover
        monkeypatch.setattr(
            adapter, "discover", lambda **kw: tuple(reversed(original(**kw)))
        )
        shuffled = select_training_samples(adapter, budget=12, seed=20261008)
        assert shuffled.sample_ids == baseline.sample_ids

    def test_exact_budget_and_unique_ids(self, adapter):
        selection = select_training_samples(adapter, budget=15, seed=20261008)
        assert len(selection.sample_ids) == 15
        assert len(set(selection.sample_ids)) == 15

    def test_every_training_activation_is_represented(self, adapter):
        selection = select_training_samples(adapter, budget=12, seed=20261008)
        chosen = {sid.split(":")[0] for sid in selection.sample_ids}
        assert chosen == {str(a) for a in TRAIN_ACTIVATIONS}

    def test_selection_is_not_a_sorted_prefix(self, adapter, train_ids):
        """The specific failure mode ``limit_train`` has."""

        selection = select_training_samples(adapter, budget=12, seed=20261008)
        assert set(selection.sample_ids) != set(train_ids[:12])

    def test_allocation_recorded_matches_the_samples_returned(self, adapter):
        selection = select_training_samples(adapter, budget=12, seed=20261008)
        actual = {}
        for sid in selection.sample_ids:
            key = sid.split(":")[0]
            actual[key] = actual.get(key, 0) + 1
        recorded = {
            alloc.activation_id: alloc.allocated for alloc in selection.allocations
        }
        assert actual == recorded

    def test_only_training_samples_are_selected(self, adapter):
        selection = select_training_samples(adapter, budget=12, seed=20261008)
        split_of = {ref.sample_id: ref.split for ref in adapter.discover()}
        assert {split_of[sid] for sid in selection.sample_ids} == {"train"}

    def test_a_budget_above_the_eligible_count_is_refused(self, adapter):
        with pytest.raises(SampleSelectionError, match="exceeds"):
            select_training_samples(
                adapter, budget=sum(TRAIN_ACTIVATIONS.values()) + 1, seed=1
            )

    def test_provenance_records_what_is_needed_to_reproduce(self, adapter):
        selection = select_training_samples(adapter, budget=12, seed=20261008)
        provenance = selection.provenance
        assert provenance["seed"] == 20261008
        assert provenance["budget"] == 12
        assert provenance["split"] == "train"
        assert provenance["dataset_id"] == "kuro_siwo"
        assert provenance["allocation_method"] == "largest_remainder"
        assert provenance["selector_version"]
        assert provenance["n_eligible"] == sum(TRAIN_ACTIVATIONS.values())
        assert provenance["sample_id_sha256"]


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


class TestQuarantineExclusion:
    def test_a_quarantined_sample_is_never_selected(
        self, adapter, train_ids, one_quarantined
    ):
        victim = train_ids[0]
        # Budget deliberately equals the whole eligible pool, so the excluded
        # sample would HAVE to appear if it were still a candidate.
        budget = sum(TRAIN_ACTIVATIONS.values()) - 1
        selection = select_training_samples(
            adapter, budget=budget, seed=20261008, quarantine=one_quarantined
        )
        assert victim not in selection.sample_ids
        assert len(selection.sample_ids) == budget

    def test_the_eligible_pool_excludes_the_quarantined_sample(
        self, adapter, one_quarantined
    ):
        selection = select_training_samples(
            adapter, budget=5, seed=20261008, quarantine=one_quarantined
        )
        assert selection.provenance["n_excluded_by_quarantine"] == 1
        assert (
            selection.provenance["n_eligible"]
            == sum(TRAIN_ACTIVATIONS.values()) - 1
        )

    def test_quarantine_shrinks_the_available_budget(self, adapter, one_quarantined):
        """Asking for the full split while one sample is excluded must refuse."""

        with pytest.raises(SampleSelectionError, match="exceeds"):
            select_training_samples(
                adapter,
                budget=sum(TRAIN_ACTIVATIONS.values()),
                seed=1,
                quarantine=one_quarantined,
            )

    def test_the_policy_is_recorded_in_provenance(self, adapter, one_quarantined):
        selection = select_training_samples(
            adapter, budget=5, seed=20261008, quarantine=one_quarantined
        )
        record = selection.provenance["quarantine"]
        assert record["policy_applied"] is True
        assert record["policy_version"] == "9.9.9"
        assert record["n_excluded"] == 1

    def test_no_policy_is_recorded_as_no_policy(self, adapter):
        """Absent must not be indistinguishable from 'nothing to exclude'."""

        selection = select_training_samples(adapter, budget=5, seed=20261008)
        record = selection.provenance["quarantine"]
        assert record["policy_applied"] is False
        assert selection.provenance["n_excluded_by_quarantine"] == 0

    def test_the_real_policy_excludes_the_documented_sample(self):
        """Guard on the committed policy the probe config actually loads."""

        parsed = yaml.safe_load(
            (REPO_ROOT / "configs" / "quarantine" / "kuro_siwo.yaml").read_text(
                encoding="utf-8"
            )
        )
        ids = [entry["sample_id"] for entry in parsed["samples"]]
        assert "1111004:df3d7e17-0a15-5d76-aad9-4571285aae39" in ids


# ---------------------------------------------------------------------------
# Manifest round trip and validation
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_path(adapter, tmp_path) -> Path:
    selection = select_training_samples(adapter, budget=12, seed=20261008)
    return write_manifest(tmp_path / "selection.json", selection)


class TestManifestRoundTrip:
    def test_a_written_manifest_reads_back_identically(self, adapter, manifest_path):
        expected = select_training_samples(adapter, budget=12, seed=20261008)
        loaded = read_manifest(manifest_path)
        assert loaded.sample_ids == expected.sample_ids

    def test_reading_checks_dataset_and_split(self, manifest_path):
        read_manifest(manifest_path, dataset_id="kuro_siwo", split="train")
        with pytest.raises(SampleSelectionError, match="another corpus"):
            read_manifest(manifest_path, dataset_id="sen1floods11")
        with pytest.raises(SampleSelectionError, match="split"):
            read_manifest(manifest_path, split="validation")

    def test_an_existing_manifest_is_not_silently_overwritten(
        self, adapter, manifest_path
    ):
        other = select_training_samples(adapter, budget=11, seed=7)
        with pytest.raises(SampleSelectionError, match="already exists"):
            write_manifest(manifest_path, other)
        # The original selection is intact.
        assert len(read_manifest(manifest_path).sample_ids) == 12

    def test_identity_summarises_without_inlining_every_id(self, manifest_path):
        identity = read_manifest(manifest_path).identity()
        assert identity["applied"] is True
        assert identity["n_samples"] == 12
        assert identity["seed"] == 20261008
        assert identity["sample_id_sha256"]
        assert "sample_ids" not in identity


class TestInvalidManifestIsRejected:
    def _rewrite(self, path: Path, mutate) -> Path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(SampleSelectionError, match="not found"):
            read_manifest(tmp_path / "absent.json")

    def test_malformed_json_is_refused(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SampleSelectionError, match="not valid JSON"):
            read_manifest(path)

    def test_an_empty_selection_is_refused(self, manifest_path):
        self._rewrite(manifest_path, lambda p: p.__setitem__("sample_ids", []))
        with pytest.raises(SampleSelectionError, match="non-empty"):
            read_manifest(manifest_path)

    def test_duplicate_ids_are_refused(self, manifest_path):
        def duplicate(payload):
            payload["sample_ids"][1] = payload["sample_ids"][0]

        self._rewrite(manifest_path, duplicate)
        with pytest.raises(SampleSelectionError, match="duplicate"):
            read_manifest(manifest_path)

    def test_a_non_string_id_is_refused(self, manifest_path):
        self._rewrite(manifest_path, lambda p: p["sample_ids"].__setitem__(0, 17))
        with pytest.raises(SampleSelectionError, match="non-empty string"):
            read_manifest(manifest_path)

    def test_an_edited_id_list_fails_the_digest(self, manifest_path):
        """The specific defence against a hand-edited manifest."""

        def swap(payload):
            payload["sample_ids"][0] = "1111004:not-a-real-grid"

        self._rewrite(manifest_path, swap)
        with pytest.raises(SampleSelectionError, match="sample_id_sha256"):
            read_manifest(manifest_path)

    def test_a_dropped_id_fails_the_budget_check(self, manifest_path):
        def drop(payload):
            payload["sample_ids"].pop()
            # Keep the digest honest so the BUDGET check is what fires.
            import hashlib

            joined = "\n".join(sorted(payload["sample_ids"]))
            payload["provenance"]["sample_id_sha256"] = hashlib.sha256(
                joined.encode("utf-8")
            ).hexdigest()

        self._rewrite(manifest_path, drop)
        with pytest.raises(SampleSelectionError, match="budget"):
            read_manifest(manifest_path)

    def test_absent_provenance_is_refused(self, manifest_path):
        self._rewrite(manifest_path, lambda p: p.pop("provenance"))
        with pytest.raises(SampleSelectionError, match="provenance must be a mapping"):
            read_manifest(manifest_path)

    @pytest.mark.parametrize(
        "key",
        [
            "selector_version",
            "allocation_method",
            "seed",
            "budget",
            "dataset_id",
            "split",
            "sample_id_sha256",
            "activations",
        ],
    )
    def test_each_required_provenance_key_is_required(self, manifest_path, key):
        self._rewrite(manifest_path, lambda p: p["provenance"].pop(key))
        with pytest.raises(SampleSelectionError, match="missing required key"):
            read_manifest(manifest_path)

    def test_a_manifest_naming_the_test_split_is_refused(self, manifest_path):
        self._rewrite(manifest_path, lambda p: p["provenance"].__setitem__("split", "test"))
        with pytest.raises(SampleSelectionError, match="test split"):
            read_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Dataset construction from a manifest
# ---------------------------------------------------------------------------


class TestDatasetConstructionFromManifest:
    def test_the_dataset_contains_exactly_the_manifest_samples(
        self, adapter, manifest_path
    ):
        manifest = read_manifest(manifest_path)
        dataset = KuroSiwoSarDataset(
            adapter, "train", sample_ids=list(manifest.sample_ids)
        )
        assert len(dataset) == len(manifest.sample_ids)
        assert set(dataset.sample_ids) == set(manifest.sample_ids)

    def test_dataset_order_is_deterministic_regardless_of_manifest_order(
        self, adapter, manifest_path
    ):
        manifest = read_manifest(manifest_path)
        forward = KuroSiwoSarDataset(
            adapter, "train", sample_ids=list(manifest.sample_ids)
        )
        backward = KuroSiwoSarDataset(
            adapter, "train", sample_ids=list(reversed(manifest.sample_ids))
        )
        assert forward.sample_ids == backward.sample_ids

    def test_every_training_activation_reaches_the_dataset(
        self, adapter, manifest_path
    ):
        manifest = read_manifest(manifest_path)
        dataset = KuroSiwoSarDataset(
            adapter, "train", sample_ids=list(manifest.sample_ids)
        )
        assert set(dataset.activations) == {str(a) for a in TRAIN_ACTIVATIONS}

    def test_an_unknown_manifest_id_is_rejected_by_the_dataset(self, adapter):
        with pytest.raises(KeyError):
            KuroSiwoSarDataset(
                adapter, "train", sample_ids=["1111004:not-a-real-grid"]
            )

    def test_a_quarantined_manifest_id_raises_rather_than_shrinking(
        self, adapter, train_ids, one_quarantined
    ):
        """A named sample must never be silently dropped."""

        with pytest.raises(QuarantineError):
            KuroSiwoSarDataset(
                adapter,
                "train",
                sample_ids=[train_ids[0], train_ids[1]],
                quarantine=one_quarantined,
            )

    def test_construction_reads_no_pixels(self, adapter, manifest_path, monkeypatch):
        manifest = read_manifest(manifest_path)

        def explode(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("dataset construction must not load pixels")

        monkeypatch.setattr(adapter, "load_sample", explode)
        KuroSiwoSarDataset(adapter, "train", sample_ids=list(manifest.sample_ids))


class TestValidationSamplesAreRetained:
    def test_all_eligible_validation_samples_are_used_when_uncapped(self, adapter):
        dataset = KuroSiwoSarDataset(adapter, "validation", limit=None)
        assert len(dataset) == N_VALIDATION

    def test_a_manifest_does_not_touch_the_validation_split(
        self, adapter, manifest_path
    ):
        """The training allowlist must not leak into validation."""

        manifest = read_manifest(manifest_path)
        train = KuroSiwoSarDataset(
            adapter, "train", sample_ids=list(manifest.sample_ids)
        )
        validation = KuroSiwoSarDataset(adapter, "validation", limit=None)
        assert len(validation) == N_VALIDATION
        assert not set(validation.sample_ids) & set(train.sample_ids)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _probe_payload() -> dict:
    return load_experiment(PROBE_CONFIG).model_dump()


class TestProbeConfiguration:
    def test_the_probe_config_loads(self):
        config = load_experiment(PROBE_CONFIG)
        assert config.name == PROBE_CONFIG

    def test_it_names_the_manifest_and_leaves_limit_train_unset(self):
        config = load_experiment(PROBE_CONFIG)
        assert (
            config.dataset.train_sample_manifest
            == "artifacts/selections/m5_mps_probe2.json"
        )
        assert config.dataset.limit_train is None

    def test_validation_is_uncapped(self):
        assert load_experiment(PROBE_CONFIG).dataset.limit_validation is None

    def test_bounded_run_settings(self):
        config = load_experiment(PROBE_CONFIG)
        assert config.training.device == "mps"
        assert config.training.epochs == 3
        assert config.dataloader.batch_size == 1
        assert config.dataloader.num_workers == 0
        assert config.dataloader.persistent_workers is False
        assert config.training.seed == 20261008

    def test_fitted_normalisation_is_required_and_runnable(self):
        config = load_experiment(PROBE_CONFIG)
        assert config.normalisation.enabled is True
        assert config.normalisation.required_kind == "fitted"
        assert config.normalisation.allow_development_statistics is False
        assert config.normalisation.blocked_reason() is None

    def test_the_frozen_contract_is_unchanged_from_the_bounded_config(self):
        """Only the training subset and length may differ."""

        probe = load_experiment(PROBE_CONFIG).model_dump()
        bounded = load_experiment(BOUNDED_CONFIG).model_dump()
        for section in ("reprojection", "quarantine", "model", "loss", "optimizer",
                        "augmentation"):
            assert probe[section] == bounded[section], section
        assert probe["dataset"]["channels"] == bounded["dataset"]["channels"]
        assert probe["dataset"]["representation"] == bounded["dataset"]["representation"]
        assert probe["dataset"]["ignore_index"] == bounded["dataset"]["ignore_index"]
        assert probe["primary_metric"] == bounded["primary_metric"]
        assert probe["normalisation"]["statistics_path"] == (
            bounded["normalisation"]["statistics_path"]
        )

    def test_canonical_reprojection_is_preserved(self):
        reprojection = load_experiment(PROBE_CONFIG).reprojection
        assert reprojection.enabled is True
        assert reprojection.resolution_m == 10.0
        assert reprojection.tile_size == 224
        assert reprojection.sar_resampling == "bilinear"
        assert reprojection.label_resampling == "nearest"
        assert reprojection.nodata_mode == "strict"

    def test_the_manifest_it_names_exists_and_validates(self):
        config = load_experiment(PROBE_CONFIG)
        path = REPO_ROOT / config.dataset.train_sample_manifest
        if not path.is_file():
            pytest.skip(f"{path} has not been generated in this environment")
        manifest = read_manifest(
            path, dataset_id=config.dataset.id, split=config.dataset.train_split
        )
        assert len(manifest.sample_ids) == 256
        assert manifest.provenance["seed"] == 20261008
        allocated = {
            entry["activation_id"]: entry["allocated"]
            for entry in manifest.provenance["activations"]
        }
        assert sum(allocated.values()) == 256
        assert set(allocated) == {"1111004", "1111005", "1111006"}


class TestManifestAndLimitTrainAreMutuallyExclusive:
    def test_both_set_is_rejected(self):
        payload = _probe_payload()
        payload["dataset"]["limit_train"] = 32
        with pytest.raises(ValueError, match="mutually exclusive"):
            ExperimentConfig.model_validate(payload)

    def test_manifest_only_is_accepted(self):
        payload = _probe_payload()
        payload["dataset"]["limit_train"] = None
        assert ExperimentConfig.model_validate(payload).dataset.limit_train is None

    def test_limit_train_only_is_still_accepted(self):
        payload = _probe_payload()
        payload["dataset"]["train_sample_manifest"] = None
        payload["dataset"]["limit_train"] = 32
        config = ExperimentConfig.model_validate(payload)
        assert config.dataset.limit_train == 32
        assert config.dataset.train_sample_manifest is None


class TestBackwardCompatibility:
    @pytest.mark.parametrize(
        "name",
        [
            "m5_kuro_sar_unet",
            "m5_kuro_sar_unet_dev",
            BOUNDED_CONFIG,
        ],
    )
    def test_existing_configs_still_load(self, name):
        load_experiment(name)

    @pytest.mark.parametrize(
        "name",
        [
            "m5_kuro_sar_unet",
            "m5_kuro_sar_unet_dev",
            BOUNDED_CONFIG,
        ],
    )
    def test_existing_configs_default_to_no_manifest(self, name):
        """The new field must not change what an existing config means."""

        assert load_experiment(name).dataset.train_sample_manifest is None

    def test_a_config_without_the_new_key_is_valid(self):
        payload = _probe_payload()
        payload["dataset"].pop("train_sample_manifest")
        config = ExperimentConfig.model_validate(payload)
        assert config.dataset.train_sample_manifest is None

    def test_the_baseline_config_file_was_not_modified(self):
        """The frozen baseline must not have acquired a manifest."""

        parsed = yaml.safe_load(
            (
                REPO_ROOT / "configs" / "experiments" / "m5_kuro_sar_unet.yaml"
            ).read_text(encoding="utf-8")
        )
        assert "train_sample_manifest" not in parsed["dataset"]

    def test_the_frozen_segmentation_config_is_untouched(self):
        parsed = yaml.safe_load(
            (REPO_ROOT / "configs" / "segmentation.yaml").read_text(encoding="utf-8")
        )
        assert parsed["model"]["architecture"] is None
        assert "train_sample_manifest" not in yaml.safe_dump(parsed)

    def test_a_mistyped_manifest_key_is_not_silently_ignored(self):
        payload = _probe_payload()
        payload["dataset"]["train_sample_manifests"] = "typo.json"
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(payload)


class TestTrainResolvesTheManifest:
    def test_no_manifest_configured_resolves_to_none(self):
        payload = _probe_payload()
        payload["dataset"]["train_sample_manifest"] = None
        assert resolve_train_sample_manifest(
            ExperimentConfig.model_validate(payload)
        ) is None

    def test_a_configured_manifest_is_loaded(self, manifest_path):
        payload = _probe_payload()
        payload["dataset"]["train_sample_manifest"] = str(manifest_path)
        manifest = resolve_train_sample_manifest(
            ExperimentConfig.model_validate(payload)
        )
        assert manifest is not None
        assert len(manifest.sample_ids) == 12

    def test_a_missing_manifest_refuses_rather_than_training_on_everything(self):
        payload = _probe_payload()
        payload["dataset"]["train_sample_manifest"] = "artifacts/selections/absent.json"
        with pytest.raises(SampleSelectionError, match="not found"):
            resolve_train_sample_manifest(ExperimentConfig.model_validate(payload))


# ---------------------------------------------------------------------------
# Test-split discipline
# ---------------------------------------------------------------------------


class TestTestSplitIsUntouched:
    def test_selecting_from_the_test_split_is_refused(self, adapter):
        with pytest.raises(SampleSelectionError, match="test split"):
            select_training_samples(adapter, budget=2, seed=1, split="test")

    def test_the_probe_config_names_no_test_split(self):
        config = load_experiment(PROBE_CONFIG)
        assert config.dataset.train_split != "test"
        assert config.dataset.validation_split != "test"
        assert config.checkpoint.selection_split == "validation"
