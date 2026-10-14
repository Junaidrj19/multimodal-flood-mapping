"""Synthetic contract tests for the M4 dataset adapters.

The external corpora remain outside Git.  These fixtures reproduce their file
layouts and metadata semantics using tiny rasters so CI can exercise rejection
paths without requiring either private local delivery.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodmap.datasets import (
    DatasetDiscoveryError,
    DatasetIntegrityError,
    KuroSiwoAdapter,
    Sen1Floods11Adapter,
)
from floodmap.utils.contract import load_dataset_capabilities


def _write_raster(path: Path, array: np.ndarray, *, crs: str, transform, nodata=None, descriptions=None):
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
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = tuple(descriptions)


def _write_kuro_catalogue(root: Path, activation: int, rows: list[tuple[str, int, int, int]], *, extra_rows=()):
    path = root / str(activation) / "catalogue.gkpg"
    connection = sqlite3.connect(path)
    # Rewritable so a test can replace the fixture's catalogue in place.
    connection.execute("DROP TABLE IF EXISTS catalogue")
    connection.execute(
        "CREATE TABLE catalogue (grid_id TEXT, actid INTEGER, aoiid INTEGER, "
        "exported INTEGER, master INTEGER, crank INTEGER, source_date TEXT, s1_ids TEXT)"
    )
    for grid_id, exported, master, crank in rows:
        connection.execute(
            "INSERT INTO catalogue VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (grid_id, activation, 1, exported, master, crank, "2020-01-01", "[]"),
        )
    for grid_id, aoiid, exported, master, crank in extra_rows:
        connection.execute(
            "INSERT INTO catalogue VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (grid_id, activation, aoiid, exported, master, crank, "2020-01-01", "[]"),
        )
    connection.commit()
    connection.close()


@pytest.fixture
def kuro_root(tmp_path: Path) -> Path:
    root = tmp_path / "kuro"
    activation = 1111003
    grid = "11111111-1111-1111-1111-111111111111"
    sample = root / str(activation) / "01" / "tile-a"
    sample.mkdir(parents=True)
    transform = from_origin(4801325, 1292245, 10, 10)
    shape = (4, 4)
    valid = np.ones(shape, dtype=np.uint8)
    valid[0, 0] = 0
    labels = np.zeros(shape, dtype=np.uint8)
    labels[1, 1] = 1
    labels[2, 2] = 2
    labels[0, 0] = 3
    source = np.full(shape, 0.1, dtype=np.float32)
    source[0, 0] = 0.0
    source[1, 1] = 0.4
    for role, date in (("SL1", "2019-11-15"), ("SL2", "2019-11-09"), ("MS1", "2019-11-27")):
        for pol in ("IVV", "IVH"):
            _write_raster(
                sample / f"{role}_{pol}_{activation}_01_{date.replace('-', '')}.tif",
                source,
                crs="EPSG:3857",
                transform=transform,
                nodata=0.0,
            )
    _write_raster(sample / f"MK0_MLU_{activation}_01_20191127.tif", labels, crs="EPSG:3857", transform=transform, nodata=3)
    _write_raster(sample / f"MK0_MNA_{activation}_01_20191127.tif", valid, crs="EPSG:3857", transform=transform, nodata=0)
    _write_raster(sample / f"MK0_DEM_{activation}_01_20191127.tif", source, crs="EPSG:3857", transform=transform)
    _write_raster(sample / f"MK0_SLOPE_{activation}_01_20191127.tif", source, crs="EPSG:3857", transform=transform)
    sources = {
        "SL1": {"source_date": "2019-11-15", "master": False, "crank": 1},
        "SL2": {"source_date": "2019-11-09", "master": False, "crank": 2},
        "MS1": {"source_date": "2019-11-27", "master": True, "crank": 1},
    }
    (sample / "info.json").write_text(
        json.dumps({"grid_id": grid, "actid": activation, "aoiid": 1, "sources": sources}),
        encoding="utf-8",
    )
    # Partition 00 has a plausible metadata file but is deliberately absent
    # from the index and has no label files.
    unlabelled = root / str(activation) / "00" / "unlabelled"
    unlabelled.mkdir(parents=True)
    (unlabelled / "info.json").write_text(
        json.dumps({"grid_id": "ignored", "actid": activation, "aoiid": None}), encoding="utf-8"
    )
    _write_kuro_catalogue(
        root,
        activation,
        [
            (grid, 1, 0, 1),
            (grid, 1, 0, 2),
            (grid, 1, 1, 1),
            ("not-exported", 0, 1, 1),
        ],
    )
    return root


@pytest.fixture
def sen_root(tmp_path: Path) -> Path:
    root = tmp_path / "sen1"
    transform = from_origin(-65.0, -14.0, 0.00009, 0.00009)
    sample_id = "Bolivia_1"
    shape = (4, 4)
    s1 = np.stack([np.full(shape, -10, dtype=np.float32), np.full(shape, -20, dtype=np.float32)])
    s2 = np.full((13, *shape), 10000, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.int16)
    labels[0, 0] = -1
    otsu = labels.copy()
    jrc = np.zeros(shape, dtype=np.uint8)
    files = {
        "S1Hand": (f"{sample_id}_S1Hand.tif", s1, ("VV", "VH"), np.nan),
        "S2Hand": (f"{sample_id}_S2Hand.tif", s2, tuple(f"B{i}" for i in range(1, 14)), 0),
        "LabelHand": (f"{sample_id}_LabelHand.tif", labels, None, None),
        "S1OtsuLabelHand": (f"{sample_id}_S1OtsuLabelHand.tif", otsu, ("s1flood",), -1),
        "JRCWaterHand": (f"{sample_id}_JRCWaterHand.tif", jrc, ("occurrence",), None),
    }
    for kind, (name, array, descriptions, nodata) in files.items():
        _write_raster(
            root / "HandLabeled" / kind / name,
            array,
            crs="EPSG:4326",
            transform=transform,
            nodata=nodata,
            descriptions=descriptions,
        )
    split = root / "splits" / "flood_handlabeled"
    split.mkdir(parents=True)
    for filename in ("flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv", "flood_bolivia_data.csv"):
        (split / filename).write_text(
            f"{sample_id}_S1Hand.tif,{sample_id}_LabelHand.tif\n"
            if filename == "flood_test_data.csv"
            else "",
            encoding="utf-8",
        )
    return root


class TestKuroSiwoAdapter:
    def test_partition_and_exported_filters(self, kuro_root):
        adapter = KuroSiwoAdapter(kuro_root)
        refs = adapter.discover()
        assert len(refs) == 1
        assert refs[0].split == "validation"
        assert refs[0].sample_id.startswith("1111003:")
        assert adapter.index().missing_official_activations["test"]

    def test_sl1_is_used_and_sl2_is_recorded_as_dropped(self, kuro_root):
        sample = KuroSiwoAdapter(kuro_root).load_sample("1111003:11111111-1111-1111-1111-111111111111")
        assert set(sample.sar) == {"pre", "post"}
        assert sample.temporal_roles["SL1"].rank == 1
        assert sample.temporal_roles["SL2"].rank == 2
        assert "SL2" in sample.dropped_sources
        assert sample.source_crs == "EPSG:3857"
        assert sample.source_transform[0:6] == tuple(sample.temporal_roles["SL1"].assets["vv"].transform)
        assert sample.target_grid.resolution_m == 10.0
        assert sample.target_grid.crs.startswith("EPSG:326")
        assert sample.target_grid.resampling.categorical_label == "nearest"
        assert sample.target_grid.resampling.sar_domain == "linear_power"
        assert sample.provenance.source_files

    def test_labels_validity_and_clip_are_explicit(self, kuro_root):
        sample = KuroSiwoAdapter(kuro_root).load_sample("1111003:11111111-1111-1111-1111-111111111111")
        assert set(np.unique(sample.labels)) == {0, 1, 2, 3}
        assert sample.label_scheme.semantic_values == (0, 1, 2)
        assert sample.label_scheme.nodata_value == 3
        assert sample.nodata.class_mask_nodata == 255
        assert sample.nodata.ignore_index == 3
        assert float(sample.sar["post"]["vv"].max()) == pytest.approx(0.15)
        assert sample.preprocessing_state.sar_clip_applied is True
        assert sample.preprocessing_state.additional_speckle_filter_applied is False
        assert sample.capabilities.terrain.dem_available_to_canonical_sample is False

    def test_integrity_rejects_mismatched_mna(self, kuro_root):
        ref = KuroSiwoAdapter(kuro_root).discover()[0]
        with rasterio.open(ref.validity_asset.path, "r+") as dataset:
            values = dataset.read(1)
            values[0, 1] = 0
            dataset.write(values, 1)
        with pytest.raises(DatasetIntegrityError, match="MLU|SAR"):
            KuroSiwoAdapter(kuro_root).load_sample(ref.sample_id)

    def test_unlabelled_partition_rows_with_null_aoiid_are_filtered(self, kuro_root):
        """Regression: the real delivery crashed discovery on `int(None)`.

        The catalogue describes BOTH partitions. Partition 00 (unlabelled) rows
        carry ``aoiid = NULL`` with ``exported = 1``, which is exactly what
        ``configs/data.yaml -> partition_contract`` records. Measured on the
        local delivery, 3 of 5 activations contain such rows (1111002: 2 grids,
        1111005: 1, 1111006: 4), and none of their grid ids overlap a labelled
        grid. They must be filtered, not asserted against: the previous code
        grouped them first and then called ``int(row["aoiid"])``, which raised
        TypeError before any useful message was produced.
        """
        activation = 1111003
        grid = "11111111-1111-1111-1111-111111111111"
        _write_kuro_catalogue(
            kuro_root,
            activation,
            [(grid, 1, 0, 1), (grid, 1, 0, 2), (grid, 1, 1, 1)],
            # One partition-00 triplet: exported, but no AOI id and no labels.
            extra_rows=[
                ("unlabelled-00", None, 1, 0, 1),
                ("unlabelled-00", None, 1, 0, 2),
                ("unlabelled-00", None, 1, 1, 1),
            ],
        )
        refs = KuroSiwoAdapter(kuro_root).discover()
        assert [ref.grid_id for ref in refs] == [grid]

    def test_a_labelled_grid_with_a_non_unit_aoiid_is_still_rejected(self, kuro_root):
        """Filtering partition 00 must not weaken the labelled-partition check."""
        activation = 1111003
        grid = "11111111-1111-1111-1111-111111111111"
        _write_kuro_catalogue(
            kuro_root,
            activation,
            [(grid, 1, 0, 1), (grid, 1, 0, 2)],
            # Third row of the SAME grid claims a different AOI, so the triplet
            # is incomplete under the labelled filter.
            extra_rows=[(grid, 2, 1, 1, 1)],
        )
        with pytest.raises(DatasetDiscoveryError, match="expected 3"):
            KuroSiwoAdapter(kuro_root).discover()


class TestSen1Floods11Adapter:
    def test_five_file_discovery_and_split_identity(self, sen_root):
        refs = Sen1Floods11Adapter(sen_root).discover()
        assert len(refs) == 1
        assert set(refs[0].assets) == {"s1", "s2", "label", "s1_otsu", "jrc_water"}
        assert refs[0].split == "test"

    def test_native_representations_and_binary_labels(self, sen_root):
        sample = Sen1Floods11Adapter(sen_root).load_sample("Bolivia_1")
        assert sample.source_crs == "EPSG:4326"
        assert sample.target_grid.resolution_m == 10.0
        assert sample.target_grid.crs == "EPSG:32720"
        assert sample.sar_representation == "decibel"
        assert sample.sar["post"]["vv"].min() == -10
        assert sample.optical["B1"].dtype == np.int16
        assert sample.optical["B1"].max() == 10000
        assert sample.preprocessing_state.optical_scale_factor == 10000.0
        assert sample.label_scheme.semantic_values == (0, 1)
        assert sample.label_scheme.nodata_value == -1
        assert sample.validity[0, 0] is np.False_ or not bool(sample.validity[0, 0])

    def test_no_pre_event_change_or_permanent_water_capability(self, sen_root):
        sample = Sen1Floods11Adapter(sen_root).load_sample("Bolivia_1")
        assert set(sample.sar) == {"post"}
        assert sample.capabilities.imagery.pre_event is False
        assert sample.capabilities.imagery.change_features is False
        assert sample.capabilities.labels.separates_flood_from_permanent_water is False
        assert sample.capabilities.terrain.dem_bundled is False

    def test_alignment_mismatch_is_rejected(self, sen_root):
        path = sen_root / "HandLabeled" / "S2Hand" / "Bolivia_1_S2Hand.tif"
        with rasterio.open(path, "r+") as dataset:
            dataset.transform = from_origin(-65.0, -14.0, 0.0001, 0.0001)
        with pytest.raises(DatasetIntegrityError, match="unaligned"):
            Sen1Floods11Adapter(sen_root).discover()

    def test_missing_required_component_is_rejected(self, sen_root):
        (sen_root / "HandLabeled" / "JRCWaterHand" / "Bolivia_1_JRCWaterHand.tif").unlink()
        with pytest.raises(DatasetDiscoveryError, match="five-file"):
            Sen1Floods11Adapter(sen_root).discover()


def test_capability_representation_differs_without_fabricating_optional_fields():
    caps = load_dataset_capabilities()
    assert caps["kuro_siwo"].validity_mechanisms == (
        "separate_binary_raster",
        "label_nodata_sentinel",
        "sar_nodata_zero",
    )
    assert caps["sen1floods11"].validity_mechanisms == ("label_nodata_sentinel",)
    assert caps["kuro_siwo"].optical_representation is None
    assert caps["sen1floods11"].optical_representation is not None


@pytest.mark.skipif(
    not __import__("os").environ.get("FLOODMAP_KURO_SIWO_ROOT"),
    reason="external Kuro Siwo root was not configured",
)
def test_optional_external_kuro_smoke():
    adapter = KuroSiwoAdapter()
    assert adapter.discover()


@pytest.mark.skipif(
    not __import__("os").environ.get("FLOODMAP_SEN1FLOODS11_ROOT"),
    reason="external Sen1Floods11 root was not configured",
)
def test_optional_external_sen1_smoke():
    adapter = Sen1Floods11Adapter()
    assert adapter.discover()


# ---------------------------------------------------------------------------
# Finiteness scope: consumed baseline epochs vs the dropped SL2 epoch
# ---------------------------------------------------------------------------


def _corrupt_plane(root: Path, activation: int, role: str, pol: str, value: float):
    """Write one non-finite / negative pixel into a source SAR plane.

    Targets a pixel the fixture marks VALID (MK0_MNA == 1), which is the exact
    condition observed in the real delivery: a value that no declared validity
    mechanism accounts for.
    """
    path = sorted((root / str(activation) / "01").glob(f"*/{role}_{pol}_*.tif"))[0]
    with rasterio.open(path, "r+") as dataset:
        values = dataset.read(1)
        values[2, 2] = value  # (2, 2) is labelled class 2 and MNA-valid
        dataset.write(values, 1)
    return path


class TestConsumedVersusDroppedEpochIntegrity:
    """Every CONSUMED channel is validated; a dropped epoch is reported.

    The M5 baseline consumes SL1 (pre) and MS1 (post) only; ``SL2`` is recorded
    as a dropped source and its pixels never reach a tensor. Refusing a whole
    sample because an epoch it discards is defective is an over-broad check: it
    costs a sample whose four consumed channels are provably clean.

    The narrowing is deliberately asymmetric -- a defect in a consumed channel
    is still fatal, and the dropped-epoch defect is still reported rather than
    dropped, so nothing becomes invisible.
    """

    SAMPLE = "1111003:11111111-1111-1111-1111-111111111111"

    @pytest.mark.parametrize("role", ["SL1", "MS1"])
    @pytest.mark.parametrize("value", [float("nan"), -1.0])
    def test_a_defect_in_a_consumed_epoch_is_still_fatal(self, kuro_root, role, value):
        _corrupt_plane(kuro_root, 1111003, role, "IVV", value)
        with pytest.raises(DatasetIntegrityError, match="finite|non-negative"):
            KuroSiwoAdapter(kuro_root).load_sample(self.SAMPLE)

    @pytest.mark.parametrize("pol", ["IVV", "IVH"])
    def test_every_consumed_polarisation_is_checked(self, kuro_root, pol):
        _corrupt_plane(kuro_root, 1111003, "MS1", pol, float("nan"))
        with pytest.raises(DatasetIntegrityError, match="finite|non-negative"):
            KuroSiwoAdapter(kuro_root).load_sample(self.SAMPLE)

    def test_a_nan_in_the_dropped_epoch_does_not_reject_the_sample(self, kuro_root):
        _corrupt_plane(kuro_root, 1111003, "SL2", "IVV", float("nan"))
        sample = KuroSiwoAdapter(kuro_root).load_sample(self.SAMPLE)

        # The four consumed channels load and are clean.
        assert set(sample.sar) == {"pre", "post"}
        for epoch in ("pre", "post"):
            for pol in ("vv", "vh"):
                plane = sample.sar[epoch][pol]
                assert np.isfinite(plane).all()
                assert (plane >= 0).all()

    def test_the_dropped_epoch_defect_is_recorded_not_hidden(self, kuro_root):
        path = _corrupt_plane(kuro_root, 1111003, "SL2", "IVV", float("nan"))
        sample = KuroSiwoAdapter(kuro_root).load_sample(self.SAMPLE)

        diagnostics = sample.dropped_source_diagnostics
        assert diagnostics, "a dropped-epoch defect must be reported"
        assert "SL2" in diagnostics
        entry = diagnostics["SL2"]["vv"]
        assert entry["non_finite_pixels"] == 1
        assert entry["non_finite_on_declared_valid_pixels"] == 1
        assert entry["path"] == str(path)
        # And it reaches provenance, so an artifact carries the disclosure.
        assert sample.provenance.dropped_source_diagnostics == diagnostics

    def test_a_clean_sample_reports_no_dropped_epoch_defects(self, kuro_root):
        sample = KuroSiwoAdapter(kuro_root).load_sample(self.SAMPLE)
        assert sample.dropped_source_diagnostics == {}

    def test_a_negative_value_in_the_dropped_epoch_is_recorded_not_fatal(self, kuro_root):
        _corrupt_plane(kuro_root, 1111003, "SL2", "IVH", -0.5)
        sample = KuroSiwoAdapter(kuro_root).load_sample(self.SAMPLE)
        entry = sample.dropped_source_diagnostics["SL2"]["vh"]
        assert entry["negative_pixels"] == 1

    def test_the_dropped_epoch_still_appears_in_temporal_metadata(self, kuro_root):
        """Narrowing the finiteness check must not remove SL2 from the record."""
        _corrupt_plane(kuro_root, 1111003, "SL2", "IVV", float("nan"))
        sample = KuroSiwoAdapter(kuro_root).load_sample(self.SAMPLE)
        assert sample.temporal_roles["SL2"].rank == 2
        assert "SL2" in sample.dropped_sources
        assert sample.provenance.temporal_roles == ("SL1", "MS1", "SL2")

    def test_label_and_validity_cross_checks_are_unchanged(self, kuro_root):
        """The other two integrity invariants must still bite."""
        ref = KuroSiwoAdapter(kuro_root).discover()[0]
        with rasterio.open(ref.validity_asset.path, "r+") as dataset:
            values = dataset.read(1)
            values[1, 2] = 0
            dataset.write(values, 1)
        with pytest.raises(DatasetIntegrityError, match="MLU|SAR"):
            KuroSiwoAdapter(kuro_root).load_sample(ref.sample_id)
