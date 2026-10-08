"""Kuro Siwo discovery and adapter for the frozen M4 contract."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import rasterio

from floodmap.utils.contract import DatasetCapabilities, load_dataset_capabilities
from floodmap.utils.provenance import ArtifactProvenance, ArtifactType, ProductionInput

from .base import (
    DatasetAdapter,
    asset_from_path,
    derive_target_grid,
    external_root,
    raster_centroid,
    require_equal_metadata,
)
from .contracts import (
    CanonicalSample,
    LabelScheme,
    NodataSemantics,
    PreprocessingState,
    RasterAsset,
    SampleProvenance,
    TemporalRole,
)
from .errors import DatasetDiscoveryError, DatasetIntegrityError

__all__ = ["KuroSiwoAdapter", "KuroSiwoIndex", "KuroSiwoSampleRef"]


# Official activation split from the upstream Kuro Siwo training configuration.
# The local delivery is allowed to be incomplete; these lists are not changed by
# what happens to be present on disk.
OFFICIAL_SPLITS: Mapping[str, frozenset[int]] = {
    "train": frozenset(
        {
            130,
            470,
            555,
            118,
            174,
            324,
            421,
            554,
            427,
            518,
            502,
            498,
            497,
            496,
            492,
            147,
            267,
            273,
            275,
            417,
            567,
            1111011,
            1111004,
            1111009,
            1111010,
            1111006,
            1111005,
        }
    ),
    "validation": frozenset({514, 559, 279, 520, 437, 1111003, 1111008}),
    "test": frozenset({321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013}),
}
OFFICIAL_ACTIVATIONS = frozenset().union(*OFFICIAL_SPLITS.values())


@dataclass(frozen=True)
class KuroSiwoSampleRef:
    """Lazy index record for one labelled Kuro Siwo grid."""

    sample_id: str
    activation_id: int
    split: str
    grid_id: str
    directory: Path
    info: Mapping[str, Any]
    temporal_assets: Mapping[str, Mapping[str, RasterAsset]]
    label_asset: RasterAsset
    validity_asset: RasterAsset
    source_asset: RasterAsset
    target_grid: Any


@dataclass(frozen=True)
class KuroSiwoIndex:
    samples: tuple[KuroSiwoSampleRef, ...]
    discovered_activations: tuple[int, ...]
    missing_official_activations: Mapping[str, tuple[int, ...]]
    expected_official_activations: Mapping[str, tuple[int, ...]]

    @property
    def locally_available_samples(self) -> int:
        return len(self.samples)


class KuroSiwoAdapter(DatasetAdapter):
    """Read-only adapter for the labelled Kuro Siwo partition (``01``)."""

    dataset_id = "kuro_siwo"
    _ENV = "FLOODMAP_KURO_SIWO_ROOT"
    _CLIP_MAX = 0.15

    def __init__(self, root: Optional[Path] = None):
        self.root = external_root(root, self._ENV)
        self.capabilities: DatasetCapabilities = load_dataset_capabilities()[self.dataset_id]
        self._index: Optional[KuroSiwoIndex] = None

    def index(self, *, refresh: bool = False) -> KuroSiwoIndex:
        if self._index is None or refresh:
            self._index = self._build_index()
        return self._index

    def discover(self, *, refresh: bool = False) -> tuple[KuroSiwoSampleRef, ...]:
        return self.index(refresh=refresh).samples

    def _build_index(self) -> KuroSiwoIndex:
        refs: list[KuroSiwoSampleRef] = []
        discovered: list[int] = []
        for activation_dir in sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()):
            activation_id = int(activation_dir.name)
            discovered.append(activation_id)
            refs.extend(self._discover_activation(activation_dir, activation_id))

        expected = {name: tuple(sorted(ids)) for name, ids in OFFICIAL_SPLITS.items()}
        present = set(discovered)
        missing = {name: tuple(sorted(ids - present)) for name, ids in OFFICIAL_SPLITS.items()}
        return KuroSiwoIndex(
            samples=tuple(refs),
            discovered_activations=tuple(sorted(set(discovered))),
            missing_official_activations=missing,
            expected_official_activations=expected,
        )

    def _discover_activation(self, activation_dir: Path, activation_id: int) -> list[KuroSiwoSampleRef]:
        partition = activation_dir / "01"
        if not partition.is_dir():
            # Partition 00 is intentionally not considered.  An activation with
            # only 00 data has no supervised samples and is simply locally absent.
            return []
        catalogue = activation_dir / "catalogue.gkpg"
        if not catalogue.is_file():
            raise DatasetDiscoveryError(f"Kuro catalogue is missing: {catalogue}")
        exported = self._read_exported_catalogue(catalogue, activation_id)
        on_disk = self._sample_directories(partition)
        on_disk_ids = set(on_disk)
        exported_ids = set(exported)
        if on_disk_ids != exported_ids:
            missing = sorted(exported_ids - on_disk_ids)
            extra = sorted(on_disk_ids - exported_ids)
            raise DatasetDiscoveryError(
                f"catalogue/export consistency failed for activation {activation_id}: "
                f"missing exported grids={missing[:5]}, unlisted on-disk grids={extra[:5]}"
            )

        split = self._split_for_activation(activation_id)
        refs: list[KuroSiwoSampleRef] = []
        for grid_id in sorted(on_disk_ids):
            refs.append(self._make_ref(on_disk[grid_id], activation_id, split, exported[grid_id]))
        return refs

    @staticmethod
    def _sample_directories(partition: Path) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for directory in sorted(p for p in partition.iterdir() if p.is_dir()):
            info_path = directory / "info.json"
            if not info_path.is_file():
                raise DatasetDiscoveryError(f"Kuro sample is missing info.json: {directory}")
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                grid_id = str(info["grid_id"])
            except (OSError, ValueError, KeyError) as exc:
                raise DatasetDiscoveryError(f"malformed Kuro info.json: {info_path}") from exc
            if grid_id in result:
                raise DatasetDiscoveryError(f"duplicate Kuro grid_id on disk: {grid_id}")
            result[grid_id] = directory
        return result

    @staticmethod
    def _read_exported_catalogue(catalogue: Path, activation_id: int) -> dict[str, dict[str, Any]]:
        try:
            connection = sqlite3.connect(str(catalogue))
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT grid_id, actid, aoiid, exported, master, crank, source_date, s1_ids "
                "FROM catalogue"
            ).fetchall()
        except sqlite3.Error as exc:
            raise DatasetDiscoveryError(f"could not read Kuro catalogue: {catalogue}") from exc
        finally:
            try:
                connection.close()
            except UnboundLocalError:
                pass

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if int(row["actid"]) != activation_id:
                continue
            if int(row["exported"]) != 1:
                continue
            record = dict(row)
            groups.setdefault(str(record["grid_id"]), []).append(record)
        result: dict[str, dict[str, Any]] = {}
        for grid_id, records in groups.items():
            if len(records) != 3:
                raise DatasetDiscoveryError(
                    f"Kuro catalogue grid {grid_id} has {len(records)} exported rows, expected 3"
                )
            if any(int(row["aoiid"]) != 1 for row in records):
                raise DatasetDiscoveryError(f"Kuro catalogue AOI is not labelled partition 01: {grid_id}")
            roles = {(bool(row["master"]), int(row["crank"])) for row in records}
            if roles != {(False, 1), (False, 2), (True, 1)}:
                raise DatasetDiscoveryError(f"Kuro catalogue roles are malformed for grid {grid_id}")
            result[grid_id] = {"rows": tuple(records)}
        return result

    @staticmethod
    def _split_for_activation(activation_id: int) -> str:
        for split, ids in OFFICIAL_SPLITS.items():
            if activation_id in ids:
                return split
        return "unassigned"

    def _make_ref(
        self, directory: Path, activation_id: int, split: str, catalogue_record: Mapping[str, Any]
    ) -> KuroSiwoSampleRef:
        try:
            info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DatasetDiscoveryError(f"malformed Kuro info.json: {directory / 'info.json'}") from exc
        if int(info.get("actid")) != activation_id or int(info.get("aoiid")) != 1:
            raise DatasetIntegrityError(f"Kuro labelled sample metadata is inconsistent: {directory}")
        if str(info.get("grid_id")) != directory.name and str(info.get("grid_id")) != str(
            catalogue_record.get("grid_id", info.get("grid_id"))
        ):
            raise DatasetIntegrityError(f"Kuro grid identity is inconsistent: {directory}")
        sources = info.get("sources")
        if not isinstance(sources, dict) or set(sources) != {"MS1", "SL1", "SL2"}:
            raise DatasetIntegrityError(f"Kuro temporal source map is incomplete: {directory}")
        if sources["MS1"].get("master") is not True:
            raise DatasetIntegrityError("Kuro MS1 must be the master/post-event source")
        if sources["SL1"].get("master") is not False or sources["SL1"].get("crank") != 1:
            raise DatasetIntegrityError("Kuro SL1 must be crank 1 pre-event source")
        if sources["SL2"].get("master") is not False or sources["SL2"].get("crank") != 2:
            raise DatasetIntegrityError("Kuro SL2 must be crank 2 pre-event source")

        temporal_assets: dict[str, dict[str, RasterAsset]] = {}
        required_assets: list[RasterAsset] = []
        for role in ("MS1", "SL1", "SL2"):
            temporal_assets[role] = {}
            for pol in ("vv", "vh"):
                paths = sorted(directory.glob(f"{role}_I{pol.upper()}_*.tif"))
                if len(paths) != 1:
                    raise DatasetDiscoveryError(
                        f"Kuro {role}/{pol} requires exactly one raster in {directory}"
                    )
                asset = asset_from_path(paths[0])
                if asset.dtype != "float32":
                    raise DatasetIntegrityError(
                        f"Kuro SAR must be float32 linear sigma0: {asset.path}"
                    )
                temporal_assets[role][pol] = asset
                required_assets.append(asset)
        label_paths = sorted(directory.glob("MK0_MLU_*.tif"))
        validity_paths = sorted(directory.glob("MK0_MNA_*.tif"))
        if len(label_paths) != 1 or len(validity_paths) != 1:
            raise DatasetDiscoveryError(f"Kuro labels/validity are incomplete: {directory}")
        label_asset = asset_from_path(label_paths[0])
        validity_asset = asset_from_path(validity_paths[0])
        required_assets.extend((label_asset, validity_asset))
        source_asset = require_equal_metadata(required_assets)
        if source_asset.crs != "EPSG:3857":
            raise DatasetIntegrityError(f"Kuro source CRS must be EPSG:3857: {source_asset.path}")
        if source_asset.transform[0] != 10.0 or source_asset.transform[4] != -10.0:
            raise DatasetIntegrityError("Kuro source spacing must be 10 projected units")
        if label_asset.nodata is not None and label_asset.nodata != 3:
            raise DatasetIntegrityError("Kuro label raster nodata must be 3")
        centroid_lon, centroid_lat = raster_centroid(source_asset)
        target = derive_target_grid(
            centroid_lon=centroid_lon, centroid_lat=centroid_lat, source_crs=source_asset.crs
        )
        return KuroSiwoSampleRef(
            sample_id=f"{activation_id}:{info['grid_id']}",
            activation_id=activation_id,
            split=split,
            grid_id=str(info["grid_id"]),
            directory=directory,
            info=info,
            temporal_assets=temporal_assets,
            label_asset=label_asset,
            validity_asset=validity_asset,
            source_asset=source_asset,
            target_grid=target,
        )

    def _ref_by_id(self, sample_id: str) -> KuroSiwoSampleRef:
        for ref in self.discover():
            if ref.sample_id == sample_id:
                return ref
        raise KeyError(sample_id)

    def load_sample(self, sample_id: str) -> CanonicalSample:
        ref = self._ref_by_id(sample_id)
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for role, assets in ref.temporal_assets.items():
            arrays[role] = {}
            for pol, asset in assets.items():
                with rasterio.open(asset.path) as dataset:
                    values = dataset.read(1).astype(np.float32, copy=False)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise DatasetIntegrityError(f"Kuro linear SAR is not finite/non-negative: {asset.path}")
                arrays[role][pol] = np.minimum(values, self._CLIP_MAX).astype(np.float32, copy=False)

        with rasterio.open(ref.label_asset.path) as dataset:
            labels = dataset.read(1)
        with rasterio.open(ref.validity_asset.path) as dataset:
            mna = dataset.read(1)

        stored = set(int(value) for value in np.unique(labels))
        if not stored.issubset({0, 1, 2, 3}):
            raise DatasetIntegrityError(f"Kuro label contains undeclared values: {sorted(stored)}")
        if not set(int(value) for value in np.unique(mna)).issubset({0, 1}):
            raise DatasetIntegrityError("Kuro MNA must contain only 0=invalid and 1=valid")
        valid = mna == 1
        label_invalid = labels == 3
        sar_invalid = np.logical_or.reduce(
            [np.logical_and.reduce([arrays[role][pol] == 0.0 for pol in ("vv", "vh")]) for role in arrays]
        )
        if not np.array_equal(label_invalid, ~valid):
            raise DatasetIntegrityError("Kuro integrity failure: MLU==3 does not equal MNA==0")
        if not np.array_equal(sar_invalid, ~valid):
            raise DatasetIntegrityError("Kuro integrity failure: SAR==0.0 does not equal MNA==0")

        temporal_roles = {
            role: TemporalRole(
                name=role,
                source_date=str(ref.info["sources"][role].get("source_date")),
                assets=assets,
                master=bool(ref.info["sources"][role].get("master")),
                rank=int(ref.info["sources"][role].get("crank")),
            )
            for role, assets in ref.temporal_assets.items()
        }
        source_files = tuple(
            asset.path
            for assets in ref.temporal_assets.values()
            for asset in assets.values()
        ) + (ref.label_asset.path, ref.validity_asset.path)
        artifact = ArtifactProvenance(
            artifact_type=ArtifactType.PREPROCESSED_RASTER,
            artifact_id=f"{self.dataset_id}:{ref.sample_id}",
            production_inputs=[ProductionInput.PERMITTED_TRAINING_DATASET],
            source_product_ids=[str(path) for path in source_files],
            preprocessing_operations=["clip linear sigma0 to 0.15 at dataset adapter"],
            notes="Kuro Siwo labelled partition 01; SL2 retained as dropped metadata.",
        )
        return CanonicalSample(
            dataset_id=self.dataset_id,
            sample_id=ref.sample_id,
            split=ref.split,
            event_id=str(ref.activation_id),
            source_crs=ref.source_asset.crs,
            source_transform=ref.source_asset.transform,
            source_shape=ref.source_asset.shape,
            source_spacing=(abs(ref.source_asset.transform[0]), abs(ref.source_asset.transform[4])),
            source_spacing_units="projected_units",
            target_grid=ref.target_grid,
            temporal_roles=temporal_roles,
            sar={"pre": arrays["SL1"], "post": arrays["MS1"]},
            sar_representation="linear_sigma0",
            optical=None,
            dem=None,
            labels=labels.astype(np.int16, copy=False),
            validity=valid.astype(bool, copy=False),
            label_scheme=LabelScheme(
                semantic_values=(0, 1, 2),
                stored_values=(0, 1, 2, 3),
                nodata_value=3,
                nodata_form="in_band_positive_sentinel",
                names={0: "No water", 1: "Permanent Waters", 2: "Floods"},
            ),
            nodata=NodataSemantics(
                source_label_nodata=3,
                source_continuous_nodata=0.0,
            ),
            capabilities=self.capabilities,
            provenance=SampleProvenance(
                source_dataset=self.dataset_id,
                source_sample_or_grid=ref.grid_id,
                activation_or_event=str(ref.activation_id),
                source_crs=ref.source_asset.crs,
                temporal_roles=("SL1", "MS1", "SL2"),
                preprocessing_representation="linear_sigma0",
                clipping_state="applied_0.15_linear_sigma0",
                validity_mechanism=tuple(self.capabilities.validity_mechanisms),
                source_files=source_files,
                artifact=artifact,
            ),
            preprocessing_state=PreprocessingState(
                sar_representation="linear_sigma0",
                sar_clip_applied=True,
                sar_clip_domain="linear_sigma0",
                sar_clip_max_linear=self._CLIP_MAX,
                sar_conversion_to_db_applied=False,
                optical_scale_factor=None,
                additional_speckle_filter_applied=False,
                excluded_sources=("MK0_DEM", "MK0_SLOPE"),
            ),
            dropped_sources=("SL2", "MK0_DEM", "MK0_SLOPE"),
        )
