"""Sen1Floods11 hand-labelled discovery and adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

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

__all__ = ["Sen1Floods11Adapter", "Sen1Floods11SampleRef"]


@dataclass(frozen=True)
class Sen1Floods11SampleRef:
    """Lazy index record for one five-raster hand-labelled chip."""

    sample_id: str
    split: str
    event_id: str
    assets: Mapping[str, RasterAsset]
    source_asset: RasterAsset
    target_grid: object


class Sen1Floods11Adapter(DatasetAdapter):
    """Read-only adapter for the ``HandLabeled`` five-file sample contract."""

    dataset_id = "sen1floods11"
    _ENV = "FLOODMAP_SEN1FLOODS11_ROOT"
    _KINDS = ("S1Hand", "S2Hand", "LabelHand", "S1OtsuLabelHand", "JRCWaterHand")

    def __init__(self, root: Optional[Path] = None):
        self.root = external_root(root, self._ENV)
        self.capabilities: DatasetCapabilities = load_dataset_capabilities()[self.dataset_id]
        self._refs: Optional[tuple[Sen1Floods11SampleRef, ...]] = None

    def discover(self, *, refresh: bool = False) -> tuple[Sen1Floods11SampleRef, ...]:
        if self._refs is None or refresh:
            self._refs = self._build_index()
        return self._refs

    def _build_index(self) -> tuple[Sen1Floods11SampleRef, ...]:
        split_map = self._read_split_map()
        s1_root = self.root / "HandLabeled" / "S1Hand"
        if not s1_root.is_dir():
            raise DatasetDiscoveryError(f"Sen1Floods11 directory is missing: {s1_root}")
        refs: list[Sen1Floods11SampleRef] = []
        for s1_path in sorted(s1_root.glob("*.tif")):
            sample_id = s1_path.name.removesuffix("_S1Hand.tif")
            if sample_id not in split_map:
                raise DatasetDiscoveryError(
                    f"Sen1Floods11 sample has no official split entry: {s1_path.name}"
                )
            paths = {"s1": s1_path}
            for kind, suffix in (
                ("s2", "_S2Hand.tif"),
                ("label", "_LabelHand.tif"),
                ("s1_otsu", "_S1OtsuLabelHand.tif"),
                ("jrc_water", "_JRCWaterHand.tif"),
            ):
                paths[kind] = self.root / "HandLabeled" / kind_to_directory(kind) / f"{sample_id}{suffix}"
            if any(not path.is_file() for path in paths.values()):
                missing = [str(path) for path in paths.values() if not path.is_file()]
                raise DatasetDiscoveryError(
                    f"Sen1Floods11 five-file sample is incomplete for {sample_id}: {missing}"
                )
            assets = {name: asset_from_path(path) for name, path in paths.items()}
            source = require_equal_metadata(assets.values(), transform_tolerance=1e-10)
            if source.crs != "EPSG:4326":
                raise DatasetIntegrityError(f"Sen1Floods11 source CRS must be EPSG:4326: {source.path}")
            s1 = assets["s1"]
            s2 = assets["s2"]
            label = assets["label"]
            if s1.count != 2 or s1.dtype != "float32" or s1.descriptions[:2] != ("VV", "VH"):
                raise DatasetIntegrityError(f"Sen1Floods11 S1 must contain VV/VH bands: {s1.path}")
            if s2.count != 13 or s2.dtype != "int16":
                raise DatasetIntegrityError(f"Sen1Floods11 S2 must contain 13 bands: {s2.path}")
            if label.count != 1:
                raise DatasetIntegrityError(f"Sen1Floods11 label must be single-band: {label.path}")
            centroid_lon, centroid_lat = raster_centroid(source)
            target = derive_target_grid(
                centroid_lon=centroid_lon, centroid_lat=centroid_lat, source_crs=source.crs
            )
            refs.append(
                Sen1Floods11SampleRef(
                    sample_id=sample_id,
                    split=split_map[sample_id][0],
                    event_id=split_map[sample_id][1],
                    assets=assets,
                    source_asset=source,
                    target_grid=target,
                )
            )
        return tuple(refs)

    def _read_split_map(self) -> dict[str, tuple[str, str]]:
        split_root = self.root / "splits" / "flood_handlabeled"
        if not split_root.is_dir():
            raise DatasetDiscoveryError(f"Sen1Floods11 split directory is missing: {split_root}")
        result: dict[str, tuple[str, str]] = {}
        files = {
            "flood_train_data.csv": "train",
            "flood_valid_data.csv": "validation",
            "flood_test_data.csv": "test",
            "flood_bolivia_data.csv": "bolivia",
        }
        for filename, split in files.items():
            path = split_root / filename
            if not path.is_file():
                raise DatasetDiscoveryError(f"Sen1Floods11 split manifest is missing: {path}")
            try:
                with path.open(newline="", encoding="utf-8") as stream:
                    for row in csv.reader(stream):
                        if len(row) < 1 or not row[0].strip():
                            raise DatasetDiscoveryError(f"malformed split row in {path}")
                        name = Path(row[0].strip()).name
                        if not name.endswith("_S1Hand.tif"):
                            raise DatasetDiscoveryError(f"split row is not an S1Hand file: {name}")
                        sample_id = name.removesuffix("_S1Hand.tif")
                        event = sample_id.split("_", 1)[0]
                        if sample_id in result and result[sample_id][0] != split:
                            raise DatasetDiscoveryError(
                                f"Sen1Floods11 sample occurs in multiple splits: {sample_id}"
                            )
                        result[sample_id] = (split, event)
            except OSError as exc:
                raise DatasetDiscoveryError(f"could not read Sen1Floods11 split file: {path}") from exc
        return result

    def _ref_by_id(self, sample_id: str) -> Sen1Floods11SampleRef:
        for ref in self.discover():
            if ref.sample_id == sample_id:
                return ref
        raise KeyError(sample_id)

    def load_sample(self, sample_id: str) -> CanonicalSample:
        ref = self._ref_by_id(sample_id)
        with rasterio.open(ref.assets["s1"].path) as dataset:
            s1_values = dataset.read().astype(np.float32, copy=False)
        with rasterio.open(ref.assets["s2"].path) as dataset:
            s2_values = dataset.read()
        with rasterio.open(ref.assets["label"].path) as dataset:
            labels = dataset.read(1)
        with rasterio.open(ref.assets["s1_otsu"].path) as dataset:
            otsu = dataset.read(1)
        with rasterio.open(ref.assets["jrc_water"].path) as dataset:
            jrc = dataset.read(1)

        # Sen1Floods11's only declared validity mechanism is LabelHand == -1.
        # Do not promote NaNs in the continuous S1 raster into a second,
        # undocumented validity mechanism: the source contract publishes no SAR
        # nodata convention for this corpus.
        stored = set(int(value) for value in np.unique(labels))
        if not stored.issubset({-1, 0, 1}):
            raise DatasetIntegrityError(
                f"Sen1Floods11 LabelHand contains undeclared values: {sorted(stored)}"
            )
        if set(int(value) for value in np.unique(otsu)) - {-1, 0, 1}:
            raise DatasetIntegrityError("Sen1Floods11 S1OtsuLabelHand contains undeclared values")
        valid = labels != -1
        source_files = tuple(asset.path for asset in ref.assets.values())
        artifact = ArtifactProvenance(
            artifact_type=ArtifactType.PREPROCESSED_RASTER,
            artifact_id=f"{self.dataset_id}:{ref.sample_id}",
            production_inputs=[ProductionInput.PERMITTED_TRAINING_DATASET],
            source_product_ids=[str(path) for path in source_files],
            preprocessing_operations=[],
            notes="Sen1Floods11 hand-labelled five-raster chip; post-event only.",
        )
        optical = {
            name: s2_values[index]
            for index, name in enumerate(
                ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
            )
        }
        return CanonicalSample(
            dataset_id=self.dataset_id,
            sample_id=ref.sample_id,
            split=ref.split,
            event_id=ref.event_id,
            source_crs=ref.source_asset.crs,
            source_transform=ref.source_asset.transform,
            source_shape=ref.source_asset.shape,
            source_spacing=(abs(ref.source_asset.transform[0]), abs(ref.source_asset.transform[4])),
            source_spacing_units="degrees",
            target_grid=ref.target_grid,
            temporal_roles={
                "post": TemporalRole(
                    name="post",
                    source_date=None,
                    assets={"s1": ref.assets["s1"], "s2": ref.assets["s2"]},
                )
            },
            sar={"post": {"vv": s1_values[0], "vh": s1_values[1]}},
            sar_representation="decibel",
            optical=optical,
            dem=None,
            labels=labels.astype(np.int16, copy=False),
            validity=valid.astype(bool, copy=False),
            label_scheme=LabelScheme(
                semantic_values=(0, 1),
                stored_values=(-1, 0, 1),
                nodata_value=-1,
                nodata_form="in_band_negative_sentinel",
                names={0: "Not Water", 1: "Water"},
            ),
            nodata=NodataSemantics(
                source_label_nodata=-1,
                source_continuous_nodata=None,
            ),
            capabilities=self.capabilities,
            provenance=SampleProvenance(
                source_dataset=self.dataset_id,
                source_sample_or_grid=ref.sample_id,
                activation_or_event=ref.event_id,
                source_crs=ref.source_asset.crs,
                temporal_roles=("post",),
                preprocessing_representation="decibel + TOA reflectance x10000",
                clipping_state="not_applied",
                validity_mechanism=tuple(self.capabilities.validity_mechanisms),
                source_files=source_files,
                artifact=artifact,
            ),
            preprocessing_state=PreprocessingState(
                sar_representation="decibel",
                sar_clip_applied=False,
                sar_clip_domain=None,
                sar_clip_max_linear=None,
                sar_conversion_to_db_applied=False,
                optical_scale_factor=10000.0,
                additional_speckle_filter_applied=False,
            ),
            dropped_sources=(),
            auxiliary_rasters={"s1_otsu_label": otsu, "jrc_water": jrc},
        )


def kind_to_directory(kind: str) -> str:
    return {
        "s2": "S2Hand",
        "label": "LabelHand",
        "s1_otsu": "S1OtsuLabelHand",
        "jrc_water": "JRCWaterHand",
    }[kind]
