"""Strict consumption of selected M1 manifest products by M2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ..acquisition.manifest import AcquisitionManifest
from ..acquisition.scenes import SensorKind
from ..utils.provenance import Unknown
from .errors import SourceProductError

__all__ = ["ManifestSourceProduct", "PreprocessingInputs"]


def _known(value: Any) -> Any:
    return None if isinstance(value, Unknown) else value


@dataclass(frozen=True)
class ManifestSourceProduct:
    """One selected source product and its explicit local asset path."""

    manifest_id: str
    sensor: SensorKind
    scene_id: str
    product_id: str
    path: Path
    acquired_at: Optional[str]
    product_type: Optional[str]
    relative_orbit: Optional[int]
    scene_cloud_percent: Optional[float]
    platform: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "sensor": self.sensor.value,
            "scene_id": self.scene_id,
            "product_id": self.product_id,
            "path": str(self.path),
            "acquired_at": self.acquired_at,
            "product_type": self.product_type,
            "relative_orbit": self.relative_orbit,
            "scene_cloud_percent": self.scene_cloud_percent,
            "platform": (
                self.platform.model_dump(mode="json")
                if isinstance(self.platform, Unknown)
                else self.platform
            ),
        }


@dataclass(frozen=True)
class PreprocessingInputs:
    """Selected before/after inputs extracted from one M1 manifest."""

    manifest_id: str
    aoi: Mapping[str, Any]
    sentinel1: Optional[Tuple[ManifestSourceProduct, ManifestSourceProduct]]
    sentinel2: Optional[Tuple[ManifestSourceProduct, ManifestSourceProduct]]

    @classmethod
    def from_manifest(
        cls,
        manifest: AcquisitionManifest,
        source_paths: Mapping[str, Path | str],
    ) -> "PreprocessingInputs":
        """Resolve only scenes selected by M1; no catalogue discovery occurs."""
        serialised = manifest.to_dict()
        manifest_id = str(serialised.get("artifact_id") or "")
        if not manifest_id:
            raise SourceProductError("M1 acquisition manifest has no artifact_id")
        aoi = serialised.get("aoi")
        if not isinstance(aoi, Mapping):
            raise SourceProductError("M1 manifest has no explicit AOI geometry")

        def pair(
            sensor: SensorKind,
        ) -> Optional[Tuple[ManifestSourceProduct, ManifestSourceProduct]]:
            selection = serialised.get("selections", {}).get(sensor.value)
            if not isinstance(selection, Mapping) or selection.get("status") != "selected":
                return None
            before = cls._product(
                manifest_id, sensor, selection.get("selected_before"), source_paths
            )
            after = cls._product(manifest_id, sensor, selection.get("selected_after"), source_paths)
            return before, after

        return cls(
            manifest_id=manifest_id,
            aoi=aoi,
            sentinel1=pair(SensorKind.SENTINEL1),
            sentinel2=pair(SensorKind.SENTINEL2),
        )

    @staticmethod
    def _product(
        manifest_id: str,
        sensor: SensorKind,
        scene: Any,
        source_paths: Mapping[str, Path | str],
    ) -> ManifestSourceProduct:
        if not isinstance(scene, Mapping):
            raise SourceProductError(f"manifest has no selected {sensor.value} scene")
        scene_id = str(scene.get("scene_id") or "")
        product_id = str(scene.get("product_id") or "")
        if not scene_id or not product_id:
            raise SourceProductError(f"selected {sensor.value} scene lacks scene/product ID")
        raw_path = source_paths.get(scene_id)
        if raw_path is None:
            raise SourceProductError(
                f"no explicit local source path supplied for selected scene {scene_id!r}"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise SourceProductError(f"source product for {scene_id!r} does not exist: {path}")
        return ManifestSourceProduct(
            manifest_id=manifest_id,
            sensor=sensor,
            scene_id=scene_id,
            product_id=product_id,
            path=path,
            acquired_at=scene.get("acquired_at"),
            product_type=_known(scene.get("product_type")),
            relative_orbit=_known(scene.get("relative_orbit")),
            scene_cloud_percent=_known(scene.get("scene_cloud_percent")),
            platform=scene.get("platform_short_name"),
        )
