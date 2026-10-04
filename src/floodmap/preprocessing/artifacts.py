"""Analysis-ready raster artifact writing and provenance linkage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..acquisition.manifest import AcquisitionManifest
from ..acquisition.scenes import SensorKind
from ..utils.provenance import ArtifactProvenance, ArtifactType, ProductionInput
from .grid import AnalysisGrid, require_rasterio

try:
    import rasterio
except ImportError as exc:  # pragma: no cover
    rasterio = None  # type: ignore[assignment]
    _RASTERIO_ERROR = exc
else:
    _RASTERIO_ERROR = None

__all__ = ["ProcessedArtifact", "write_raster_artifact"]


@dataclass(frozen=True)
class ProcessedArtifact:
    """Paths and quality record for one analysis-ready output."""

    artifact_path: Path
    valid_mask_path: Path
    quality_path: Path
    provenance_path: Path
    quality: Mapping[str, Any]
    provenance: ArtifactProvenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "valid_mask_path": str(self.valid_mask_path),
            "quality_path": str(self.quality_path),
            "provenance_path": str(self.provenance_path),
            "quality": dict(self.quality),
            "provenance": self.provenance.to_dict(),
        }


def _require_rasterio() -> None:
    require_rasterio()
    if _RASTERIO_ERROR is not None:
        raise RuntimeError("rasterio is required to write M2 artifacts") from _RASTERIO_ERROR


def _quality_stats(data: np.ndarray, valid: np.ndarray, grid: AnalysisGrid) -> Dict[str, Any]:
    total = int(valid.size)
    valid_count = int(valid.sum())
    finite = np.isfinite(data)
    return {
        "width": grid.width,
        "height": grid.height,
        "bands": int(data.shape[0]),
        "crs": grid.crs,
        "transform": [float(value) for value in grid.transform[:6]],
        "resolution_m": grid.resolution_m,
        "bounds": list(grid.bounds),
        "valid_pixel_count": valid_count,
        "invalid_pixel_count": total - valid_count,
        "invalid_pixel_percent": (100.0 * (total - valid_count) / total if total else None),
        "finite_pixel_percent": (100.0 * int(finite.all(axis=0).sum()) / total if total else None),
    }


def write_raster_artifact(
    *,
    data: np.ndarray,
    valid_mask: np.ndarray,
    grid: AnalysisGrid,
    output_dir: Path,
    stem: str,
    manifest: AcquisitionManifest,
    sensor: Optional[SensorKind],
    source_product_ids: Sequence[str],
    source_scene_ids: Sequence[str],
    operations: Sequence[str],
    config_version: str,
    preprocessing_version: str,
    production_input: Optional[ProductionInput] = None,
    dem_version: Optional[str] = None,
    quality_extra: Optional[Mapping[str, Any]] = None,
    nodata: Optional[float] = None,
) -> ProcessedArtifact:
    """Write data, mask, QA JSON and linked ArtifactProvenance."""
    _require_rasterio()
    if sensor is None and production_input is None:
        raise ValueError("an artifact must identify its production input")
    if data.ndim != 3:
        raise ValueError("artifact data must have shape (bands, height, width)")
    if valid_mask.shape != (grid.height, grid.width):
        raise ValueError("valid mask shape does not match analysis grid")
    if data.shape[1:] != (grid.height, grid.width):
        raise ValueError("artifact data shape does not match analysis grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{stem}.tif"
    valid_mask_path = output_dir / f"{stem}_valid_mask.tif"
    quality_path = output_dir / f"{stem}_quality.json"
    provenance_path = output_dir / f"{stem}_provenance.yaml"
    dtype = str(data.dtype)
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": int(data.shape[0]),
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(artifact_path, "w", **profile) as destination:
        destination.write(data)
        destination.update_tags(
            preprocessing_version=preprocessing_version,
            acquisition_manifest_id=manifest.artifact_id,
            source_product_ids=",".join(source_product_ids),
        )
    mask_profile = {**profile, "count": 1, "dtype": "uint8", "nodata": 0}
    with rasterio.open(valid_mask_path, "w", **mask_profile) as destination:
        destination.write(valid_mask.astype("uint8", copy=False), 1)
        destination.update_tags(mask_semantics="1=valid observation, 0=not observed")

    quality = _quality_stats(data, valid_mask, grid)
    quality.update(
        {
            "status": "success",
            "preprocessing_version": preprocessing_version,
            "config_version": config_version,
            "acquisition_manifest_id": manifest.artifact_id,
            "source_product_ids": list(source_product_ids),
            "source_scene_ids": list(source_scene_ids),
        }
    )
    quality.update(quality_extra or {})
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = _provenance(
        manifest=manifest,
        sensor=sensor,
        production_input=production_input,
        source_product_ids=source_product_ids,
        source_scene_ids=source_scene_ids,
        operations=operations,
        config_version=config_version,
        preprocessing_version=preprocessing_version,
        dem_version=dem_version,
        quality=quality,
    )
    provenance_path.write_text(provenance.to_yaml(), encoding="utf-8")
    return ProcessedArtifact(
        artifact_path=artifact_path,
        valid_mask_path=valid_mask_path,
        quality_path=quality_path,
        provenance_path=provenance_path,
        quality=quality,
        provenance=provenance,
    )


def _provenance(
    *,
    manifest: AcquisitionManifest,
    sensor: Optional[SensorKind],
    production_input: Optional[ProductionInput],
    source_product_ids: Sequence[str],
    source_scene_ids: Sequence[str],
    operations: Sequence[str],
    config_version: str,
    preprocessing_version: str,
    dem_version: Optional[str],
    quality: Mapping[str, Any],
) -> ArtifactProvenance:
    linked = manifest.provenance
    production_inputs = (
        [production_input or sensor.production_input] if (production_input or sensor) else []
    )
    sensor_label = sensor.value if sensor is not None else production_inputs[0].value
    return ArtifactProvenance(
        artifact_type=ArtifactType.PREPROCESSED_RASTER,
        artifact_id=f"preprocessed-{sensor_label}-{manifest.artifact_id}",
        aoi=linked.aoi,
        aoi_crs=linked.aoi_crs,
        event_date=linked.event_date,
        sentinel1=linked.sentinel1 if sensor is SensorKind.SENTINEL1 else None,
        sentinel2=linked.sentinel2 if sensor is SensorKind.SENTINEL2 else None,
        production_inputs=production_inputs,
        dem_version=dem_version,
        acquisition_manifest_id=manifest.artifact_id,
        source_product_ids=list(source_product_ids),
        preprocessing_operations=list(operations),
        quality=dict(quality),
        preprocessing_version=preprocessing_version,
        config_version=config_version,
        limitations=[
            "This artifact is analysis-ready raster input; it is not a flood/debris prediction.",
            f"Source scene IDs: {', '.join(source_scene_ids)}.",
        ],
        notes=(
            "M2 preprocessing preserves the M1 manifest link. Source metadata that "
            "cannot be established is represented in quality or limitations."
        ),
    )
