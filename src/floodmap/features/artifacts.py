"""Feature-set artifact writing and provenance linkage.

Artifact layout
---------------
``docs/data-contract.md`` §1.8/§2.5/§3.5 fix M2's convention as
``<stem>.tif``, ``<stem>_valid_mask.tif``, ``<stem>_quality.json``,
``<stem>_provenance.yaml``. M3 extends that same convention rather than
inventing a parallel one, and adds the registry:

```text
<output_dir>/features/<feature-set-id>/
  ├── features.tif                  # one band per feature, descriptions = names
  ├── features_valid_mask.tif       # one band per feature, same order
  ├── features_quality.json
  ├── features_provenance.yaml      # ArtifactProvenance, feature_stack
  └── feature_registry.json         # the machine-readable feature contract
```

The valid mask is a band-per-feature stack, not a single plane. Features do not
share validity: a Sentinel-1 change feature can be valid where a Sentinel-2
index is cloud-masked. Collapsing them into one mask would either discard valid
SAR evidence or mark cloud-obscured optical pixels as observed, and
``docs/scientific-assumptions.md`` §8 requires "not observed" to stay
distinguishable from "observed, not flooded". A combined-validity summary is
recorded in the QA record as a statistic, never substituted for the per-feature
masks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ..preprocessing.grid import AnalysisGrid, require_rasterio
from ..utils.provenance import (
    ArtifactProvenance,
    ArtifactType,
    BeforeAfterPair,
    ProductionInput,
)
from .errors import FeatureDependencyError
from .registry import FeatureRegistry

try:
    import rasterio
except ImportError as exc:  # pragma: no cover
    rasterio = None  # type: ignore[assignment]
    _RASTERIO_ERROR = exc
else:
    _RASTERIO_ERROR = None

__all__ = ["FeatureSetArtifact", "write_feature_set"]


def _require_import() -> None:
    require_rasterio()
    if _RASTERIO_ERROR is not None:
        raise FeatureDependencyError(
            "rasterio is required to write M3 feature artifacts"
        ) from _RASTERIO_ERROR


@dataclass(frozen=True)
class FeatureSetArtifact:
    """Paths, registry, QA and provenance for one generated feature set."""

    feature_set_id: str
    features_path: Path
    valid_mask_path: Path
    registry_path: Path
    quality_path: Path
    provenance_path: Path
    registry: FeatureRegistry
    quality: Mapping[str, Any]
    provenance: ArtifactProvenance

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.registry.names()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "features_path": str(self.features_path),
            "valid_mask_path": str(self.valid_mask_path),
            "registry_path": str(self.registry_path),
            "quality_path": str(self.quality_path),
            "provenance_path": str(self.provenance_path),
            "feature_names": list(self.feature_names),
            "quality": dict(self.quality),
            "provenance": self.provenance.to_dict(),
        }


def write_feature_set(
    *,
    feature_set_id: str,
    data: np.ndarray,
    valid_masks: np.ndarray,
    grid: AnalysisGrid,
    registry: FeatureRegistry,
    output_dir: Path | str,
    quality: Mapping[str, Any],
    acquisition_manifest_ids: Sequence[str],
    source_product_ids: Sequence[str],
    production_inputs: Sequence[ProductionInput],
    operations: Sequence[str],
    pipeline_version: str,
    config_version: str,
    nodata: float,
    sentinel1_pair: Optional[BeforeAfterPair] = None,
    sentinel2_pair: Optional[BeforeAfterPair] = None,
    dem_version: Optional[str] = None,
    aoi: Any = None,
    aoi_crs: Optional[str] = None,
    event_date: Optional[str] = None,
) -> FeatureSetArtifact:
    """Write the feature stack, per-feature masks, registry, QA and provenance."""
    _require_import()
    names = registry.names()
    if data.ndim != 3 or valid_masks.ndim != 3:
        raise ValueError("feature data and masks must have shape (features, height, width)")
    if data.shape != valid_masks.shape:
        raise ValueError("feature data and valid-mask stacks must have identical shapes")
    if data.shape[0] != len(names):
        raise ValueError(
            f"feature stack has {data.shape[0]} bands but the registry defines {len(names)}"
        )
    if data.shape[1:] != (grid.height, grid.width):
        raise ValueError("feature stack shape does not match the analysis grid")

    target = Path(output_dir) / "features" / feature_set_id
    target.mkdir(parents=True, exist_ok=True)
    features_path = target / "features.tif"
    valid_mask_path = target / "features_valid_mask.tif"
    registry_path = target / "feature_registry.json"
    quality_path = target / "features_quality.json"
    provenance_path = target / "features_provenance.yaml"

    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": len(names),
        "dtype": str(data.dtype),
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(features_path, "w", **profile) as destination:
        destination.write(data)
        for index, name in enumerate(names, start=1):
            destination.set_band_description(index, name)
        destination.update_tags(
            feature_set_id=feature_set_id,
            feature_pipeline_version=pipeline_version,
            feature_set_version=registry.feature_set_version,
            registry_version=registry.registry_version,
            acquisition_manifest_ids=",".join(acquisition_manifest_ids),
            feature_names=",".join(names),
        )

    mask_profile = {**profile, "dtype": "uint8", "nodata": 0}
    with rasterio.open(valid_mask_path, "w", **mask_profile) as destination:
        destination.write(valid_masks.astype("uint8", copy=False))
        for index, name in enumerate(names, start=1):
            destination.set_band_description(index, f"{name}_valid")
        destination.update_tags(
            mask_semantics="1=valid feature value, 0=not computable from valid inputs",
            per_feature="one band per feature, in the same order as features.tif",
        )

    registry_path.write_text(
        json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quality_record = dict(quality)
    quality_path.write_text(
        json.dumps(quality_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    provenance = ArtifactProvenance(
        artifact_type=ArtifactType.FEATURE_STACK,
        artifact_id=f"features-{feature_set_id}",
        aoi=aoi,
        aoi_crs=aoi_crs,
        event_date=event_date,
        sentinel1=sentinel1_pair,
        sentinel2=sentinel2_pair,
        dem_version=dem_version,
        production_inputs=list(production_inputs),
        acquisition_manifest_id=(
            acquisition_manifest_ids[0] if len(acquisition_manifest_ids) == 1 else None
        ),
        source_product_ids=list(source_product_ids),
        preprocessing_operations=list(operations),
        quality=quality_record,
        preprocessing_version=pipeline_version,
        config_version=config_version,
        limitations=[
            "This artifact is a set of model-ready features. It is not a "
            "flood/debris prediction and contains no classification: no "
            "threshold, decision rule or class label is applied in M3.",
            "A feature value is evidence for the segmentation model, not a "
            "finding about the surface.",
            "Each feature carries its own validity band. An invalid pixel means "
            "'not computable from valid observations', never 'observed, not flooded'.",
            f"Acquisition manifest IDs: {', '.join(acquisition_manifest_ids)}.",
            f"Feature set version: {registry.feature_set_version}; "
            f"registry version: {registry.registry_version}.",
        ],
        notes=(
            "M3 feature generation consumes M2 analysis-ready artifacts only. "
            "Band semantics, formulas, units, nodata policy and scientific "
            "rationale for every band are in feature_registry.json."
        ),
    )
    provenance_path.write_text(provenance.to_yaml(), encoding="utf-8")

    return FeatureSetArtifact(
        feature_set_id=feature_set_id,
        features_path=features_path,
        valid_mask_path=valid_mask_path,
        registry_path=registry_path,
        quality_path=quality_path,
        provenance_path=provenance_path,
        registry=registry,
        quality=quality_record,
        provenance=provenance,
    )
