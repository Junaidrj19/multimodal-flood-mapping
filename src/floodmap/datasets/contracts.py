"""Canonical M4 sample and dataset capability interfaces.

The adapters deliberately keep source representations and label schemes intact.
The canonical sample is a boundary object: it carries enough metadata for the
later raster preprocessing stage to create the 10 m UTM grid, but does not
silently reproject or resample source files during discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from floodmap.utils.contract import DatasetCapabilities
from floodmap.utils.provenance import ArtifactProvenance

__all__ = [
    "CanonicalSample",
    "DatasetCapabilitySummary",
    "LabelScheme",
    "NodataSemantics",
    "PreprocessingState",
    "RasterAsset",
    "ResamplingPolicies",
    "SampleProvenance",
    "TargetGridMetadata",
    "TemporalRole",
]


@dataclass(frozen=True)
class RasterAsset:
    """A source raster reference and its lightweight geospatial metadata."""

    path: Path
    crs: str
    transform: Tuple[float, ...]
    shape: Tuple[int, int]
    dtype: str
    count: int = 1
    nodata: Optional[float] = None
    descriptions: Tuple[Optional[str], ...] = ()


@dataclass(frozen=True)
class TemporalRole:
    """One source epoch, including its source files and acquisition identity."""

    name: str
    source_date: Optional[str]
    assets: Mapping[str, RasterAsset]
    master: Optional[bool] = None
    rank: Optional[int] = None


@dataclass(frozen=True)
class ResamplingPolicies:
    """C3 type-specific methods carried with the canonical grid rule."""

    continuous_sar: str = "bilinear"
    continuous_optical: str = "bilinear"
    categorical_label: str = "nearest"
    binary_validity_mask: str = "nearest"
    dem: str = "bilinear"
    sar_domain: str = "linear_power"
    sar_before_db_conversion: bool = True


@dataclass(frozen=True)
class TargetGridMetadata:
    """The frozen canonical-grid rule and an optional concrete grid instance."""

    crs: str
    resolution_m: float = 10.0
    zone: Optional[int] = None
    hemisphere: Optional[str] = None
    centroid_lon: Optional[float] = None
    centroid_lat: Optional[float] = None
    transform: Optional[Tuple[float, ...]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    is_native_source_resolution: bool = False
    resampling: ResamplingPolicies = field(default_factory=ResamplingPolicies)


@dataclass(frozen=True)
class LabelScheme:
    """Stored and semantic labels are separate concepts."""

    semantic_values: Tuple[int, ...]
    stored_values: Tuple[int, ...]
    nodata_value: int
    nodata_form: str
    names: Mapping[int, str]


@dataclass(frozen=True)
class NodataSemantics:
    """Source nodata and canonical output nodata are intentionally distinct."""

    source_label_nodata: Optional[int]
    source_continuous_nodata: Optional[float]
    class_mask_nodata: int = 255
    continuous_nodata: float = -9999.0
    ignore_index: int = 3


@dataclass(frozen=True)
class PreprocessingState:
    """Explicit state of transformations applied at the adapter boundary."""

    sar_representation: str
    sar_clip_applied: bool
    sar_clip_domain: Optional[str]
    sar_clip_max_linear: Optional[float]
    sar_conversion_to_db_applied: bool
    optical_scale_factor: Optional[float] = None
    resampled: bool = False
    additional_speckle_filter_applied: bool = False
    excluded_sources: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleProvenance:
    """Compact, sample-level provenance without raster payloads."""

    source_dataset: str
    source_sample_or_grid: str
    activation_or_event: Optional[str]
    source_crs: str
    temporal_roles: Tuple[str, ...]
    preprocessing_representation: str
    clipping_state: str
    validity_mechanism: Tuple[str, ...]
    source_files: Tuple[Path, ...]
    artifact: ArtifactProvenance


@dataclass(frozen=True)
class CanonicalSample:
    """Dataset-independent sample boundary object.

    Optional modalities are represented by ``None`` or an empty mapping.  In
    particular, Sen1Floods11 has no ``pre`` SAR epoch and no DEM; neither is
    represented by a zero-filled placeholder.
    """

    dataset_id: str
    sample_id: str
    split: str
    event_id: Optional[str]
    source_crs: str
    source_transform: Tuple[float, ...]
    source_shape: Tuple[int, int]
    source_spacing: Tuple[float, float]
    source_spacing_units: str
    target_grid: TargetGridMetadata
    temporal_roles: Mapping[str, TemporalRole]
    sar: Mapping[str, Mapping[str, np.ndarray]]
    sar_representation: str
    optical: Optional[Mapping[str, np.ndarray]]
    dem: Optional[np.ndarray]
    labels: np.ndarray
    validity: np.ndarray
    label_scheme: LabelScheme
    nodata: NodataSemantics
    capabilities: DatasetCapabilities
    provenance: SampleProvenance
    preprocessing_state: PreprocessingState
    dropped_sources: Tuple[str, ...] = ()
    auxiliary_rasters: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels.shape != self.source_shape:
            raise ValueError("labels do not match source_shape")
        if self.validity.shape != self.source_shape:
            raise ValueError("validity does not match source_shape")
        if self.validity.dtype != np.bool_:
            raise ValueError("validity must be a boolean mask")


@dataclass(frozen=True)
class DatasetCapabilitySummary:
    """Small public summary used by discovery/indexing clients."""

    dataset_id: str
    capabilities: DatasetCapabilities
    validity_mechanisms: Tuple[str, ...]
