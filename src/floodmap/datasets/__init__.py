"""M4 dataset contracts, indexes and external-corpus adapters."""

from .contracts import (
    CanonicalSample,
    DatasetCapabilitySummary,
    LabelScheme,
    NodataSemantics,
    PreprocessingState,
    RasterAsset,
    ResamplingPolicies,
    SampleProvenance,
    TargetGridMetadata,
    TemporalRole,
)
from .errors import DatasetAdapterError, DatasetDiscoveryError, DatasetIntegrityError
from .kuro_siwo import KuroSiwoAdapter, KuroSiwoIndex, KuroSiwoSampleRef
from .sen1floods11 import Sen1Floods11Adapter, Sen1Floods11SampleRef

__all__ = [
    "CanonicalSample",
    "DatasetAdapterError",
    "DatasetCapabilitySummary",
    "DatasetDiscoveryError",
    "DatasetIntegrityError",
    "KuroSiwoAdapter",
    "KuroSiwoIndex",
    "KuroSiwoSampleRef",
    "LabelScheme",
    "NodataSemantics",
    "PreprocessingState",
    "RasterAsset",
    "ResamplingPolicies",
    "SampleProvenance",
    "Sen1Floods11Adapter",
    "Sen1Floods11SampleRef",
    "TargetGridMetadata",
    "TemporalRole",
]
