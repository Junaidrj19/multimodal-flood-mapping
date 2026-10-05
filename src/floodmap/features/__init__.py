"""Construction of model inputs from permitted observations.

Scientific responsibility
-------------------------
Derive reproducible, configuration-driven features: SAR backscatter channels,
pre/post SAR differences or ratios, optical bands and spectral indices,
pre/post optical differences, and terrain features where justified
(architecture.md §5).

Boundary
--------
```text
M2 analysis-ready products + provenance + QA
        -> M3 feature generation
        -> features + per-feature valid masks + registry + QA + provenance
        -> M4 segmentation
```

Key constraints
---------------
* Feature construction is configuration-driven and reproducible.
* No validation-only source may contribute to any feature. Doing so is
  training-time leakage and a critical defect (AGENTS.md §3).
* M3 performs **no classification**. It applies no flood/debris threshold and
  produces no class label; a feature is evidence for the segmentation model,
  not a finding about the surface.
* M3 performs **no resampling**. Spatial normalisation is M2's responsibility,
  so inputs that do not share the analysis grid are an explicit failure.
* Validity propagates: a feature pixel is valid only where every contributing
  input pixel is valid. An invalid source pixel is never interpolated or
  replaced with a plausible value in order to make a feature computable.

Every generated feature is described in a machine-readable registry
(:class:`floodmap.features.registry.FeatureRegistry`) carrying its physical
meaning, source sensor, inputs, formula, units, dtype, declared range, nodata
policy, version and scientific rationale.

Milestone 3 provides the feature-generation engine. Production execution
remains gated on the unresolved band/polarisation bindings and target grid that
``configs/features.yaml`` and ``configs/preprocessing.yaml`` leave null.
"""

from .artifacts import FeatureSetArtifact, write_feature_set
from .config import FeatureConfig, NormalisationConfig, load_feature_config
from .errors import (
    FeatureBoundaryError,
    FeatureConfigurationError,
    FeatureError,
    FeatureGridError,
    FeatureInputError,
    FeatureRegistryError,
    UnsupportedFeatureError,
)
from .inputs import FeatureInputs, PreprocessedInput, load_preprocessed_input
from .pipeline import build_feature_registry, generate_feature_set
from .registry import (
    FEATURE_TEMPLATES,
    ArtifactRole,
    BackscatterRepresentation,
    FeatureDefinition,
    FeatureFamily,
    FeatureInput,
    FeatureRegistry,
    FeatureTemplate,
    TemplateExpansion,
    TransformKind,
    build_registry,
    template_ids,
)

__all__ = [
    "ArtifactRole",
    "BackscatterRepresentation",
    "FEATURE_TEMPLATES",
    "FeatureBoundaryError",
    "FeatureConfig",
    "FeatureConfigurationError",
    "FeatureDefinition",
    "FeatureError",
    "FeatureFamily",
    "FeatureGridError",
    "FeatureInput",
    "FeatureInputError",
    "FeatureInputs",
    "FeatureRegistry",
    "FeatureRegistryError",
    "FeatureSetArtifact",
    "FeatureTemplate",
    "NormalisationConfig",
    "PreprocessedInput",
    "TemplateExpansion",
    "TransformKind",
    "UnsupportedFeatureError",
    "build_feature_registry",
    "build_registry",
    "generate_feature_set",
    "load_feature_config",
    "load_preprocessed_input",
    "template_ids",
    "write_feature_set",
]
