"""Production of spatially aligned, analysis-ready inputs.

Scientific responsibility
-------------------------
Spatial reference normalisation, clipping to the area of interest, raster
alignment and co-registration, missing-data handling, modality-specific
normalisation, quality masking, and provenance recording
(architecture.md §4).

Key constraints
---------------
* Rasters must not be assumed aligned. CRS, transform, resolution, extent,
  pixel alignment and acquisition geometry are verified before any
  pixel-level comparison (AGENTS.md §7).
* Resampling must be documented with method and reason.
* No hidden preprocessing: any step a model depends on belongs here and in
  the documented pipeline (architecture.md §4).

Milestone 2 provides explicit, manifest-bound raster validation, masking,
alignment and analysis-ready artifact writing. Production execution remains
configuration- and source-product-gated.
"""

from .artifacts import ProcessedArtifact
from .config import PreprocessingConfig, load_preprocessing_config
from .grid import AlignmentReport, AnalysisGrid, compare_grids, require_aligned
from .inputs import ManifestSourceProduct, PreprocessingInputs
from .pipeline import (
    preprocess_dem,
    preprocess_sentinel1_pair,
    preprocess_sentinel2_pair,
    validate_preprocessing_pair_alignment,
)

__all__ = [
    "AlignmentReport",
    "AnalysisGrid",
    "ManifestSourceProduct",
    "PreprocessingConfig",
    "PreprocessingInputs",
    "ProcessedArtifact",
    "compare_grids",
    "load_preprocessing_config",
    "preprocess_dem",
    "preprocess_sentinel1_pair",
    "preprocess_sentinel2_pair",
    "require_aligned",
    "validate_preprocessing_pair_alignment",
]
