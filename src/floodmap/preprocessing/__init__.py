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

Status: NOT IMPLEMENTED. See docs/data-contract.md for the required interface.
"""
