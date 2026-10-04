"""Flood and debris segmentation.

Scientific responsibility
-------------------------
Map an analysis-ready multimodal raster to a class probability raster, a
segmentation mask, and model metadata sufficient to reproduce the inference
(architecture.md §6).

Key constraints
---------------
* The architecture is deliberately NOT pre-selected. Model choice must follow
  measured performance, data compatibility, compute feasibility and
  reproducibility rather than popularity (AGENTS.md §5, PRD.md §8).
* Before committing to a model, establish label compatibility, input modality
  compatibility, spatial resolution, splits, compute cost, baseline
  performance and unseen-Himalaya performance (AGENTS.md §5).
* Thresholds are selected on validation data, never on the unseen-Himalaya
  evaluation set (docs/evaluation-protocol.md).

Status: NOT IMPLEMENTED.
"""
