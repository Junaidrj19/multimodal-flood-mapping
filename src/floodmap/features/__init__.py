"""Construction of model inputs from permitted observations.

Scientific responsibility
-------------------------
Derive reproducible, configuration-driven features: SAR backscatter channels,
pre/post SAR differences or ratios, optical bands and spectral indices,
pre/post optical differences, and terrain features where justified
(architecture.md §5).

Key constraints
---------------
* Feature construction is configuration-driven and reproducible.
* No validation-only source may contribute to any feature. Doing so is
  training-time leakage and a critical defect (AGENTS.md §3).

Status: NOT IMPLEMENTED.
"""
