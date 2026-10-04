"""Optional DEM-based downstream flood-path tracing (bonus capability).

Scientific responsibility
-------------------------
From an upstream point, use terrain-derived downstream direction to estimate a
plausible flood path and list settlements along it (architecture.md §10).

Key constraints
---------------
* This is terrain-based inference, NOT a validated hydrodynamic simulation,
  and must never be described as one unless such a model is actually
  implemented and validated (AGENTS.md §10, architecture.md §10).

Status: NOT IMPLEMENTED. Secondary goal (PRD.md §3).
"""
