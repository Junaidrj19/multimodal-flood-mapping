"""Situation report generation from structured system outputs.

Scientific responsibility
-------------------------
Render a one-page English and Nepali situation report from structured analysis
results (architecture.md §16, PRD.md FR-12).

Key constraints
---------------
* Every numerical statement must be traceable to a system output.
* A language model, if used, is a rendering and summarisation layer only. It
  must never become the source of numerical truth, and must not invent
  statistics (AGENTS.md §11, architecture.md §16, README.md §8).
* Reports must state limitations that affect the result (AGENTS.md §17).
* Required Copernicus, WorldDEM-30 and OpenStreetMap attributions must be
  present (AGENTS.md §18).

Status: NOT IMPLEMENTED.
"""
