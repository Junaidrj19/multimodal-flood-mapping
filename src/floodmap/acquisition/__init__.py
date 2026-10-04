"""Acquisition of permitted observations for a requested area and event date.

Scientific responsibility
-------------------------
Obtain Sentinel-1, Sentinel-2, Copernicus DEM and a PRE-EVENT OpenStreetMap
snapshot, preserving acquisition metadata (timestamps, orbit number and track,
processing level) so downstream stages can reason about temporal and geometric
comparability.

Key constraints
---------------
* Sentinel-1 change detection should prefer matching orbit tracks; different
  tracks view terrain from different angles and must not be compared
  pixel-by-pixel as equivalent observations (AGENTS.md §4, README.md §4).
* The OSM snapshot must predate the event (architecture.md §3).
* Unavailable or unsuitable acquisitions must be recorded, not silently
  skipped, so that insufficient coverage surfaces as an explicit failure
  (architecture.md §17).
* Validation-only sources must never be acquired through this module.

Status: NOT IMPLEMENTED. See docs/data-contract.md for the required interface.
"""
