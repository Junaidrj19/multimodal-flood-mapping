"""floodmap — flood damage and accessibility intelligence from permitted satellite data.

Multimodal AI Hackathon 2026, Track B: Mapping Flood Damage from Space.

Scientific responsibility boundary
----------------------------------
This package holds the project's scientific logic. Notebooks and the dashboard
consume it; they must not re-implement it (AGENTS.md §12, §16).

Production input boundary (AGENTS.md §3, architecture.md §2, §18)
-----------------------------------------------------------------
Only the following may enter the production pipeline:

    Sentinel-1, Sentinel-2, Copernicus DEM, pre-event OpenStreetMap,
    permitted training datasets

The following are VALIDATION / COMPARISON ONLY and must never reach training,
feature construction, preprocessing, inference or threshold selection:

    Copernicus EMS / EMSR927, UNOSAT, other published damage maps,
    post-event OSM edits

Importing a validation-only source into any production path is defined as a
critical defect, not a stylistic issue. ``tests/test_data_boundary.py`` enforces
this mechanically.

Implementation status
---------------------
This release establishes the foundation only. ``floodmap.utils`` contains the
provenance schema and configuration loader. Every other subpackage is a
documented, deliberately empty namespace awaiting its own milestone — see
``README.md`` §14 for the implementation order.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
