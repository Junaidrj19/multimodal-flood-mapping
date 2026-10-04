"""Shared utilities: provenance schema and configuration loading.

This is the only subpackage with executable logic in the foundation milestone.

Modules
-------
provenance
    Explicit, serializable provenance records for generated artifacts, plus a
    closed enumeration separating permitted production inputs from
    validation-only sources (architecture.md §12, §18).
config
    Minimal YAML configuration loader, so experimental parameters live in
    ``configs/`` rather than in source code (AGENTS.md §14).
"""
