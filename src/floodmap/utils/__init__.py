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
contract
    Typed, ``extra="forbid"`` schema for the frozen M4 dataset contract in
    ``configs/data.yaml -> m4_contract``, plus the per-dataset capability model.
    A configuration schema only: no model, adapter, training or inference code
    (docs/m4-architecture-decision.md §10).
"""
