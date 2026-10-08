"""Errors raised by the M4 dataset discovery and adapter layer."""

from __future__ import annotations


class DatasetAdapterError(ValueError):
    """Base class for an invalid dataset root, index or sample."""


class DatasetDiscoveryError(DatasetAdapterError):
    """The external dataset cannot be indexed under its published contract."""


class DatasetIntegrityError(DatasetAdapterError):
    """A discovered sample fails a content or alignment integrity check."""
