"""Explicit failures for the analysis-ready preprocessing boundary."""

from __future__ import annotations

__all__ = [
    "PreprocessingError",
    "PreprocessingConfigurationError",
    "PreprocessingDependencyError",
    "InputValidationError",
    "AlignmentError",
    "SourceProductError",
    "UnsupportedPreprocessingError",
]


class PreprocessingError(Exception):
    """Base class for failures that must not produce a successful artifact."""


class PreprocessingConfigurationError(PreprocessingError):
    """A scientific preprocessing choice remains unset or contradictory."""


class PreprocessingDependencyError(PreprocessingError):
    """The configured geospatial operation requires an unavailable library."""


class InputValidationError(PreprocessingError):
    """A source raster lacks metadata or content required by the contract."""


class AlignmentError(PreprocessingError):
    """Two grids failed an explicit spatial alignment check."""


class SourceProductError(PreprocessingError):
    """A path is not one of the selected products in the M1 manifest."""


class UnsupportedPreprocessingError(PreprocessingError):
    """The available local stack cannot honestly perform a requested operation."""
