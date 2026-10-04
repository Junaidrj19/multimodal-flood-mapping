"""Explicit failures for the feature-generation boundary.

M3 fails rather than producing a feature whose scientific meaning is not
established. Each class below marks a distinct reason so a caller can tell an
unresolved configuration apart from an input that genuinely cannot support the
requested feature.
"""

from __future__ import annotations

__all__ = [
    "FeatureError",
    "FeatureConfigurationError",
    "FeatureDependencyError",
    "FeatureInputError",
    "FeatureRegistryError",
    "FeatureGridError",
    "FeatureBoundaryError",
    "UnsupportedFeatureError",
]


class FeatureError(Exception):
    """Base class for failures that must not produce a feature artifact."""


class FeatureConfigurationError(FeatureError):
    """A scientific feature parameter is unset, contradictory or invented."""


class FeatureDependencyError(FeatureError):
    """A required library for raster feature IO is unavailable."""


class FeatureInputError(FeatureError):
    """An M2 analysis-ready input is absent, malformed or lacks required metadata."""


class FeatureRegistryError(FeatureError):
    """A requested feature is unknown, or its registry metadata is incomplete."""


class FeatureGridError(FeatureError):
    """Two M2 inputs do not share the expected analysis grid.

    M3 never resamples. Spatial normalisation is M2's responsibility, so an
    unaligned pair is an explicit failure here.
    """


class FeatureBoundaryError(FeatureError):
    """A validation-only source was offered as a feature-generation input.

    ``AGENTS.md`` §3 defines this as a critical defect rather than a
    recoverable condition.
    """


class UnsupportedFeatureError(FeatureError):
    """The available inputs cannot honestly support the requested transform."""
