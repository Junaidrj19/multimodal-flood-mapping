"""Acquisition failure types.

Why this module exists
----------------------
``architecture.md`` §17 requires the pipeline to distinguish failure modes
rather than "silently producing a misleading map". A single generic exception
would collapse "the provider timed out", "our credentials are wrong" and "no
suitable imagery exists" into one indistinguishable event, and those three
demand completely different responses from an operator.

The hierarchy below is therefore shaped by *what the caller should do*, not by
where in the code the error happened:

* :class:`ProviderTimeout`, :class:`ProviderHttpError` — transient or
  provider-side; retrying may help.
* :class:`ProviderAuthError` — a credential/configuration problem; retrying
  will not help.
* :class:`ProviderMalformedResponse` — the provider answered, but not in the
  contract we parse. Never guess the missing pieces.
* :class:`AoiNotConfigured`, :class:`SearchWindowNotConfigured`,
  :class:`SelectionStrategyNotConfigured` — the *caller* has not supplied a
  scientific decision that must not be defaulted.
* :class:`NoSameTrackPairError` — a genuine scientific outcome, not a bug.

A note on what is deliberately NOT here
---------------------------------------
There is no "fall back to something reasonable" path. Every error in this
module is raised instead of substituting a plausible value, because a
fabricated scene identifier or an invented orbit number is far more damaging
than a failed run (see the no-fake-success rule in the milestone brief and
``AGENTS.md`` §11).
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "AcquisitionError",
    "AoiNotConfigured",
    "ConfigurationIncomplete",
    "NoSameTrackPairError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderHttpError",
    "ProviderMalformedResponse",
    "ProviderTimeout",
    "SearchWindowNotConfigured",
    "SelectionError",
    "SelectionStrategyNotConfigured",
]


class AcquisitionError(Exception):
    """Base class for every acquisition failure."""


# ---------------------------------------------------------------------------
# Configuration problems: a required scientific decision is missing
# ---------------------------------------------------------------------------


class ConfigurationIncomplete(AcquisitionError):
    """A required configuration value is absent or still a TODO placeholder.

    Raised rather than defaulted. The configuration files intentionally ship
    with ``null`` for values that are scientific decisions (search windows,
    cloud thresholds, selection strategy). Substituting a default here would
    make an unmade decision look like a made one.
    """


class AoiNotConfigured(ConfigurationIncomplete):
    """No area of interest was supplied.

    The AOI geometry for this project is deliberately not fixed anywhere in
    the repository. It must come from the caller or from configuration, and
    there is no built-in geometry to fall back on.
    """


class SearchWindowNotConfigured(ConfigurationIncomplete):
    """No temporal search window was supplied for a sensor.

    The correct window depends on actual satellite revisit availability around
    the event, which has not been established. See ``docs/data-contract.md``
    §1.2.
    """


class SelectionStrategyNotConfigured(ConfigurationIncomplete):
    """No candidate-selection strategy was supplied.

    ``docs/data-contract.md`` §1.2 records the behaviour when several
    candidates qualify as an open decision that "must not default silently".
    This enforces that.
    """


# ---------------------------------------------------------------------------
# Provider problems
# ---------------------------------------------------------------------------


class ProviderError(AcquisitionError):
    """Base class for failures originating at the data provider."""

    def __init__(self, message: str, *, provider: Optional[str] = None) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)


class ProviderTimeout(ProviderError):
    """The provider did not respond within the configured timeout.

    Observed in practice against the real catalogue: an insufficiently
    constrained Sentinel-2 query times out rather than returning slowly, so
    this is an expected operational condition and not an exotic edge case.
    """


class ProviderAuthError(ProviderError):
    """Authentication or authorisation failed (HTTP 401/403).

    Separated from :class:`ProviderHttpError` because retrying is pointless
    and the remedy is operator action on credentials.
    """


class ProviderHttpError(ProviderError):
    """The provider returned an unsuccessful HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        detail = f"HTTP {status_code}: {message}" if status_code is not None else message
        super().__init__(detail, provider=provider)


class ProviderMalformedResponse(ProviderError):
    """The response could not be parsed, or omitted a field we require.

    Raised instead of filling the gap. A scene record missing its acquisition
    timestamp or identifier cannot be repaired by inference, and a guessed
    value would propagate silently into provenance.
    """


# ---------------------------------------------------------------------------
# Selection outcomes that are errors only under a strict policy
# ---------------------------------------------------------------------------


class SelectionError(AcquisitionError):
    """Base class for pair-selection failures."""


class NoSameTrackPairError(SelectionError):
    """No before/after Sentinel-1 pair shares a relative orbit.

    This is a scientific result, not a defect. ``AGENTS.md`` §4 warns that
    different tracks view terrain from different angles and must not be
    compared pixel-by-pixel as equivalent observations, so the absence of a
    same-track pair means the requested change-detection comparison cannot be
    made from this data.

    Whether this is raised or merely recorded is governed by
    ``acquisition.sentinel1.on_no_same_track_pair``. It is never resolved by
    quietly pairing scenes from different tracks.
    """
