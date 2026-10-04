"""Acquisition status and rejection vocabulary.

Why these are enums rather than strings or booleans
---------------------------------------------------
The milestone brief draws a hard line between six states that a naive
implementation collapses into "worked / didn't work":

``DISCOVERY SUCCESS``, ``DISCOVERY EMPTY``, ``DISCOVERY FAILED``,
``DOWNLOAD SUCCESS``, ``DOWNLOAD FAILED``, ``NOT ATTEMPTED``.

Those distinctions carry real scientific weight. "No suitable imagery exists
for this area and date" (:attr:`DiscoveryStatus.EMPTY`) is a finding about the
world that belongs in the final report as a limitation. "The catalogue API
timed out" (:attr:`DiscoveryStatus.FAILED`) is a finding about our
infrastructure and says nothing about data availability. Reporting the second
as the first would be a fabricated scientific claim.

Likewise :attr:`DownloadStatus.NOT_ATTEMPTED` is the honest default. Discovery
succeeding does not imply anything was downloaded, and a manifest must never
imply it holds data it never fetched.

:class:`RejectionReason` exists so that a discarded candidate carries *why* it
was discarded. "Best scene" with no reason recorded is exactly the opaque
decision the brief prohibits, and it is also the form that hides bugs: a
selection that silently drops every candidate with an unknown orbit looks
identical to one that found nothing.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "DiscoveryStatus",
    "DownloadStatus",
    "ManifestStatus",
    "RejectionReason",
    "SelectionStrategy",
    "SENSOR_NEUTRAL_REJECTIONS",
]


class DiscoveryStatus(str, Enum):
    """Outcome of a catalogue discovery query.

    ``EMPTY`` and ``FAILED`` are deliberately separate; see module docstring.
    """

    #: The query ran and returned at least one candidate.
    SUCCESS = "discovery_success"

    #: The query ran correctly and the provider holds nothing matching it.
    #: A statement about data availability, not about our code.
    EMPTY = "discovery_empty"

    #: The query could not be completed: timeout, auth, transport, bad response.
    #: Says nothing about whether suitable data exists.
    FAILED = "discovery_failed"

    #: Discovery was not run (for example, a dry run that only built the query).
    NOT_ATTEMPTED = "discovery_not_attempted"


class DownloadStatus(str, Enum):
    """Outcome of a product download.

    Defaults to :attr:`NOT_ATTEMPTED` everywhere. Downloading is an explicit,
    opt-in action: a successful discovery must never imply a fetch.
    """

    SUCCESS = "download_success"
    FAILED = "download_failed"
    NOT_ATTEMPTED = "download_not_attempted"


class ManifestStatus(str, Enum):
    """Overall outcome of one acquisition run.

    ``PARTIAL`` is needed because the two sensors fail independently: a
    monsoon event can plausibly yield a usable Sentinel-1 pair and no usable
    Sentinel-2 scene at all. Forcing that into a single success/failure flag
    would either overstate or discard a real result.
    """

    #: Both requested sensors produced a usable selection.
    COMPLETE = "complete"

    #: At least one sensor produced a usable selection and at least one did not.
    #: The manifest records which, and why.
    PARTIAL = "partial"

    #: No sensor produced a usable selection, but all queries ran correctly.
    #: A data-availability finding.
    NO_USABLE_DATA = "no_usable_data"

    #: One or more queries could not be completed. Not a data finding.
    FAILED = "failed"

    #: Query construction only; nothing was sent to the provider.
    DRY_RUN = "dry_run"


class SelectionStrategy(str, Enum):
    """How to choose among several qualifying candidates.

    ``docs/data-contract.md`` §1.2 records this as an open decision that must
    not default silently, so there is no default member: the caller or the
    configuration has to name one. Both members are fully deterministic.
    """

    #: Smallest absolute time difference from the event date.
    NEAREST_IN_TIME = "nearest_in_time"

    #: Lowest provider-reported scene cloud cover, then nearest in time.
    #: Sentinel-2 only — Sentinel-1 is cloud-independent, so requesting this
    #: for SAR is a configuration error rather than a no-op.
    LEAST_CLOUD_THEN_NEAREST = "least_cloud_then_nearest"


class RejectionReason(str, Enum):
    """Why a discovered candidate was not selected.

    Recorded per candidate so the selection is auditable after the fact
    without re-running it or reading the source.
    """

    # --- temporal -----------------------------------------------------------
    #: Acquired outside the configured before/after search window.
    OUTSIDE_SEARCH_WINDOW = "outside_search_window"

    #: Acquired on the wrong side of the event for the slot being filled.
    WRONG_SIDE_OF_EVENT = "wrong_side_of_event"

    #: Provider did not report an acquisition timestamp, so the scene cannot be
    #: placed relative to the event at all. Never assumed.
    ACQUISITION_TIME_UNKNOWN = "acquisition_time_unknown"

    # --- Sentinel-1 orbit geometry -----------------------------------------
    #: Provider did not report a relative orbit, so same-track compatibility
    #: cannot be *verified*. Rejected rather than assumed compatible.
    ORBIT_UNKNOWN = "orbit_unknown"

    #: Known relative orbit, but not the orbit of the pair being formed.
    ORBIT_MISMATCH = "orbit_mismatch"

    #: Same relative orbit but contradictory pass direction. Defensive: this
    #: should not occur in a consistent catalogue, and if it does the record is
    #: not trustworthy enough to pair.
    ORBIT_DIRECTION_MISMATCH = "orbit_direction_mismatch"

    # --- quality ------------------------------------------------------------
    #: Above the cloud limit set in configuration. Only ever applied when a
    #: limit was explicitly configured; no threshold is invented.
    CLOUD_ABOVE_CONFIGURED_LIMIT = "cloud_above_configured_limit"

    #: Cloud metadata was required by the configured ranking/filter but was not
    #: reported by the provider. Missing metadata is not treated as clear sky.
    CLOUD_UNKNOWN = "cloud_unknown"

    # --- ranking ------------------------------------------------------------
    #: Valid and compatible, but another candidate ranked higher under the
    #: configured strategy. Retained so the runner-up is visible.
    NOT_TOP_RANKED = "not_top_ranked"

    #: Valid, but its sensor slot was already filled by a higher-ranked pair.
    SUPERSEDED_BY_SELECTED_PAIR = "superseded_by_selected_pair"


#: Rejection reasons that apply to either sensor. Used by tests to assert that
#: sensor-specific reasons (the orbit ones) are not raised for Sentinel-2.
SENSOR_NEUTRAL_REJECTIONS: tuple[RejectionReason, ...] = (
    RejectionReason.OUTSIDE_SEARCH_WINDOW,
    RejectionReason.WRONG_SIDE_OF_EVENT,
    RejectionReason.ACQUISITION_TIME_UNKNOWN,
    RejectionReason.NOT_TOP_RANKED,
    RejectionReason.SUPERSEDED_BY_SELECTED_PAIR,
)
