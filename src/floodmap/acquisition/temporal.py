"""Event specification and temporal search windows.

Why the event day is excluded by default
----------------------------------------
The locked project fact is a *date*, 2026-08-26, not an instant. A Sentinel
scene acquired on that date may have been captured hours before the flood or
hours after it. Classifying it as "before" because its date is not greater
than the event date would be a guess presented as a fact, and it is the kind
of guess that inverts a change-detection result: a post-flood scene used as
the pre-event reference makes flooding appear as drying.

So when only a date is known, the whole event day sits in neither window and
is reported as :attr:`TemporalPlan.event_day_excluded`. The exclusion is
recorded as a provenance limitation, because it also costs us the acquisition
closest to the event.

A caller who knows the event time can supply ``event_time_utc``, and the
windows then split at that instant instead, with the ambiguity gone. That is
the only way to recover same-day scenes, and it requires real information
rather than an assumption.

``AGENTS.md`` §4 also forbids silently mixing temporal windows, so the before
and after intervals are explicit, half-open and non-overlapping by
construction.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import SearchWindowNotConfigured

__all__ = [
    "EventSpec",
    "SearchWindow",
    "TemporalPlan",
    "TimeInterval",
    "parse_acquired_at",
]

#: Upper bound on a search window. Not a scientific limit: a window of years
#: would pull in seasonal and land-cover change that is indistinguishable from
#: flood signal in a pre/post comparison, so an obviously wrong value is
#: rejected rather than silently honoured.
_MAX_WINDOW_DAYS = 365


def parse_acquired_at(raw: str) -> datetime:
    """Parse a provider acquisition timestamp into a timezone-aware UTC datetime.

    Accepts the trailing ``Z`` form the catalogue emits
    (``2024-03-04T12:22:36.716570Z``), which :func:`datetime.fromisoformat`
    rejects before Python 3.11.

    A naive timestamp is rejected rather than assumed to be UTC. Silently
    attaching a timezone to an ambiguous value can move a scene across the
    event boundary, which is exactly the error this module exists to prevent.
    """
    text = raw.strip()
    if not text:
        raise ValueError("acquisition timestamp is empty")

    normalised = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValueError(f"unparseable acquisition timestamp {raw!r}: {exc}") from exc

    if parsed.tzinfo is None:
        raise ValueError(
            f"acquisition timestamp {raw!r} has no timezone. It is not assumed to be "
            "UTC, because a wrong offset can move a scene to the other side of the "
            "event date (AGENTS.md §4)."
        )
    return parsed.astimezone(timezone.utc)


class TimeInterval(BaseModel):
    """A half-open UTC interval, ``[start, end)``.

    Half-open so that adjacent intervals cannot both claim a boundary instant.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _ordered_and_aware(self) -> "TimeInterval":
        for label, value in (("start", self.start), ("end", self.end)):
            if value.tzinfo is None:
                raise ValueError(f"{label} must be timezone-aware")
        if self.end <= self.start:
            raise ValueError(f"end must be after start (got start={self.start}, end={self.end})")
        return self

    def contains(self, moment: datetime) -> bool:
        """True when ``moment`` falls in ``[start, end)``."""
        if moment.tzinfo is None:
            raise ValueError("cannot test a naive datetime against a UTC interval")
        aware = moment.astimezone(timezone.utc)
        return self.start <= aware < self.end

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def to_dict(self) -> Dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


class SearchWindow(BaseModel):
    """How far either side of the event to search, in whole days.

    Both bounds are required. There is no default, because the correct window
    depends on actual satellite revisit availability around the event, which
    has not been established (``docs/data-contract.md`` §1.2). A default would
    present an unmade decision as a made one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    before_days: int = Field(description="Days before the event to search. Must be >= 1.")
    after_days: int = Field(description="Days after the event to search. Must be >= 1.")

    @field_validator("before_days", "after_days")
    @classmethod
    def _positive_and_bounded(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"search window must be at least 1 day, got {v}. A zero or negative "
                "window cannot contain an acquisition."
            )
        if v > _MAX_WINDOW_DAYS:
            raise ValueError(
                f"search window of {v} days exceeds the {_MAX_WINDOW_DAYS}-day limit. "
                "Over such a span, seasonal and land-cover change becomes "
                "indistinguishable from flood signal in a pre/post comparison."
            )
        return v

    @classmethod
    def from_config(
        cls,
        before_days: Optional[int],
        after_days: Optional[int],
        *,
        sensor: str,
    ) -> "SearchWindow":
        """Build from configuration values, failing loudly on unset ones.

        Raises
        ------
        SearchWindowNotConfigured
            When either bound is ``None``. The config files ship these as
            ``null`` TODOs on purpose.
        """
        missing = [
            name
            for name, value in (
                ("window_before_days", before_days),
                ("window_after_days", after_days),
            )
            if value is None
        ]
        if missing:
            raise SearchWindowNotConfigured(
                f"acquisition.{sensor}.search_window is incomplete: {', '.join(missing)} "
                f"is null. The correct window depends on actual {sensor} revisit "
                "availability around the event and must be set deliberately, not "
                "defaulted (docs/data-contract.md §1.2)."
            )
        return cls(before_days=int(before_days), after_days=int(after_days))


class EventSpec(BaseModel):
    """The event being analysed.

    ``event_time_utc`` is optional and genuinely changes the semantics: with it
    the windows split at an instant and same-day scenes are usable; without it
    the event day is excluded from both windows. See module docstring.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_date: date = Field(description="Date of the event, ISO 8601.")
    event_time_utc: Optional[datetime] = Field(
        default=None,
        description=(
            "Exact event instant in UTC, when known. Supplying it allows "
            "same-day acquisitions to be classified; omitting it excludes the "
            "whole event day from both search windows."
        ),
    )
    label: str = Field(default="", description="Optional human-readable event name.")

    @field_validator("event_time_utc")
    @classmethod
    def _event_time_is_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and v.tzinfo is None:
            raise ValueError(
                "event_time_utc must be timezone-aware; a naive instant is not " "assumed to be UTC"
            )
        return v.astimezone(timezone.utc) if v is not None else None

    @model_validator(mode="after")
    def _event_time_matches_event_date(self) -> "EventSpec":
        """An event time on a different day than event_date is contradictory."""
        if self.event_time_utc is not None:
            if self.event_time_utc.date() != self.event_date:
                raise ValueError(
                    f"event_time_utc ({self.event_time_utc.isoformat()}) falls on "
                    f"{self.event_time_utc.date()}, which contradicts event_date "
                    f"({self.event_date}). One of the two is wrong."
                )
        return self

    @property
    def day_start(self) -> datetime:
        """Midnight UTC at the start of the event day."""
        return datetime.combine(self.event_date, time.min, tzinfo=timezone.utc)

    @property
    def day_end(self) -> datetime:
        """Midnight UTC at the start of the day after the event."""
        return self.day_start + timedelta(days=1)

    @property
    def split_instant(self) -> datetime:
        """The instant separating before from after.

        The known event time when available, otherwise the start of the event
        day, with the remainder of that day excluded by :meth:`plan`.
        """
        return self.event_time_utc or self.day_start

    def plan(self, window: SearchWindow) -> "TemporalPlan":
        """Resolve this event and window into explicit search intervals."""
        return TemporalPlan.build(self, window)


class TemporalPlan(BaseModel):
    """Resolved before/after search intervals for one sensor.

    Constructed via :meth:`build` so the intervals and the event-day exclusion
    are derived in one place rather than recomputed at each call site.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: EventSpec
    window: SearchWindow
    before: TimeInterval
    after: TimeInterval
    event_day_excluded: bool = Field(
        description=(
            "True when the event time is unknown and the whole event day "
            "therefore belongs to neither window."
        )
    )

    @classmethod
    def build(cls, event: EventSpec, window: SearchWindow) -> "TemporalPlan":
        if event.event_time_utc is not None:
            # Exact instant known: windows meet at it, nothing is discarded.
            before_end = event.event_time_utc
            after_start = event.event_time_utc
            excluded = False
        else:
            # Only the date is known. The event day is ambiguous, so it is
            # excluded from both sides rather than guessed into one.
            before_end = event.day_start
            after_start = event.day_end
            excluded = True

        return cls(
            event=event,
            window=window,
            before=TimeInterval(
                start=before_end - timedelta(days=window.before_days),
                end=before_end,
            ),
            after=TimeInterval(
                start=after_start,
                end=after_start + timedelta(days=window.after_days),
            ),
            event_day_excluded=excluded,
        )

    @model_validator(mode="after")
    def _windows_do_not_overlap(self) -> "TemporalPlan":
        """Guard against the two windows sharing any instant.

        ``AGENTS.md`` §4 forbids silently mixing temporal windows. An overlap
        would let one scene qualify as both the before and the after
        observation, which would make a change-detection result meaningless.
        """
        if self.before.end > self.after.start:
            raise ValueError(
                f"before window ends at {self.before.end.isoformat()}, after the "
                f"after window starts at {self.after.start.isoformat()}; overlapping "
                "windows would let a single scene serve as both observations"
            )
        return self

    @property
    def full_extent(self) -> TimeInterval:
        """Single interval spanning both windows, for one provider query.

        The provider is queried once over the whole span and the two sides are
        separated locally. This halves the request count and, more importantly,
        means both sides are partitioned by the same code path rather than by
        two independently constructed queries that could drift apart.
        """
        return TimeInterval(start=self.before.start, end=self.after.end)

    def classify(self, acquired_at: datetime) -> Optional[str]:
        """Return ``"before"``, ``"after"``, or ``None`` if in neither window."""
        if self.before.contains(acquired_at):
            return "before"
        if self.after.contains(acquired_at):
            return "after"
        return None

    def offset_from_event(self, acquired_at: datetime) -> timedelta:
        """Absolute distance from the event split instant.

        The ranking key for nearest-in-time selection. Absolute value, so a
        scene two days before and one two days after are equidistant; ties are
        broken by the deterministic secondary keys in ``selection``.
        """
        aware = acquired_at.astimezone(timezone.utc)
        return abs(aware - self.event.split_instant)

    def limitations(self) -> Tuple[str, ...]:
        """Limitations this plan imposes, for the provenance record."""
        notes: list[str] = []
        if self.event_day_excluded:
            notes.append(
                f"The exact time of the {self.event.event_date.isoformat()} event is "
                "unknown, so acquisitions on the event day were excluded from both "
                "search windows. A same-day scene cannot be classified as pre- or "
                "post-event without guessing, and this exclusion discards the "
                "acquisitions closest in time to the event."
            )
        notes.append(
            f"Search windows were {self.window.before_days} day(s) before and "
            f"{self.window.after_days} day(s) after the event. A scene outside these "
            "bounds was not considered, regardless of suitability."
        )
        return tuple(notes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_date": self.event.event_date.isoformat(),
            "event_time_utc": (
                self.event.event_time_utc.isoformat() if self.event.event_time_utc else None
            ),
            "event_label": self.event.label,
            "window_before_days": self.window.before_days,
            "window_after_days": self.window.after_days,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "event_day_excluded": self.event_day_excluded,
        }
