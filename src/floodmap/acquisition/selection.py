"""Deterministic, auditable selection of pre/post scene candidates.

What changed in M6.1 and why
----------------------------
Selection is now a *validation* stage, not only a ranking stage. Before a pair
may be reported ``selected``, every candidate must clear:

1. the temporal window (unchanged);
2. the configured cloud limit and cloud-metadata requirement (unchanged);
3. the processing-level tier, when a preference is configured (unchanged);
4. **AOI footprint coverage**, measured geometrically
   (:mod:`floodmap.acquisition.coverage`) — previously a constant ``True``;
5. for Sentinel-1, the configured **required polarisations** — previously not
   checked at all;
6. for Sentinel-2, **pair compatibility**: MGRS tile agreement and a recorded
   decision about mixed processing levels — previously a prose note.

:attr:`SelectionOutcome.is_usable` is derived from the recorded validation
result rather than from the mere presence of two scene references. A gate that
fails therefore cannot be reported as usable, which is what makes the flag safe
to use as a download authorisation downstream.

Every discarded candidate carries a reason. Two filters that previously dropped
candidates silently — the processing-level tier filter and the footprint filter
— now record ``PROCESSING_LEVEL_EXCLUDED`` and a footprint reason respectively,
because a selection that quietly discards candidates is indistinguishable from
one that found nothing.

Selection is independent per temporal side and then reconciled as a pair: the
best individual scene on each side is not automatically the best *pair*, so
pairs are ranked by their worse side first and incompatible pairs are excluded
or recorded before a winner is chosen.

No AOI, no selection
--------------------
``aoi`` is a required keyword argument on both selectors. Coverage is a
mandatory gate, and a gate that cannot be evaluated must not silently pass, so
there is deliberately no "AOI not supplied" path that returns a usable pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from .aoi import AreaOfInterest
from .coverage import FootprintCoverage, evaluate_footprint_coverage
from .outcomes import RejectionReason, SelectionStrategy
from .scenes import DiscoveredScene, Sentinel1Scene, Sentinel2Scene, SensorKind
from .temporal import TemporalPlan
from ..utils.provenance import Unknown

__all__ = [
    "CandidateRejection",
    "NoSameTrackPair",
    "RejectedCandidate",
    "SelectionOutcome",
    "SelectionResult",
    "normalise_polarisations",
    "select_sentinel1",
    "select_sentinel1_pair",
    "select_sentinel2",
    "select_sentinel2_pair",
]


@dataclass(frozen=True)
class CandidateRejection:
    """One discarded candidate and the reason it was not selected."""

    scene_id: str
    sensor: SensorKind
    reason: RejectionReason
    rationale: str
    side: Optional[str] = None
    #: Measured AOI coverage at the moment of rejection, when it was measured.
    coverage_fraction: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "sensor": self.sensor.value,
            "reason": self.reason.value,
            "rationale": self.rationale,
            "side": self.side,
            "coverage_fraction": (
                None if self.coverage_fraction is None else round(self.coverage_fraction, 6)
            ),
        }


def _dedupe(rejections: Iterable[CandidateRejection]) -> Tuple[CandidateRejection, ...]:
    """Collapse repeated rejections of the same candidate for the same reason.

    Enumerating candidate *pairs* can reject one candidate many times over. The
    record should say why it was discarded, not how many pairings it appeared in.
    """
    seen: dict[tuple[str, str, Optional[str]], CandidateRejection] = {}
    for rejection in rejections:
        seen.setdefault(
            (rejection.scene_id, rejection.reason.value, rejection.side), rejection
        )
    return tuple(seen.values())


@dataclass(frozen=True)
class SelectionOutcome:
    """Selection result with both the decision and its discarded candidates."""

    sensor: SensorKind
    status: str
    selected_before: Optional[DiscoveredScene] = None
    selected_after: Optional[DiscoveredScene] = None
    rationale: str = ""
    rejected_candidates: Tuple[CandidateRejection, ...] = field(default_factory=tuple)
    # Tile consistency information for pre/post pair (Sentinel-2 only)
    tile_consistency: Optional[Dict[str, Any]] = None
    # Whether an L1C product was selected as a fallback when L2A was preferred
    fallback_used: bool = False
    # Per-side AOI coverage verdicts, recorded so the decision is auditable.
    coverage: Optional[Dict[str, Any]] = None
    # The gates applied and whether every one of them passed.
    pair_validation: Optional[Dict[str, Any]] = None
    # The inputs that produced this selection, so the run can be reproduced.
    selection_policy: Optional[Dict[str, Any]] = None

    @property
    def validation_passed(self) -> bool:
        """True only when a recorded validation says every gate passed."""
        return bool(self.pair_validation and self.pair_validation.get("passed"))

    @property
    def is_usable(self) -> bool:
        """Whether this outcome authorises using (and downloading) the pair.

        Deliberately *not* merely "two scenes are present": a pair that failed a
        validation gate must never be reported usable, so this reads the
        recorded validation rather than inferring success from structure.
        """
        return (
            self.status == "selected"
            and self.selected_before is not None
            and self.selected_after is not None
            and self.validation_passed
        )

    @property
    def pair(self) -> Optional[tuple[DiscoveredScene, DiscoveredScene]]:
        if not self.is_usable:
            return None
        # The property is safe after is_usable; the explicit branch keeps type
        # checkers and callers from seeing an optional pair.
        return (self.selected_before, self.selected_after)  # type: ignore[return-value]

    @property
    def selected_pair(self) -> Optional[tuple[DiscoveredScene, DiscoveredScene]]:
        """Descriptive alias used by manifest/report consumers."""
        return self.pair

    @property
    def rejections(self) -> Tuple[CandidateRejection, ...]:
        """Short alias for the structured discarded-candidate list."""
        return self.rejected_candidates

    @property
    def reasons(self) -> Tuple[RejectionReason, ...]:
        """Distinct rejection reasons, for compact assertions and reporting."""
        return tuple(dict.fromkeys(item.reason for item in self.rejected_candidates))

    def to_dict(self) -> dict:
        result = {
            "sensor": self.sensor.value,
            "status": self.status,
            "selected_before": (
                self.selected_before.to_dict() if self.selected_before is not None else None
            ),
            "selected_after": (
                self.selected_after.to_dict() if self.selected_after is not None else None
            ),
            "rationale": self.rationale,
            "rejected_candidates": [item.to_dict() for item in self.rejected_candidates],
            "is_usable": self.is_usable,
        }
        if self.tile_consistency is not None:
            result["tile_consistency"] = self.tile_consistency
        if self.fallback_used:
            result["fallback_used"] = True
        if self.coverage is not None:
            result["coverage"] = self.coverage
        if self.pair_validation is not None:
            result["pair_validation"] = self.pair_validation
        if self.selection_policy is not None:
            result["selection_policy"] = self.selection_policy
        return result


@dataclass(frozen=True)
class NoSameTrackPair(SelectionOutcome):
    """Explicit Sentinel-1 outcome when no compatible relative-orbit pair exists."""

    status: str = field(default="no_same_track_pair", init=False)


def _side_candidates(
    scenes: Sequence[DiscoveredScene], plan: TemporalPlan, sensor: SensorKind
) -> tuple[list[DiscoveredScene], list[DiscoveredScene], list[CandidateRejection]]:
    before: list[DiscoveredScene] = []
    after: list[DiscoveredScene] = []
    rejected: list[CandidateRejection] = []
    for scene in scenes:
        side = plan.classify(scene.acquired_at)
        if side == "before":
            before.append(scene)
        elif side == "after":
            after.append(scene)
        else:
            rejected.append(
                CandidateRejection(
                    scene_id=scene.scene_id,
                    sensor=sensor,
                    reason=RejectionReason.OUTSIDE_SEARCH_WINDOW,
                    rationale=(
                        f"Acquired at {scene.acquired_at.isoformat()}, which is outside "
                        "the configured before/after intervals."
                    ),
                )
            )
    return before, after, rejected


def _nearest_key(scene: DiscoveredScene, plan: TemporalPlan) -> tuple:
    return (plan.offset_from_event(scene.acquired_at), scene.acquired_at, scene.scene_id)


def _unknown_orbit_rejection(scene: Sentinel1Scene, side: str) -> CandidateRejection:
    return CandidateRejection(
        scene_id=scene.scene_id,
        sensor=SensorKind.SENTINEL1,
        reason=RejectionReason.ORBIT_UNKNOWN,
        side=side,
        rationale=(
            "Relative orbit was not reported by the provider; same-track "
            "compatibility cannot be verified, so the candidate was rejected."
        ),
    )


def normalise_polarisations(values: Iterable[str] | None) -> Tuple[str, ...]:
    """Upper-case and de-duplicate a required-polarisation set.

    Kept public because the CLI and the configuration both need it, and because
    ``"vv"`` and ``"VV"`` must not be treated as different requirements.
    """
    if not values:
        return ()
    return tuple(
        dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip())
    )


def _coverage_rejection(
    coverage: FootprintCoverage, scene: DiscoveredScene, side: str
) -> CandidateRejection:
    assert coverage.reason is not None  # only called for a rejection
    return CandidateRejection(
        scene_id=scene.scene_id,
        sensor=scene.sensor,
        reason=coverage.reason,
        side=side,
        rationale=coverage.detail,
        coverage_fraction=coverage.coverage_fraction,
    )


def _polarisation_verdict(
    scene: Sentinel1Scene, required: Tuple[str, ...]
) -> Optional[CandidateRejection]:
    """Return a rejection when the required polarisation set is not satisfied."""
    if not required:
        return None
    reported = scene.polarisations
    if isinstance(reported, Unknown) or not reported:
        return CandidateRejection(
            scene_id=scene.scene_id,
            sensor=SensorKind.SENTINEL1,
            reason=RejectionReason.POLARISATION_UNVERIFIABLE,
            rationale=(
                "The provider did not report polarisation channels, so the required set "
                f"{list(required)} cannot be verified. A single-channel product would "
                "silently leave a model input empty."
            ),
        )
    reported_set = normalise_polarisations(reported)
    missing = [channel for channel in required if channel not in reported_set]
    if missing:
        return CandidateRejection(
            scene_id=scene.scene_id,
            sensor=SensorKind.SENTINEL1,
            reason=RejectionReason.POLARISATION_MISMATCH,
            rationale=(
                f"Reported polarisations {list(reported_set)} do not contain the required "
                f"{missing}; the configured requirement is {list(required)}."
            ),
        )
    return None


def _s1_pair_key(before: Sentinel1Scene, after: Sentinel1Scene, plan: TemporalPlan) -> tuple:
    return (
        max(plan.offset_from_event(before.acquired_at), plan.offset_from_event(after.acquired_at)),
        plan.offset_from_event(before.acquired_at) + plan.offset_from_event(after.acquired_at),
        before.acquired_at,
        after.acquired_at,
        before.scene_id,
        after.scene_id,
    )


def select_sentinel1_pair(
    scenes: Sequence[Sentinel1Scene],
    plan: TemporalPlan,
    *,
    aoi: AreaOfInterest,
    minimum_aoi_coverage: Optional[float] = None,
    required_polarisations: Iterable[str] | None = None,
    require_same_relative_orbit: bool = True,
) -> SelectionOutcome:
    """Select a same-relative-orbit Sentinel-1 pair that covers the AOI.

    Cross-track substitution is deliberately unsupported. ``NoSameTrackPair`` is
    returned when the candidate pool cannot satisfy the geometry rule.

    Parameters
    ----------
    aoi
        Required. Every candidate must be shown to cover it; there is no path
        that selects a pair without measuring coverage.
    minimum_aoi_coverage
        Optional required coverage fraction in ``(0, 1]``. ``None`` requires
        intersection and records the fraction without thresholding it.
    required_polarisations
        Channels that must be present, e.g. ``["VV", "VH"]``. Empty means the
        polarisation set is recorded but not required — an explicit choice, not
        a default that assumes dual polarisation.
    """
    if not isinstance(aoi, AreaOfInterest):
        raise TypeError(
            f"aoi must be an AreaOfInterest so coverage can be measured, got {type(aoi).__name__}"
        )
    if not require_same_relative_orbit:
        raise ValueError(
            "Sentinel-1 selection requires same relative orbit; cross-track "
            "fallback would make terrain geometry look like change"
        )

    required = normalise_polarisations(required_polarisations)

    before_raw, after_raw, rejected = _side_candidates(scenes, plan, SensorKind.SENTINEL1)
    before: list[Sentinel1Scene] = []
    after: list[Sentinel1Scene] = []
    coverage_records: Dict[str, Any] = {}
    for side, candidates, target in (
        ("before", before_raw, before),
        ("after", after_raw, after),
    ):
        for candidate in candidates:
            if not isinstance(candidate, Sentinel1Scene):
                raise TypeError(f"expected Sentinel1Scene, got {type(candidate).__name__}")
            if not isinstance(candidate.relative_orbit, int):
                rejected.append(_unknown_orbit_rejection(candidate, side))
                continue
            polarisation_rejection = _polarisation_verdict(candidate, required)
            if polarisation_rejection is not None:
                rejected.append(
                    CandidateRejection(
                        scene_id=polarisation_rejection.scene_id,
                        sensor=polarisation_rejection.sensor,
                        reason=polarisation_rejection.reason,
                        rationale=polarisation_rejection.rationale,
                        side=side,
                    )
                )
                continue
            coverage = evaluate_footprint_coverage(
                candidate.footprint, aoi, minimum_coverage=minimum_aoi_coverage
            )
            coverage_records[candidate.scene_id] = coverage.to_dict()
            if not coverage.accepted:
                rejected.append(_coverage_rejection(coverage, candidate, side))
                continue
            target.append(candidate)

    pairs: list[tuple[Sentinel1Scene, Sentinel1Scene]] = []
    for candidate_before in before:
        for candidate_after in after:
            if candidate_before.relative_orbit != candidate_after.relative_orbit:
                continue
            if (
                isinstance(candidate_before.orbit_direction, str)
                and isinstance(candidate_after.orbit_direction, str)
                and candidate_before.orbit_direction != candidate_after.orbit_direction
            ):
                rejected.extend(
                    [
                        CandidateRejection(
                            scene_id=candidate_before.scene_id,
                            sensor=SensorKind.SENTINEL1,
                            reason=RejectionReason.ORBIT_DIRECTION_MISMATCH,
                            side="before",
                            rationale=(
                                "The matching relative orbit number has contradictory "
                                "reported pass directions."
                            ),
                        ),
                        CandidateRejection(
                            scene_id=candidate_after.scene_id,
                            sensor=SensorKind.SENTINEL1,
                            reason=RejectionReason.ORBIT_DIRECTION_MISMATCH,
                            side="after",
                            rationale=(
                                "The matching relative orbit number has contradictory "
                                "reported pass directions."
                            ),
                        ),
                    ]
                )
                continue
            pairs.append((candidate_before, candidate_after))

    policy = {
        "sensor": SensorKind.SENTINEL1.value,
        "aoi_id": aoi.aoi_id,
        "aoi_wkt": aoi.to_wkt(),
        "aoi_crs": aoi.crs,
        "minimum_aoi_coverage": minimum_aoi_coverage,
        "required_polarisations": list(required),
        "require_same_relative_orbit": require_same_relative_orbit,
        "strategy": "nearest_in_time",
        "window_before_days": plan.window.before_days,
        "window_after_days": plan.window.after_days,
        "event_date": plan.event.event_date.isoformat(),
        "event_day_excluded": plan.event_day_excluded,
    }

    if not pairs:
        for candidate in before + after:
            has_same_orbit_on_other_side = any(
                candidate.relative_orbit == other.relative_orbit
                for other in (after if candidate in before else before)
            )
            if not has_same_orbit_on_other_side:
                rejected.append(
                    CandidateRejection(
                        scene_id=candidate.scene_id,
                        sensor=SensorKind.SENTINEL1,
                        reason=RejectionReason.ORBIT_MISMATCH,
                        side="before" if candidate in before else "after",
                        rationale=(
                            f"Relative orbit {candidate.relative_orbit} has no candidate "
                            "on the opposite side of the event."
                        ),
                    )
                )
        return NoSameTrackPair(
            sensor=SensorKind.SENTINEL1,
            rationale=(
                "No before/after Sentinel-1 candidates share a verified relative "
                "orbit with compatible pass direction. Cross-track scenes were not "
                "substituted."
            ),
            rejected_candidates=_dedupe(rejected),
            coverage=coverage_records or None,
            selection_policy=policy,
        )

    selected_before, selected_after = min(
        pairs, key=lambda pair: _s1_pair_key(pair[0], pair[1], plan)
    )
    selected_ids = {selected_before.scene_id, selected_after.scene_id}
    for candidate in before + after:
        if candidate.scene_id not in selected_ids:
            rejected.append(
                CandidateRejection(
                    scene_id=candidate.scene_id,
                    sensor=SensorKind.SENTINEL1,
                    reason=RejectionReason.NOT_TOP_RANKED,
                    side="before" if candidate in before else "after",
                    rationale=(
                        "A compatible same-track pair was selected using nearest-in-time "
                        "pair ranking; this candidate was a deterministic runner-up."
                    ),
                )
            )

    pair_validation = {
        "passed": True,
        "gates": {
            "temporal_window": "passed",
            "relative_orbit_match": "passed",
            "orbit_direction_compatible": "passed",
            "required_polarisations": (
                "not_required" if not required else "passed"
            ),
            "aoi_coverage": "not_thresholded" if minimum_aoi_coverage is None else "passed",
        },
        "required_polarisations": list(required),
        "reported_polarisations": {
            "before": list(normalise_polarisations(selected_before.polarisations))
            if not isinstance(selected_before.polarisations, Unknown)
            else "unknown",
            "after": list(normalise_polarisations(selected_after.polarisations))
            if not isinstance(selected_after.polarisations, Unknown)
            else "unknown",
        },
        "relative_orbit": selected_before.relative_orbit,
        "orbit_direction": selected_before.orbit_direction,
        "coverage": {
            "before": coverage_records.get(selected_before.scene_id),
            "after": coverage_records.get(selected_after.scene_id),
        },
    }

    return SelectionOutcome(
        sensor=SensorKind.SENTINEL1,
        status="selected",
        selected_before=selected_before,
        selected_after=selected_after,
        rationale=(
            f"Selected Sentinel-1 scenes {selected_before.scene_id} and "
            f"{selected_after.scene_id}: both report relative orbit "
            f"{selected_before.relative_orbit}, both cover the AOI "
            f"({pair_validation['coverage']['before']['coverage_fraction']:.4%} and "
            f"{pair_validation['coverage']['after']['coverage_fraction']:.4%}), and the pair "
            "minimises the maximum absolute time distance from the event among compatible "
            "pairs."
        ),
        rejected_candidates=_dedupe(rejected),
        coverage=coverage_records or None,
        pair_validation=pair_validation,
        selection_policy=policy,
    )


def _cloud_rejection(
    scene: Sentinel2Scene, reason: RejectionReason, side: str, detail: str
) -> CandidateRejection:
    return CandidateRejection(
        scene_id=scene.scene_id,
        sensor=SensorKind.SENTINEL2,
        reason=reason,
        side=side,
        rationale=detail,
    )


def _normalize_processing_level(level: Optional[str]) -> str:
    """Normalize a processing level string to 'L2A' or 'L1C'."""
    if level is None:
        return "UNKNOWN"
    normalized = level.strip().upper()
    if normalized in {"L2A", "S2MSI2A", "LEVEL-2A", "LEVEL2A"}:
        return "L2A"
    if normalized in {"L1C", "S2MSI1C", "LEVEL-1C", "LEVEL1C"}:
        return "L1C"
    return "UNKNOWN"


def _tile_of(scene: Sentinel2Scene) -> Optional[str]:
    """The MGRS tile as a plain string, or ``None`` when it is not usable."""
    tile = scene.tile_id
    if tile is None or isinstance(tile, Unknown):
        return None
    text = str(tile).strip()
    return text or None


def select_sentinel2_pair(
    scenes: Sequence[Sentinel2Scene],
    plan: TemporalPlan,
    *,
    aoi: AreaOfInterest,
    strategy: Union[SelectionStrategy, str],
    max_cloud_percent: Optional[float] = None,
    processing_level_preference: Optional[str] = None,
    allow_processing_level_fallback: bool = True,
    minimum_aoi_coverage: Optional[float] = None,
    require_same_tile: bool = True,
    allow_mixed_processing_levels: bool = True,
) -> SelectionOutcome:
    """Choose one Sentinel-2 scene per temporal side and validate the pair.

    Selection order (LEAST_CLOUD_THEN_NEAREST):
      1. Processing level tier (if a preference is set): preferred level first
      2. Scene-level cloud percentage (lower is better)
      3. Temporal proximity to event (nearer is better)
      4. Scene ID (deterministic tiebreaker)

    Sides are ranked independently, then reconciled as a *pair*: candidate pairs
    are formed from the eligible scenes on each side and ranked by their worse
    side first, so the best individual scenes do not automatically form the
    selected pair. Pairs that fail tile compatibility are excluded with a
    recorded reason. Mixed processing levels are ranked below same-level pairs
    and, when selected, recorded as a disclosed limitation rather than a note.

    If processing_level_preference is set and no eligible product matches,
    allow_processing_level_fallback controls whether the other level is considered.
    """
    if not isinstance(aoi, AreaOfInterest):
        raise TypeError(
            f"aoi must be an AreaOfInterest so coverage can be measured, got {type(aoi).__name__}"
        )

    try:
        resolved_strategy = (
            strategy if isinstance(strategy, SelectionStrategy) else SelectionStrategy(strategy)
        )
    except ValueError as exc:
        raise ValueError(
            f"unsupported Sentinel-2 selection strategy {strategy!r}; choose one of "
            f"{[item.value for item in SelectionStrategy]}"
        ) from exc

    if max_cloud_percent is not None and not 0.0 <= max_cloud_percent <= 100.0:
        raise ValueError("max_cloud_percent must be in [0, 100]")

    pref = None
    if processing_level_preference is not None:
        pref = processing_level_preference.strip().upper()
        if pref not in {"L2A", "L1C"}:
            raise ValueError(
                f"processing_level_preference must be 'L2A' or 'L1C', got {processing_level_preference!r}"
            )

    before_raw, after_raw, rejected = _side_candidates(scenes, plan, SensorKind.SENTINEL2)
    before: list[Sentinel2Scene] = []
    after: list[Sentinel2Scene] = []
    coverage_records: Dict[str, Any] = {}

    def _proc_rank(scene: Sentinel2Scene) -> int:
        """Return processing level tier: 0 = preferred, 1 = fallback, 2 = no preference."""
        if pref is None:
            return 0
        proc = _normalize_processing_level(scene.processing_level or scene.product_type)
        if proc == pref:
            return 0
        # Not preferred level
        if allow_processing_level_fallback:
            return 1
        return 2  # Effectively ineligible

    def rank(scene: Sentinel2Scene) -> tuple:
        if resolved_strategy is SelectionStrategy.NEAREST_IN_TIME:
            return _nearest_key(scene, plan)
        # Tiered ranking: processing level tier -> cloud % -> temporal proximity
        tier = _proc_rank(scene)
        cloud = scene.scene_cloud_percent
        if not isinstance(cloud, (int, float)):
            cloud = float("inf")
        return (tier, cloud, *_nearest_key(scene, plan))

    for side, candidates, target in (
        ("before", before_raw, before),
        ("after", after_raw, after),
    ):
        for candidate in candidates:
            if not isinstance(candidate, Sentinel2Scene):
                raise TypeError(f"expected Sentinel2Scene, got {type(candidate).__name__}")
            cloud = candidate.scene_cloud_percent
            if max_cloud_percent is not None and not isinstance(cloud, (int, float)):
                rejected.append(
                    _cloud_rejection(
                        candidate,
                        RejectionReason.CLOUD_UNKNOWN,
                        side,
                        "A configured cloud limit cannot be checked because the provider "
                        "did not report scene-level cloud cover.",
                    )
                )
                continue
            if (
                max_cloud_percent is not None
                and isinstance(cloud, (int, float))
                and cloud > max_cloud_percent
            ):
                rejected.append(
                    _cloud_rejection(
                        candidate,
                        RejectionReason.CLOUD_ABOVE_CONFIGURED_LIMIT,
                        side,
                        f"Scene cloud cover {cloud:.2f}% exceeds configured limit "
                        f"{max_cloud_percent:.2f}%.",
                    )
                )
                continue
            if resolved_strategy is SelectionStrategy.LEAST_CLOUD_THEN_NEAREST and not isinstance(
                cloud, (int, float)
            ):
                rejected.append(
                    _cloud_rejection(
                        candidate,
                        RejectionReason.CLOUD_UNKNOWN,
                        side,
                        "least_cloud_then_nearest requires a reported scene cloud "
                        "percentage; missing cloud metadata is not treated as clear.",
                    )
                )
                continue
            # Processing-level tier: previously an unrecorded silent drop.
            if _proc_rank(candidate) >= 2:
                rejected.append(
                    CandidateRejection(
                        scene_id=candidate.scene_id,
                        sensor=SensorKind.SENTINEL2,
                        reason=RejectionReason.PROCESSING_LEVEL_EXCLUDED,
                        side=side,
                        rationale=(
                            f"Processing level "
                            f"{_normalize_processing_level(candidate.processing_level or candidate.product_type)} "
                            f"is not the preferred level {pref} and processing-level fallback "
                            "is disabled."
                        ),
                    )
                )
                continue
            # AOI coverage: previously a constant True.
            coverage = evaluate_footprint_coverage(
                candidate.footprint, aoi, minimum_coverage=minimum_aoi_coverage
            )
            coverage_records[candidate.scene_id] = coverage.to_dict()
            if not coverage.accepted:
                rejected.append(_coverage_rejection(coverage, candidate, side))
                continue
            target.append(candidate)

    # Pair compatibility. A pair is only formed from candidates that passed every
    # per-candidate gate above; the checks here are inherently pair-level.
    pairs: list[tuple[Sentinel2Scene, Sentinel2Scene]] = []
    for candidate_before in before:
        for candidate_after in after:
            if require_same_tile:
                tile_before = _tile_of(candidate_before)
                tile_after = _tile_of(candidate_after)
                if tile_before is None or tile_after is None:
                    for scene, side in (
                        (candidate_before, "before"),
                        (candidate_after, "after"),
                    ):
                        rejected.append(
                            CandidateRejection(
                                scene_id=scene.scene_id,
                                sensor=SensorKind.SENTINEL2,
                                reason=RejectionReason.TILE_UNVERIFIABLE,
                                side=side,
                                rationale=(
                                    "Same-tile pairing is required but the provider did not "
                                    "report an MGRS tile for this candidate, so tile "
                                    "compatibility cannot be verified."
                                ),
                            )
                        )
                    continue
                if tile_before != tile_after:
                    for scene, side, own, other in (
                        (candidate_before, "before", tile_before, tile_after),
                        (candidate_after, "after", tile_after, tile_before),
                    ):
                        rejected.append(
                            CandidateRejection(
                                scene_id=scene.scene_id,
                                sensor=SensorKind.SENTINEL2,
                                reason=RejectionReason.TILE_MISMATCH,
                                side=side,
                                rationale=(
                                    f"MGRS tile {own} does not match the opposite side's tile "
                                    f"{other}. A scene on a different tile is not on the same "
                                    "grid, so the pair is not a like-for-like comparison."
                                ),
                            )
                        )
                    continue
            level_before = _normalize_processing_level(
                candidate_before.processing_level or candidate_before.product_type
            )
            level_after = _normalize_processing_level(
                candidate_after.processing_level or candidate_after.product_type
            )
            mixed_levels = level_before != level_after
            if mixed_levels and not allow_mixed_processing_levels:
                for scene, side, own in (
                    (candidate_before, "before", level_before),
                    (candidate_after, "after", level_after),
                ):
                    rejected.append(
                        CandidateRejection(
                            scene_id=scene.scene_id,
                            sensor=SensorKind.SENTINEL2,
                            reason=RejectionReason.MIXED_PROCESSING_LEVELS_NOT_ALLOWED,
                            side=side,
                            rationale=(
                                f"Pre/post processing levels differ ({level_before} vs "
                                f"{level_after}); mixed levels are not permitted by "
                                "configuration. BOA and TOA reflectance are not "
                                "interchangeable in a spectral index."
                            ),
                        )
                    )
                continue
            pairs.append((candidate_before, candidate_after))

    def _pair_key(pair: tuple[Sentinel2Scene, Sentinel2Scene]) -> tuple:
        """Rank pairs by their **worse** side first, then by the other side.

        Minimising the worse side is what "the best pair" means here: a pair is
        limited by its weaker observation, so an excellent before-scene paired
        with a poor after-scene should not beat two adequate ones.
        """
        first, second = pair
        rank_first, rank_second = rank(first), rank(second)
        lower, upper = (
            (rank_first, rank_second)
            if rank_first <= rank_second
            else (rank_second, rank_first)
        )
        mixed = (
            0
            if _normalize_processing_level(first.processing_level or first.product_type)
            == _normalize_processing_level(second.processing_level or second.product_type)
            else 1
        )
        return (mixed, upper, lower, first.scene_id, second.scene_id)

    selection_policy = {
        "sensor": SensorKind.SENTINEL2.value,
        "aoi_id": aoi.aoi_id,
        "aoi_wkt": aoi.to_wkt(),
        "aoi_crs": aoi.crs,
        "strategy": resolved_strategy.value,
        "max_cloud_percent": max_cloud_percent,
        "processing_level_preference": pref,
        "allow_processing_level_fallback": allow_processing_level_fallback,
        "minimum_aoi_coverage": minimum_aoi_coverage,
        "require_same_tile": require_same_tile,
        "allow_mixed_processing_levels": allow_mixed_processing_levels,
        "window_before_days": plan.window.before_days,
        "window_after_days": plan.window.after_days,
        "event_date": plan.event.event_date.isoformat(),
        "event_day_excluded": plan.event_day_excluded,
    }

    if not pairs:
        no_candidates = not before or not after
        return SelectionOutcome(
            sensor=SensorKind.SENTINEL2,
            status="no_usable_scene" if no_candidates else "no_compatible_pair",
            rationale=(
                f"The configured {resolved_strategy.value} strategy could not select "
                "one qualifying Sentinel-2 scene on both temporal sides."
                if no_candidates
                else (
                    f"Candidates passed every per-scene gate on both sides but no pair "
                    f"satisfied the pair-compatibility rules (same tile required: "
                    f"{require_same_tile}; mixed processing levels allowed: "
                    f"{allow_mixed_processing_levels})."
                )
            ),
            rejected_candidates=_dedupe(rejected),
            coverage=coverage_records or None,
            selection_policy=selection_policy,
        )

    selected_before, selected_after = min(pairs, key=_pair_key)
    selected_ids = {selected_before.scene_id, selected_after.scene_id}
    for candidate in before + after:
        if candidate.scene_id not in selected_ids:
            rejected.append(
                CandidateRejection(
                    scene_id=candidate.scene_id,
                    sensor=SensorKind.SENTINEL2,
                    reason=RejectionReason.NOT_TOP_RANKED,
                    side="before" if candidate in before else "after",
                    rationale=(
                        f"The configured {resolved_strategy.value} pair ranking placed "
                        f"({selected_before.scene_id}, {selected_after.scene_id}) above this "
                        "candidate's pairings."
                    ),
                )
            )

    selected_level_before = _normalize_processing_level(
        selected_before.processing_level or selected_before.product_type
    )
    selected_level_after = _normalize_processing_level(
        selected_after.processing_level or selected_after.product_type
    )
    mixed_processing_levels = selected_level_before != selected_level_after
    fallback_used = any(
        pref is not None and _proc_rank(scene) == 1
        for scene in (selected_before, selected_after)
    )

    # Tile consistency check
    tile_consistency = None
    pre_tile = _tile_of(selected_before)
    post_tile = _tile_of(selected_after)
    if pre_tile is not None or post_tile is not None:
        tile_consistency = {
            "pre_tile": pre_tile or "unknown",
            "post_tile": post_tile or "unknown",
            "match": bool(pre_tile is not None and pre_tile == post_tile),
            "required": require_same_tile,
            "note": (
                "Matching tile IDs are necessary but not sufficient for full AOI "
                "coverage; coverage itself is measured from the footprint and recorded "
                "in the coverage block."
            ),
        }

    pair_validation = {
        "passed": True,
        "gates": {
            "temporal_window": "passed",
            "cloud": (
                "not_configured" if max_cloud_percent is None else "passed"
            ),
            "processing_level": (
                "no_preference" if pref is None else "passed"
            ),
            "aoi_coverage": "not_thresholded" if minimum_aoi_coverage is None else "passed",
            "tile_compatibility": (
                "not_required"
                if not require_same_tile
                else ("passed" if tile_consistency and tile_consistency["match"] else "unknown")
            ),
            "processing_level_compatibility": (
                "passed" if not mixed_processing_levels else "mixed_levels_disclosed"
            ),
        },
        "minimum_aoi_coverage": minimum_aoi_coverage,
        "processing_levels": {
            "before": selected_level_before,
            "after": selected_level_after,
        },
        "mixed_processing_levels": mixed_processing_levels,
        "allow_mixed_processing_levels": allow_mixed_processing_levels,
        "coverage": {
            "before": coverage_records.get(selected_before.scene_id),
            "after": coverage_records.get(selected_after.scene_id),
        },
        "tile": {"before": pre_tile, "after": post_tile},
    }

    cloud_text = (
        "scene-level cloud cover"
        if resolved_strategy is SelectionStrategy.LEAST_CLOUD_THEN_NEAREST
        else "time distance"
    )
    pref_text = f"processing level preference = {pref}" if pref else "no processing level preference"
    fallback_text = " (fallback used)" if fallback_used else ""
    mixed_text = (
        "; processing levels are MIXED ("
        f"{selected_level_before} vs {selected_level_after}) and this is a disclosed "
        "limitation, not an equivalent comparison"
        if mixed_processing_levels
        else ""
    )
    rationale = (
        f"Selected {selected_before.scene_id} before and {selected_after.scene_id} "
        f"after using {resolved_strategy.value}; {pref_text}; "
        f"primary ranking key = {cloud_text}{fallback_text}; "
        f"tile match = {tile_consistency['match'] if tile_consistency else 'unknown'}; "
        f"AOI coverage = {pair_validation['coverage']['before']['coverage_fraction']:.4%} "
        f"(pre) / {pair_validation['coverage']['after']['coverage_fraction']:.4%} (post)"
        f"{mixed_text}."
    )

    return SelectionOutcome(
        sensor=SensorKind.SENTINEL2,
        status="selected",
        selected_before=selected_before,
        selected_after=selected_after,
        rationale=rationale,
        rejected_candidates=_dedupe(rejected),
        tile_consistency=tile_consistency,
        fallback_used=fallback_used,
        coverage=coverage_records or None,
        pair_validation=pair_validation,
        selection_policy=selection_policy,
    )


def select_sentinel1(*args, **kwargs) -> SelectionOutcome:
    """Compatibility alias for :func:`select_sentinel1_pair`."""
    return select_sentinel1_pair(*args, **kwargs)


def select_sentinel2(*args, **kwargs) -> SelectionOutcome:
    """Compatibility alias for :func:`select_sentinel2_pair`."""
    return select_sentinel2_pair(*args, **kwargs)


# Public vocabulary aliases keep the result object easy to discover without
# creating parallel implementations or duplicate serialization formats.
RejectedCandidate = CandidateRejection
SelectionResult = SelectionOutcome
