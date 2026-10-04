"""Deterministic, auditable selection of pre/post scene candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

from .outcomes import RejectionReason, SelectionStrategy
from .scenes import DiscoveredScene, Sentinel1Scene, Sentinel2Scene, SensorKind
from .temporal import TemporalPlan

__all__ = [
    "CandidateRejection",
    "NoSameTrackPair",
    "RejectedCandidate",
    "SelectionOutcome",
    "SelectionResult",
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

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "sensor": self.sensor.value,
            "reason": self.reason.value,
            "rationale": self.rationale,
            "side": self.side,
        }


@dataclass(frozen=True)
class SelectionOutcome:
    """Selection result with both the decision and its discarded candidates."""

    sensor: SensorKind
    status: str
    selected_before: Optional[DiscoveredScene] = None
    selected_after: Optional[DiscoveredScene] = None
    rationale: str = ""
    rejected_candidates: Tuple[CandidateRejection, ...] = field(default_factory=tuple)

    @property
    def is_usable(self) -> bool:
        return self.selected_before is not None and self.selected_after is not None

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

    def to_dict(self) -> dict:
        return {
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
        }


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
    require_same_relative_orbit: bool = True,
) -> SelectionOutcome:
    """Select a same-relative-orbit Sentinel-1 pair.

    Cross-track substitution is deliberately unsupported.  ``NoSameTrackPair``
    is returned when the candidate pool cannot satisfy the geometry rule.
    """
    if not require_same_relative_orbit:
        raise ValueError(
            "Sentinel-1 selection requires same relative orbit; cross-track "
            "fallback would make terrain geometry look like change"
        )

    before_raw, after_raw, rejected = _side_candidates(scenes, plan, SensorKind.SENTINEL1)
    before: list[Sentinel1Scene] = []
    after: list[Sentinel1Scene] = []
    for side, candidates, target in (
        ("before", before_raw, before),
        ("after", after_raw, after),
    ):
        for candidate in candidates:
            if not isinstance(candidate, Sentinel1Scene):
                raise TypeError(f"expected Sentinel1Scene, got {type(candidate).__name__}")
            if not isinstance(candidate.relative_orbit, int):
                rejected.append(_unknown_orbit_rejection(candidate, side))
            else:
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
            rejected_candidates=tuple(rejected),
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
    return SelectionOutcome(
        sensor=SensorKind.SENTINEL1,
        status="selected",
        selected_before=selected_before,
        selected_after=selected_after,
        rationale=(
            f"Selected Sentinel-1 scenes {selected_before.scene_id} and "
            f"{selected_after.scene_id}: both report relative orbit "
            f"{selected_before.relative_orbit}, and the pair minimises the maximum "
            "absolute time distance from the event among compatible pairs."
        ),
        rejected_candidates=tuple(rejected),
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


def select_sentinel2_pair(
    scenes: Sequence[Sentinel2Scene],
    plan: TemporalPlan,
    *,
    strategy: Union[SelectionStrategy, str],
    max_cloud_percent: Optional[float] = None,
) -> SelectionOutcome:
    """Choose one Sentinel-2 scene per temporal side under an explicit strategy."""
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
    before_raw, after_raw, rejected = _side_candidates(scenes, plan, SensorKind.SENTINEL2)
    before: list[Sentinel2Scene] = []
    after: list[Sentinel2Scene] = []
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
            target.append(candidate)

    def rank(scene: Sentinel2Scene) -> tuple:
        if resolved_strategy is SelectionStrategy.NEAREST_IN_TIME:
            return _nearest_key(scene, plan)
        # The cloud-unknown case was removed above, so this is a real float.
        return (scene.scene_cloud_percent, *_nearest_key(scene, plan))  # type: ignore[tuple-item]

    selected: list[Sentinel2Scene] = []
    for side, candidates in (("before", before), ("after", after)):
        if not candidates:
            continue
        winner = min(candidates, key=rank)
        selected.append(winner)
        for candidate in candidates:
            if candidate.scene_id != winner.scene_id:
                rejected.append(
                    CandidateRejection(
                        scene_id=candidate.scene_id,
                        sensor=SensorKind.SENTINEL2,
                        reason=RejectionReason.NOT_TOP_RANKED,
                        side=side,
                        rationale=(
                            f"The configured {resolved_strategy.value} strategy ranked "
                            f"{winner.scene_id} above this candidate."
                        ),
                    )
                )

    selected_before = next(
        (scene for scene in selected if plan.classify(scene.acquired_at) == "before"), None
    )
    selected_after = next(
        (scene for scene in selected if plan.classify(scene.acquired_at) == "after"), None
    )
    if selected_before is None or selected_after is None:
        return SelectionOutcome(
            sensor=SensorKind.SENTINEL2,
            status="no_usable_scene",
            rationale=(
                f"The configured {resolved_strategy.value} strategy could not select "
                "one qualifying Sentinel-2 scene on both temporal sides."
            ),
            rejected_candidates=tuple(rejected),
        )
    cloud_text = (
        "scene-level cloud cover"
        if resolved_strategy is SelectionStrategy.LEAST_CLOUD_THEN_NEAREST
        else "time distance"
    )
    return SelectionOutcome(
        sensor=SensorKind.SENTINEL2,
        status="selected",
        selected_before=selected_before,
        selected_after=selected_after,
        rationale=(
            f"Selected {selected_before.scene_id} before and {selected_after.scene_id} "
            f"after using {resolved_strategy.value}; the primary ranking key was "
            f"{cloud_text}, with deterministic time and scene-id tie breakers."
        ),
        rejected_candidates=tuple(rejected),
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
