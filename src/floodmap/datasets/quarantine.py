"""Explicit sample-level quarantine.

A quarantine is an **eligibility list consulted before loading**, not an
exception handler wrapped around one. That distinction is the whole design:

* the adapter's integrity refusal stays exactly as loud as it was. Nothing here
  catches ``DatasetIntegrityError``;
* an excluded sample is excluded **by name**, with a reason code from a closed
  set, and is reported wherever sample counts appear;
* asking for a quarantined sample explicitly RAISES. A caller that names a
  sample must not silently receive nothing back;
* frozen label and validity semantics are untouched. Quarantine operates on
  whole samples; it does not reinterpret a pixel, promote NaN to a validity
  mechanism, impute a value, or reconstruct a sample.

The policy lives in ``configs/quarantine/<dataset_id>.yaml`` — sample IDs and
reasons only, no pixel data, so it is committable and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from floodmap.utils.config import project_root

__all__ = [
    "QuarantineEntry",
    "QuarantinePolicy",
    "QuarantineError",
    "SampleEligibility",
    "load_quarantine",
    "quarantine_path",
]


class QuarantineError(RuntimeError):
    """A quarantine policy is malformed, or a quarantined sample was required."""


@dataclass(frozen=True)
class QuarantineEntry:
    """One excluded sample, with the reason it is excluded."""

    sample_id: str
    reason_code: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            **{k: v for k, v in self.details.items() if k != "sample_id"},
        }


@dataclass(frozen=True)
class SampleEligibility:
    """The result of applying a policy to a set of sample IDs."""

    eligible: Tuple[str, ...]
    excluded: Tuple[QuarantineEntry, ...]
    policy_version: Optional[str]
    policy_path: Optional[str]

    @property
    def n_eligible(self) -> int:
        return len(self.eligible)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)

    def audit_record(self) -> Dict[str, Any]:
        """The audit trail entry. Always present, even when nothing is excluded.

        An absent record is indistinguishable from an unapplied policy, so the
        key is written unconditionally.
        """

        by_code: Dict[str, int] = {}
        for entry in self.excluded:
            by_code[entry.reason_code] = by_code.get(entry.reason_code, 0) + 1
        return {
            "policy_applied": self.policy_path is not None,
            "policy_version": self.policy_version,
            "policy_path": self.policy_path,
            "n_eligible": self.n_eligible,
            "n_excluded": self.n_excluded,
            "excluded_by_reason_code": by_code,
            "excluded_samples": [entry.to_dict() for entry in self.excluded],
        }


@dataclass(frozen=True)
class QuarantinePolicy:
    """A loaded, validated quarantine policy for one dataset."""

    dataset_id: str
    version: str
    entries: Mapping[str, QuarantineEntry]
    reason_codes: Mapping[str, str]
    path: Optional[Path] = None
    approved_at: Optional[str] = None
    evidence: Optional[str] = None

    #: A policy that excludes nothing. Used where no policy file exists, so
    #: callers never branch on ``None``.
    @classmethod
    def empty(cls, dataset_id: str) -> "QuarantinePolicy":
        return cls(
            dataset_id=dataset_id,
            version="0.0.0",
            entries={},
            reason_codes={},
            path=None,
        )

    def is_quarantined(self, sample_id: str) -> bool:
        return sample_id in self.entries

    def entry(self, sample_id: str) -> Optional[QuarantineEntry]:
        return self.entries.get(sample_id)

    def partition(self, sample_ids: Iterable[str]) -> SampleEligibility:
        """Split sample IDs into eligible and excluded, preserving input order."""

        eligible: list[str] = []
        excluded: list[QuarantineEntry] = []
        for sample_id in sample_ids:
            entry = self.entries.get(sample_id)
            if entry is None:
                eligible.append(sample_id)
            else:
                excluded.append(entry)
        return SampleEligibility(
            eligible=tuple(eligible),
            excluded=tuple(excluded),
            policy_version=self.version if self.path is not None else None,
            policy_path=str(self.path) if self.path is not None else None,
        )

    def require_eligible(self, sample_ids: Sequence[str]) -> None:
        """Raise if any explicitly named sample is quarantined.

        A caller that names samples by ID gets an error rather than a shorter
        list, because silently returning fewer samples than were asked for is
        the failure mode this whole module exists to prevent.
        """

        named = [sid for sid in sample_ids if sid in self.entries]
        if named:
            detail = "; ".join(
                f"{sid} ({self.entries[sid].reason_code})" for sid in named
            )
            raise QuarantineError(
                f"{len(named)} explicitly requested sample(s) are quarantined: {detail}. "
                "Remove them from the request or, if the exclusion is wrong, amend "
                f"{self.path}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "path": str(self.path) if self.path else None,
            "approved_at": self.approved_at,
            "evidence": self.evidence,
            "n_entries": len(self.entries),
            "reason_codes": dict(self.reason_codes),
        }


def quarantine_path(dataset_id: str) -> Path:
    """Conventional location of a dataset's quarantine policy."""

    return project_root() / "configs" / "quarantine" / f"{dataset_id}.yaml"


def load_quarantine(
    dataset_id: str,
    *,
    path: Optional[Path] = None,
    required: bool = False,
) -> QuarantinePolicy:
    """Load and validate a quarantine policy.

    Parameters
    ----------
    dataset_id
        Dataset the policy applies to. Checked against the file's own
        ``dataset_id`` so a policy cannot be applied to the wrong corpus.
    path
        Explicit policy path. Defaults to the conventional location.
    required
        When ``True``, a missing file raises. Default ``False`` returns an
        empty policy, so a dataset with nothing to exclude needs no file.
    """

    resolved = Path(path) if path is not None else quarantine_path(dataset_id)
    if not resolved.is_file():
        if required:
            raise QuarantineError(f"quarantine policy not found: {resolved}")
        return QuarantinePolicy.empty(dataset_id)

    parsed = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise QuarantineError(f"quarantine policy must parse to a mapping: {resolved}")

    declared = parsed.get("dataset_id")
    if declared != dataset_id:
        raise QuarantineError(
            f"{resolved} declares dataset_id {declared!r} but was loaded for "
            f"{dataset_id!r}; a policy must not be applied to another corpus"
        )
    version = parsed.get("version")
    if not version:
        raise QuarantineError(f"{resolved} has no version; an unversioned policy is unauditable")

    reason_codes = parsed.get("reason_codes") or {}
    if not isinstance(reason_codes, Mapping):
        raise QuarantineError(f"{resolved}: reason_codes must be a mapping")

    entries: Dict[str, QuarantineEntry] = {}
    for raw in parsed.get("samples") or []:
        if not isinstance(raw, Mapping):
            raise QuarantineError(f"{resolved}: each quarantine entry must be a mapping")
        sample_id = raw.get("sample_id")
        code = raw.get("reason_code")
        if not sample_id:
            raise QuarantineError(f"{resolved}: a quarantine entry has no sample_id")
        if not code:
            raise QuarantineError(
                f"{resolved}: {sample_id} has no reason_code; an unexplained exclusion "
                "is indistinguishable from a silent skip"
            )
        if code not in reason_codes:
            raise QuarantineError(
                f"{resolved}: {sample_id} uses undeclared reason_code {code!r}; "
                f"declared codes are {sorted(reason_codes)}"
            )
        if sample_id in entries:
            raise QuarantineError(f"{resolved}: duplicate quarantine entry {sample_id}")
        entries[str(sample_id)] = QuarantineEntry(
            sample_id=str(sample_id),
            reason_code=str(code),
            reason=str(reason_codes[code]).strip(),
            details={k: v for k, v in raw.items() if k not in {"sample_id", "reason_code"}},
        )

    return QuarantinePolicy(
        dataset_id=dataset_id,
        version=str(version),
        entries=entries,
        reason_codes={str(k): str(v).strip() for k, v in reason_codes.items()},
        path=resolved,
        approved_at=parsed.get("approved_at"),
        evidence=parsed.get("evidence"),
    )
