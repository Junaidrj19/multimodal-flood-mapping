"""Normalisation statistics: fitting, artifacts and the development escape hatch.

The C5 contract (``configs/data.yaml`` -> ``m4_contract.statistics``) fixes the
artifact SCHEMA, not the values and not the method. This module implements that
schema and the fitting mechanism, and it does not invent any numbers.

FOUR KINDS OF STATISTICS, three of which are contractual:

``reference``
    Published upstream values (Kuro's ``data_mean``/``data_std``). A
    tolerance-banded cross-check only. Their pre/post basis, split basis and
    aggregation basis are all UNSTATED upstream, so adopting them as our
    parameters would normalise by an unknown recipe. :class:`StatisticsArtifact`
    refuses to serve them where fitted parameters are required.

``fitted``
    OUR parameters, from the training split and nothing else. Produced only by
    :func:`fit_training_statistics`, which raises on any sample whose split is
    not ``train``.

``validation``
    Measured on validation data for DRIFT REPORTING only. Never normalises.

``development``
    NOT part of the frozen contract, and deliberately named so it cannot be
    mistaken for ``fitted``. It exists because the smoke and overfit tests need
    deterministic, reproducible scaling before any real fit has been run. Every
    development artifact carries ``development_only = True`` and
    ``usable_for_reported_results = False``, and
    :meth:`StatisticsArtifact.require_fitted` rejects it.

NO FITTED STATISTICS EXIST YET. Nothing here changes that; it builds the
machinery that will produce them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from floodmap.datasets.contracts import CanonicalSample

from .channels import CHANNEL_ORDER, REQUIRED_SAR_REPRESENTATION
from .tensors import stack_channels

__all__ = [
    "ChannelStatistics",
    "StatisticsArtifact",
    "StatisticsError",
    "development_statistics",
    "fit_training_statistics",
]

#: Schema identity, from the frozen C5 block. Not a local choice.
SCHEMA_ID = "floodmap.statistics/1"
SCHEMA_VERSION = "1.0.0"

#: Filenames of the sidecar pair. The split is forced rather than preferred:
#: ``floodmap.features.pipeline`` coerces ``float()`` over every value of every
#: top-level key in the values file, so an embedded provenance block raises.
VALUES_FILE = "statistics.json"
PROVENANCE_FILE = "statistics_meta.json"

#: The three contract kinds, plus the explicitly non-contract development kind.
CONTRACT_KINDS: Tuple[str, ...] = ("reference", "fitted", "validation")
DEVELOPMENT_KIND = "development"
ALL_KINDS: Tuple[str, ...] = CONTRACT_KINDS + (DEVELOPMENT_KIND,)

#: Provenance keys the C5 block requires any artifact to answer.
REQUIRED_PROVENANCE: Tuple[str, ...] = (
    "schema_id",
    "schema_version",
    "kind",
    "dataset_id",
    "split",
    "activations",
    "channels",
    "representation",
    "preprocessing_state",
    "clip_state",
    "temporal_role_basis",
    "aggregation_basis",
    "n_tiles",
    "n_valid_pixels",
    "created_at",
    "code_version",
)


class StatisticsError(RuntimeError):
    """A statistics artifact is missing, malformed, or used outside its kind."""


@dataclass(frozen=True)
class ChannelStatistics:
    """Mean and standard deviation for one input channel."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std):
            raise StatisticsError(f"channel statistics must be finite, got {self}")
        if self.std <= 0.0:
            raise StatisticsError(
                f"channel standard deviation must be positive, got {self.std}; a zero "
                "std would divide by zero at normalisation time"
            )


@dataclass(frozen=True)
class StatisticsArtifact:
    """A versioned statistics artifact: values plus answerable provenance."""

    kind: str
    values: Mapping[str, ChannelStatistics]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ALL_KINDS:
            raise StatisticsError(
                f"unknown statistics kind {self.kind!r}; expected one of {ALL_KINDS}"
            )
        if tuple(self.values) != CHANNEL_ORDER:
            raise StatisticsError(
                f"statistics channels {tuple(self.values)} do not match the frozen "
                f"channel order {CHANNEL_ORDER}; a reordered artifact would re-map a "
                "trained model's input weights"
            )
        missing = [key for key in REQUIRED_PROVENANCE if key not in self.provenance]
        if missing:
            raise StatisticsError(
                "statistics provenance cannot answer " + ", ".join(missing)
            )

    # -- kind policy -------------------------------------------------------

    @property
    def is_development_only(self) -> bool:
        return self.kind == DEVELOPMENT_KIND

    @property
    def usable_for_reported_results(self) -> bool:
        """Only a fitted artifact may back a reported number."""

        return self.kind == "fitted"

    def require_fitted(self) -> "StatisticsArtifact":
        """Return self if this is a fitted artifact, else explain the refusal."""

        if self.kind == "fitted":
            return self
        if self.kind == DEVELOPMENT_KIND:
            raise StatisticsError(
                "this is a DEVELOPMENT statistics artifact and must not be used where "
                "fitted training statistics are required; it exists only so pipeline "
                "correctness tests can run before a real fit, and any metric produced "
                "with it is not a reportable result"
            )
        raise StatisticsError(
            f"{self.kind!r} statistics must never be used as normalisation parameters: "
            "reference values have an unstated pre/post, split and aggregation basis, "
            "and validation values would leak validation data into the model"
        )

    # -- tensors -----------------------------------------------------------

    def mean_vector(self) -> np.ndarray:
        return np.array([self.values[name].mean for name in CHANNEL_ORDER], dtype=np.float32)

    def std_vector(self) -> np.ndarray:
        return np.array([self.values[name].std for name in CHANNEL_ORDER], dtype=np.float32)

    def identity(self) -> Dict[str, Any]:
        """Compact identity recorded in checkpoints and run records."""

        return {
            "schema_id": self.provenance.get("schema_id"),
            "schema_version": self.provenance.get("schema_version"),
            "kind": self.kind,
            "dataset_id": self.provenance.get("dataset_id"),
            "split": self.provenance.get("split"),
            "channels": list(CHANNEL_ORDER),
            "representation": self.provenance.get("representation"),
            "n_tiles": self.provenance.get("n_tiles"),
            "n_valid_pixels": self.provenance.get("n_valid_pixels"),
            "development_only": self.is_development_only,
            "usable_for_reported_results": self.usable_for_reported_results,
        }

    # -- sidecar pair ------------------------------------------------------

    def write(self, directory: Path) -> Tuple[Path, Path]:
        """Write the sidecar pair and return both paths."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        values_path = directory / VALUES_FILE
        provenance_path = directory / PROVENANCE_FILE

        # Flat and float-only: the features reader coerces float() over every
        # value of every top-level key, so provenance cannot live in here.
        flat = {
            name: {"mean": float(stat.mean), "std": float(stat.std)}
            for name, stat in self.values.items()
        }
        values_path.write_text(json.dumps(flat, indent=2, sort_keys=False), encoding="utf-8")
        provenance_path.write_text(
            json.dumps(dict(self.provenance), indent=2, sort_keys=False), encoding="utf-8"
        )
        return values_path, provenance_path

    @classmethod
    def read(cls, directory: Path) -> "StatisticsArtifact":
        """Read a sidecar pair written by :meth:`write`."""

        directory = Path(directory)
        values_path = directory / VALUES_FILE
        provenance_path = directory / PROVENANCE_FILE
        for path in (values_path, provenance_path):
            if not path.is_file():
                raise StatisticsError(
                    f"incomplete statistics artifact: {path} is missing. The layout is a "
                    "sidecar PAIR; values without provenance cannot be audited"
                )
        raw_values = json.loads(values_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        values = {
            str(name): ChannelStatistics(
                mean=float(entry["mean"]), std=float(entry["std"])
            )
            for name, entry in raw_values.items()
        }
        kind = str(provenance.get("kind", ""))
        return cls(kind=kind, values=values, provenance=provenance)


def _provenance(
    *,
    kind: str,
    dataset_id: str,
    split: str,
    activations: Sequence[str],
    n_tiles: int,
    n_valid_pixels: int,
    representation: str,
    clip_state: str,
    preprocessing_state: Mapping[str, Any],
    aggregation_basis: str,
    code_version: Optional[str],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "dataset_id": dataset_id,
        "split": split,
        "activations": list(activations),
        "channels": list(CHANNEL_ORDER),
        "representation": representation,
        "preprocessing_state": dict(preprocessing_state),
        "clip_state": clip_state,
        # Named explicitly because the upstream reference statistics leave it
        # unstated, which is precisely why they cannot be reused as ours.
        "temporal_role_basis": "pre=SL1 (crank 1), post=MS1 (master); SL2 dropped",
        "aggregation_basis": aggregation_basis,
        "n_tiles": int(n_tiles),
        "n_valid_pixels": int(n_valid_pixels),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version,
    }
    if extra:
        record.update(dict(extra))
    return record


def _quarantine_record(
    quarantine: Optional[object],
    skipped: Sequence[Mapping[str, Any]],
    eligibility: Optional[object],
) -> Dict[str, Any]:
    """Merge exclusions the fitter skipped with exclusions the caller pre-filtered.

    Both sources are real. The fitter skips a quarantined sample it is handed;
    a caller streaming from disk must pre-filter, because a quarantined sample
    may be unloadable and would raise before the fitter saw it. Reporting only
    one of the two would understate the audit trail.
    """

    entries = [dict(entry) for entry in skipped]
    seen = {entry.get("sample_id") for entry in entries}
    if eligibility is not None:
        for entry in getattr(eligibility, "excluded", ()):  # SampleEligibility
            if entry.sample_id not in seen:
                entries.append(
                    {
                        "sample_id": entry.sample_id,
                        "reason_code": entry.reason_code,
                        "excluded_by": "caller pre-filter",
                    }
                )
                seen.add(entry.sample_id)

    by_code: Dict[str, int] = {}
    for entry in entries:
        code = str(entry.get("reason_code"))
        by_code[code] = by_code.get(code, 0) + 1

    applied = quarantine is not None or eligibility is not None
    path = getattr(quarantine, "path", None) if quarantine is not None else None
    return {
        "policy_applied": applied,
        "policy_version": getattr(quarantine, "version", None) if quarantine else None,
        "policy_path": str(path) if path else None,
        "n_excluded": len(entries),
        "excluded_by_reason_code": by_code,
        "excluded_samples": entries,
    }


def fit_training_statistics(
    samples: Iterable[CanonicalSample],
    *,
    dataset_id: str = "kuro_siwo",
    code_version: Optional[str] = None,
    reprojection: Optional[object] = None,
    quarantine: Optional[object] = None,
    eligibility: Optional[object] = None,
) -> StatisticsArtifact:
    """Fit per-channel mean and std over the TRAINING split only.

    Streaming and mask-respecting by construction:

    * accumulates count, sum and sum-of-squares per channel, so no tile stack
      is ever materialised and the 7,000-tile training split never has to fit
      in RAM;
    * counts only pixels the M4 canonical validity mask marks valid, so nodata
      (which is 6.48% of Kuro label pixels, and is stored as 0.0 in the SAR
      rasters) does not drag the mean toward zero;
    * refuses any sample whose split is not ``train``. Fitting on validation or
      test data is test-time leakage, and the check is here rather than in the
      caller because this is the function that produces the artifact.

    The aggregation basis is per valid PIXEL, pooled across tiles — recorded in
    provenance because a per-tile-then-average basis gives a different answer
    on tiles with differing valid fractions.

    ``reprojection`` MUST match what training will use. Statistics fitted on
    the delivered source grid do not describe a reprojected training input:
    bilinear resampling changes the variance of the field it is applied to, so
    a std fitted pre-resampling is systematically wrong for post-resampling
    data. The setting is therefore recorded in provenance and the training path
    passes its own configuration straight through.

    ``quarantine`` applies the same eligibility policy training uses, skipping
    and recording any sample it excludes. ``eligibility`` supplies a
    pre-computed :class:`SampleEligibility` for the case where the caller
    already filtered the stream — which it must, because a quarantined sample
    may be unloadable, and the fitter would never get to see it. Without this
    the artifact would record zero exclusions while the caller had in fact
    excluded samples, understating its own audit trail.
    """

    count = np.zeros(len(CHANNEL_ORDER), dtype=np.float64)
    total = np.zeros(len(CHANNEL_ORDER), dtype=np.float64)
    total_sq = np.zeros(len(CHANNEL_ORDER), dtype=np.float64)
    activations: set[str] = set()
    n_tiles = 0
    quarantined_seen: list[dict] = []

    for sample in samples:
        if sample.split != "train":
            raise StatisticsError(
                f"sample {sample.sample_id!r} is in split {sample.split!r}; fitted "
                "statistics must come from the training split only, and fitting on "
                "validation or test data is test-time leakage"
            )
        if sample.dataset_id != dataset_id:
            raise StatisticsError(
                f"sample {sample.sample_id!r} belongs to dataset "
                f"{sample.dataset_id!r}, not {dataset_id!r}; a mixed-corpus fit would "
                "average two different SAR representations"
            )
        if quarantine is not None and quarantine.is_quarantined(sample.sample_id):
            # The SAME eligibility policy as training. A sample excluded from
            # training must not contribute to the parameters training uses, and
            # the exclusion is recorded rather than quietly skipped.
            entry = quarantine.entry(sample.sample_id)
            quarantined_seen.append(
                {"sample_id": sample.sample_id, "reason_code": entry.reason_code}
            )
            continue
        if reprojection is not None:
            # Identical call shape to the training path, via the shared
            # spatial component, so the fitted parameters describe exactly the
            # arrays the model will receive.
            from .spatial import reproject_canonical_sample

            warped = reproject_canonical_sample(
                sample,
                resolution_m=getattr(reprojection, "resolution_m", 10.0),
                tile_size=getattr(reprojection, "tile_size", None),
                nodata_policy=reprojection.nodata_policy()
                if hasattr(reprojection, "nodata_policy")
                else None,
                sar_resampling=getattr(reprojection, "sar_resampling", "bilinear"),
                label_resampling=getattr(reprojection, "label_resampling", "nearest"),
            )
            stack = warped.image
            valid = warped.valid_mask
        else:
            stack = stack_channels(sample)
            valid = np.asarray(sample.validity, dtype=bool)
        if not valid.any():
            continue
        selected = stack[:, valid].astype(np.float64, copy=False)
        count += selected.shape[1]
        total += selected.sum(axis=1)
        total_sq += np.square(selected).sum(axis=1)
        if sample.event_id is not None:
            activations.add(str(sample.event_id))
        n_tiles += 1

    if n_tiles == 0 or not count.all():
        raise StatisticsError(
            "no valid training pixels were seen; a statistics artifact fitted on "
            "nothing would be a fabricated parameter set"
        )

    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if not (std > 0).all():
        zero = [CHANNEL_ORDER[i] for i, value in enumerate(std) if value <= 0]
        raise StatisticsError(
            f"channels {zero} have zero variance over the training split; "
            "normalising by them would divide by zero"
        )

    values = {
        name: ChannelStatistics(mean=float(mean[index]), std=float(std[index]))
        for index, name in enumerate(CHANNEL_ORDER)
    }
    provenance = _provenance(
        kind="fitted",
        dataset_id=dataset_id,
        split="train",
        activations=sorted(activations),
        n_tiles=n_tiles,
        n_valid_pixels=int(count[0]),
        representation=REQUIRED_SAR_REPRESENTATION,
        clip_state="applied_0.15_linear_sigma0",
        preprocessing_state={
            "sar_clip_applied": True,
            "sar_clip_max_linear": 0.15,
            "sar_conversion_to_db_applied": False,
            "additional_speckle_filter_applied": False,
            # Reflects what actually happened, not a constant: a provenance
            # record claiming unresampled data for a reprojected fit would
            # describe the wrong artifact.
            "resampled": reprojection is not None,
        },
        aggregation_basis="per valid pixel, pooled across tiles (not per-tile mean of means)",
        code_version=code_version,
        extra={
            "validity_basis": "M4 canonical validity mask (Kuro MK0_MNA == 1)",
            "fitted_from_validation_or_test": False,
            # Eligibility is recorded unconditionally: an absent key would be
            # indistinguishable from an unapplied policy. A caller that
            # pre-filtered the stream supplies its own record, because the
            # fitter cannot count samples it was never handed.
            "quarantine": _quarantine_record(quarantine, quarantined_seen, eligibility),
            # Recorded because a std fitted before resampling does not describe
            # a resampled training input: bilinear changes field variance.
            "spatial_transform": (
                {"reprojected_to_canonical_grid": False, "grid": "delivered source grid"}
                if reprojection is None
                else {
                    "reprojected_to_canonical_grid": True,
                    "shared_implementation": "floodmap.preprocessing.reprojection",
                    "resolution_m": getattr(reprojection, "resolution_m", None),
                    "tile_size": getattr(reprojection, "tile_size", None),
                    "sar_resampling": getattr(reprojection, "sar_resampling", None),
                    "label_resampling": getattr(reprojection, "label_resampling", None),
                    "nodata_mode": getattr(reprojection, "nodata_mode", None),
                }
            ),
        },
    )
    return StatisticsArtifact(kind="fitted", values=values, provenance=provenance)


def development_statistics(
    *,
    dataset_id: str = "kuro_siwo",
    reason: str = "pipeline correctness testing before any training fit exists",
    code_version: Optional[str] = None,
) -> StatisticsArtifact:
    """A deterministic, explicitly non-contract artifact for tests.

    Unit mean 0 / std 1, i.e. an identity transform. That choice is deliberate:
    an identity leaves the linear sigma0 values exactly as the adapter produced
    them, so a failing test points at the pipeline rather than at an invented
    scaling. It is NOT a fitted artifact and
    :meth:`StatisticsArtifact.require_fitted` rejects it.
    """

    values = {name: ChannelStatistics(mean=0.0, std=1.0) for name in CHANNEL_ORDER}
    provenance = _provenance(
        kind=DEVELOPMENT_KIND,
        dataset_id=dataset_id,
        split="none (no data was read)",
        activations=[],
        n_tiles=0,
        n_valid_pixels=0,
        representation=REQUIRED_SAR_REPRESENTATION,
        clip_state="applied_0.15_linear_sigma0",
        preprocessing_state={
            "sar_clip_applied": True,
            "sar_clip_max_linear": 0.15,
            "sar_conversion_to_db_applied": False,
        },
        aggregation_basis="none — identity transform, no data was aggregated",
        code_version=code_version,
        extra={
            "development_only": True,
            "usable_for_reported_results": False,
            "is_part_of_frozen_contract": False,
            "reason": reason,
            "transform": "identity (mean 0, std 1)",
        },
    )
    return StatisticsArtifact(kind=DEVELOPMENT_KIND, values=values, provenance=provenance)
