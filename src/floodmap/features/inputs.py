"""Strict consumption of M2 analysis-ready artifacts by M3.

M3's only input is what M2 wrote. This module reads those artifacts back from
disk and refuses anything that cannot be traced to an M1 acquisition manifest.
There is deliberately no catalogue query, no scene selection, no download and
no construction of a raster from arbitrary arrays: the boundary described in
``docs/data-contract.md`` §0.5 runs through here.

The sibling-file convention mirrors
:func:`floodmap.preprocessing.artifacts.write_raster_artifact` exactly, so M3
reads the format M2 writes rather than a second, parallel one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..preprocessing.grid import AnalysisGrid, require_aligned, require_rasterio
from ..preprocessing.validation import RasterMetadata, read_valid_mask, require_valid_raster
from ..utils.provenance import (
    ArtifactProvenance,
    ArtifactType,
    ProductionInput,
    ValidationOnlySource,
)
from .errors import (
    FeatureBoundaryError,
    FeatureDependencyError,
    FeatureGridError,
    FeatureInputError,
)
from .registry import ArtifactRole

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - exercised without the geo extra
    rasterio = None  # type: ignore[assignment]
    _RASTERIO_ERROR = exc
else:
    _RASTERIO_ERROR = None

__all__ = ["FeatureInputs", "PreprocessedInput", "load_preprocessed_input"]


def _require_import() -> None:
    require_rasterio()
    if _RASTERIO_ERROR is not None:
        raise FeatureDependencyError(
            "rasterio is required to read M2 artifacts and write feature artifacts"
        ) from _RASTERIO_ERROR


@dataclass(frozen=True)
class PreprocessedInput:
    """One M2 analysis-ready artifact, with its mask, QA and provenance."""

    artifact_role: ArtifactRole
    artifact_path: Path
    valid_mask_path: Path
    quality_path: Path
    provenance_path: Path
    metadata: RasterMetadata
    grid: AnalysisGrid
    quality: Mapping[str, Any]
    provenance: ArtifactProvenance

    @property
    def band_names(self) -> Tuple[str, ...]:
        return tuple(
            description.strip()
            for description in self.metadata.descriptions
            if description and description.strip()
        )

    def band_index(self, band_name: str) -> int:
        """Resolve a band description to a 1-based index.

        Band *order* is never assumed. M2 writes descriptions, and a feature
        refers to a band by name, so a reordered artifact cannot silently feed
        the wrong channel into a formula.
        """
        for index, description in enumerate(self.metadata.descriptions, start=1):
            if description and description.strip() == band_name:
                return index
        raise FeatureInputError(
            f"{self.artifact_path} has no band named {band_name!r}; "
            f"available band descriptions: {list(self.band_names)}"
        )

    def read_band(self, band_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(values, valid)`` for one named band.

        The validity mask is the intersection of the mask M2 wrote, the
        per-band nodata/finite check, and — for completeness — the raster's own
        GDAL mask. The M2 mask is authoritative about what was observed; the
        others cannot make an unobserved pixel valid, only invalidate further.
        """
        _require_import()
        index = self.band_index(band_name)
        with rasterio.open(self.artifact_path) as dataset:
            values = dataset.read(index).astype("float64", copy=False)
        valid = self.read_valid_mask() & read_valid_mask(self.artifact_path, band_indexes=(index,))
        return values, valid

    def read_valid_mask(self) -> np.ndarray:
        """Read the M2 ``1 = valid observation`` mask."""
        _require_import()
        with rasterio.open(self.valid_mask_path) as dataset:
            if dataset.count != 1:
                raise FeatureInputError(
                    f"{self.valid_mask_path} must contain exactly one mask band"
                )
            return dataset.read(1) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_role": self.artifact_role.value,
            "artifact_path": str(self.artifact_path),
            "valid_mask_path": str(self.valid_mask_path),
            "quality_path": str(self.quality_path),
            "provenance_path": str(self.provenance_path),
            "band_names": list(self.band_names),
            "grid": self.grid.to_dict(),
            "acquisition_manifest_id": self.provenance.acquisition_manifest_id,
            "source_product_ids": list(self.provenance.source_product_ids),
            "preprocessing_version": self.provenance.preprocessing_version,
            "production_inputs": [item.value for item in self.provenance.production_inputs],
        }


def _sibling_paths(artifact_path: Path) -> Tuple[Path, Path, Path]:
    stem = artifact_path.with_suffix("").name
    parent = artifact_path.parent
    return (
        parent / f"{stem}_valid_mask.tif",
        parent / f"{stem}_quality.json",
        parent / f"{stem}_provenance.yaml",
    )


def load_preprocessed_input(
    artifact_role: ArtifactRole,
    artifact_path: Path | str,
    *,
    allowed_sources: Sequence[ProductionInput],
) -> PreprocessedInput:
    """Load and validate one M2 artifact for feature generation.

    Rejects, with distinct errors:

    * a missing raster, mask, QA or provenance sibling file;
    * a provenance record that is not a ``preprocessed_raster``, which would
      mean M3 was handed something other than an analysis-ready product;
    * a provenance record with no M1 acquisition manifest ID, since the
      traceability chain required by ``AGENTS.md`` §22 would be broken;
    * a declared production input outside the permitted set for this feature
      set, including any validation-only source.
    """
    _require_import()
    path = Path(artifact_path)
    if not path.is_file():
        raise FeatureInputError(f"M2 artifact does not exist: {path}")
    mask_path, quality_path, provenance_path = _sibling_paths(path)
    for required, label in (
        (mask_path, "valid mask"),
        (quality_path, "quality record"),
        (provenance_path, "provenance record"),
    ):
        if not required.is_file():
            raise FeatureInputError(
                f"M2 artifact {path.name} has no {label} at {required}. M3 consumes "
                "complete M2 outputs only; a raster without its mask and provenance "
                "cannot be traced or masked correctly."
            )

    metadata = require_valid_raster(path, min_bands=1)
    try:
        provenance = ArtifactProvenance.from_yaml(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FeatureInputError(f"could not parse provenance at {provenance_path}: {exc}") from exc
    if provenance.artifact_type is not ArtifactType.PREPROCESSED_RASTER:
        raise FeatureInputError(
            f"{provenance_path} declares artifact_type "
            f"{provenance.artifact_type.value!r}; M3 consumes "
            f"{ArtifactType.PREPROCESSED_RASTER.value!r} artifacts only"
        )
    if not provenance.acquisition_manifest_id:
        raise FeatureInputError(
            f"{provenance_path} has no acquisition_manifest_id; a feature artifact "
            "must remain traceable to the M1 manifest and the original source scenes"
        )
    if not provenance.production_inputs:
        raise FeatureInputError(
            f"{provenance_path} declares no production input; the source of this "
            "artifact cannot be established"
        )

    forbidden = {item.value for item in ValidationOnlySource}
    permitted = set(allowed_sources)
    for declared in provenance.production_inputs:
        if declared.value in forbidden:
            raise FeatureBoundaryError(
                f"{provenance_path} declares a validation-only source as a "
                "production input; this is a critical defect (AGENTS.md §3)"
            )
        if declared not in permitted:
            raise FeatureBoundaryError(
                f"{provenance_path} declares production input {declared.value!r}, "
                f"which is not in this feature set's allowed_sources "
                f"{sorted(item.value for item in permitted)}"
            )

    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FeatureInputError(f"could not parse QA at {quality_path}: {exc}") from exc

    if not any(description and description.strip() for description in metadata.descriptions):
        raise FeatureInputError(
            f"{path} has no band descriptions. M3 addresses bands by name so that a "
            "reordered stack cannot feed the wrong channel into a formula."
        )

    return PreprocessedInput(
        artifact_role=artifact_role,
        artifact_path=path,
        valid_mask_path=mask_path,
        quality_path=quality_path,
        provenance_path=provenance_path,
        metadata=metadata,
        grid=metadata.as_grid(),
        quality=quality,
        provenance=provenance,
    )


@dataclass(frozen=True)
class FeatureInputs:
    """The set of M2 artifacts one feature set is generated from."""

    inputs: Mapping[ArtifactRole, PreprocessedInput]

    @classmethod
    def load(
        cls,
        artifact_paths: Mapping[ArtifactRole, Path | str],
        *,
        allowed_sources: Sequence[ProductionInput],
    ) -> "FeatureInputs":
        if not artifact_paths:
            raise FeatureInputError("no M2 artifacts were supplied to feature generation")
        loaded = {
            role: load_preprocessed_input(role, path, allowed_sources=allowed_sources)
            for role, path in artifact_paths.items()
        }
        return cls(inputs=loaded)

    def require(self, role: ArtifactRole) -> PreprocessedInput:
        value = self.inputs.get(role)
        if value is None:
            raise FeatureInputError(
                f"the enabled features require the {role.value!r} M2 artifact, which "
                "was not supplied"
            )
        return value

    def require_all(self, roles: Sequence[ArtifactRole]) -> None:
        missing = [role.value for role in roles if role not in self.inputs]
        if missing:
            raise FeatureInputError(
                f"the enabled features require M2 artifacts that were not supplied: " f"{missing}"
            )

    def reference_grid(self) -> AnalysisGrid:
        """The grid every input must share.

        Chosen as the first artifact in :class:`ArtifactRole` declaration order
        so the choice is deterministic and independent of mapping iteration.
        """
        for role in ArtifactRole:
            if role in self.inputs:
                return self.inputs[role].grid
        raise FeatureInputError("no M2 artifacts available to establish a reference grid")

    def require_common_grid(self, *, tolerance: Optional[float] = None) -> Dict[str, Any]:
        """Verify every input shares the analysis grid; never resample.

        ``docs/data-contract.md`` §0.5.1 makes M2 responsible for spatial
        normalisation. If two inputs disagree on CRS, transform, resolution,
        dimensions, extent or pixel alignment, the correct action is to fail and
        send the operator back to M2 — resampling here would hide a registration
        error behind an interpolation and silently invalidate every change
        feature computed from the pair.
        """
        reference = self.reference_grid()
        reports: Dict[str, Any] = {}
        for role in ArtifactRole:
            candidate = self.inputs.get(role)
            if candidate is None:
                continue
            try:
                report = require_aligned(reference, candidate.grid, tolerance=tolerance)
            except Exception as exc:
                raise FeatureGridError(
                    f"{candidate.artifact_path} does not share the analysis grid of the "
                    f"other M2 inputs ({exc}). M3 does not resample; re-run M2 onto a "
                    "common target grid."
                ) from exc
            reports[role.value] = report.to_dict()
        return {
            "reference_grid": reference.to_dict(),
            "tolerance": tolerance,
            "per_artifact": reports,
        }

    def acquisition_manifest_ids(self) -> Tuple[str, ...]:
        seen: list[str] = []
        for role in ArtifactRole:
            candidate = self.inputs.get(role)
            if candidate is None:
                continue
            value = candidate.provenance.acquisition_manifest_id
            if isinstance(value, str) and value and value not in seen:
                seen.append(value)
        return tuple(seen)

    def source_product_ids(self) -> Tuple[str, ...]:
        seen: list[str] = []
        for role in ArtifactRole:
            candidate = self.inputs.get(role)
            if candidate is None:
                continue
            for product_id in candidate.provenance.source_product_ids:
                if product_id not in seen:
                    seen.append(product_id)
        return tuple(seen)

    def production_inputs(self) -> Tuple[ProductionInput, ...]:
        seen: list[ProductionInput] = []
        for role in ArtifactRole:
            candidate = self.inputs.get(role)
            if candidate is None:
                continue
            for item in candidate.provenance.production_inputs:
                if item not in seen:
                    seen.append(item)
        return tuple(seen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            role.value: self.inputs[role].to_dict() for role in ArtifactRole if role in self.inputs
        }
