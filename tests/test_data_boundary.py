"""Production-input boundary guard.

``AGENTS.md`` §3 states that accidentally importing validation data into
inference or feature generation is a **critical defect**. ``architecture.md``
§18 lists the same rule as an architectural invariant.

A prose rule relies on a reviewer remembering it. This test turns it into a
condition that fails in CI, which is the only form of the rule that survives
contact with a deadline.

Scope and exemptions
--------------------
Scanned: ``src/floodmap/`` production modules and ``configs/``.

Exempt, with reason:

* ``src/floodmap/evaluation/`` — ``architecture.md`` §11 designates this the one
  layer permitted to read validation references, and only after a production
  prediction exists.
* ``src/floodmap/utils/provenance.py`` — must name the forbidden sources in
  order to reject them.
* ``docs/`` and ``tests/`` — must name the forbidden sources to document and
  test the rule.
* In ``configs/``, the dedicated ``validation_only:`` block is where these
  sources are legitimately declared. The guard checks they do not appear in a
  ``production:`` block.

What this guard does and does not prove
---------------------------------------
It is a textual check. It catches the realistic failure — a path, URL or loader
for a forbidden product appearing in a production module — but it cannot prove
the absence of leakage through an indirect route such as a generically named
file that happens to contain EMSR927 data. Dataset identity is additionally
tracked in ``docs/dataset-registry.md`` and in the provenance record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
import yaml

from tests.conftest import REPO_ROOT

#: Case-insensitive markers for validation-only sources (AGENTS.md §3).
FORBIDDEN_MARKERS = (
    "emsr927",
    "emsr 927",
    "copernicus ems",
    "copernicus_ems",
    "unosat",
    "unitar",
)

#: Production modules exempt from the scan, with justification above.
#:
#: Kept as an explicit, short list of individual paths rather than a pattern
#: such as "any __init__.py", so that each exemption is a deliberate decision
#: and the hole it opens stays visible.
EXEMPT_PATHS = (
    # architecture.md §11 designates evaluation the one layer permitted to read
    # validation references, and only after a production prediction exists.
    "src/floodmap/evaluation/",
    # Must name the forbidden sources in order to reject them.
    "src/floodmap/utils/provenance.py",
    # Package docstring states the production/validation boundary, which requires
    # naming the forbidden sources. Documentation only: this module must never
    # contain data access. Caught by this guard on its first run, which is the
    # behaviour we want from it.
    "src/floodmap/__init__.py",
)


def _production_python_files() -> Iterator[Path]:
    """Python modules in the production path, excluding documented exemptions."""
    for path in sorted((REPO_ROOT / "src" / "floodmap").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if any(relative.startswith(exempt) or relative == exempt for exempt in EXEMPT_PATHS):
            continue
        yield path


def _contains_forbidden_marker(text: str) -> list[str]:
    lowered = text.lower()
    return [marker for marker in FORBIDDEN_MARKERS if marker in lowered]


class TestProductionCodeIsClean:
    def test_no_production_module_references_a_validation_only_source(self):
        offenders: dict[str, list[str]] = {}
        for path in _production_python_files():
            found = _contains_forbidden_marker(path.read_text(encoding="utf-8"))
            if found:
                offenders[path.relative_to(REPO_ROOT).as_posix()] = found

        assert not offenders, (
            "Validation-only source referenced in a production module. "
            "AGENTS.md §3 classifies this as a critical defect.\n"
            + "\n".join(f"  {where}: {markers}" for where, markers in offenders.items())
        )

    def test_scan_actually_covers_production_modules(self):
        """Guard against the guard silently scanning nothing.

        A low bound, not an exact count: adding a legitimate exemption should not
        break this, but a path-resolution bug that scans zero files must.
        """
        scanned = list(_production_python_files())
        assert (
            len(scanned) >= 8
        ), f"expected the production package to be scanned, saw {len(scanned)}"

    def test_exempt_paths_exist(self):
        """A stale exemption would silently widen the hole it was meant to describe."""
        for exempt in EXEMPT_PATHS:
            assert (REPO_ROOT / exempt).exists(), f"exempt path no longer exists: {exempt}"


class TestConfigBoundary:
    def test_production_block_has_no_validation_only_source(self):
        config = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
        found = _contains_forbidden_marker(yaml.safe_dump(config["production"]))
        assert (
            not found
        ), f"validation-only source inside configs/data.yaml production block: {found}"

    def test_production_block_lists_only_permitted_sources(self):
        from floodmap.utils.provenance import (
            ProductionInput,
        )  # local import keeps the guard standalone

        permitted_keys = {
            "sentinel1",
            "sentinel2",
            "dem",
            "osm",
            "training_datasets",
        }
        config = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
        assert set(config["production"]) == permitted_keys
        # The enum and the config describe the same five permitted sources.
        assert len(ProductionInput) == len(permitted_keys)

    def test_validation_only_block_is_separate_and_disabled(self):
        config = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
        assert "validation_only" in config
        assert config["validation_only"]["enabled"] is False

    def test_post_event_osm_is_not_permitted_in_production(self):
        config = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
        assert config["production"]["osm"]["allow_post_event_edits"] is False

    def test_evaluation_reference_comparison_is_disabled_by_default(self):
        """Comparison runs only after a production prediction exists."""
        config = yaml.safe_load(
            (REPO_ROOT / "configs" / "evaluation.yaml").read_text(encoding="utf-8")
        )
        assert config["reference_comparison"]["enabled"] is False

    def test_leakage_control_for_validation_sources_is_asserted(self):
        config = yaml.safe_load(
            (REPO_ROOT / "configs" / "evaluation.yaml").read_text(encoding="utf-8")
        )
        controls = config["leakage_controls"]
        assert controls["assert_no_validation_source_in_production"] is True


class TestGuardDetectsViolations:
    """The guard must be able to fail. A check that cannot fail is decoration."""

    @pytest.mark.parametrize(
        "sample",
        [
            "path = 'data/raw/EMSR927/flood_extent.shp'",
            "URL = 'https://example.org/UNOSAT/damage.geojson'",
            "# load the Copernicus EMS product for thresholding",
        ],
    )
    def test_forbidden_markers_are_detected(self, sample: str):
        assert _contains_forbidden_marker(sample)

    @pytest.mark.parametrize(
        "sample",
        [
            "path = 'data/raw/sentinel1/before.tif'",
            "dem = load('copernicus-dem')",
            "osm = load_snapshot('2026-07-01')",
        ],
    )
    def test_permitted_references_are_not_flagged(self, sample: str):
        assert not _contains_forbidden_marker(sample)
