"""Package and repository structure tests.

These verify that the scaffold the rest of the project depends on actually
exists and imports. They assert software structure only — nothing here claims
anything about scientific correctness or model performance.
"""

from __future__ import annotations

import importlib

import pytest

from tests.conftest import REPO_ROOT

#: Subpackages required by architecture.md §13.
EXPECTED_SUBPACKAGES = (
    "acquisition",
    "preprocessing",
    "features",
    "segmentation",
    "infrastructure",
    "network",
    "hydrology",
    "evaluation",
    "reporting",
    "utils",
)

#: Directories required by README.md §10 and architecture.md §14.
EXPECTED_DIRECTORIES = (
    "configs",
    "data",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/samples",
    "models",
    "artifacts",
    "tests",
    "scripts",
    "notebooks",
    "docs",
    "src/floodmap",
)


class TestPackageImports:
    def test_top_level_package_imports(self):
        module = importlib.import_module("floodmap")
        assert module.__version__

    @pytest.mark.parametrize("name", EXPECTED_SUBPACKAGES)
    def test_subpackage_imports(self, name: str):
        module = importlib.import_module(f"floodmap.{name}")
        assert module.__doc__, f"floodmap.{name} must document its scientific responsibility"

    def test_utils_modules_import(self):
        importlib.import_module("floodmap.utils.provenance")
        importlib.import_module("floodmap.utils.config")


class TestRepositoryLayout:
    @pytest.mark.parametrize("relative", EXPECTED_DIRECTORIES)
    def test_directory_exists(self, relative: str):
        assert (REPO_ROOT / relative).is_dir(), f"missing directory: {relative}"

    def test_required_documents_exist(self):
        for name in (
            "README.md",
            "PRD.md",
            "architecture.md",
            "AGENTS.md",
            "docs/data-contract.md",
            "docs/dataset-registry.md",
            "docs/scientific-assumptions.md",
            "docs/evaluation-protocol.md",
            "docs/m4-architecture-decision.md",
        ):
            assert (REPO_ROOT / name).is_file(), f"missing document: {name}"

    def test_gitignore_excludes_generated_data(self):
        """AGENTS.md §15 — raw and generated data must not be committable.

        Checks the declared patterns. Effective behaviour is additionally
        verified by ``test_gitignore_keeps_directory_skeleton``.
        """
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("data/raw/", "data/interim/", "data/processed/", "models/", "artifacts/"):
            assert pattern in text, f".gitignore must exclude {pattern}"

    def test_gitignore_keeps_directory_skeleton(self):
        """The .gitkeep files must survive the data exclusions.

        Git cannot re-include a file whose parent directory is excluded, so
        ``data/raw/`` plus ``!**/.gitkeep`` would silently drop the skeleton.
        The patterns must therefore exclude directory *contents* (``dir/*``).
        """
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for directory in ("data/raw", "data/interim", "data/processed", "models", "artifacts"):
            assert (
                f"{directory}/*" in text
            ), f"{directory} must be excluded as contents ({directory}/*)"
            assert f"!{directory}/.gitkeep" in text, f"{directory}/.gitkeep must be re-included"
