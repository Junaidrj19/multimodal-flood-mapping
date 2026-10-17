"""Real-bundle assembly and strict real/demo separation tests.

Two invariants matter here (M6.6 §E):

1. A real bundle is declared ``is_synthetic=False`` and resolves to the real
   files it was built from.
2. The real code path cannot reach the synthetic demo generator, and the demo
   path cannot borrow real values. The AST guard below makes that a test rather
   than a promise.

All fixtures are synthetic; nothing here is a real Trishuli result.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from floodmap.dashboard import build_demo_bundle, build_real_dashboard
from floodmap.dashboard.contract import EvidenceTier
from floodmap.dashboard.realdata import discover_m63_artifacts
from floodmap.dashboard.workflow import M64_BLOCKER, M64_DIR, build_workflow
from tests.conftest import REPO_ROOT
from tests.m63_fixtures import SYNTHETIC_MARKER, valid_blocker, write_blocker, write_m63_fixture

#: Modules that make up the real-data path.
REAL_PATH_MODULES = ("realdata.py", "realbundle.py", "workflow.py", "render.py")


@pytest.fixture()
def real_bundle(tmp_path: Path):
    write_m63_fixture(tmp_path / "artifacts" / "m6_3" / "rtc_change_candidates")
    write_blocker(tmp_path / M64_DIR, M64_BLOCKER, valid_blocker("m6.4"))
    artifacts = discover_m63_artifacts(tmp_path)
    workflow = build_workflow(artifacts, project_root=tmp_path)
    return build_real_dashboard(artifacts, workflow)


class TestRealBundle:
    def test_bundle_is_not_synthetic(self, real_bundle):
        assert real_bundle.manifest.is_synthetic is False
        assert "SYNTHETIC" not in real_bundle.manifest.event_name.upper()

    def test_available_layers_resolve_to_real_files(self, real_bundle):
        available = [layer for layer in real_bundle.manifest.layers if layer.available]
        assert {layer.layer_id for layer in available} == {
            "change_candidate_mask",
            "change_vv",
            "change_vh",
        }
        for layer in available:
            assert real_bundle.resolve_path(layer).is_file()
            assert real_bundle.provenance[layer.layer_id] is not None

    def test_layers_are_observed_derived_not_predictions(self, real_bundle):
        for layer_id in ("change_candidate_mask", "change_vv", "change_vh"):
            layer = next(l for l in real_bundle.manifest.layers if l.layer_id == layer_id)
            assert layer.evidence_tier is EvidenceTier.OBSERVED_DERIVED

    def test_blocked_downstream_stages_are_declared_unavailable(self, real_bundle):
        declared = {
            layer.layer_id: layer
            for layer in real_bundle.manifest.layers
            if not layer.available
        }
        assert "m6_4_exposure" in declared
        reason = declared["m6_4_exposure"].unavailable_reason
        assert reason and "prerequisite missing" in str(reason)
        # An unavailable layer must not carry a path (contract rule).
        assert declared["m6_4_exposure"].path is None

    def test_aoi_comes_from_the_artifact_summary(self, real_bundle):
        assert real_bundle.manifest.aoi_id == "fixture-aoi-SYNTHETIC"
        assert real_bundle.manifest.event_date == "2026-08-26"

    def test_cannot_build_from_unusable_artifacts(self, tmp_path: Path):
        artifacts = discover_m63_artifacts(tmp_path)
        with pytest.raises(ValueError, match="missing"):
            build_real_dashboard(artifacts)

    def test_provenance_carries_required_attributions(self, real_bundle):
        for layer in real_bundle.manifest.layers:
            if not layer.available:
                continue
            record = real_bundle.provenance[layer.layer_id]
            assert len(record.attribution) >= 3


class TestRealDemoSeparation:
    def test_demo_bundle_is_labelled_synthetic_and_real_is_not(self, tmp_path: Path):
        demo_bundle = build_demo_bundle(tmp_path / "demo")
        from floodmap.dashboard import load_manifest

        demo = load_manifest(demo_bundle)
        assert demo.manifest.is_synthetic is True
        assert demo.manifest.aoi_id == "trishuli-demo"
        assert "SYNTHETIC" in demo.manifest.event_name

    def test_real_source_never_references_the_demo_generator(self):
        """The real path must be incapable of falling back to synthetic data."""
        package = REPO_ROOT / "src" / "floodmap" / "dashboard"
        for name in REAL_PATH_MODULES:
            source = (package / name).read_text(encoding="utf-8")
            assert "demobundle" not in source, name
            assert "build_demo_bundle" not in source, name

    def test_real_path_has_no_streamlit_or_folium_dependency(self):
        """The real path must import without the optional dashboard extra."""
        package = REPO_ROOT / "src" / "floodmap" / "dashboard"
        for name in REAL_PATH_MODULES:
            tree = ast.parse((package / name).read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imported.add(node.module.split(".")[0])
            assert "streamlit" not in imported, name
            assert "folium" not in imported, name

    def test_real_modules_import_without_optional_dependencies(self):
        """Exercise the above in a fresh interpreter, not just by inspection."""
        code = (
            "import sys;"
            "sys.path.insert(0, 'src');"
            "import floodmap.dashboard.realdata, floodmap.dashboard.workflow,"
            " floodmap.dashboard.render, floodmap.dashboard.realbundle;"
            "assert 'streamlit' not in sys.modules, 'streamlit imported';"
            "assert 'folium' not in sys.modules, 'folium imported';"
            "print('ok')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout

    def test_fixture_marker_is_explicit(self, tmp_path: Path):
        """Guard against a fixture silently looking like real data."""
        path = write_m63_fixture(tmp_path / "m63")
        provenance = (path / "rtc_change_candidates_provenance.yaml").read_text(encoding="utf-8")
        assert SYNTHETIC_MARKER in provenance
        assert "fixture-aoi-SYNTHETIC" in provenance
