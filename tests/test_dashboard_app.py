"""Streamlit app smoke tests via streamlit's AppTest harness.

These do not judge the layout. They assert the app runs end to end against both
data sources without raising, that the synthetic banner is shown only on the
synthetic path, that the real path never falls back to synthetic data, and that
the workflow panel distinguishes a blocked stage from a completed one.

Requires the optional ``dashboard`` extra (``streamlit``, ``folium``). When it is
not installed these tests are skipped rather than silently passing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

streamlit = pytest.importorskip(
    "streamlit", reason="the optional dashboard extra (streamlit) is not installed"
)
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = REPO_ROOT / "dashboard" / "app.py"
DEMO_SOURCE = "Synthetic demo bundle"
REAL_SOURCE = "Real M6.3 artifacts"


def _run_app(source: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    if source is not None:
        at.sidebar.radio[0].set_value(source).run(timeout=60)
    return at


def _body_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown]
    parts += [c.value for c in at.caption]
    parts += [w.value for w in at.warning]
    parts += [e.value for e in at.error]
    parts += [s.value for s in at.success]
    parts += [i.value for i in at.info]
    return " ".join(str(p) for p in parts)


class TestDemoSource:
    def test_app_runs_without_exception(self):
        at = _run_app(DEMO_SOURCE)
        assert not at.exception, at.exception

    def test_synthetic_banner_is_shown(self):
        at = _run_app(DEMO_SOURCE)
        warnings = " ".join(w.value for w in at.warning)
        assert "SYNTHETIC PLACEHOLDER" in warnings

    def test_evidence_groups_are_labelled(self):
        body = _body_text(_run_app(DEMO_SOURCE))
        assert "Model prediction" in body
        assert "Network inference" in body

    def test_unavailable_layer_is_declared(self):
        text = " ".join(c.value for c in _run_app(DEMO_SOURCE).caption)
        assert "Unavailable layers" in text or "unavailable" in text

    def test_attribution_is_rendered(self):
        text = " ".join(c.value for c in _run_app(DEMO_SOURCE).caption)
        assert "OpenStreetMap" in text
        assert "Copernicus Sentinel" in text


class TestRealSource:
    def test_real_is_the_default_source(self):
        at = _run_app()
        assert at.sidebar.radio[0].value == REAL_SOURCE

    def test_app_runs_without_exception(self):
        at = _run_app(REAL_SOURCE)
        assert not at.exception, at.exception

    def test_real_banner_names_the_real_artifacts(self):
        at = _run_app(REAL_SOURCE)
        source_notices = " ".join(s.value for s in at.info)
        assert "REAL M6.3 artifacts" in source_notices

    def test_synthetic_banner_never_appears_on_the_real_path(self):
        at = _run_app(REAL_SOURCE)
        warnings = " ".join(w.value for w in at.warning)
        assert "SYNTHETIC PLACEHOLDER" not in warnings

    def test_workflow_panel_distinguishes_completed_from_blocked(self, tmp_path, monkeypatch):
        from floodmap.dashboard.realdata import M63_RELATIVE_DIR, PROJECT_ROOT_ENV
        from floodmap.dashboard.workflow import M64_BLOCKER, M64_DIR
        from tests.m63_fixtures import valid_blocker, write_blocker, write_m63_fixture

        write_m63_fixture(tmp_path / M63_RELATIVE_DIR)
        write_blocker(tmp_path / M64_DIR, M64_BLOCKER, valid_blocker("m6.4"))
        monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
        body = _body_text(_run_app(REAL_SOURCE))
        assert "Workflow dependency panel" in body
        assert "Completed on real data" in body
        assert "Blocked — missing prerequisite" in body

    def test_blocked_stages_are_named(self):
        body = _body_text(_run_app(REAL_SOURCE))
        assert "M6.4 infrastructure exposure" in body
        assert "M6.5 network disruption" in body

    def test_real_case_reports_candidate_metrics(self):
        """Either real metrics are shown, or an explicit unusable-artifact error."""
        at = _run_app(REAL_SOURCE)
        body = _body_text(at)
        assert "Real M6.3 artifacts" in body
        usable = "valid pixels" in body and "candidate pixels" in body
        unavailable = at.error and "not usable" in " ".join(e.value for e in at.error)
        assert usable or unavailable

    def test_observed_support_rows_keep_the_four_areas_distinct(self):
        """Extent, delivered, valid and candidate areas stay separately labelled.

        This table is the one place a reader could conflate "the study area" with
        "the observed area", so its row labels, statuses and the measurement-type
        caption are pinned. Skipped only when the git-ignored real artifacts are
        absent, which is the same tolerance the other real-path tests use.
        """
        at = _run_app(REAL_SOURCE)
        if not any("Study extent (AOI polygon)" in str(t.value) for t in at.table):
            pytest.skip("real M6.3 artifacts are absent in this checkout")
        frame = next(t.value for t in at.table if "Study extent (AOI polygon)" in str(t.value))
        assert [str(v) for v in frame["area"]] == [
            "Study extent (AOI polygon)",
            "Delivered (AOI ∩ raster window)",
            "Unobserved (no delivered pixels)",
            "Validly observed (pre/post VV+VH, pixel-centre estimate)",
            "Darkening candidates",
        ]
        statuses = [str(v) for v in frame["status"]]
        assert "Study extent — context, not evidence" in statuses
        assert "Observed / derived" in statuses
        assert "Unknown — not unaffected, flood-free or undamaged" in statuses
        assert "Change candidate — NOT confirmed flood or damage" in statuses
        captions = " ".join(c.value for c in at.caption)
        assert "pixel-centre estimate" in captions
        assert "geometric polygon intersection" in captions
        assert "lower bound" in captions

    def test_missing_real_artifacts_are_never_backfilled_with_demo_data(self):
        """A broken real path must error, not quietly render the synthetic demo."""
        at = _run_app(REAL_SOURCE)
        body = _body_text(at)
        if at.error:
            assert "SYNTHETIC PLACEHOLDER" not in body
            assert "not usable" in body
        else:
            assert "Candidate" in body or "candidate" in body
