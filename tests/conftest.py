"""Shared test helpers.

Resolves the repository root once so tests do not depend on the working
directory from which pytest was invoked.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, resolved from this file's location (tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
