#!/usr/bin/env python3
"""Build the M6.5 readiness gate and write a current execution record.

The M6.5 artifact directory holds a ``candidate_disruption_blocker.json`` from
2026-10-17, when the stage genuinely refused to run. The stage has since run.
This script does **not** rewrite that blocker — ``AGENTS.md`` §22 wants the audit
trail intact — it writes a *separate* reconciliation record alongside it, stating
which artifact is current and why, and recording the three readiness verdicts.

It runs no science, downloads nothing, trains nothing, and overwrites no
scientific artifact. The only file it writes is the execution record named
below, inside the git-ignored ``artifacts/`` tree.

Usage
-----

::

    .venv/bin/python scripts/m65_readiness_gate.py
    .venv/bin/python scripts/m65_readiness_gate.py --out-dir <dir>
    .venv/bin/python scripts/m65_readiness_gate.py --check   # no write

Exit codes
----------

===  =========================================================================
  0  Gate built. Stage completion and labelled research use resolved; operational
     use is blocked, which is the expected and correct outcome.
  1  A path argument is unusable.
  2  M6.5 execution could not be verified from the artifacts (not executed,
     currently blocked, or indeterminate).
===  =========================================================================

``--require-operational`` additionally exits 3 when operational use is blocked.
It exists so the blocked state is *testable*, not so it can be bypassed: there
is no flag that grants operational authorisation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from floodmap.dashboard.readiness import (  # noqa: E402
    ExecutionVerdict,
    GateDecision,
    UseClass,
    build_readiness_report,
    screen_claim,
)
from floodmap.dashboard.workflow import M65_BLOCKER, M65_DIR, M65_OUTPUT  # noqa: E402

#: Written beside the blocker, never over it.
EXECUTION_RECORD = "candidate_disruption_execution_record.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile M6.5 execution state and report the readiness gate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"M6.5 artifact directory (default: {M65_DIR}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report the gate without writing the execution record.",
    )
    parser.add_argument(
        "--require-operational",
        action="store_true",
        help="Exit 3 if operational use is blocked. For testing the gate, not bypassing it.",
    )
    return parser.parse_args(argv)


def _print_report(report, record_path: Path | None) -> None:
    execution = report.execution
    print("=" * 78)
    print("M6.5 READINESS GATE")
    print("=" * 78)
    print(f"stage            : {report.stage}")
    print(f"execution verdict: {execution.verdict.value}")
    print(f"  summary        : {execution.summary_generated_at} ({execution.summary_status})")
    print(f"  blocker        : {execution.blocker_generated_at} ({execution.blocker_status})")
    print(f"  blocker stale  : {execution.blocker_is_stale}")
    print(f"  hubs assessed  : {execution.hub_reachability_assessed}")
    if execution.problems:
        print("  problems       :")
        for problem in execution.problems:
            print(f"    - {problem}")
    print(f"detail           : {execution.detail}")

    print()
    print("-" * 78)
    print("VERDICTS")
    print("-" * 78)
    for outcome in report.outcomes:
        print(f"  {outcome.use.value:24s} {outcome.decision.value}")
        for reason in outcome.reasons:
            print(f"      reason  : {reason}")
        for blocker in outcome.blockers:
            print(f"      BLOCKER : {blocker}")

    print()
    print("-" * 78)
    print("OPEN LIMITATIONS")
    print("-" * 78)
    for limitation in report.limitations:
        print(f"  {limitation.limitation_id:24s} {limitation.state.value}")
        print(f"      {limitation.detail}")
        for item in limitation.evidence:
            print(f"      evidence: {item}")

    print()
    print("-" * 78)
    print("MANDATORY PRESENTATION LABELS")
    print("-" * 78)
    for label in report.required_labels:
        print(f"  - {label}")

    print()
    if record_path is not None:
        print(f"execution record written: {record_path}")
    else:
        print("execution record not written (--check)")
    print(
        "blocker report preserved unmodified: "
        f"{execution.blocker_path}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    out_dir = args.out_dir if args.out_dir is not None else REPO_ROOT / M65_DIR
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        print(f"error: not a directory: {out_dir}", file=sys.stderr)
        return 1

    report = build_readiness_report(
        summary_path=out_dir / M65_OUTPUT,
        blocker_path=out_dir / M65_BLOCKER,
        # Independent clean-run reproduction has not been performed. This stays
        # False until one actually is, and no code path infers it.
        clean_run_reproduced=False,
    )

    # The gate's own prose must satisfy the gate's own vocabulary rule.
    offenders: list[str] = []
    for text in list(report.required_labels) + [item.detail for item in report.limitations]:
        if screen_claim(text):
            offenders.append(text)
    if offenders:
        print(
            "error: the gate's own text failed its claim screen; refusing to write a "
            "record that violates AGENTS.md §8:",
            file=sys.stderr,
        )
        for text in offenders:
            print(f"  - {text}", file=sys.stderr)
        return 1

    record_path: Path | None = None
    if not args.check:
        payload = {
            "record_type": "m6.5_execution_status_reconciliation",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "written_by": "scripts/m65_readiness_gate.py",
            "note": (
                "This record reconciles M6.5 execution state. It does not replace "
                f"{M65_BLOCKER}, which is retained unmodified as historical audit "
                "evidence of a real refusal on 2026-10-17."
            ),
            "preserved_blocker": M65_BLOCKER,
            "reproducibility": {
                "prior_execution_recorded": report.execution.executed,
                "independent_clean_run_reproduced": False,
                "per_file_checksum_manifest_available": False,
                "note": (
                    "Successful prior execution and a passing test suite are recorded. "
                    "Neither establishes independent clean-run artifact "
                    "reproducibility, which was not performed."
                ),
            },
            **report.to_dict(),
        }
        record_path = out_dir / EXECUTION_RECORD
        record_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    _print_report(report, record_path)

    if report.execution.verdict in (
        ExecutionVerdict.NOT_EXECUTED,
        ExecutionVerdict.BLOCKED_CURRENT,
        ExecutionVerdict.INDETERMINATE,
    ):
        print(
            f"\nM6.5 execution not verified ({report.execution.verdict.value}).",
            file=sys.stderr,
        )
        return 2

    operational = report.outcome(UseClass.OPERATIONAL)
    if args.require_operational and (
        operational is None or operational.decision is not GateDecision.PERMITTED_WITH_LABELS
    ):
        print(
            "\noperational use is blocked, as expected: "
            f"{len(operational.blockers) if operational else 0} open blockers.",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
