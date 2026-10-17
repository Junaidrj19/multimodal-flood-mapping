#!/usr/bin/env python3
"""M6.5 — road-network disruption and settlement connectivity under
hypothetical closure of M6.4 candidate-exposed roads.

Reads the M6.4 candidate-exposure artifacts **from disk**, builds the pre-event
road graph, applies a stated hypothetical closure rule, and compares
connectivity and settlement reachability against the untouched baseline graph.

This command performs no download. The study AOI is configured ``UNVERIFIED``
and therefore does not authorise one, so ``--exposure-dir`` must already contain
``candidate_exposure_roads.geojson`` and
``candidate_exposure_settlements.geojson`` written by
``scripts/candidate_exposure.py``. When those artifacts are absent, the command
writes a machine-readable blocker report describing exactly what is missing and
exits non-zero rather than fabricating a road network.

What the output is
------------------
A scenario comparison. M6.4 measured how much of each road lies on M6.3
Sentinel-1 RTC darkening-candidate pixels; nothing in that measurement
establishes that such a road is impassable. A settlement reported
``SCENARIO_UNREACHABLE`` would lose its modelled road route to a hub **if**
candidate-intersecting roads were closed. It is not a finding that a road was
destroyed or that anyone is cut off.

Example
-------
``.venv/bin/python scripts/candidate_disruption.py \
    --exposure-dir artifacts/m6_4/candidate_infrastructure_exposure \
    --osm-dir data/raw/osm/trishuli-corridor-UNVERIFIED-9035eeee \
    --out-dir artifacts/m6_5/candidate_network_disruption``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from floodmap.infrastructure.candidate_mask import (  # noqa: E402
    CandidateMaskError,
    load_candidate_mask,
)
from floodmap.network.candidate_disruption import (  # noqa: E402
    CANDIDATE_AND_UNOBSERVED_CLOSURE,
    CANDIDATE_CLOSURE,
    SETTLEMENT_STATUSES,
    CandidateDisruptionError,
    ClosureRules,
    MissingNetworkInputs,
    analysis_crs,
    assess_candidate_disruption,
    build_candidate_road_network,
    candidate_disruption_provenance,
    closure_table,
    disruption_blocker_report,
    load_network_inputs,
    write_candidate_disruption_artifacts,
    write_disruption_blocker_report,
)
from floodmap.utils.config import load_config  # noqa: E402

#: Concrete, policy-compliant steps that would unblock the stage. Written into
#: the blocker artifact so the next operator does not have to rediscover them.
#:
#: The extent is settled: M6.15 adopted the human-approved corridor and M6.16
#: reconciled M6.3 to it. The open prerequisites are an explicit authorisation
#: for pre-event OSM retrieval and the M6.4 exposure outputs that depend on it.
NEXT_STEP = (
    "Do not bypass the authorisation gate. The study extent is the human-approved "
    "corridor configured in acquisition.aoi, with its producer, source and construction "
    "parameters recorded in its provenance sidecar; provenance_status stays 'unverified' "
    "so that adopting a geometry does not also authorise retrieval. Obtain an explicit "
    "human authorisation for pre-event OSM retrieval for this AOI.",
    "With that authorisation, fetch pre-event OSM for an explicit pre-event snapshot: "
    "python scripts/impact.py --aoi-id <aoi-id> --aoi-bbox <min_lon> <min_lat> "
    "<max_lon> <max_lat> --event-date 2026-08-26 --out <osm-dir>  (run without --mask). "
    "That writes osm_roads/settlements/hubs.geojson plus osm_provenance.json.",
    "Run M6.4: python scripts/candidate_exposure.py --candidate-mask "
    "artifacts/m6_3/rtc_change_candidates/rtc_change_candidates.tif --valid-mask "
    "artifacts/m6_3/rtc_change_candidates/rtc_valid_mask.tif --osm-dir <osm-dir> "
    "--out-dir artifacts/m6_4/candidate_infrastructure_exposure.",
    "Re-run this command with --exposure-dir pointing at the M6.4 output and --osm-dir "
    "at the pre-event OSM directory that supplies the hub layer. No change to this "
    "stage is required.",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exposure-dir",
        type=Path,
        required=True,
        help="M6.4 output directory holding candidate_exposure_*.geojson",
    )
    parser.add_argument(
        "--osm-dir",
        type=Path,
        help=(
            "pre-event OSM directory supplying osm_hubs.geojson. Without it, road "
            "disruption is reported and hub reachability is recorded as not assessed."
        ),
    )
    parser.add_argument(
        "--candidate-mask",
        type=Path,
        help=(
            "M6.3 rtc_change_candidates.tif. Required only for --closure-unit piece, "
            "which re-measures every graph piece against the raster."
        ),
    )
    parser.add_argument(
        "--valid-mask",
        type=Path,
        help="M6.3 rtc_valid_mask.tif; enables the candidate/valid cross-check",
    )
    parser.add_argument(
        "--closure-unit",
        choices=("road", "piece"),
        help="override network.closure_unit from config",
    )
    parser.add_argument(
        "--analysis-crs",
        help=(
            "override the metric CRS the graph is built in; defaults to the candidate "
            "mask CRS recorded by M6.4"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="output artifact directory")
    parser.add_argument("--config", type=Path, help="override configs/candidate_disruption.yaml")
    parser.add_argument(
        "--require-hubs",
        action="store_true",
        help="fail instead of reporting hub reachability as not assessed",
    )
    return parser


def _load_override(path: Path) -> dict:
    import yaml

    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise SystemExit(f"error: {path} must contain a mapping")
    return parsed


def _print_summary(summary: dict) -> None:
    graph = summary["graph"]
    rule = summary["closure_rule"]
    print()
    print("M6.5 candidate road-network disruption (hypothetical closure scenario)")
    print(f"  exposure input:  {summary['inputs']['candidate_exposure_directory']}")
    print(f"  upstream stage:  {summary['inputs']['candidate_exposure_stage']}")
    print(f"  OSM snapshot:    {summary['inputs']['osm_snapshot']} (retrieval performed: False)")
    print(f"  AOI provenance:  {summary['inputs']['aoi_provenance_status']}")
    print(f"  analysis CRS:    {graph['analysis_crs']}")
    print()
    print(
        f"  Baseline graph: {graph['nodes']} nodes, {graph['pieces_routable']} routable "
        f"pieces of {graph['pieces_total']}, {graph['road_km']} km, "
        f"{graph['connected_components_baseline']} components"
    )
    print(f"  Closure rule:   unit={rule['unit']}, threshold={rule['threshold_m']} m")
    for scenario in (CANDIDATE_CLOSURE, CANDIDATE_AND_UNOBSERVED_CLOSURE):
        entry = summary["disruption"][scenario]
        print(
            f"    {scenario:<36} {entry['pieces_closed']:>6} pieces, "
            f"{entry['closed_km']:>9} km, {entry['connected_components']:>4} components"
        )
    for highway, km in list(
        summary["disruption"][CANDIDATE_CLOSURE]["closed_km_by_highway"].items()
    )[:6]:
        print(f"      {highway:<16} {km} km")

    accounting = summary["road_length_accounting"]
    print()
    print(
        f"  Road length: {accounting['total_km']} km mapped, "
        f"{accounting['candidate_intersecting_km']} km on candidate pixels, "
        f"{accounting['not_observed_km']} km not observed"
    )
    uncertainty = summary["coverage_uncertainty"]
    print(
        f"  Coverage: {uncertainty['pieces_closed_only_if_unobserved_closed']} pieces "
        f"({uncertainty['km_closed_only_if_unobserved_closed']} km) close only if "
        "not-observed roads close"
    )
    settlements = summary["settlements"]
    print(
        f"  Settlements: {settlements['on_network']} of {settlements['total']} on the "
        f"network, {settlements['component_shrunk_under_candidate_closure']} lose "
        "reachable network"
    )

    reachability = summary["hub_reachability"]
    if not reachability["assessed"]:
        print()
        print(f"  Hub reachability NOT assessed: {reachability['reason']}")
    else:
        short = {
            "SCENARIO_REACHABLE": "REACH",
            "SCENARIO_DETOURED": "DETOUR",
            "SCENARIO_UNREACHABLE": "UNREACH",
            "UNREACHABLE_ONLY_IF_UNOBSERVED_CLOSED": "UNOBS?",
            "INSUFFICIENT_DATA": "NO_DATA",
        }
        print()
        print(
            f"  {'hub group':<12} {'hubs':>7} "
            + " ".join(f"{short[s]:>8}" for s in SETTLEMENT_STATUSES)
        )
        for group, entry in reachability["groups"].items():
            counts = entry["status_counts"]
            hubs = f"{entry['hubs_on_network']}/{entry['hubs_total']}"
            print(
                f"  {group:<12} {hubs:>7} "
                + " ".join(f"{counts[s]:>8}" for s in SETTLEMENT_STATUSES)
            )
        print("\n  scenario-unreachable count, by closure threshold (m):")
        for group, per_threshold in reachability["sensitivity_by_closure_m"].items():
            cells = ", ".join(
                f"{t} m: {v['SCENARIO_UNREACHABLE']}" for t, v in per_threshold.items()
            )
            print(f"    {group:<12} {cells}")
        for group, entry in reachability["groups"].items():
            names = [c["name"] or c["osm_id"] for c in entry["scenario_unreachable"]]
            if names:
                print(f"\n  {group}, unreachable under candidate closure: {', '.join(names)}")
    print(
        "\n  Interpretation: conditional on the stated closure rule. Candidate "
        "intersection is radar-change evidence, not observed impassability, so this is "
        "not confirmed road failure or settlement isolation."
    )


def _print_blocker(report: dict) -> None:
    print()
    print("M6.5 PARTIALLY COMPLETE — BLOCKER", file=sys.stderr)
    print(f"  reason:   {report['reason']}", file=sys.stderr)
    print(f"  searched: {report['searched_directory']}", file=sys.stderr)
    missing = ", ".join(report["missing_inputs"]) or "(directory absent)"
    print(f"  missing:  {missing}", file=sys.stderr)
    auth = report["authorisation"]
    print(
        f"  gate:     AOI provenance {auth['aoi_provenance_status']}, "
        f"authorises_download={auth['authorises_download']}",
        file=sys.stderr,
    )
    print("  next step:", file=sys.stderr)
    for index, step in enumerate(report["next_step"], start=1):
        print(f"    {index}. {step}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = (
        load_config("candidate_disruption")
        if args.config is None
        else _load_override(args.config)
    )
    rules = ClosureRules.from_config(config)
    if args.closure_unit is not None:
        rules = ClosureRules(
            snap_max_m=rules.snap_max_m,
            closure_unit=args.closure_unit,
            closure_min_candidate_m=rules.closure_min_candidate_m,
            sensitivity_closure_m=rules.sensitivity_closure_m,
            detour_ratio=rules.detour_ratio,
            close_outside_grid_roads=rules.close_outside_grid_roads,
        )

    try:
        inputs = load_network_inputs(
            args.exposure_dir,
            config=config,
            osm_dir=args.osm_dir,
            require_hubs=True if args.require_hubs else None,
        )
    except MissingNetworkInputs as exc:
        report = disruption_blocker_report(
            reason=str(exc),
            missing=exc.missing,
            searched=exc.searched,
            upstream=_upstream_hint(args.exposure_dir, config),
            next_step=NEXT_STEP,
        )
        path = write_disruption_blocker_report(report, args.out_dir)
        _print_blocker(report)
        print(f"  blocker report: {path}", file=sys.stderr)
        return 2
    except CandidateDisruptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mask = None
    if args.candidate_mask is not None:
        try:
            valid_mask = args.valid_mask
            if valid_mask is None:
                beside = args.candidate_mask.parent / "rtc_valid_mask.tif"
                valid_mask = beside if beside.is_file() else None
            mask = load_candidate_mask(
                args.candidate_mask,
                valid_mask,
                expected_stage=str(
                    (inputs.exposure_summary or {})
                    .get("data_coverage", {})
                    .get("candidate_mask", {})
                    .get("stage")
                    or "m6.3-rtc-change-candidates-0.2.0"
                ),
            )
        except (CandidateDisruptionError, CandidateMaskError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    elif rules.closure_unit == "piece":
        print(
            "error: closure_unit='piece' re-measures every graph piece against the M6.3 "
            "candidate raster; pass --candidate-mask or use --closure-unit road",
            file=sys.stderr,
        )
        return 1

    try:
        crs = analysis_crs(inputs.exposure_summary, args.analysis_crs)
        if mask is not None and not crs.equals(mask.crs):
            print(
                f"error: the analysis CRS {crs.to_string()} differs from the candidate "
                f"mask CRS {mask.crs.to_string()}; piece re-measurement requires the graph "
                "and the raster to share a grid (AGENTS.md §7)",
                file=sys.stderr,
            )
            return 1
        network, settlement_snap, hub_snap = build_candidate_road_network(
            inputs.roads,
            crs,
            settlements=inputs.settlements,
            hubs=inputs.hubs,
            snap_max_m=rules.snap_max_m,
        )
        table = closure_table(network, inputs.roads, rules, mask=mask)
        pieces, settlements, routes, summary = assess_candidate_disruption(
            network, table, inputs, settlement_snap, hub_snap, rules
        )
        provenance = candidate_disruption_provenance(
            inputs=inputs,
            summary=summary,
            config_version=str(config.get("version", "unversioned")),
        )
        paths = write_candidate_disruption_artifacts(
            pieces, settlements, routes, summary, args.out_dir, provenance=provenance
        )
    except CandidateDisruptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    print()
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


def _upstream_hint(exposure_dir: Path, config: dict) -> dict:
    """Whatever M6.4 lineage is readable, for the blocker record.

    The blocker should still name the Sentinel-1 pair and AOI when the summary
    survived but the geometry did not, so a reader can see which real inputs
    were present.
    """
    import json

    summary_path = exposure_dir / str(config["inputs"]["summary_file"])
    if not summary_path.is_file():
        return {}
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
