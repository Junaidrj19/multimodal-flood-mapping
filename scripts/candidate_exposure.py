#!/usr/bin/env python3
"""M6.4 — infrastructure exposure to M6.3 Sentinel-1 RTC change candidates.

Reads the existing M6.3 candidate mask and its valid-data mask, reads pre-event
OpenStreetMap roads, buildings and settlements **from disk**, and reports which
features intersect candidate regions and which lie in never-observed areas.

This command performs no download. The study AOI is configured ``UNVERIFIED``
and therefore does not authorise one, so ``--osm-dir`` must already contain
layers produced earlier by ``scripts/impact.py`` (``osm_roads.geojson``,
``osm_buildings.geojson``, ``osm_settlements.geojson`` and
``osm_provenance.json``). When those layers are absent, the command writes a
machine-readable blocker report describing exactly what is missing and exits
non-zero rather than fabricating infrastructure.

Results are candidate exposure. A candidate region intersecting a road is
evidence of spatial coincidence with radar change, not of flooding, destruction
or inaccessibility.

Example
-------
``.venv/bin/python scripts/candidate_exposure.py \
    --candidate-mask artifacts/m6_3/rtc_change_candidates/rtc_change_candidates.tif \
    --valid-mask artifacts/m6_3/rtc_change_candidates/rtc_valid_mask.tif \
    --osm-dir data/raw/osm/trishuli-corridor-UNVERIFIED-9035eeee \
    --out-dir artifacts/m6_4/candidate_infrastructure_exposure``
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

from floodmap.infrastructure.candidate_exposure import (  # noqa: E402
    CANDIDATE_STATUSES,
    CandidateExposureError,
    IntersectionRules,
    MissingInfrastructureLayers,
    assess_candidate_intersections,
    blocker_report,
    candidate_exposure_provenance,
    check_layers_overlap_mask,
    load_candidate_pair_metadata,
    load_local_osm_layers,
    summarize_candidate_exposure,
    write_blocker_report,
    write_candidate_exposure_artifacts,
)
from floodmap.infrastructure.candidate_mask import (  # noqa: E402
    CandidateMaskError,
    load_candidate_mask,
)
from floodmap.utils.config import load_config  # noqa: E402

#: Concrete, policy-compliant steps that would unblock the stage. Written into
#: the blocker artifact so the next operator does not have to rediscover them.
#:
#: The extent itself is no longer the open question: M6.15 adopted the
#: human-approved corridor and recorded its producer, source, construction
#: parameters and digests in the AOI provenance sidecar. What is still missing
#: is a separate authorisation to retrieve pre-event OSM, which the adopted
#: ``unverified`` provenance status deliberately does not grant.
NEXT_STEP = (
    "Do not bypass the authorisation gate. The study extent is the human-approved "
    "corridor configured in acquisition.aoi, whose producer, source and construction "
    "parameters are already recorded in its provenance sidecar; its provenance_status "
    "remains 'unverified' precisely so that adopting a geometry does not also authorise "
    "data retrieval. Obtain an explicit human authorisation for pre-event OSM retrieval "
    "for this AOI, separately from the AOI adoption decision.",
    "With that authorisation, fetch pre-event OSM for an explicit pre-event snapshot: "
    "python scripts/impact.py --aoi-id <aoi-id> --aoi-bbox <min_lon> <min_lat> "
    "<max_lon> <max_lat> --event-date 2026-08-26 --out <osm-dir>  (run without --mask, "
    "which performs OSM retrieval only and writes osm_*.geojson plus "
    "osm_provenance.json). The snapshot must predate the event date.",
    "Re-run this command with --osm-dir pointing at that directory. No change to this "
    "stage is required. Exposure will be computed only over the AOI-scoped valid "
    "support; roads in the unobserved 36.6% of the extent remain unassessed and must "
    "not be reported as unaffected.",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-mask",
        type=Path,
        required=True,
        help="M6.3 rtc_change_candidates.tif",
    )
    parser.add_argument(
        "--valid-mask",
        type=Path,
        help="M6.3 rtc_valid_mask.tif; strongly recommended, enables a cross-check",
    )
    parser.add_argument(
        "--m63-summary",
        type=Path,
        help="M6.3 rtc_change_summary.json; defaults to the candidate mask's directory",
    )
    parser.add_argument(
        "--osm-dir",
        type=Path,
        required=True,
        help="directory holding pre-event osm_*.geojson and osm_provenance.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="output artifact directory")
    parser.add_argument(
        "--config",
        type=Path,
        help="override configs/candidate_exposure.yaml",
    )
    parser.add_argument(
        "--allow-missing-osm-provenance",
        action="store_true",
        help=(
            "proceed when osm_provenance.json is absent. The OSM snapshot and AOI are "
            "then unrecorded and the pre-event rule cannot be checked, so the result is "
            "not reportable. Development use only."
        ),
    )
    return parser


def _print_summary(summary: dict) -> None:
    mask = summary["data_coverage"]["candidate_mask"]
    counts = mask["pixel_counts"]
    print()
    print("M6.4 candidate infrastructure exposure (candidate evidence only)")
    print(f"  candidate mask:  {mask['path']}")
    print(f"  stage:           {mask['stage']}")
    print(f"  grid:            {mask['crs']}, {mask['pixel_size_m']} m, {mask['shape']}")
    print(
        f"  pixels:          candidate {counts['candidate']}, "
        f"not_candidate {counts['not_candidate']}, not_observed {counts['not_observed']}"
    )
    print(f"  candidate grid area: {mask['candidate_grid_area_km2']} km2 (NOT flood area)")
    osm = summary.get("osm") or {}
    print(f"  OSM snapshot:    {osm.get('snapshot')} (retrieval performed: False)")
    print(f"  AOI provenance:  {summary.get('aoi_provenance_status')}")
    print()
    header = f"  {'layer':<12} {'total':>7} " + " ".join(
        f"{s.replace('CANDIDATE_', 'CAND_'):>22}" for s in CANDIDATE_STATUSES
    )
    print(header)
    for name, entry in summary["layers"].items():
        row = entry["count_by_status"]
        print(
            f"  {name:<12} {entry['features_total']:>7} "
            + " ".join(f"{row[s]:>22}" for s in CANDIDATE_STATUSES)
        )
    for name, entry in summary["layers"].items():
        if "candidate_intersecting_length_km" in entry:
            print(
                f"\n  {name}: {entry['candidate_intersecting_length_km']} km on candidate "
                f"pixels of {entry['total_length_km']} km mapped "
                f"({entry['not_observed_length_km']} km not observed)"
            )
            for highway, km in list(
                entry.get("candidate_intersecting_km_by_highway", {}).items()
            )[:6]:
                print(f"    {highway:<16} {km} km")
    print(
        "\n  Interpretation: spatial intersection with SAR change candidates. Not "
        "confirmed flood, damage or inaccessibility."
    )


def _print_blocker(report: dict) -> None:
    print()
    print("M6.4 PARTIALLY COMPLETE — BLOCKER", file=sys.stderr)
    print(f"  reason:   {report['reason']}", file=sys.stderr)
    print(f"  searched: {report['searched_directory']}", file=sys.stderr)
    print(f"  missing:  {', '.join(report['missing_layers']) or '(directory absent)'}", file=sys.stderr)
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
        load_config("candidate_exposure") if args.config is None else _load_override(args.config)
    )
    inputs = config["inputs"]
    osm_config = inputs["osm"]
    rules = IntersectionRules.from_config(config)

    mask_dir = args.candidate_mask.parent
    summary_path = args.m63_summary or (mask_dir / "rtc_change_summary.json")
    valid_mask = args.valid_mask
    if valid_mask is None and (mask_dir / "rtc_valid_mask.tif").is_file():
        valid_mask = mask_dir / "rtc_valid_mask.tif"

    # The candidate mask and its M6.3 lineage are loaded first, so a blocker on
    # the infrastructure side can still report exactly which real inputs were
    # available.
    try:
        pair = load_candidate_pair_metadata(summary_path)
        mask = load_candidate_mask(
            args.candidate_mask,
            valid_mask,
            expected_stage=str(inputs["candidate_mask_stage"]),
            alignment_tolerance=(
                None
                if inputs.get("alignment_tolerance") is None
                else float(inputs["alignment_tolerance"])
            ),
        )
    except (CandidateExposureError, CandidateMaskError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        osm = load_local_osm_layers(
            args.osm_dir,
            required_layers=tuple(osm_config["required_layers"]),
            optional_layers=tuple(osm_config["optional_layers"]),
            require_provenance_sidecar=(
                bool(osm_config["require_provenance_sidecar"])
                and not args.allow_missing_osm_provenance
            ),
            event_date=pair.get("event_date"),
            require_pre_event_snapshot=bool(osm_config["require_pre_event_snapshot"]),
        )
    except MissingInfrastructureLayers as exc:
        report = blocker_report(
            reason=str(exc),
            missing=exc.missing,
            searched=exc.searched,
            mask=mask,
            pair=pair,
            next_step=NEXT_STEP,
        )
        path = write_blocker_report(report, args.out_dir)
        _print_blocker(report)
        print(f"  blocker report: {path}", file=sys.stderr)
        return 2
    except CandidateExposureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        aoi_report = check_layers_overlap_mask(osm, mask)
        assessed = assess_candidate_intersections(osm.layers, mask, rules)
        summary = summarize_candidate_exposure(
            assessed, mask, rules, osm=osm, aoi_report=aoi_report, pair=pair
        )
        provenance = candidate_exposure_provenance(
            mask=mask,
            osm=osm,
            summary=summary,
            pair=pair,
            config_version=str(config.get("version", "unversioned")),
        )
        paths = write_candidate_exposure_artifacts(
            assessed, summary, args.out_dir, provenance=provenance
        )
    except CandidateExposureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    print()
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


def _load_override(path: Path) -> dict:
    import yaml

    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise SystemExit(f"error: {path} must contain a mapping")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
