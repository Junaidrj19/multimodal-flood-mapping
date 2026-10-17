#!/usr/bin/env python3
"""Generate auditable change candidates from the existing local RTC pair.

This command does not download imagery, train a model, use a DEM, or run
infrastructure/network inference.  It consumes the already assembled real
Sentinel-1 RTC VV/VH GeoTIFFs and writes:

* ``rtc_change_features.tif`` — VV and VH log-ratio change in dB;
* ``rtc_change_candidates.tif`` — 0/1/255 exploratory darkening candidates;
* ``rtc_valid_mask.tif`` — explicit valid pre/post VV+VH observations;
* ``rtc_aoi_mask.tif`` — pixels whose centre lies inside the configured AOI;
* JSON QA/statistics and linked provenance records.

The study extent is read from ``configs/data.yaml`` and verified against the
geometry's SHA-256.  There is deliberately no free-text ``--aoi-id`` option: a
typed label is not evidence of the extent the statistics were computed against,
and a mislabelled product is worse than a missing one.

All outputs are scoped to that extent.  The delivered RTC rasters are
bounding-box crops cut for an earlier extent, so the raster window and the AOI
are different areas; the summary reports both scopes with named denominators
and never presents the full AOI area as observed area.

The candidate mask must be described as change evidence, not confirmed flood
damage.  RTC input values are gamma0 linear power, not sigma0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from floodmap.baseline.rtc_change_candidates import (  # noqa: E402
    RtcPairMetadata,
    compute_rtc_change_candidates,
    load_configured_aoi_binding,
    load_rtc_change_config,
    write_rtc_change_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, required=True, help="pre-event assembled RTC VV/VH GeoTIFF")
    parser.add_argument("--post", type=Path, required=True, help="post-event assembled RTC VV/VH GeoTIFF")
    parser.add_argument("--out-dir", type=Path, required=True, help="output artifact directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPOSITORY_ROOT / "configs" / "rtc_change_candidates.yaml",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=_REPOSITORY_ROOT / "configs" / "data.yaml",
        help="configuration carrying acquisition.aoi; the AOI identity and digest come from here",
    )
    parser.add_argument(
        "--expect-aoi-id",
        default=None,
        help=(
            "optional guard: fail unless the configured acquisition.aoi.aoi_id equals this "
            "value. Use it to assert which extent a run was meant to use; it never sets it."
        ),
    )
    parser.add_argument("--event-date", required=True)
    parser.add_argument("--pre-scene-id", required=True)
    parser.add_argument("--post-scene-id", required=True)
    parser.add_argument("--pre-acquired-at", required=True)
    parser.add_argument("--post-acquired-at", required=True)
    parser.add_argument("--relative-orbit", type=int, required=True)
    parser.add_argument("--orbit-direction", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_rtc_change_config(args.config)
        inputs = config["inputs"]
        rule = config["candidate_rule"]
        aoi = load_configured_aoi_binding(args.data_config, project_root=_REPOSITORY_ROOT)
        if args.expect_aoi_id is not None and aoi.aoi_id != args.expect_aoi_id:
            raise ValueError(
                f"configured AOI is {aoi.aoi_id!r} but --expect-aoi-id required "
                f"{args.expect_aoi_id!r}; refusing to write outputs keyed to an "
                "extent the caller did not expect"
            )
        result = compute_rtc_change_candidates(
            args.pre,
            args.post,
            aoi=aoi,
            vv_threshold_db=float(rule["vv_change_db_max"]),
            vh_threshold_db=float(rule["vh_change_db_max"]),
            alignment_tolerance=(
                None
                if inputs.get("alignment_tolerance") is None
                else float(inputs["alignment_tolerance"])
            ),
        )
        pair = RtcPairMetadata.from_binding(
            aoi,
            event_date=args.event_date,
            pre_scene_id=args.pre_scene_id,
            post_scene_id=args.post_scene_id,
            pre_acquired_at=args.pre_acquired_at,
            post_acquired_at=args.post_acquired_at,
            relative_orbit=args.relative_orbit,
            orbit_direction=args.orbit_direction,
        )
        paths = write_rtc_change_artifacts(
            result,
            args.out_dir,
            pair=pair,
            vv_threshold_db=float(rule["vv_change_db_max"]),
            vh_threshold_db=float(rule["vh_change_db_max"]),
            config_version=str(config.get("version", "unversioned")),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    scope = summary["aoi_scope"]
    print("RTC change-candidate stage complete (candidate evidence only)")
    print(f"  radiometry: {summary['radiometric_convention']} (NOT sigma0)")
    print(f"  AOI: {summary['aoi_id']}")
    print(f"  AOI sha256: {summary['aoi_source_sha256']}")
    print(f"  AOI provenance: {summary['aoi_provenance_status']}")
    print("  -- delivered raster window scope (bounding-box crop, NOT the AOI) --")
    print(f"    window pixels: {summary['total_pixels']}")
    print(f"    pixels outside AOI: {scope['window_pixels_outside_aoi']} "
          f"({scope['window_pixels_outside_aoi_fraction']:.4%} of window)")
    print("  -- configured AOI scope --")
    print(f"    full AOI area (extent, NOT observed): {scope['aoi_geometric_area_km2']:.6f} km2")
    print(f"    delivered AOI area: {scope['aoi_delivered_area_km2']:.6f} km2 "
          f"({scope['aoi_delivered_fraction']:.4%} of AOI)")
    print(f"    unobserved AOI area: {scope['aoi_unobserved_area_km2']:.6f} km2 "
          f"({scope['aoi_unobserved_fraction']:.4%} of AOI) — unknown, not unaffected")
    print(f"    valid AOI pixels: {scope['aoi_valid_pixels']} "
          f"({scope['aoi_valid_area_km2']:.6f} km2 estimate)")
    print(f"    candidates: {scope['candidate_pixels']} "
          f"({scope['candidate_area_km2']:.6f} km2), "
          f"{scope['candidate_fraction_of_aoi_valid_pixels']:.4%} of valid AOI pixels")
    print(f"  disclosure: {summary['coverage_disclosure']}")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
