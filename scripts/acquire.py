#!/usr/bin/env python3
"""Discover permitted Earth Observation metadata and write an acquisition manifest.

Examples
--------
Metadata-only dry run (no network):

``python scripts/acquire.py --dry-run --aoi-bbox 0.0 0.0 0.1 0.1 \
    --aoi-id synthetic --before-days 12 --after-days 12``

Metadata discovery is the default. Product downloads require the explicit
``--download`` flag and are independently recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

# Support running directly from a source checkout without requiring an
# editable install first. Installed users still resolve the same package.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from floodmap.acquisition.aoi import AreaOfInterest
from floodmap.acquisition.config import AcquisitionConfig, load_acquisition_config
from floodmap.acquisition.errors import AcquisitionError
from floodmap.acquisition.manifest import AcquisitionManifest
from floodmap.acquisition.outcomes import DiscoveryStatus
from floodmap.acquisition.providers import CdseOdataProvider, DiscoveryResult, SearchRequest
from floodmap.acquisition.scenes import SensorKind
from floodmap.acquisition.selection import (
    SelectionOutcome,
    select_sentinel1_pair,
    select_sentinel2_pair,
)
from floodmap.acquisition.temporal import EventSpec, TemporalPlan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data", help="configuration name under configs/")
    parser.add_argument("--aoi-id", help="stable AOI identifier")
    parser.add_argument(
        "--aoi-bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="AOI bounds in EPSG:4326",
    )
    parser.add_argument(
        "--aoi-geojson",
        type=Path,
        help="path to a GeoJSON Polygon geometry or Feature in EPSG:4326",
    )
    parser.add_argument("--event-date", type=date.fromisoformat)
    parser.add_argument("--event-time-utc", type=_parse_datetime)
    parser.add_argument("--before-days", type=int)
    parser.add_argument("--after-days", type=int)
    parser.add_argument(
        "--s2-strategy",
        choices=("nearest_in_time", "least_cloud_then_nearest"),
        help="override configured Sentinel-2 selection strategy",
    )
    parser.add_argument("--dry-run", action="store_true", help="build queries without network I/O")
    parser.add_argument(
        "--download",
        action="store_true",
        help="explicitly enable product downloads after metadata selection",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/acquisition_manifest.json"),
        help="output .json or .yaml manifest path",
    )
    return parser


def _parse_datetime(value: str) -> datetime:
    normalised = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalised)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("event time must include a timezone, e.g. Z")
    return parsed


def _load_aoi(args: argparse.Namespace, config: AcquisitionConfig) -> AreaOfInterest:
    if args.aoi_bbox is not None:
        return AreaOfInterest.from_bbox(
            args.aoi_id or "cli-aoi",
            args.aoi_bbox,
            is_synthetic=bool(args.aoi_id and "synthetic" in args.aoi_id.lower()),
        )
    if args.aoi_geojson is not None:
        raw = json.loads(args.aoi_geojson.read_text(encoding="utf-8"))
        geometry = raw.get("geometry") if raw.get("type") == "Feature" else raw
        return AreaOfInterest.from_geojson_geometry(
            args.aoi_id or "cli-aoi",
            geometry,
            is_synthetic=bool(args.aoi_id and "synthetic" in args.aoi_id.lower()),
        )
    return config.require_aoi()


def _window(
    sensor: str,
    config: AcquisitionConfig,
    data: Mapping[str, Any],
    args: argparse.Namespace,
):
    sensor_config = config.sentinel1 if sensor == "sentinel1" else config.sentinel2
    before = args.before_days if args.before_days is not None else sensor_config.window_before_days
    after = args.after_days if args.after_days is not None else sensor_config.window_after_days
    # Legacy event-level values remain a documented fallback for compatibility
    # with the M0 data configuration.
    event = data.get("event", {})
    fallback = event if isinstance(event, Mapping) else {}
    return sensor_config.resolved_search_window(
        sensor=sensor,
        fallback={
            "window_before_days": (
                before if before is not None else fallback.get("window_before_days")
            ),
            "window_after_days": after if after is not None else fallback.get("window_after_days"),
        },
    )


def _request(
    *,
    aoi: AreaOfInterest,
    sensor: SensorKind,
    plan: TemporalPlan,
    config: AcquisitionConfig,
) -> SearchRequest:
    sensor_config = config.sentinel1 if sensor is SensorKind.SENTINEL1 else config.sentinel2
    product_types = sensor_config.product_types
    product_type = product_types[0] if len(product_types) == 1 else None
    return SearchRequest(
        aoi=aoi,
        sensor=sensor,
        interval=plan.full_extent,
        product_type=product_type,
        max_cloud_percent=(
            sensor_config.max_scene_cloud_percent if sensor is SensorKind.SENTINEL2 else None
        ),
        max_results=config.provider.max_results,
        page_size=config.provider.page_size,
    )


def _not_attempted(request: SearchRequest, provider: str, executed_url: str) -> DiscoveryResult:
    return DiscoveryResult(
        status=DiscoveryStatus.NOT_ATTEMPTED,
        request=request,
        provider=provider,
        executed_url=executed_url,
        executed_urls=(executed_url,),
    )


def _selection(
    sensor: SensorKind,
    result: DiscoveryResult,
    plan: TemporalPlan,
    config: AcquisitionConfig,
    strategy_override: Optional[str],
) -> SelectionOutcome:
    scenes = list(result.scenes) if result.status is DiscoveryStatus.SUCCESS else []
    if sensor is SensorKind.SENTINEL1:
        return select_sentinel1_pair(
            scenes,
            plan,
            require_same_relative_orbit=config.sentinel1.require_same_relative_orbit,
        )
    strategy = strategy_override or config.selection_strategy("sentinel2")
    return select_sentinel2_pair(
        scenes,
        plan,
        strategy=strategy,
        max_cloud_percent=config.sentinel2.max_scene_cloud_percent,
    )


def run_acquisition(args: argparse.Namespace) -> AcquisitionManifest:
    """Run one configured acquisition and return its manifest."""
    config, data = load_acquisition_config(args.config)
    aoi = _load_aoi(args, config)
    raw_event = data["event"]
    event = EventSpec(
        event_date=args.event_date or date.fromisoformat(str(raw_event["date"])),
        event_time_utc=args.event_time_utc,
        label=str(raw_event.get("name", "")),
    )
    plans: dict[str, TemporalPlan] = {}
    requests: dict[str, SearchRequest] = {}
    for sensor_name, sensor_kind in (
        ("sentinel1", SensorKind.SENTINEL1),
        ("sentinel2", SensorKind.SENTINEL2),
    ):
        sensor_config = (
            config.sentinel1 if sensor_kind is SensorKind.SENTINEL1 else config.sentinel2
        )
        if not sensor_config.enabled:
            continue
        window = _window(sensor_name, config, data, args)
        plan = event.plan(window)
        plans[sensor_kind.value] = plan
        requests[sensor_kind.value] = _request(
            aoi=aoi, sensor=sensor_kind, plan=plan, config=config
        )

    provider_config = config.provider
    username, password = provider_config.credentials.values_from_environment()
    provider = CdseOdataProvider(
        base_url=provider_config.catalogue_url,
        download_url_template=provider_config.download_url_template,
        timeout_seconds=provider_config.timeout_seconds,
        username=username,
        password=password,
    )
    discoveries: dict[str, DiscoveryResult] = {}
    selections: dict[str, SelectionOutcome] = {}
    downloads = []
    for sensor_name, request in requests.items():
        if args.dry_run:
            discoveries[sensor_name] = _not_attempted(
                request, provider.name, provider.build_query_url(request)
            )
            continue
        result = provider.search(request)
        discoveries[sensor_name] = result
        selections[sensor_name] = _selection(
            SensorKind(sensor_name),
            result,
            plans[sensor_name],
            config,
            args.s2_strategy,
        )
        outcome = selections[sensor_name]
        if outcome.is_usable:
            for scene in (outcome.selected_before, outcome.selected_after):
                if scene is not None:
                    downloads.append(
                        provider.download(
                            scene,
                            Path(config.download.destination),
                            # Configuration selects the destination, but an
                            # explicit CLI flag is required to authorize a fetch.
                            enabled=bool(args.download),
                        )
                    )

    manifest = AcquisitionManifest.from_run(
        aoi=aoi,
        event=event,
        temporal_plans=plans,
        requests=requests,
        discoveries=discoveries,
        selections=selections,
        downloads=downloads,
        provider=provider.name,
        config_version=str(data.get("version", "")),
        dry_run=args.dry_run,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.suffix.lower() in {".yaml", ".yml"}:
        args.manifest.write_text(manifest.to_yaml(), encoding="utf-8")
    else:
        args.manifest.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_acquisition(args)
    except (AcquisitionError, OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "status": manifest.status.value,
                "provider": manifest.provider,
                "selected_sensors": [
                    sensor for sensor, outcome in manifest.selections.items() if outcome.is_usable
                ],
                "downloads": [download.status.value for download in manifest.downloads],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
