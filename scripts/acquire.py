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

from floodmap.acquisition.aoi import AreaOfInterest, geometry_from_geojson_document
from floodmap.acquisition.config import AcquisitionConfig, load_acquisition_config
from floodmap.acquisition.errors import AcquisitionError, NoSameTrackPairError
from floodmap.acquisition.manifest import AcquisitionManifest
from floodmap.acquisition.outcomes import DiscoveryStatus
from floodmap.acquisition.providers import CdseOdataProvider, DiscoveryResult, SearchRequest
from floodmap.acquisition.scenes import SensorKind
from floodmap.acquisition.selection import (
    NoSameTrackPair,
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
        "--query-envelope-prefilter",
        action="store_true",
        help=(
            "send the AOI's bounding envelope as the catalogue spatial filter instead "
            "of the AOI geometry. The envelope is a strict superset, so no candidate is "
            "excluded, but it is NOT used for coverage: the measured coverage gate still "
            "runs against the AOI itself. Required when the AOI polygon is too large for "
            "the provider's URL limit (the 1,515-vertex Trishuli corridor produces a "
            "~41 kB query and is refused). The manifest records the substitution."
        ),
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


def _geometry_from_geojson(raw: Any) -> Mapping[str, Any]:
    """Extract the geometry mapping from a GeoJSON document.

    Thin delegate to :func:`floodmap.acquisition.aoi.geometry_from_geojson_document`
    so the CLI and the configuration loader cannot drift apart about which
    containers are acceptable. Kept as a named function because the tests pin
    the CLI's behaviour through it.

    Three containers are accepted, because an AOI polygon arrives in whichever
    one the producing tool emitted: a bare ``Polygon`` geometry, a ``Feature``
    wrapping one, and a single-feature ``FeatureCollection``. The last is the
    ``ogr2ogr`` default output shape, so rejecting it would turn a routine
    shapefile/GeoPackage conversion into an unexplained CLI failure.

    A ``FeatureCollection`` carrying more than one feature is **rejected rather
    than reduced to its first member.** Which feature is the study area would
    be our guess, not the operator's instruction, and silently adopting one
    would change the analysis extent without anybody choosing it.
    """
    return geometry_from_geojson_document(raw)


def _load_aoi(args: argparse.Namespace, config: AcquisitionConfig) -> AreaOfInterest:
    if args.aoi_bbox is not None:
        return AreaOfInterest.from_bbox(
            args.aoi_id or "cli-aoi",
            args.aoi_bbox,
            is_synthetic=bool(args.aoi_id and "synthetic" in args.aoi_id.lower()),
        )
    if args.aoi_geojson is not None:
        raw = json.loads(args.aoi_geojson.read_text(encoding="utf-8"))
        return AreaOfInterest.from_geojson_geometry(
            args.aoi_id or "cli-aoi",
            _geometry_from_geojson(raw),
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
    aoi: AreaOfInterest,
) -> SelectionOutcome:
    """Resolve the selection for one sensor, always passing the AOI geometry.

    The AOI is threaded through explicitly: AOI coverage is a mandatory gate, so
    a selection that cannot evaluate it must not be produced at all.
    """
    scenes = list(result.scenes) if result.status is DiscoveryStatus.SUCCESS else []
    if sensor is SensorKind.SENTINEL1:
        return select_sentinel1_pair(
            scenes,
            plan,
            aoi=aoi,
            minimum_aoi_coverage=config.sentinel1.minimum_aoi_coverage,
            required_polarisations=config.sentinel1.required_polarisations,
            require_same_relative_orbit=config.sentinel1.require_same_relative_orbit,
        )
    strategy = strategy_override or config.selection_strategy("sentinel2")
    return select_sentinel2_pair(
        scenes,
        plan,
        aoi=aoi,
        strategy=strategy,
        max_cloud_percent=config.sentinel2.max_scene_cloud_percent,
        processing_level_preference=config.sentinel2.processing_level_preference,
        allow_processing_level_fallback=config.sentinel2.allow_processing_level_fallback,
        minimum_aoi_coverage=config.sentinel2.minimum_aoi_coverage,
        require_same_tile=config.sentinel2.require_same_tile,
        allow_mixed_processing_levels=config.sentinel2.allow_mixed_processing_levels,
    )


def _enforce_no_pair_policy(
    sensor_name: str,
    outcome: SelectionOutcome,
    config: AcquisitionConfig,
    *,
    discovery: Optional[DiscoveryResult] = None,
) -> None:
    """Apply ``on_no_same_track_pair`` to a Sentinel-1 outcome.

    Previously the setting was validated by the configuration and then never
    read, so ``"fail"`` silently behaved like ``"record"`` and the run exited 0.

    ``discovery`` is needed because "no same-track pair" is not the only way to
    end up with no pair, and it is not the most likely one: a query that never
    ran produces an empty candidate list too. Diagnosing that as an orbit
    geometry finding sends the operator looking at the wrong thing — an M6.2
    run did exactly that, blaming relative orbits for what was a 41 kB query URL
    the endpoint refused. So a failed discovery is reported as a failed query.
    """
    if sensor_name != "sentinel-1" or not isinstance(outcome, NoSameTrackPair):
        return
    if discovery is not None and discovery.status is DiscoveryStatus.FAILED:
        raise AcquisitionError(
            "the Sentinel-1 catalogue query did not complete, so no candidate pair "
            f"could be evaluated. This is a query failure, not an orbit finding: "
            f"{discovery.error}\n"
            "The empty candidate list is a consequence of the failure, and no "
            "conclusion about Sentinel-1 availability may be drawn from it."
        )
    policy = config.sentinel1.on_no_same_track_pair
    if policy == "fail":
        raise NoSameTrackPairError(
            "acquisition.sentinel1.on_no_same_track_pair is 'fail' and no Sentinel-1 "
            "before/after pair shares a verified relative orbit. No manifest was written. "
            "Inspect the discovery candidates, widen the search window, or set the policy "
            "to 'warn_and_continue' to record the failure and continue."
        )
    if policy == "warn_and_continue":
        print(
            f"warning: no same-track Sentinel-1 pair for {sensor_name}; recorded and "
            "continuing because on_no_same_track_pair is 'warn_and_continue'",
            file=sys.stderr,
        )


def _write_manifest(path: Path, manifest: AcquisitionManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(manifest.to_yaml(), encoding="utf-8")
    else:
        path.write_text(manifest.to_json() + "\n", encoding="utf-8")


def _authorise_download(args: argparse.Namespace, aoi: AreaOfInterest, config) -> None:
    """Refuse a fetch that configuration or AOI provenance does not permit.

    Called before any catalogue request so an unauthorised run fails fast
    rather than after issuing queries.

    Two gates, both of which previously did not exist:

    ``acquisition.download.enabled``
        The key was defined in :class:`DownloadConfig` and set in
        ``configs/data.yaml``, but **no production code read it**: only
        ``--download`` was consulted, so a configuration that said
        ``enabled: false`` still fetched. A config key that reads as a download
        switch and silently is not one is worse than no key at all.

    AOI provenance
        Product bytes are only fetched against a geometry whose origin is
        established. An unverified extent is permitted for metadata discovery —
        that is reversible and tells an operator what the catalogue holds — but
        the rasters a download produces are what every downstream flood claim is
        computed from, so fetching them would commit the project to a study area
        nobody has verified (``docs/dataset-registry.md`` §1.6).
    """
    if not args.download or args.dry_run:
        return
    if not config.download.enabled:
        raise AcquisitionError(
            "--download was passed but acquisition.download.enabled is false in the "
            "configuration. Both must agree before product bytes are fetched: set "
            "the configuration flag deliberately rather than relying on the CLI "
            "flag alone."
        )
    if aoi.authorises_download:
        return

    # The AOI is not verified. A fetch is still permissible, but only under a
    # recorded operator exception that names the exact products and pins the AOI
    # bytes. The exception does NOT change provenance_status: the AOI stays
    # unverified and every downstream disclosure keeps firing.
    authorisation = config.download.exploratory_authorisation
    gaps = "\n  - ".join(aoi.provenance_gaps) or "(none recorded)"
    if not authorisation.enabled:
        raise AcquisitionError(
            f"--download was passed, but AOI {aoi.aoi_id!r} has provenance_status "
            f"'{aoi.provenance_status.value}' rather than 'verified', so a product "
            "download is not authorised.\n"
            "Metadata discovery against this AOI is permitted; fetching the rasters "
            "that downstream flood claims are computed from is not, because the "
            "study extent itself is not established.\n"
            f"Unresolved provenance questions:\n  - {gaps}\n"
            "Either resolve these and set provenance_status to 'verified', or record a "
            "narrow exception under acquisition.download.exploratory_authorisation "
            "naming the exact product ids and the AOI sha256. Do not flip the status "
            "without the answers."
        )
    if authorisation.aoi_sha256 != aoi.source_sha256:
        raise AcquisitionError(
            "an exploratory download authorisation is recorded, but it was granted "
            f"against AOI sha256 {authorisation.aoi_sha256!r} and the configured AOI is "
            f"{aoi.source_sha256!r}. The exception was granted for a specific study "
            "extent; it does not carry over to a different geometry."
        )
    print(
        "warning: downloading against an UNVERIFIED AOI under a recorded operator "
        f"exception (authorised_by={authorisation.authorised_by!r}). The AOI provenance "
        "status is unchanged and remains unverified; no flood-evidence claim is "
        "authorised from these products.",
        file=sys.stderr,
    )


def _query_aoi(args: argparse.Namespace, aoi: AreaOfInterest) -> tuple[AreaOfInterest, Optional[dict]]:
    """Resolve the geometry sent to the catalogue, and how it differs from the AOI.

    Default is the AOI itself. ``--query-envelope-prefilter`` substitutes the AOI's
    bounding envelope because the full polygon exceeds the provider's URL length
    limit; the returned record documents the substitution for the manifest so the
    query geometry can never be mistaken for the study extent. Coverage is
    measured against ``aoi`` regardless of this choice.
    """
    if not args.query_envelope_prefilter:
        return aoi, None
    envelope = aoi.query_envelope()
    return envelope, {
        "kind": "bounding_envelope",
        "aoi_id": aoi.aoi_id,
        "query_aoi_id": envelope.aoi_id,
        "query_bounds": envelope.bounds().as_list(),
        "superset_of_aoi": True,
        "coverage_measured_against": aoi.aoi_id,
        "reason": (
            "the provider spatial filter travels in the request URL and the AOI "
            f"polygon ({len(aoi.geometry.exterior) - 1} vertices) produces a query "
            "larger than the endpoint accepts, which it answers by closing the "
            "connection; the envelope is a strict superset of the AOI, so it cannot "
            "exclude a candidate the AOI would have matched"
        ),
    }


def run_acquisition(args: argparse.Namespace) -> AcquisitionManifest:
    """Run one configured acquisition and return its manifest."""
    config, data = load_acquisition_config(args.config)
    aoi = _load_aoi(args, config)
    _authorise_download(args, aoi, config)
    query_aoi, query_prefilter = _query_aoi(args, aoi)
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
            aoi=query_aoi, sensor=sensor_kind, plan=plan, config=config
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

    def _build(downloads, *, download_pending: bool) -> AcquisitionManifest:
        return AcquisitionManifest.from_run(
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
            download_pending=download_pending,
            query_prefilter=query_prefilter,
            # Recorded whenever an exception exists, including on a failed or
            # pending run, so the authorisation under which bytes were fetched is
            # auditable from the artifact alone.
            download_authorisation=(
                config.download.exploratory_authorisation.to_record()
                if config.download.exploratory_authorisation.enabled
                else None
            ),
        )

    # --- stage 1: discovery and validated selection (no network fetch) -------
    for sensor_name, request in requests.items():
        if args.dry_run:
            discoveries[sensor_name] = _not_attempted(
                request, provider.name, provider.build_query_url(request)
            )
            continue
        result = provider.search(request)
        discoveries[sensor_name] = result
        outcome = _selection(
            SensorKind(sensor_name),
            result,
            plans[sensor_name],
            config,
            args.s2_strategy,
            aoi,
        )
        selections[sensor_name] = outcome
        try:
            _enforce_no_pair_policy(sensor_name, outcome, config, discovery=result)
        except (NoSameTrackPairError, AcquisitionError):
            # A run that stops must still leave its evidence behind. The failing
            # outcome is recorded first, so the manifest carries the discovery
            # result, every candidate and every rejection reason under a status
            # that says the run did not produce a usable pair; then the error
            # propagates, so the exit status still reports the failure. (The
            # pre-download manifest exists for the same reason: an interrupted
            # run must not be indistinguishable from one that never ran.)
            _write_manifest(args.manifest, _build([], download_pending=False))
            raise

    # An interrupted download must not leave the run unrecorded. When a fetch was
    # authorised, the selection result is written first under a status that
    # cannot be mistaken for a finished acquisition, then rewritten with the
    # per-scene download outcomes.
    if args.download and not args.dry_run:
        _write_manifest(args.manifest, _build([], download_pending=True))

    # --- stage 2: explicitly authorised downloads ---------------------------
    downloads = []
    # Per-product enforcement. The run-level gate above authorises the *run*;
    # this authorises each *product*. Both are needed: an exploratory exception
    # names specific products, and a scene that is usable but outside that
    # allowlist must still not be fetched.
    authorisation = config.download.exploratory_authorisation
    aoi_is_verified = aoi.authorises_download
    for sensor_name in requests:
        outcome = selections.get(sensor_name)
        if outcome is None or not outcome.is_usable:
            continue
        for scene in (outcome.selected_before, outcome.selected_after):
            if scene is None:
                continue
            product_id = getattr(scene, "product_id", None)
            permitted = aoi_is_verified or authorisation.covers(
                aoi_sha256=aoi.source_sha256, product_id=product_id
            )
            if not permitted:
                raise AcquisitionError(
                    f"scene {scene.scene_id!r} (product_id={product_id!r}) is not covered "
                    "by the recorded exploratory download authorisation, and the AOI is "
                    "not verified. Refusing to fetch it rather than widening the "
                    "exception implicitly: add the product id deliberately if it is "
                    "meant to be included."
                )
            downloads.append(
                provider.download(
                    scene,
                    Path(config.download.destination),
                    # Configuration selects the destination, but an
                    # explicit CLI flag is required to authorize a fetch.
                    enabled=bool(args.download),
                )
            )

    manifest = _build(downloads, download_pending=False)
    _write_manifest(args.manifest, manifest)
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
                "download_status": manifest.download_status.value,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
