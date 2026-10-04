"""Offline verification for Milestone 1 Earth Observation acquisition."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from floodmap.acquisition.aoi import AreaOfInterest
from floodmap.acquisition.config import load_acquisition_config
from floodmap.acquisition.manifest import AcquisitionManifest
from floodmap.acquisition.outcomes import (
    DiscoveryStatus,
    ManifestStatus,
    RejectionReason,
    SelectionStrategy,
)
from floodmap.acquisition.providers.base import SearchRequest
from floodmap.acquisition.providers.cdse import CdseOdataProvider
from floodmap.acquisition.providers.transport import RequestsTransport
from floodmap.acquisition.scenes import Sentinel1Scene, Sentinel2Scene, SensorKind
from floodmap.acquisition.selection import (
    NoSameTrackPair,
    select_sentinel1_pair,
    select_sentinel2_pair,
)
from floodmap.acquisition.temporal import EventSpec, SearchWindow
from floodmap.utils.provenance import ArtifactType, ProductionInput, Unknown


AOI = AreaOfInterest.from_bbox("synthetic-test", [0.0, 0.0, 0.1, 0.1], is_synthetic=True)
EVENT = EventSpec(
    event_date=date(2024, 1, 10),
    event_time_utc=datetime(2024, 1, 10, 12, tzinfo=timezone.utc),
)
PLAN = EVENT.plan(SearchWindow(before_days=5, after_days=5))


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = json.dumps(payload) if payload is not None else ""
        self.headers = {}

    def json(self):
        return self._payload


def _request(sensor: SensorKind, *, cloud=None, page_size=100, max_results=200):
    return SearchRequest(
        aoi=AOI,
        sensor=sensor,
        interval=PLAN.full_extent,
        max_cloud_percent=cloud,
        page_size=page_size,
        max_results=max_results,
    )


def _product(product_id, name, acquired_at, *, sensor="SENTINEL-1", attrs=None):
    return {
        "Id": product_id,
        "Name": name,
        "ContentDate": {"Start": acquired_at.isoformat().replace("+00:00", "Z")},
        "Online": True,
        "ContentLength": 123,
        "GeoFootprint": {
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.0]]],
        },
        "Attributes": [
            {"Name": "platformShortName", "Value": sensor},
            *([] if attrs is None else attrs),
        ],
    }


def _s1(scene_id, acquired_at, orbit, direction="DESCENDING"):
    return Sentinel1Scene(
        scene_id=scene_id,
        product_id=scene_id,
        acquired_at=acquired_at,
        product_type="IW_GRDH_1S",
        platform_short_name="SENTINEL-1",
        platform_unit="S1A",
        provider="cdse-odata",
        relative_orbit=orbit,
        orbit_direction=direction,
    )


def _s2(scene_id, acquired_at, cloud):
    return Sentinel2Scene(
        scene_id=scene_id,
        product_id=scene_id,
        acquired_at=acquired_at,
        product_type="S2MSI2A",
        platform_short_name="SENTINEL-2",
        platform_unit="S2A",
        provider="cdse-odata",
        scene_cloud_percent=cloud,
    )


class PagingTransport:
    def __init__(self, products):
        self.products = products
        self.calls = []

    def get(self, url, *, params=None, headers=None, timeout_seconds=60.0):
        self.calls.append((url, params))
        skip = int(parse_qs(urlparse(url).query).get("$skip", [0])[0])
        top = int(parse_qs(urlparse(url).query).get("$top", [100])[0])
        return FakeResponse(payload={"value": self.products[skip : skip + top]})


def test_cdse_query_is_encoded_and_contains_required_filters():
    url = CdseOdataProvider().build_query_url(_request(SensorKind.SENTINEL1))
    assert "%24filter=" in url
    assert "%24expand=Attributes" in url
    params = parse_qs(urlparse(url).query)
    query_filter = params["$filter"][0]
    assert "Collection/Name eq 'SENTINEL-1'" in query_filter
    assert "IW_GRDH_1S" in query_filter
    assert "OData.CSC.Intersects" in query_filter
    assert "ContentDate/Start ge" in query_filter


def test_cdse_parses_pages_and_maps_unrepresented_fields_to_unknown():
    before = _product(
        "p1",
        "S1A_IW_GRDH_1SDV_20240105T120000_000000_000000_0000",
        datetime(2024, 1, 5, 12, tzinfo=timezone.utc),
        attrs=[
            {"Name": "productType", "Value": "IW_GRDH_1S"},
            {"Name": "relativeOrbitNumber", "Value": 12},
            {"Name": "orbitDirection", "Value": "DESCENDING"},
        ],
    )
    after = _product(
        "p2",
        "S1A_IW_GRDH_1SDV_20240111T120000_000000_000000_0000",
        datetime(2024, 1, 11, 12, tzinfo=timezone.utc),
        attrs=[
            {"Name": "productType", "Value": "IW_GRDH_1S"},
            {"Name": "relativeOrbitNumber", "Value": 12},
            {"Name": "orbitDirection", "Value": "DESCENDING"},
        ],
    )
    transport = PagingTransport([before, after])
    result = CdseOdataProvider(transport=transport).search(
        _request(SensorKind.SENTINEL1, page_size=1, max_results=2)
    )
    assert result.status is DiscoveryStatus.SUCCESS
    assert result.pages_fetched == 3  # two full pages plus an empty look-ahead page
    assert [int(parse_qs(urlparse(url).query)["$skip"][0]) for url, _ in transport.calls] == [
        0,
        1,
        2,
    ]
    assert isinstance(result.scenes[0].platform_unit, str)
    assert isinstance(result.scenes[0].incidence_angle, Unknown)
    assert result.scenes[0].footprint is not None


def test_cdse_s2_query_has_product_and_cloud_filters():
    url = CdseOdataProvider().build_query_url(_request(SensorKind.SENTINEL2, cloud=40))
    query_filter = parse_qs(urlparse(url).query)["$filter"][0]
    assert "S2MSI2A" in query_filter and "S2MSI1C" in query_filter
    assert "cloudCover" in query_filter
    assert "DoubleAttribute/Value le 40" in query_filter


@pytest.mark.parametrize(
    ("status", "exception"),
    [(401, "auth"), (500, "http")],
)
def test_requests_transport_maps_auth_and_server_errors(status, exception):
    class Session:
        def get(self, *args, **kwargs):
            return FakeResponse(status_code=status, payload={"error": "no"})

    transport = RequestsTransport(session=Session(), max_attempts=1, provider="test")
    if exception == "auth":
        from floodmap.acquisition.errors import ProviderAuthError

        expected = ProviderAuthError
    else:
        from floodmap.acquisition.errors import ProviderHttpError

        expected = ProviderHttpError
    with pytest.raises(expected):
        transport.get("https://example.invalid")


def test_requests_transport_maps_timeout():
    class Session:
        def get(self, *args, **kwargs):
            raise TimeoutError("synthetic timeout")

    from floodmap.acquisition.errors import ProviderTimeout

    transport = RequestsTransport(session=Session(), max_attempts=1, provider="test")
    with pytest.raises(ProviderTimeout):
        transport.get("https://example.invalid")


def test_s1_selection_never_substitutes_a_cross_track_pair():
    before = _s1("before", datetime(2024, 1, 9, tzinfo=timezone.utc), 12)
    after = _s1("after", datetime(2024, 1, 11, tzinfo=timezone.utc), 41)
    result = select_sentinel1_pair([before, after], PLAN)
    assert isinstance(result, NoSameTrackPair)
    assert result.status == "no_same_track_pair"
    assert {item.reason for item in result.rejected_candidates} >= {RejectionReason.ORBIT_MISMATCH}


def test_s1_unknown_orbit_is_rejected_explicitly():
    before = _s1("before", datetime(2024, 1, 9, tzinfo=timezone.utc), None)
    after = _s1("after", datetime(2024, 1, 11, tzinfo=timezone.utc), 12)
    result = select_sentinel1_pair([before, after], PLAN)
    assert isinstance(result, NoSameTrackPair)
    unknown = [item for item in result.rejected_candidates if item.scene_id == "before"]
    assert unknown and unknown[0].reason is RejectionReason.ORBIT_UNKNOWN


def test_s1_unknown_pass_direction_is_not_treated_as_mismatch():
    before = _s1(
        "before",
        datetime(2024, 1, 9, tzinfo=timezone.utc),
        12,
        direction=Unknown(reason="not_available_from_source"),
    )
    after = _s1("after", datetime(2024, 1, 11, tzinfo=timezone.utc), 12)
    result = select_sentinel1_pair([before, after], PLAN)
    assert result.is_usable
    assert not any(
        rejection.reason is RejectionReason.ORBIT_DIRECTION_MISMATCH
        for rejection in result.rejected_candidates
    )


def test_s2_strategy_and_cloud_limit_are_deterministic():
    b = datetime(2024, 1, 9, tzinfo=timezone.utc)
    a = datetime(2024, 1, 11, tzinfo=timezone.utc)
    scenes = [_s2("b-near", b, 50), _s2("b-clear", b - timedelta(days=1), 5), _s2("a", a, 10)]
    nearest = select_sentinel2_pair(scenes, PLAN, strategy=SelectionStrategy.NEAREST_IN_TIME)
    least = select_sentinel2_pair(scenes, PLAN, strategy=SelectionStrategy.LEAST_CLOUD_THEN_NEAREST)
    assert nearest.selected_before.scene_id == "b-near"
    assert least.selected_before.scene_id == "b-clear"
    limited = select_sentinel2_pair(scenes, PLAN, strategy="nearest_in_time", max_cloud_percent=20)
    assert limited.selected_before.scene_id == "b-clear"
    assert any(
        item.reason is RejectionReason.CLOUD_ABOVE_CONFIGURED_LIMIT
        for item in limited.rejected_candidates
    )


def test_manifest_serialization_and_provenance_link():
    before = _s1("before", datetime(2024, 1, 9, tzinfo=timezone.utc), 12)
    after = _s1("after", datetime(2024, 1, 11, tzinfo=timezone.utc), 12)
    selection = select_sentinel1_pair([before, after], PLAN)
    request = _request(SensorKind.SENTINEL1)
    from floodmap.acquisition.providers.base import DiscoveryResult

    discovery = DiscoveryResult(
        status=DiscoveryStatus.SUCCESS,
        request=request,
        scenes=(before, after),
        provider="cdse-odata",
    )
    manifest = AcquisitionManifest.from_run(
        aoi=AOI,
        event=EVENT,
        temporal_plans={"sentinel-1": PLAN},
        requests={"sentinel-1": request},
        discoveries={"sentinel-1": discovery},
        selections={"sentinel-1": selection},
        provider="cdse-odata",
        config_version="0.1.0",
    )
    assert manifest.status is ManifestStatus.COMPLETE
    assert manifest.provenance.artifact_type is ArtifactType.ACQUISITION_MANIFEST
    assert manifest.provenance.production_inputs == [ProductionInput.SENTINEL1]
    restored = AcquisitionManifest.from_yaml(manifest.to_yaml())
    assert restored.to_dict() == manifest.to_dict()
    assert "acquisition_manifest" in manifest.to_json()


def test_manifest_does_not_claim_selection_without_successful_discovery():
    before = _s1("before", datetime(2024, 1, 9, tzinfo=timezone.utc), 12)
    after = _s1("after", datetime(2024, 1, 11, tzinfo=timezone.utc), 12)
    selection = select_sentinel1_pair([before, after], PLAN)
    request = _request(SensorKind.SENTINEL1)
    manifest = AcquisitionManifest.from_run(
        aoi=AOI,
        event=EVENT,
        temporal_plans={"sentinel-1": PLAN},
        requests={"sentinel-1": request},
        discoveries={},
        selections={"sentinel-1": selection},
    )
    assert manifest.status is ManifestStatus.FAILED


def test_download_is_not_attempted_without_explicit_enable(tmp_path):
    from floodmap.acquisition.outcomes import DownloadStatus

    scene = _s1("scene", datetime(2024, 1, 9, tzinfo=timezone.utc), 12)
    result = CdseOdataProvider().download(scene, tmp_path)
    assert result.status is DownloadStatus.NOT_ATTEMPTED
    assert not list(tmp_path.iterdir())


def test_m1_config_declares_aoi_download_and_credentials_contract():
    config, _ = load_acquisition_config()
    assert config.aoi is None
    assert config.download.enabled is False
    assert config.provider.credentials.username_env == "CDSE_USERNAME"
    assert config.provider.credentials.password_env == "CDSE_PASSWORD"
    assert config.sentinel1.require_same_relative_orbit is True
    assert config.sentinel2.selection_strategy is SelectionStrategy.LEAST_CLOUD_THEN_NEAREST
