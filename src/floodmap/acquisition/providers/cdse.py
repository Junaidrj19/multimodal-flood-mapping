"""CDSE OData metadata discovery and explicitly enabled product downloads.

Query syntax: https://documentation.dataspace.copernicus.eu/APIs/OData.html
The requested collection/product filter is never used to fill absent metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlencode

from ..errors import ProviderAuthError, ProviderError, ProviderHttpError, ProviderMalformedResponse
from ..outcomes import DiscoveryStatus, DownloadStatus
from ..scenes import (
    DiscoveredScene,
    SceneFootprint,
    Sentinel1Scene,
    Sentinel2Scene,
    SensorKind,
    catalogue_unknown,
    derive_platform_unit,
    parse_polarisations,
)
from ..temporal import parse_acquired_at
from .base import DiscoveryResult, DownloadResult, HttpTransport, SceneProvider, SearchRequest
from .transport import RequestsTransport

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
PRODUCT_TYPES = {
    SensorKind.SENTINEL1: ("IW_GRDH_1S",),
    SensorKind.SENTINEL2: ("S2MSI2A", "S2MSI1C"),
}


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _attribute_filter(name: str, value: str) -> str:
    return (
        "Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq '{name}' and att/OData.CSC.StringAttribute/Value eq '{value}')"
    )


def _product_types(request: SearchRequest) -> tuple[str, ...]:
    allowed = PRODUCT_TYPES[request.sensor]
    if request.product_type is None:
        return allowed
    if request.product_type not in allowed:
        raise ValueError(
            f"unsupported {request.sensor.value} product type: {request.product_type!r}"
        )
    return (request.product_type,)


def _odata_filter(request: SearchRequest) -> str:
    products = [_attribute_filter("productType", item) for item in _product_types(request)]
    predicates = [
        f"Collection/Name eq '{request.sensor.value.upper()}'",
        "(" + " or ".join(products) + ")",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{request.aoi.to_wkt()}')",
        f"ContentDate/Start ge {_iso_z(request.interval.start)}",
        f"ContentDate/Start lt {_iso_z(request.interval.end)}",
    ]
    if request.max_cloud_percent is not None:
        predicates.append(
            "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {request.max_cloud_percent:g})"
        )
    return " and ".join(predicates)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or float(value) != int(value):
        raise ValueError(f"expected an integer, got {value!r}")
    return int(value)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a non-empty string")
    return value


def _attributes(product: Mapping[str, Any]) -> dict[str, Any]:
    items = product.get("Attributes") or []
    if not isinstance(items, list):
        raise ValueError("Attributes must be an array")
    result = {}
    for item in items:
        if not isinstance(item, dict) or "Name" not in item:
            raise ValueError("attribute must contain a Name")
        name = _text(item["Name"])
        if name in result and result[name] != item.get("Value"):
            raise ValueError(f"contradictory attribute values for {name}")
        result[name] = item.get("Value")
    return result


def _value(attributes: Mapping[str, Any], name: str, parse=_text):
    value = attributes.get(name)
    if value is None or value == "":
        return catalogue_unknown(name)
    return parse(value)


def _scene(product: Any, request: SearchRequest, base_url: str) -> DiscoveredScene:
    if not isinstance(product, dict):
        raise ValueError("product must be an object")
    attributes = _attributes(product)
    product_id = _text(product["Id"])
    scene_id = _text(product["Name"])
    product_type = _value(attributes, "productType")
    if isinstance(product_type, str) and product_type not in _product_types(request):
        raise ValueError("catalogue returned a product outside the requested product types")
    platform = _value(attributes, "platformShortName")
    if isinstance(platform, str) and platform != request.sensor.value.upper():
        raise ValueError("catalogue platform contradicts the requested sensor")
    online = product.get("Online")
    if online is not None and not isinstance(online, bool):
        raise ValueError("Online must be boolean")
    length = _value(product, "ContentLength", _integer)
    if isinstance(length, int) and length < 0:
        raise ValueError("ContentLength must not be negative")
    footprint = product.get("GeoFootprint")
    common = dict(
        scene_id=scene_id,
        product_id=product_id,
        acquired_at=parse_acquired_at(_text(product["ContentDate"]["Start"])),
        product_type=product_type,
        platform_short_name=platform,
        platform_unit=derive_platform_unit(scene_id) or catalogue_unknown("platform unit"),
        footprint=(
            SceneFootprint.model_validate(footprint)
            if footprint is not None
            else catalogue_unknown("GeoFootprint")
        ),
        online=online if online is not None else catalogue_unknown("Online"),
        content_length_bytes=length,
        provider="cdse-odata",
        reference_url=f"{base_url}({quote(product_id, safe='')})",
    )
    orbit = _value(attributes, "relativeOrbitNumber", _integer)
    if request.sensor is SensorKind.SENTINEL1:
        return Sentinel1Scene(
            **common,
            relative_orbit=orbit,
            orbit_direction=_value(attributes, "orbitDirection"),
            absolute_orbit=_value(attributes, "orbitNumber", _integer),
            operational_mode=(
                _value(attributes, "operationalMode")
                if "operationalMode" in attributes
                else _value(attributes, "instrumentMode")
            ),
            swath_identifier=(
                _value(attributes, "swathIdentifier")
                if "swathIdentifier" in attributes
                else _value(attributes, "swath")
            ),
            polarisations=_value(attributes, "polarisationChannels", parse_polarisations),
        )
    return Sentinel2Scene(
        **common,
        relative_orbit=orbit,
        processing_level=_value(attributes, "processingLevel"),
        tile_id=_value(attributes, "tileId"),
        scene_cloud_percent=_value(attributes, "cloudCover", float),
    )


class CdseOdataProvider(SceneProvider):
    name = "cdse-odata"

    def __init__(
        self,
        *,
        base_url: str = CATALOGUE_URL,
        download_url_template: str | None = None,
        token_url: str = TOKEN_URL,
        transport: HttpTransport | None = None,
        timeout_seconds: float = 60.0,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/?")
        self.download_url_template = download_url_template or DOWNLOAD_URL
        self.token_url = token_url
        self.transport = transport or RequestsTransport(provider=self.name)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.username, self.password = username, password

    def _url(self, request: SearchRequest, skip: int, top: int) -> str:
        params = {
            "$filter": _odata_filter(request),
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start asc,Id asc",
            "$top": str(top),
            "$skip": str(skip),
        }
        return f"{self.base_url}?{urlencode(params, quote_via=quote)}"

    def build_query_url(self, request: SearchRequest) -> str:
        # One look-ahead record distinguishes an exact cap from truncation.
        return self._url(request, 0, min(request.page_size, request.max_results + 1))

    def search(self, request: SearchRequest) -> DiscoveryResult:
        first_url = self.build_query_url(request)  # configuration errors are caller errors
        urls: list[str] = []
        scenes: list[DiscoveredScene] = []
        seen: set[str] = set()
        pages = 0
        try:
            while len(scenes) <= request.max_results:
                top = min(request.page_size, request.max_results + 1 - len(scenes))
                url = self._url(request, len(scenes), top)
                urls.append(url)
                response = self.transport.get(url, timeout_seconds=self.timeout_seconds)
                if response.status_code in (401, 403):
                    raise ProviderAuthError(f"HTTP {response.status_code}", provider=self.name)
                if not 200 <= response.status_code < 300:
                    raise ProviderHttpError(
                        "catalogue request failed",
                        status_code=response.status_code,
                        provider=self.name,
                    )
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("value"), list):
                    raise ValueError("CDSE response must contain the OData value array")
                products = payload["value"]
                pages += 1
                if len(products) > top:
                    raise ValueError("catalogue returned more records than requested")
                for product in products:
                    scene = _scene(product, request, self.base_url)
                    if scene.product_id in seen:
                        raise ValueError(
                            "duplicate product across catalogue pages; rerun discovery"
                        )
                    seen.add(scene.product_id)
                    scenes.append(scene)
                has_next = bool(payload.get("@odata.nextLink"))
                if not products:
                    if has_next:
                        raise ValueError("empty catalogue page has a continuation link")
                    break
                if len(products) < top and not has_next:
                    break
            return DiscoveryResult(
                status=DiscoveryStatus.SUCCESS if scenes else DiscoveryStatus.EMPTY,
                request=request,
                scenes=tuple(scenes[: request.max_results]),
                provider=self.name,
                executed_url=first_url,
                executed_urls=tuple(urls),
                pages_fetched=pages,
                truncated=len(scenes) > request.max_results,
            )
        except (ProviderError, ValueError, TypeError, KeyError, OverflowError) as exc:
            error = (
                exc
                if isinstance(exc, ProviderError)
                else ProviderMalformedResponse(str(exc), provider=self.name)
            )
            return DiscoveryResult(
                status=DiscoveryStatus.FAILED,
                request=request,
                provider=self.name,
                executed_url=first_url,
                executed_urls=tuple(urls),
                pages_fetched=pages,
                error=str(error),
                error_type=type(error).__name__,
            )

    def download(
        self, scene: DiscoveredScene, destination_dir: Path, *, enabled: bool = False
    ) -> DownloadResult:
        if not enabled:
            return DownloadResult(
                status=DownloadStatus.NOT_ATTEMPTED,
                scene_id=scene.scene_id,
                reason_not_attempted="Download disabled; an explicit --download is required.",
            )
        try:
            if not self.username or not self.password:
                raise ProviderAuthError("CDSE username/password environment variables are required")
            if scene.provider != self.name:
                raise ValueError("cannot download a scene from a different provider")
            if scene.online is False:
                raise ValueError("product is offline; staging is not implemented")
            # UUID filenames avoid interpreting catalogue names as local paths.
            from uuid import UUID

            product_id = str(UUID(scene.product_id))
            destination = destination_dir / f"{product_id}.zip"
            download = getattr(self.transport, "download", None)
            if download is None:
                raise ValueError("transport does not support streamed downloads")
            count, digest = download(
                self.download_url_template.format(product_id=product_id),
                destination,
                token_url=self.token_url,
                username=self.username,
                password=self.password,
                timeout_seconds=self.timeout_seconds,
            )
            return DownloadResult(
                status=DownloadStatus.SUCCESS,
                scene_id=scene.scene_id,
                destination=destination,
                bytes_written=count,
                sha256=digest,
            )
        except (ProviderError, OSError, ValueError) as exc:
            return DownloadResult(
                status=DownloadStatus.FAILED, scene_id=scene.scene_id, error=str(exc)
            )
