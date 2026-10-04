"""Provider-agnostic acquisition contract.

Why this indirection exists
---------------------------
``AGENTS.md`` §12 asks for module boundaries that follow scientific
responsibilities. Selection logic, manifests and provenance are scientific
concerns; OData filter syntax is not. Everything in this package except
``providers/cdse.py`` is written against the interfaces here, so swapping the
catalogue (the provider also exposes a STAC API, and the legacy STAC endpoint
was already deprecated once) does not touch selection or provenance.

The same indirection is what makes the test suite honest. :class:`HttpTransport`
is a protocol, so tests inject a fake that replays recorded response *shapes*
with synthetic values. No test reaches the network, and the fabrication rule is
respected because the fake never pretends to be a successful real acquisition:
synthetic scenes are produced only inside tests and the manifest records the
provider that produced them.

Discovery and download are separate methods on purpose
------------------------------------------------------
:meth:`SceneProvider.search` returns metadata and must never fetch a raster.
:meth:`SceneProvider.download` is a distinct, explicitly invoked call. A
successful search says nothing about whether anything was downloaded, which is
why :class:`~floodmap.acquisition.outcomes.DownloadStatus` defaults to
``NOT_ATTEMPTED``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from ..aoi import AreaOfInterest
from ..outcomes import DiscoveryStatus, DownloadStatus
from ..scenes import DiscoveredScene, SensorKind
from ..temporal import TimeInterval

__all__ = [
    "DiscoveryResult",
    "DownloadResult",
    "HttpResponse",
    "HttpTransport",
    "SceneProvider",
    "SearchRequest",
]


# ---------------------------------------------------------------------------
# HTTP abstraction
# ---------------------------------------------------------------------------


@runtime_checkable
class HttpResponse(Protocol):
    """The subset of an HTTP response this package relies on."""

    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any:
        """Parsed JSON body. Raises on a non-JSON body."""
        ...


@runtime_checkable
class HttpTransport(Protocol):
    """Minimal HTTP transport.

    Deliberately narrow: one GET. A wider surface would be harder to fake
    faithfully in tests, and the catalogue search this package performs needs
    nothing more.

    Implementations must translate their library's exceptions into the
    ``floodmap.acquisition.errors`` hierarchy, so callers never have to catch
    a transport-specific type.
    """

    def get(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = 60.0,
    ) -> HttpResponse: ...


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchRequest:
    """One catalogue query.

    Immutable so that the request recorded in the manifest is provably the one
    that was executed, rather than a mutated copy.

    ``max_cloud_percent`` is ``None`` unless a limit was explicitly
    configured. No threshold is invented: ``docs/data-contract.md`` §2.2 warns
    against picking one "without inspecting actual availability".
    """

    aoi: AreaOfInterest
    sensor: SensorKind
    interval: TimeInterval
    product_type: Optional[str] = None
    max_cloud_percent: Optional[float] = None
    max_results: int = 200
    page_size: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.sensor, SensorKind):
            raise ValueError("sensor must be a permitted SensorKind")
        if self.max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {self.max_results}")
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        if self.max_cloud_percent is not None and not 0.0 <= self.max_cloud_percent <= 100.0:
            raise ValueError(f"max_cloud_percent must be in [0, 100], got {self.max_cloud_percent}")
        if self.max_cloud_percent is not None and self.sensor is SensorKind.SENTINEL1:
            raise ValueError(
                "max_cloud_percent is meaningless for Sentinel-1, which is "
                "cloud-independent; setting it suggests a configuration mix-up"
            )

    def describe(self) -> str:
        """Human-readable summary, recorded in the manifest rationale."""
        parts = [
            f"sensor={self.sensor.value}",
            f"aoi={self.aoi.aoi_id}",
            f"from={self.interval.start.isoformat()}",
            f"to={self.interval.end.isoformat()}",
        ]
        if self.product_type:
            parts.append(f"product_type={self.product_type}")
        if self.max_cloud_percent is not None:
            parts.append(f"max_cloud_percent={self.max_cloud_percent}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor": self.sensor.value,
            "aoi_id": self.aoi.aoi_id,
            "aoi_is_synthetic": self.aoi.is_synthetic,
            "aoi_wkt": self.aoi.to_wkt(),
            "aoi_crs": self.aoi.crs,
            "interval": self.interval.to_dict(),
            "product_type": self.product_type,
            "max_cloud_percent": self.max_cloud_percent,
            "max_results": self.max_results,
            "page_size": self.page_size,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    """Outcome of one discovery query.

    Carries the status explicitly rather than letting an empty ``scenes`` list
    stand in for it. "The provider holds nothing for this area and date" and
    "the query failed" both produce zero scenes but mean opposite things: the
    first is a finding about data availability that belongs in the report, the
    second says nothing about the world.

    ``executed_url`` is retained so a reviewer can replay the exact query.
    """

    status: DiscoveryStatus
    request: SearchRequest
    scenes: Sequence[DiscoveredScene] = field(default_factory=tuple)
    executed_url: Optional[str] = None
    error: Optional[str] = None
    provider: str = ""
    pages_fetched: int = 0
    truncated: bool = False
    executed_urls: Sequence[str] = field(default_factory=tuple)
    error_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is DiscoveryStatus.SUCCESS and not self.scenes:
            raise ValueError(
                "DiscoveryStatus.SUCCESS with no scenes is contradictory; use EMPTY "
                "when the provider correctly returned nothing"
            )
        if self.status is DiscoveryStatus.EMPTY and self.scenes:
            raise ValueError("DiscoveryStatus.EMPTY cannot carry scenes")
        if self.status is DiscoveryStatus.FAILED and not self.error:
            raise ValueError(
                "DiscoveryStatus.FAILED must carry the reason; a failure without a "
                "recorded cause cannot be diagnosed or reported"
            )
        if self.status is DiscoveryStatus.FAILED and self.scenes:
            raise ValueError(
                "DiscoveryStatus.FAILED cannot carry scenes; partial results from a "
                "failed query would be indistinguishable from a complete set"
            )

    @property
    def succeeded(self) -> bool:
        """True only for a query that ran and returned candidates."""
        return self.status is DiscoveryStatus.SUCCESS

    @property
    def ran_without_error(self) -> bool:
        """True when the query completed, whether or not it found anything."""
        return self.status in (DiscoveryStatus.SUCCESS, DiscoveryStatus.EMPTY)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "provider": self.provider,
            "request": self.request.to_dict(),
            "executed_url": self.executed_url,
            "error": self.error,
            "error_type": self.error_type,
            "executed_urls": list(self.executed_urls),
            "scene_count": len(self.scenes),
            "scenes": [scene.to_dict() for scene in self.scenes],
            "pages_fetched": self.pages_fetched,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one product download attempt."""

    status: DownloadStatus
    scene_id: str
    destination: Optional[Path] = None
    bytes_written: Optional[int] = None
    error: Optional[str] = None
    reason_not_attempted: Optional[str] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is DownloadStatus.SUCCESS and self.destination is None:
            raise ValueError("a successful download must record where it was written")
        if self.status is DownloadStatus.FAILED and not self.error:
            raise ValueError("a failed download must record why")
        if self.status is DownloadStatus.NOT_ATTEMPTED and not self.reason_not_attempted:
            raise ValueError(
                "NOT_ATTEMPTED must record why, so that a disabled download is "
                "distinguishable from an overlooked one"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "scene_id": self.scene_id,
            "destination": str(self.destination) if self.destination else None,
            "bytes_written": self.bytes_written,
            "sha256": self.sha256,
            "error": self.error,
            "reason_not_attempted": self.reason_not_attempted,
        }


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class SceneProvider(ABC):
    """A source of satellite scene metadata.

    Implementations own all provider-specific knowledge: endpoints, query
    syntax, response shape, authentication. Nothing above this interface may
    depend on those details.
    """

    #: Short stable identifier, recorded in every scene and manifest.
    name: str = "abstract"

    @abstractmethod
    def search(self, request: SearchRequest) -> DiscoveryResult:
        """Discover scenes matching ``request``. Metadata only.

        Must not download raster products, and must not raise for an ordinary
        provider failure: translate it into a
        :attr:`~floodmap.acquisition.outcomes.DiscoveryStatus.FAILED` result
        carrying the reason, so one sensor failing does not abort the run.
        """

    @abstractmethod
    def build_query_url(self, request: SearchRequest) -> str:
        """Return the URL ``search`` would call, without calling it.

        Supports dry runs: the brief requires discovery to be usable without
        downloading, and a reviewer to be able to see the query that would be
        issued before any network access happens.
        """

    def download(
        self,
        scene: DiscoveredScene,
        destination_dir: Path,
        *,
        enabled: bool = False,
    ) -> DownloadResult:
        """Download one product.

        Default implementation refuses, returning ``NOT_ATTEMPTED``. A provider
        that cannot download must not silently appear to succeed, and the
        opt-in ``enabled`` flag means a discovery run can never trigger a fetch
        by accident.
        """
        return DownloadResult(
            status=DownloadStatus.NOT_ATTEMPTED,
            scene_id=scene.scene_id,
            reason_not_attempted=(
                f"provider {self.name!r} does not implement download"
                if not enabled
                else f"provider {self.name!r} has download support disabled"
            ),
        )

    def describe(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
