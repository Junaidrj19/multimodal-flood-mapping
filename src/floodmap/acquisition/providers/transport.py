"""HTTP transport backed by ``requests``.

Responsibility
--------------
This module is the only place that knows about the ``requests`` library. It
translates library exceptions into the ``floodmap.acquisition.errors``
hierarchy so that no caller has to catch a transport-specific type, and so a
future swap to ``httpx`` or ``urllib`` touches one file.

Retry policy and why it is this narrow
--------------------------------------
Retries apply **only** to timeouts, connection errors and HTTP 5xx — failures
where the request was valid and the provider is at fault. A 4xx is never
retried: repeating a malformed filter or a bad credential cannot change the
outcome and only consumes the provider's quota.

HTTP 429 is honoured via ``Retry-After`` when the provider supplies it. The
published quotas do not document a per-minute limit for catalogue search
specifically (the documented 2000/min figure is footnoted as applying to S3
object access), so this is defensive rather than tuned to a known ceiling.

Backoff is exponential with a cap. The caller's overall patience is bounded by
``max_attempts``, not by the clock, which keeps a failing run from hanging a
pipeline indefinitely.
"""

from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

from ..errors import (
    ProviderAuthError,
    ProviderHttpError,
    ProviderMalformedResponse,
    ProviderTimeout,
)

__all__ = ["RequestsTransport"]

#: Statuses worth retrying: the provider is at fault, the request was not.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Statuses meaning "your credentials, not your query".
_AUTH_STATUSES = frozenset({401, 403})

#: Cap on a single backoff sleep, seconds. Prevents a 60s+ stall on attempt 5.
_MAX_BACKOFF_SECONDS = 16.0

#: Ceiling on an honoured ``Retry-After``. A provider asking for ten minutes
#: should fail the run rather than silently block it.
_MAX_RETRY_AFTER_SECONDS = 60.0


class _RequestsResponseAdapter:
    """Adapts a ``requests.Response`` to the :class:`HttpResponse` protocol.

    Exists so that a malformed JSON body surfaces as
    :class:`ProviderMalformedResponse` rather than a ``requests``-internal
    ``JSONDecodeError`` leaking through the abstraction.
    """

    __slots__ = ("_response", "_provider")

    def __init__(self, response: Any, provider: str) -> None:
        self._response = response
        self._provider = provider

    @property
    def status_code(self) -> int:
        return int(self._response.status_code)

    @property
    def text(self) -> str:
        return str(self._response.text)

    @property
    def content(self) -> bytes:
        return bytes(self._response.content)

    def json(self) -> Any:
        try:
            return self._response.json()
        except Exception as exc:
            preview = self.text[:200] if self.text else "<empty body>"
            raise ProviderMalformedResponse(
                f"response body is not valid JSON: {exc}. First 200 chars: {preview!r}",
                provider=self._provider,
            ) from exc


class RequestsTransport:
    """``requests``-backed transport with bounded, selective retries."""

    def __init__(
        self,
        *,
        provider: str = "http",
        max_attempts: int = 3,
        backoff_base_seconds: float = 1.0,
        sleep: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_attempts
            Total attempts including the first. ``1`` disables retrying.
        sleep
            Injectable sleep, so retry tests run instantly instead of actually
            waiting. Defaults to :func:`time.sleep`.
        session
            Injectable ``requests.Session``. Defaults to the module-level
            ``requests`` functions.
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.provider = provider
        self.max_attempts = max_attempts
        self.backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep or time.sleep
        self._session = session

    def _requester(self) -> Any:
        """Resolve the object used to issue the GET.

        ``requests`` is imported lazily so that importing this package does not
        require it until a real network call is actually made. Tests inject a
        fake transport and never reach here.
        """
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderHttpError(
                "the 'requests' package is required for live provider access; "
                "install it with `pip install -e '.[dev]'` or inject a transport",
                provider=self.provider,
            ) from exc
        return requests

    def _retry_delay(self, attempt: int, response: Optional[Any]) -> float:
        """Seconds to wait before the next attempt.

        Honours ``Retry-After`` when present and parseable, otherwise falls
        back to capped exponential backoff.
        """
        if response is not None:
            raw = getattr(response, "headers", {}) or {}
            retry_after = raw.get("Retry-After") or raw.get("retry-after")
            if retry_after:
                try:
                    requested = float(str(retry_after).strip())
                except ValueError:
                    requested = None  # HTTP-date form; fall through to backoff
                if requested is not None and 0 <= requested <= _MAX_RETRY_AFTER_SECONDS:
                    return requested
        return min(self.backoff_base_seconds * (2**attempt), _MAX_BACKOFF_SECONDS)

    def get(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = 60.0,
    ) -> _RequestsResponseAdapter:
        """Issue a GET, retrying only provider-side failures.

        Raises
        ------
        ProviderTimeout
            Every attempt timed out or the connection failed.
        ProviderAuthError
            HTTP 401/403. Not retried.
        ProviderHttpError
            Any other unsuccessful status, or a retryable status that did not
            recover within ``max_attempts``.
        """
        requester = self._requester()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            is_final = attempt == self.max_attempts - 1
            try:
                response = requester.get(
                    url,
                    params=dict(params) if params else None,
                    headers=dict(headers) if headers else None,
                    timeout=timeout_seconds,
                )
            except Exception as exc:
                # Timeouts and connection resets are retryable; anything else
                # from the transport is not something a retry will fix.
                if not _is_retryable_exception(exc):
                    raise ProviderHttpError(
                        f"transport error: {exc}", provider=self.provider
                    ) from exc
                last_error = exc
                if is_final:
                    raise ProviderTimeout(
                        f"request to {url} failed after {self.max_attempts} attempt(s): {exc}",
                        provider=self.provider,
                    ) from exc
                self._sleep(self._retry_delay(attempt, None))
                continue

            status = int(response.status_code)

            if status in _AUTH_STATUSES:
                # Never retried: the query is fine, the credentials are not.
                raise ProviderAuthError(
                    f"authentication/authorisation failed with HTTP {status} for {url}. "
                    "Check the credentials supplied via the configured environment "
                    "variables; catalogue search itself does not require credentials.",
                    provider=self.provider,
                )

            if status in _RETRYABLE_STATUSES and not is_final:
                self._sleep(self._retry_delay(attempt, response))
                continue

            if not 200 <= status < 300:
                body = str(getattr(response, "text", ""))[:300]
                raise ProviderHttpError(
                    f"unsuccessful response for {url}: {body!r}",
                    status_code=status,
                    provider=self.provider,
                )

            return _RequestsResponseAdapter(response, self.provider)

        # Only reachable if the loop exits without returning or raising, which
        # would mean max_attempts was consumed by retryable statuses.
        raise ProviderHttpError(
            f"request to {url} did not succeed within {self.max_attempts} attempt(s)"
            + (f": {last_error}" if last_error else ""),
            provider=self.provider,
        )

    def download(
        self,
        url: str,
        destination: Path,
        *,
        token_url: str,
        username: str,
        password: str,
        timeout_seconds: float = 60.0,
        chunk_size: int = 1024 * 1024,
    ) -> tuple[int, str]:
        """Download one CDSE product with an OAuth access token.

        Product transfer is intentionally separate from ``get``: it streams to
        disk and returns byte count plus SHA-256 so the manifest can prove what
        was written without retaining raster bytes in memory.
        """
        requester = self._requester()
        try:
            token_response = requester.post(
                token_url,
                data={
                    "client_id": "cdse-public",
                    "username": username,
                    "password": password,
                    "grant_type": "password",
                },
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise ProviderTimeout(
                f"CDSE token request failed: {exc}", provider=self.provider
            ) from exc
        if int(token_response.status_code) in {401, 403}:
            raise ProviderAuthError("CDSE token request was rejected", provider=self.provider)
        if not 200 <= int(token_response.status_code) < 300:
            raise ProviderHttpError(
                "CDSE token request failed",
                status_code=int(token_response.status_code),
                provider=self.provider,
            )
        try:
            token_payload = token_response.json()
            token = token_payload["access_token"]
        except Exception as exc:
            raise ProviderMalformedResponse(
                "CDSE token response omitted access_token", provider=self.provider
            ) from exc

        try:
            response = requester.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise ProviderTimeout(f"CDSE download failed: {exc}", provider=self.provider) from exc
        status = int(response.status_code)
        if status in {401, 403}:
            raise ProviderAuthError("CDSE download was rejected", provider=self.provider)
        if not 200 <= status < 300:
            raise ProviderHttpError(
                "CDSE download failed", status_code=status, provider=self.provider
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        try:
            with destination.open("wb") as handle:
                iterator = response.iter_content(chunk_size=chunk_size)
                for chunk in iterator:
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
        except OSError:
            destination.unlink(missing_ok=True)
            raise
        return written, digest.hexdigest()


def _is_retryable_exception(exc: Exception) -> bool:
    """Whether a transport exception is worth another attempt.

    Matched on class name rather than by importing ``requests.exceptions``, so
    this module stays importable without the dependency present and works with
    an injected fake session that raises look-alike errors.
    """
    retryable_names = {
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionError",
        "ChunkedEncodingError",
        "TimeoutError",
    }
    return any(base.__name__ in retryable_names for base in type(exc).__mro__)
