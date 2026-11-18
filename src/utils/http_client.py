"""
A small HTTP client on Python's standard library (urllib), with an async face.

The blocking request runs in a worker thread through ``asyncio.to_thread`` so
async callers keep their interface. Errors mirror what callers relied on:
a timeout raises ``asyncio.TimeoutError``, a refused connection or a bad status
from ``raise_for_status`` raises ``HTTPClientError``.
"""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

DEFAULT_TIMEOUT_SEC = 300.0  # the same default total timeout aiohttp used

# Local model servers are reached directly, as aiohttp did (it ignored proxy
# variables by default); callers that fetch from the internet opt in to them.
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_ENV_PROXY_OPENER = urllib.request.build_opener()


class HTTPClientError(Exception):
    """A connection failure or an error status."""


class HTTPConnectionError(HTTPClientError):
    """The server could not be reached."""


class HTTPStatusError(HTTPClientError):
    """The server answered with an error status (raised by raise_for_status)."""

    def __init__(self, status: int, url: str, reason: str = "") -> None:
        self.status = status
        self.url = url
        self.reason = reason
        super().__init__(f"{status}, message={reason!r}, url={url!r}")


@dataclass
class HTTPResponse:
    """A finished response: status, headers and the whole body."""

    status: int
    url: str
    body: bytes
    reason: str = ""
    headers: Dict[str, str] = field(default_factory=dict)

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise HTTPStatusError(self.status, self.url, self.reason)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (socket.timeout, TimeoutError))


def request(
    method: str,
    url: str,
    json_body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    use_env_proxy: bool = False,
) -> HTTPResponse:
    """Send one request and return the response; an error status is returned, not raised."""
    data = None
    all_headers = {"Accept": "application/json", "User-Agent": "smaLLMs"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        all_headers["Content-Type"] = "application/json"
    all_headers.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=all_headers, method=method.upper())
    try:
        opener = _ENV_PROXY_OPENER if use_env_proxy else _DIRECT_OPENER
        with opener.open(req, timeout=timeout or DEFAULT_TIMEOUT_SEC) as response:
            return HTTPResponse(
                status=response.status,
                url=response.geturl(),
                body=response.read(),
                reason=response.reason or "",
                headers={key.lower(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return HTTPResponse(
            status=exc.code,
            url=exc.geturl() or url,
            body=body,
            reason=str(exc.reason or ""),
            headers={key.lower(): value for key, value in (exc.headers or {}).items()},
        )
    except Exception as exc:
        if _is_timeout(exc):
            raise asyncio.TimeoutError(f"request to {url} timed out") from exc
        if isinstance(exc, (urllib.error.URLError, OSError)):
            reason = getattr(exc, "reason", exc)
            raise HTTPConnectionError(f"Cannot connect to {url}: {reason}") from exc
        raise


class AsyncHTTPClient:
    """Async requests run in a worker thread; stands in for a client session."""

    def __init__(self, timeout: Optional[float] = None) -> None:
        self.timeout = timeout
        self.closed = False

    async def request(
        self,
        method: str,
        url: str,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        return await asyncio.to_thread(request, method, url, json, headers, timeout or self.timeout)

    async def get(self, url: str, **kwargs: Any) -> HTTPResponse:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> HTTPResponse:
        return await self.request("POST", url, **kwargs)

    async def close(self) -> None:
        self.closed = True
