"""A tiny standard-library HTTP server on 127.0.0.1 for tests; started and stopped by each test."""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

# A route answers (status, body, headers); body may be bytes, str or JSON-able data.
Route = Callable[["Request"], Tuple[int, Any, Dict[str, str]]]


class Request:
    def __init__(self, method: str, path: str, query: Dict[str, str], headers: Dict[str, str], body: bytes) -> None:
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class FakeServer:
    """Serve routes keyed by (method, path); records every request it got."""

    def __init__(self, routes: Dict[Tuple[str, str], Route]) -> None:
        self.routes = routes
        self.requests: List[Request] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output clean
                return

            def _handle(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
                length = int(self.headers.get("Content-Length") or 0)
                request = Request(
                    self.command,
                    parsed.path,
                    query,
                    {key.lower(): value for key, value in self.headers.items()},
                    self.rfile.read(length) if length else b"",
                )
                server.requests.append(request)
                route = server.routes.get((self.command, parsed.path))
                if route is None:
                    status, body, headers = 404, {"error": "not found"}, {}
                else:
                    status, body, headers = route(request)
                if isinstance(body, bytes):
                    payload = body
                    content_type = "application/octet-stream"
                elif isinstance(body, str):
                    payload = body.encode("utf-8")
                    content_type = "text/plain; charset=utf-8"
                else:
                    payload = json.dumps(body).encode("utf-8")
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", headers.pop("Content-Type", content_type))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _handle
            do_POST = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeServer":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
