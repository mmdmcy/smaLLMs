"""Tests for the standard-library HTTP client and the model providers that use it."""

from __future__ import annotations

import asyncio
import socket
import time
import unittest

from src.models.model_manager import GenerationConfig, LMStudioModel, ModelManager, OllamaModel
from src.utils import http_client
from tests.fake_http import FakeServer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _slow(request):
    time.sleep(1.5)
    return 200, {"ok": True}, {}


class HTTPClientTests(unittest.TestCase):
    def test_get_post_and_status_handling(self) -> None:
        routes = {
            ("GET", "/json"): lambda r: (200, {"hello": "world"}, {}),
            ("POST", "/echo"): lambda r: (200, {"got": r.json(), "type": r.headers.get("content-type")}, {}),
            ("GET", "/boom"): lambda r: (500, "server exploded", {}),
        }
        with FakeServer(routes) as server:
            client = http_client.AsyncHTTPClient()
            got = asyncio.run(client.get(f"{server.url}/json"))
            self.assertEqual(got.status, 200)
            self.assertEqual(got.json(), {"hello": "world"})

            echoed = asyncio.run(client.post(f"{server.url}/echo", json={"a": [1, 2]})).json()
            self.assertEqual(echoed, {"got": {"a": [1, 2]}, "type": "application/json"})

            boom = asyncio.run(client.get(f"{server.url}/boom"))
            self.assertEqual(boom.status, 500)
            self.assertEqual(boom.text(), "server exploded")
            with self.assertRaises(http_client.HTTPStatusError):
                boom.raise_for_status()

            missing = asyncio.run(client.get(f"{server.url}/nope"))
            self.assertEqual(missing.status, 404)

    def test_timeout_raises_asyncio_timeout(self) -> None:
        with FakeServer({("GET", "/slow"): _slow}) as server:
            with self.assertRaises(asyncio.TimeoutError):
                asyncio.run(http_client.AsyncHTTPClient().get(f"{server.url}/slow", timeout=0.3))

    def test_refused_connection_raises_connection_error(self) -> None:
        with self.assertRaises(http_client.HTTPConnectionError):
            http_client.request("GET", f"http://127.0.0.1:{_free_port()}/", timeout=2)


class ProviderHTTPTests(unittest.TestCase):
    """Ollama and LM Studio over HTTP against a fake local server."""

    def test_ollama_chat_and_discovery(self) -> None:
        tags = {"models": [{"name": "fake:1b", "size": 1073741824, "details": {"family": "fake"}}]}

        def chat(request):
            body = request.json()
            return 200, {
                "message": {"role": "assistant", "content": "Answer: 42"},
                "prompt_eval_count": 7,
                "eval_count": 3,
                "eval_duration": 1_000_000,
                "model": body["model"],
            }, {}

        routes = {("GET", "/api/tags"): lambda r: (200, tags, {}), ("POST", "/api/chat"): chat}
        with FakeServer(routes) as server:
            config = {"ollama": {"base_url": server.url, "command": "definitely-not-ollama"}}
            model = OllamaModel("fake:1b", config, metadata={})
            model.windows_curl = None

            async def run():
                result = await model._request_chat_http(model._build_chat_payload("Q?", GenerationConfig()), 5)
                transport = await model._detect_transport()
                await model.session.close()
                return result, transport

            result, transport = asyncio.run(run())
            self.assertEqual(transport, "http")
            self.assertEqual(result["response"], "Answer: 42")
            self.assertEqual(result["prompt_eval_count"], 7)

            models = asyncio.run(ModelManager(config)._discover_ollama_models())
            self.assertEqual([m["name"] for m in models], ["fake:1b"])
            self.assertEqual(models[0]["size_gb"], 1.0)

    def test_ollama_missing_model_and_server_error(self) -> None:
        routes = {
            ("POST", "/api/generate"): lambda r: (404, {"error": "model not found"}, {}),
            ("POST", "/api/chat"): lambda r: (500, "out of memory", {}),
        }
        with FakeServer(routes) as server:
            model = OllamaModel("fake:1b", {"ollama": {"base_url": server.url}}, metadata={})
            missing = asyncio.run(model._request_generate_http({"model": "fake:1b"}, 5))
            failed = asyncio.run(model._request_http("/api/chat", {"model": "fake:1b"}, 5))
        self.assertEqual(missing, {"response": "", "error": "model_not_found"})
        self.assertEqual(failed, {"response": "", "error": "out of memory"})

    def test_lm_studio_generation_and_discovery(self) -> None:
        def completions(request):
            return 200, {
                "choices": [{"message": {"content": " B "}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }, {}

        routes = {
            ("GET", "/v1/models"): lambda r: (200, {"data": [{"id": "fake-model", "owned_by": "me", "created": 1}]}, {}),
            ("POST", "/v1/chat/completions"): completions,
        }
        with FakeServer(routes) as server:
            config = {"lm_studio": {"base_url": server.url}}
            model = LMStudioModel("fake-model", config, metadata={})
            result = asyncio.run(model.generate_with_metadata("Q?", GenerationConfig()))
            models = asyncio.run(ModelManager(config)._discover_lm_studio_models())
        self.assertEqual(result.text, "B")
        self.assertEqual(result.total_tokens, 6)
        self.assertEqual([m["name"] for m in models], ["fake-model"])

    def test_lm_studio_error_status_is_reported(self) -> None:
        routes = {("POST", "/v1/chat/completions"): lambda r: (400, "bad request", {})}
        with FakeServer(routes) as server:
            model = LMStudioModel("fake-model", {"lm_studio": {"base_url": server.url}}, metadata={})
            result = asyncio.run(model.generate_with_metadata("Q?", GenerationConfig()))
        self.assertEqual(result.text, "")
        self.assertEqual(result.raw, {"error": "bad request", "status": 400})

    def test_discovery_without_a_server_returns_nothing(self) -> None:
        config = {"lm_studio": {"base_url": f"http://127.0.0.1:{_free_port()}"}}
        self.assertEqual(asyncio.run(ModelManager(config)._discover_lm_studio_models()), [])


if __name__ == "__main__":
    unittest.main()
