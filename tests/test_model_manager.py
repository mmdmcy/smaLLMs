"""Tests for local model provider request construction."""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from src.models.model_manager import GenerationConfig, OllamaModel


class OllamaPayloadTests(unittest.TestCase):
    """Validate provider payloads that keep benchmark runs answer-focused."""

    def test_standard_chat_payload_disables_hidden_thinking_by_default(self) -> None:
        model = OllamaModel("qwen3.5:latest", {"ollama": {}}, metadata={})

        payload = model._build_chat_payload("Answer with A.", GenerationConfig(max_tokens=24))

        self.assertIs(payload["think"], False)
        self.assertNotIn("raw", payload)
        self.assertEqual(payload["messages"][0]["content"], "Answer with A.")
        self.assertEqual(payload["options"]["num_predict"], 24)

    def test_gpt_oss_chat_payload_uses_low_thinking_with_answer_budget(self) -> None:
        model = OllamaModel("gpt-oss:latest", {"ollama": {}}, metadata={})

        payload = model._build_chat_payload("Answer with A.", GenerationConfig(max_tokens=24))

        self.assertEqual(payload["think"], "low")
        self.assertEqual(payload["options"]["num_predict"], 128)

    def test_standard_chat_payload_can_allow_thinking_for_experiments(self) -> None:
        model = OllamaModel(
            "gpt-oss:latest",
            {"ollama": {"disable_thinking": False}},
            metadata={},
        )

        payload = model._build_chat_payload("Think if configured.", GenerationConfig(max_tokens=24))

        self.assertNotIn("think", payload)
        self.assertNotIn("raw", payload)
        self.assertEqual(payload["options"]["num_predict"], 24)

    def test_raw_fallback_always_bypasses_templates_and_thinking(self) -> None:
        model = OllamaModel(
            "qwen3.5:latest",
            {"ollama": {"disable_thinking": False}},
            metadata={},
        )

        payload = model._build_payload("Rescue this.", GenerationConfig(), raw_mode=True)

        self.assertIs(payload["think"], False)
        self.assertIs(payload["raw"], True)

    def test_closed_ollama_session_is_recreated(self) -> None:
        class ClosedSession:
            closed = True

        class OpenSession:
            closed = False

        model = OllamaModel("qwen3.5:latest", {"ollama": {}}, metadata={})
        model.session = ClosedSession()
        replacement = OpenSession()

        with mock.patch("src.models.model_manager.aiohttp.ClientSession", return_value=replacement):
            session = asyncio.run(model._get_session())

        self.assertIs(session, replacement)
        self.assertIs(model.session, replacement)


if __name__ == "__main__":
    unittest.main()
