"""Tests for live terminal rendering behavior."""

from __future__ import annotations

import io
import sys
import unittest
from unittest import mock

from smaLLMs import BeautifulSmaLLMsTerminal


class FakeTTY(io.StringIO):
    """StringIO that behaves like an interactive terminal for renderer tests."""

    def isatty(self) -> bool:
        return True


class LiveTerminalTests(unittest.TestCase):
    """Validate live progress rendering behavior."""

    def test_live_renderer_uses_normal_scrollback_buffer(self) -> None:
        output = FakeTTY()
        with mock.patch.object(sys, "stdout", output):
            terminal = BeautifulSmaLLMsTerminal()
            terminal.start_run("run-test", ["model-a"], ["gsm8k"], 3)
            terminal.finish_live_screen()

        rendered = output.getvalue()
        self.assertIn("\033[?25l", rendered)
        self.assertIn("\033[?25h", rendered)
        self.assertNotIn("\033[?1049h", rendered)
        self.assertNotIn("\033[?1049l", rendered)

    def test_finish_live_screen_is_idempotent(self) -> None:
        output = FakeTTY()
        with mock.patch.object(sys, "stdout", output):
            terminal = BeautifulSmaLLMsTerminal()
            terminal.finish_live_screen()
            terminal.start_run("run-test", ["model-a"], ["gsm8k"], 1)
            terminal.finish_live_screen()
            terminal.finish_live_screen()

        rendered = output.getvalue()
        self.assertEqual(rendered.count("\033[?25l"), 1)
        self.assertEqual(rendered.count("\033[?25h"), 1)

    def test_completed_benchmark_updates_actual_sample_total(self) -> None:
        output = FakeTTY()
        with mock.patch.object(sys, "stdout", output):
            terminal = BeautifulSmaLLMsTerminal()
            terminal.start_run("run-test", ["model-a"], ["aime_2025"], 100)
            terminal.handle_event(
                {
                    "event": "sample_completed",
                    "model_name": "model-a",
                    "benchmark_name": "aime_2025",
                    "sample_index": 30,
                    "total_samples": 30,
                    "correct_count": 12,
                    "error_count": 0,
                    "sample": {"latency_sec": 1.0},
                }
            )
            terminal.handle_event(
                {
                    "event": "benchmark_completed",
                    "model_name": "model-a",
                    "benchmark_name": "aime_2025",
                    "benchmark_result": {
                        "metrics": {
                            "sample_count": 30,
                            "correct_count": 12,
                            "error_count": 0,
                        }
                    },
                }
            )
            terminal.finish_live_screen()

        rendered = output.getvalue()
        self.assertIn("30/30", rendered)
        self.assertIn("(1/1 completed)", rendered)


if __name__ == "__main__":
    unittest.main()
