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
    """Validate full-screen progress rendering without scrollback pollution."""

    def test_live_renderer_uses_alternate_screen_buffer(self) -> None:
        output = FakeTTY()
        with mock.patch.object(sys, "stdout", output):
            terminal = BeautifulSmaLLMsTerminal()
            terminal.start_run("run-test", ["model-a"], ["gsm8k"], 3)
            terminal.finish_live_screen()

        rendered = output.getvalue()
        self.assertIn("\033[?1049h", rendered)
        self.assertIn("\033[?25l", rendered)
        self.assertIn("\033[?25h", rendered)
        self.assertIn("\033[?1049l", rendered)

    def test_finish_live_screen_is_idempotent(self) -> None:
        output = FakeTTY()
        with mock.patch.object(sys, "stdout", output):
            terminal = BeautifulSmaLLMsTerminal()
            terminal.finish_live_screen()
            terminal.start_run("run-test", ["model-a"], ["gsm8k"], 1)
            terminal.finish_live_screen()
            terminal.finish_live_screen()

        rendered = output.getvalue()
        self.assertEqual(rendered.count("\033[?1049h"), 1)
        self.assertEqual(rendered.count("\033[?1049l"), 1)


if __name__ == "__main__":
    unittest.main()
