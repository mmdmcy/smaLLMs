"""Tests for arrow-key menu option construction."""

from __future__ import annotations

import unittest

from src.cli.terminal_menu import TerminalMenuApp


class TerminalMenuOptionTests(unittest.TestCase):
    """Validate menu-only behavior without opening an interactive TTY."""

    def test_benchmark_scope_includes_all_benchmarks_suite(self) -> None:
        menu = TerminalMenuApp(app=object())

        options = menu._benchmark_scope_options()

        self.assertIn("All Benchmarks", [option.label for option in options])
        self.assertIn(("suite", "all_benchmarks"), [option.value for option in options])


if __name__ == "__main__":
    unittest.main()
