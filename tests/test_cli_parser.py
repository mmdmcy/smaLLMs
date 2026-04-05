"""CLI parser smoke tests."""

from __future__ import annotations

import unittest

from smaLLMs import _parse_list, build_parser


class CLITests(unittest.TestCase):
    """Validate lightweight CLI behavior that should not require external services."""

    def test_parse_list_supports_commas_and_repeated_flags(self) -> None:
        parsed = _parse_list(["gsm8k,mmlu", "piqa"])
        self.assertEqual(parsed, ["gsm8k", "mmlu", "piqa"])

    def test_menu_command_is_supported(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["menu"])
        self.assertEqual(args.command, "menu")

    def test_doctor_command_is_supported(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        self.assertEqual(args.command, "doctor")


if __name__ == "__main__":
    unittest.main()
