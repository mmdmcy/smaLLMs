"""Launcher parser tests."""

from __future__ import annotations

import unittest

from start import build_parser


class LauncherTests(unittest.TestCase):
    """Validate the friendly launcher CLI surface."""

    def test_check_only_flag_is_supported(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--check-only"])
        self.assertTrue(args.check_only)


if __name__ == "__main__":
    unittest.main()
