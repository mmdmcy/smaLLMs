"""Launcher parser tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import start
from start import build_parser


class LauncherTests(unittest.TestCase):
    """Validate the friendly launcher CLI surface."""

    def test_check_only_flag_is_supported(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--check-only"])
        self.assertTrue(args.check_only)

    def test_cli_passthrough_is_supported(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--", "doctor", "--json"])
        self.assertEqual(args.cli_args, ["--", "doctor", "--json"])

    def test_current_python_match_uses_invoked_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            real_python = tmp_path / "python-real"
            link_python = tmp_path / "python-link"
            real_python.write_text("")
            try:
                os.symlink(real_python, link_python)
            except (AttributeError, NotImplementedError, OSError):
                self.skipTest("Symlinks are not available in this environment")

            with mock.patch.object(start.sys, "executable", str(link_python)):
                self.assertTrue(start._current_python_matches(link_python))
                self.assertFalse(start._current_python_matches(real_python))


if __name__ == "__main__":
    unittest.main()
