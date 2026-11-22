"""Exported paths never carry the user's home folder, which names the user."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline.artifacts import portable_path


class PortablePathTests(unittest.TestCase):
    """A run's manifest and the website export write paths through portable_path."""

    def _run(self, value_from_home: tuple[str, ...], cwd_from_root: tuple[str, ...]) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            home = root / "Users" / "someone"
            cwd = root.joinpath(*cwd_from_root)
            home.mkdir(parents=True)
            cwd.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(Path, "home", return_value=home), mock.patch.object(Path, "cwd", return_value=cwd):
                return portable_path(home.joinpath(*value_from_home)), "someone"

    def test_cache_folder_under_home_is_written_with_a_tilde(self) -> None:
        out, user = self._run(("AppData", "Local", "smaLLMs", "benchmark_cache"), ("work",))
        self.assertEqual(out, "~/AppData/Local/smaLLMs/benchmark_cache")
        self.assertNotIn(user, out)

    def test_cache_folder_seen_from_inside_home_names_no_user(self) -> None:
        out, user = self._run(("AppData", "Local", "smaLLMs"), ("Users", "someone", "src", "smaLLMs"))
        self.assertNotIn(user, out)
        self.assertFalse(Path(out).is_absolute())

    def test_relative_path_stays_relative(self) -> None:
        self.assertEqual(portable_path("artifacts/run_1"), "artifacts/run_1")


if __name__ == "__main__":
    unittest.main()
