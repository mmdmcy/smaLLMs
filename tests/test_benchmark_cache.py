"""Tests for offline benchmark cache metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.benchmarks import benchmark_cache_status, configure_dataset_runtime


class BenchmarkCacheStatusTests(unittest.TestCase):
    """Validate cache readiness metadata used by offline runs."""

    def tearDown(self) -> None:
        configure_dataset_runtime({"dataset_cache_dir": None, "allow_remote_dataset_downloads": True})

    def test_cache_status_reports_readiness_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            configure_dataset_runtime({"dataset_cache_dir": tempdir, "allow_remote_dataset_downloads": False})
            status_before = benchmark_cache_status(["piqa"], 2)[0]
            rows_path = Path(status_before["rows_path"])
            meta_path = Path(status_before["meta_path"])
            rows_path.write_text(
                json.dumps({"goal": "g1", "sol1": "a", "sol2": "b", "label": 0}) + "\n"
                + json.dumps({"goal": "g2", "sol1": "a", "sol2": "b", "label": 1}) + "\n",
                encoding="utf-8",
            )
            meta_path.write_text(json.dumps({"benchmark_key": "piqa"}), encoding="utf-8")

            status = benchmark_cache_status(["piqa"], 2)[0]

        self.assertTrue(status["ready"])
        self.assertEqual(status["cached_rows"], 2)
        self.assertIsNotNone(status["rows_sha256"])
        self.assertIsNotNone(status["meta_sha256"])
        self.assertEqual(status["dataset_name"], "nthngdy/piqa")


if __name__ == "__main__":
    unittest.main()
