"""Tests for fetching benchmark rows from a (fake) Hugging Face dataset viewer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from src.pipeline import benchmarks, hf_datasets
from tests.fake_http import FakeServer


def _viewer_rows(rows, offset, length, total=None, truncated=None):
    page = rows[offset : offset + length]
    return {
        "rows": [
            {"row_idx": offset + i, "row": row, "truncated_cells": list(truncated or [])}
            for i, row in enumerate(page)
        ],
        "num_rows_total": len(rows) if total is None else total,
        "partial": False,
    }


class FakeViewerTestCase(unittest.TestCase):
    """Point the fetcher at a fake viewer, with HOME and the cache in a temp folder."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        env = mock.patch.dict(os.environ, {"HOME": self.tempdir.name, "XDG_CACHE_HOME": self.tempdir.name})
        env.start()
        self.addCleanup(env.stop)
        for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_KEY", "SMALLMS_OFFLINE"):
            os.environ.pop(name, None)
        token = mock.patch.object(hf_datasets, "huggingface_token", return_value=None)
        self.token = token.start()
        self.addCleanup(token.stop)
        self.addCleanup(hf_datasets._RESOLVED_NAMES.clear)
        self.addCleanup(hf_datasets._DEFAULT_CONFIGS.clear)
        sleep = mock.patch.object(hf_datasets.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        benchmarks.configure_dataset_runtime({"dataset_cache_dir": self.tempdir.name})
        self.addCleanup(
            benchmarks.configure_dataset_runtime, {"dataset_cache_dir": None, "allow_remote_dataset_downloads": True}
        )

    def serve(self, routes):
        server = FakeServer(routes)
        server.__enter__()
        self.addCleanup(server.__exit__, None, None, None)
        for name, value in (("VIEWER_URL", server.url), ("HUB_URL", server.url)):
            patcher = mock.patch.object(hf_datasets, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return server


class ViewerFetchTests(FakeViewerTestCase):
    def test_rows_are_paged_lazily_and_in_order(self) -> None:
        rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(250)]

        def rows_route(request):
            self.assertEqual((request.query["dataset"], request.query["config"]), ("openai/gsm8k", "main"))
            return 200, _viewer_rows(rows, int(request.query["offset"]), int(request.query["length"])), {}

        server = self.serve(
            {
                ("GET", "/api/datasets/gsm8k"): lambda r: (307, "", {"Location": "/api/datasets/openai/gsm8k"}),
                ("GET", "/api/datasets/openai/gsm8k"): lambda r: (200, {"id": "openai/gsm8k"}, {}),
                ("GET", "/rows"): rows_route,
            }
        )
        everything = list(hf_datasets.iter_rows("gsm8k", "main", "test"))
        self.assertEqual(everything, rows)
        lengths = [int(r.query["length"]) for r in server.requests if r.path == "/rows"]
        self.assertEqual(lengths, [100, 100, 100])
        self.assertNotIn("authorization", server.requests[-1].headers)

        server.requests.clear()
        first = next(iter(hf_datasets.iter_rows("gsm8k", "main", "test", page_size=10)))
        self.assertEqual(first, rows[0])
        self.assertEqual([r.path for r in server.requests], ["/rows"])  # the name is resolved once

    def test_default_config_and_token_header(self) -> None:
        self.token.return_value = "fake-token"
        server = self.serve(
            {
                ("GET", "/splits"): lambda r: (200, {"splits": [{"config": "default", "split": "train"}]}, {}),
                ("GET", "/rows"): lambda r: (200, _viewer_rows([{"a": 1}], 0, 100), {}),
            }
        )
        self.assertEqual(list(hf_datasets.iter_rows("owner/name", None, "train")), [{"a": 1}])
        self.assertEqual(server.requests[-1].query["config"], "default")
        self.assertEqual(server.requests[-1].headers["authorization"], "Bearer fake-token")

    def test_truncated_partial_and_unordered_results_are_refused(self) -> None:
        cases = {
            "truncated": _viewer_rows([{"prompt": "x"}], 0, 100, truncated=["prompt"]),
            "partial": dict(_viewer_rows([{"a": 1}], 0, 100), partial=True),
            "out of order": {
                "rows": [{"row_idx": 5, "row": {}, "truncated_cells": []}, {"row_idx": 2, "row": {}, "truncated_cells": []}],
                "num_rows_total": 2,
                "partial": False,
            },
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                self.serve({("GET", "/rows"): lambda r, p=payload: (200, p, {})})
                with self.assertRaises(hf_datasets.DatasetFetchError):
                    list(hf_datasets.iter_rows("owner/name", "main", "test"))

    def test_temporary_errors_are_retried_and_lasting_ones_reported(self) -> None:
        answers = [(500, {"error": "the dataset index is loading, this can take a minute"}, {})] * 2
        answers.append((200, _viewer_rows([{"a": 1}], 0, 100), {}))
        self.serve({("GET", "/filter"): lambda r: answers.pop(0)})
        rows = list(hf_datasets.iter_filtered_rows("owner/name", "default", "train", '"a"=1'))
        self.assertEqual(rows, [{"a": 1}])
        self.assertEqual(self.sleep.call_count, 2)

        self.serve({("GET", "/rows"): lambda r: (400, {"error": "Not found."}, {})})
        with self.assertRaisesRegex(hf_datasets.DatasetFetchError, "HTTP 400 .*Not found"):
            list(hf_datasets.iter_rows("owner/name", "main", "test"))


class BenchmarkLoadingTests(FakeViewerTestCase):
    def test_second_run_is_served_from_the_cache_offline(self) -> None:
        rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(5)]
        server = self.serve(
            {
                ("GET", "/api/datasets/gsm8k"): lambda r: (200, {"id": "openai/gsm8k"}, {}),
                ("GET", "/rows"): lambda r: (200, _viewer_rows(rows, int(r.query["offset"]), int(r.query["length"])), {}),
            }
        )
        benchmark = benchmarks.SUPPORTED_BENCHMARKS["gsm8k"]
        self.assertEqual(benchmark.load_rows(3), rows[:3])
        fetched = len(server.requests)

        benchmarks.configure_dataset_runtime(
            {"dataset_cache_dir": self.tempdir.name, "allow_remote_dataset_downloads": False}
        )
        self.assertEqual(benchmark.load_rows(3), rows[:3])
        self.assertEqual(len(server.requests), fetched)
        status = benchmarks.benchmark_cache_status(["gsm8k"], 3)[0]
        self.assertTrue(status["ready"])

    def test_filtered_benchmarks_ask_the_viewer_filter(self) -> None:
        matching = [{"n_needles": 8, "n_chars": 20000, "prompt": "[]", "answer": "x"}]
        server = self.serve(
            {
                ("GET", "/splits"): lambda r: (200, {"splits": [{"config": "default", "split": "train"}]}, {}),
                ("GET", "/filter"): lambda r: (200, _viewer_rows(matching, 0, 100), {}),
            }
        )
        rows = benchmarks.SUPPORTED_BENCHMARKS["mrcr_v2_8needle_4k_8k"]._load_rows_uncached(1)
        self.assertEqual(rows, matching)
        query = server.requests[-1].query
        self.assertEqual(query["dataset"], "openai/mrcr")
        self.assertEqual(query["where"], '"n_needles"=8 AND "n_chars">=16384 AND "n_chars"<32768')
        self.assertEqual(query["length"], "1")

    def test_social_iqa_reads_the_source_archive(self) -> None:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(
                "socialiqa-train-dev/dev.jsonl",
                "\n".join(
                    json.dumps({"context": f"c{i}", "question": "q", "answerA": "a", "answerB": "b", "answerC": "c"})
                    for i in range(3)
                )
                + "\n",
            )
            handle.writestr("socialiqa-train-dev/dev-labels.lst", "1\n3\n2\n")
        server = self.serve({("GET", "/socialiqa.zip"): lambda r: (200, archive.getvalue(), {})})
        benchmark = benchmarks.SUPPORTED_BENCHMARKS["social_iqa"]
        with mock.patch.object(type(benchmark), "source_archive_url", f"{server.url}/socialiqa.zip"):
            rows = benchmark._load_rows_uncached(2)
        self.assertEqual(
            rows,
            [
                {"context": "c0", "question": "q", "answerA": "a", "answerB": "b", "answerC": "c", "label": "1"},
                {"context": "c1", "question": "q", "answerA": "a", "answerB": "b", "answerC": "c", "label": "3"},
            ],
        )

    def test_offline_without_cache_fails_before_any_request(self) -> None:
        benchmarks.configure_dataset_runtime(
            {"dataset_cache_dir": self.tempdir.name, "allow_remote_dataset_downloads": False}
        )
        with self.assertRaisesRegex(RuntimeError, "cached row"):
            benchmarks.SUPPORTED_BENCHMARKS["boolq"].load_rows(2)


if __name__ == "__main__":
    unittest.main()
