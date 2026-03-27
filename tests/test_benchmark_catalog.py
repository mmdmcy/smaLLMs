"""Smoke tests for the current benchmark catalog."""

from __future__ import annotations

import json
import unittest

from src.pipeline.benchmarks import (
    DEFAULT_BENCHMARKS,
    SUPPORTED_BENCHMARKS,
    expand_benchmark_selection,
    list_benchmark_catalog,
    list_supported_benchmarks,
)


class BenchmarkCatalogTests(unittest.TestCase):
    """Validate the runnable benchmark registry and tracked catalog."""

    def test_default_suite_contains_piqa(self) -> None:
        self.assertIn("piqa", DEFAULT_BENCHMARKS)

    def test_catalog_exposes_runnable_and_tracked_entries(self) -> None:
        catalog = list_benchmark_catalog()
        runnable = {entry["key"] for entry in catalog if entry["local_runnable"]}
        tracked = {entry["key"] for entry in catalog if not entry["local_runnable"]}

        self.assertIn("gsm8k", runnable)
        self.assertIn("aime_2024", runnable)
        self.assertIn("mrcr_v2_8needle_4k_8k", runnable)
        self.assertIn("tau_bench", tracked)
        self.assertIn("swe_bench_verified", tracked)

    def test_supported_benchmark_list_includes_new_runnable_benchmarks(self) -> None:
        supported = {entry["key"] for entry in list_supported_benchmarks()}
        self.assertIn("piqa", supported)
        self.assertIn("social_iqa", supported)
        self.assertIn("aime_2025", supported)
        self.assertIn("graphwalks_bfs_0_128k", supported)
        self.assertIn("mrcr_v2_8needle_4k_8k", supported)

    def test_frontier_alias_expands_to_runnable_suite(self) -> None:
        expanded = expand_benchmark_selection(["frontier"])
        self.assertIn("gsm8k", expanded)
        self.assertIn("piqa", expanded)
        self.assertIn("aime_2024", expanded)
        self.assertIn("graphwalks_bfs_0_128k", expanded)
        self.assertNotIn("tau_bench", expanded)

    def test_openai_public_alias_expands_to_public_suite(self) -> None:
        expanded = expand_benchmark_selection(["openai_public"])
        self.assertIn("aime_2024", expanded)
        self.assertIn("graphwalks_parents_0_128k", expanded)
        self.assertIn("mrcr_v2_8needle_8k_16k", expanded)

    def test_tracked_only_benchmark_raises_helpful_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            expand_benchmark_selection(["tau_bench"])

        self.assertIn("tracked but not locally runnable yet", str(context.exception))

    def test_graphwalks_parser_normalizes_set_style_answers(self) -> None:
        benchmark = SUPPORTED_BENCHMARKS["graphwalks_parents_0_128k"]
        row = {"answer_nodes": ["node-a", "node-b"]}
        expected = benchmark.extract_expected_answer(row)
        predicted = benchmark.parse_prediction("Reasoning...\nFinal Answer: ['node-b', 'node-a']", row)
        self.assertEqual(predicted, expected)

    def test_mrcr_prompt_and_answer_normalization(self) -> None:
        benchmark = SUPPORTED_BENCHMARKS["mrcr_v2_8needle_4k_8k"]
        row = {
            "prompt": json.dumps(
                [
                    {"role": "user", "content": "Find the earlier answer."},
                    {"role": "assistant", "content": "First response."},
                    {"role": "user", "content": "Repeat it exactly."},
                ]
            ),
            "answer": "First response.\nWith a second line.",
        }
        prompt = benchmark.build_prompt(row)
        prediction = benchmark.parse_prediction("Assistant: ```\nFirst response.\nWith a second line.\n```", row)
        self.assertIn("User: Find the earlier answer.", prompt)
        self.assertTrue(prompt.rstrip().endswith("Assistant:"))
        self.assertEqual(prediction, benchmark.extract_expected_answer(row))


if __name__ == "__main__":
    unittest.main()
