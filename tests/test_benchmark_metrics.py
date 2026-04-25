"""Tests for benchmark metric aggregation."""

from __future__ import annotations

import unittest

from src.models.model_manager import GenerationResult
from src.pipeline.benchmarks import DatasetBenchmark, summarize_samples


class BenchmarkMetricAggregationTests(unittest.TestCase):
    """Validate enriched summary metrics built from per-sample records."""

    def test_summarize_samples_captures_extended_timings_and_sizes(self) -> None:
        samples = [
            {
                "is_correct": True,
                "error": None,
                "response_text": "Final answer: 42",
                "parsed_prediction": "42",
                "latency_sec": 1.5,
                "load_duration_sec": 0.2,
                "prompt_eval_duration_sec": 0.3,
                "eval_duration_sec": 0.6,
                "total_duration_sec": 1.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "tokens_per_second": 8.0,
                "prompt_chars": 100,
                "response_chars": 16,
                "expected_answer_chars": 2,
                "parsed_prediction_chars": 2,
            },
            {
                "is_correct": False,
                "error": "parse_failed",
                "response_text": "",
                "parsed_prediction": "",
                "latency_sec": 2.5,
                "load_duration_sec": 0.4,
                "prompt_eval_duration_sec": 0.5,
                "eval_duration_sec": 0.7,
                "total_duration_sec": 1.6,
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
                "tokens_per_second": 0.0,
                "prompt_chars": 140,
                "response_chars": 0,
                "expected_answer_chars": 1,
                "parsed_prediction_chars": 0,
            },
        ]

        metrics = summarize_samples(samples)

        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["correct_count"], 1)
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["responded_count"], 1)
        self.assertEqual(metrics["valid_prediction_count"], 1)
        self.assertEqual(metrics["invalid_prediction_count"], 1)
        self.assertAlmostEqual(metrics["success_rate"], 0.5)
        self.assertAlmostEqual(metrics["response_rate"], 0.5)
        self.assertAlmostEqual(metrics["invalid_prediction_rate"], 0.5)
        self.assertIn("accuracy_ci95_low", metrics)
        self.assertIn("accuracy_ci95_high", metrics)
        self.assertAlmostEqual(metrics["avg_load_duration_sec"], 0.3)
        self.assertAlmostEqual(metrics["avg_prompt_eval_duration_sec"], 0.4)
        self.assertAlmostEqual(metrics["avg_eval_duration_sec"], 0.65)
        self.assertAlmostEqual(metrics["avg_total_duration_sec"], 1.35)
        self.assertEqual(metrics["total_prompt_tokens"], 30)
        self.assertEqual(metrics["total_completion_tokens"], 13)
        self.assertEqual(metrics["total_prompt_chars"], 240)
        self.assertEqual(metrics["total_response_chars"], 16)
        self.assertAlmostEqual(metrics["avg_prompt_chars"], 120.0)
        self.assertAlmostEqual(metrics["avg_response_chars"], 8.0)
        self.assertAlmostEqual(metrics["avg_tokens_per_second"], 8.0)
        self.assertAlmostEqual(metrics["max_tokens_per_second"], 8.0)


class _TinyBenchmark(DatasetBenchmark):
    key = "tiny"
    display_name = "Tiny"
    dataset_name = "local/tiny"
    config_name = None
    split = "test"
    description = "Tiny deterministic test benchmark."

    def load_rows(self, limit: int):
        return [{"question": "Pick A", "answer": "A"}][:limit]

    def build_prompt(self, row):
        return f"{row['question']}\nAnswer:"

    def extract_expected_answer(self, row):
        return row["answer"]

    def parse_prediction(self, response_text, row):
        return response_text.strip()

    def sample_metadata(self, row):
        return {"question": row["question"]}


class _FakeModel:
    async def generate_with_metadata(self, prompt, generation_config):
        return GenerationResult(
            text="A",
            prompt=prompt,
            started_at="2026-04-26T00:00:00+00:00",
            ended_at="2026-04-26T00:00:01+00:00",
            latency_sec=1.0,
            provider="fake",
            raw={},
        )


class BenchmarkSampleIdentityTests(unittest.IsolatedAsyncioTestCase):
    """Validate per-sample audit fields emitted by benchmark evaluation."""

    async def test_evaluate_records_stable_sample_and_prompt_hashes(self) -> None:
        benchmark = _TinyBenchmark()

        execution = await benchmark.evaluate(
            model=_FakeModel(),
            model_info={"name": "fake-model", "provider": "fake"},
            run_id="run_test",
            num_samples=1,
        )
        sample = execution.samples[0]

        self.assertTrue(sample["sample_id"].startswith("tiny:"))
        self.assertEqual(len(sample["sample_input_sha256"]), 64)
        self.assertEqual(len(sample["prompt_sha256"]), 64)
        self.assertEqual(len(sample["prompt_template_sha256"]), 64)
        self.assertTrue(sample["prediction_valid"])
        self.assertEqual(execution.benchmark_result["prompt"]["prompt_template_id"], "tiny:_TinyBenchmark.build_prompt")


if __name__ == "__main__":
    unittest.main()
