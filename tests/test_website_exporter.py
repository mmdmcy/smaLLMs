"""Tests for the website session exporter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.exporter import WebsiteExporter


class WebsiteExporterTests(unittest.TestCase):
    """Validate the full-session website export format."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.artifacts_dir = self.root / "artifacts"
        self.output_dir = self.root / "website_exports"
        self.sync_dir = self.root / "websmaLLMs" / "public" / "data"

        self.run_id = "run_test_123"
        run_dir = self.artifacts_dir / "runs" / self.run_id
        benchmark_dir = run_dir / "benchmarks" / "gsm8k"
        sample_dir = run_dir / "samples" / "gsm8k"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "created_at": "2026-04-01T10:00:00+00:00",
            "models": ["qwen3.5:0.8b"],
            "benchmarks": ["gsm8k"],
            "supported_benchmarks": [
                {
                    "key": "gsm8k",
                    "display_name": "GSM8K",
                    "description": "Grade-school math word problems.",
                    "category": "reasoning",
                    "status": "runnable_local",
                    "harness": "single_turn_local_prompt",
                    "local_runnable": True,
                }
            ],
            "benchmark_suites": [{"key": "quick_suite", "benchmarks": ["gsm8k"]}],
        }

        summary = {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "generated_at": "2026-04-01T10:05:00+00:00",
            "manifest_path": f"artifacts/runs/{self.run_id}/manifest.json",
            "totals": {
                "models": 1,
                "benchmarks": 1,
                "evaluations": 1,
                "completed_evaluations": 1,
                "failed_evaluations": 0,
                "samples": 1,
                "correct": 1,
                "accuracy": 1.0,
                "errors": 0,
                "total_tokens": 42,
                "total_duration_sec": 1.25,
            },
            "leaderboard": [
                {
                    "rank": 1,
                    "model_name": "qwen3.5:0.8b",
                    "provider": "ollama",
                    "size_gb": 0.965,
                    "parameters": "873.44M",
                    "family": "qwen35",
                    "quantization": "Q8_0",
                    "overall_accuracy": 1.0,
                    "benchmarks_run": 1,
                    "total_samples": 1,
                    "correct_count": 1,
                    "error_count": 0,
                    "avg_latency_sec": 1.25,
                    "total_tokens": 42,
                    "benchmarks": {
                        "gsm8k": {
                            "accuracy": 1.0,
                            "sample_count": 1,
                            "avg_latency_sec": 1.25,
                            "avg_tokens_per_second": 18.4,
                        }
                    },
                }
            ],
            "evaluations": [
                {
                    "schema_version": "2.0",
                    "run_id": self.run_id,
                    "benchmark_name": "gsm8k",
                    "benchmark_display_name": "GSM8K",
                    "description": "Grade-school math word problems.",
                    "dataset": {"name": "gsm8k", "config_name": "main", "split": "test"},
                    "model": {
                        "name": "qwen3.5:0.8b",
                        "provider": "ollama",
                        "size_gb": 0.965,
                        "parameters": "873.44M",
                        "family": "qwen35",
                        "quantization": "Q8_0",
                    },
                    "metrics": {
                        "sample_count": 1,
                        "correct_count": 1,
                        "accuracy": 1.0,
                        "error_count": 0,
                        "avg_latency_sec": 1.25,
                        "max_latency_sec": 1.25,
                        "min_latency_sec": 1.25,
                        "total_prompt_tokens": 22,
                        "total_completion_tokens": 20,
                        "total_tokens": 42,
                        "avg_tokens_per_second": 18.4,
                        "local_cost_estimate": 0.0,
                    },
                    "status": "completed",
                    "artifact_paths": {
                        "samples_jsonl": f"artifacts/runs/{self.run_id}/samples/gsm8k/qwen3.5_0.8b.jsonl",
                        "benchmark_json": f"artifacts/runs/{self.run_id}/benchmarks/gsm8k/qwen3.5_0.8b.json",
                    },
                }
            ],
        }

        sample = {
            "run_id": self.run_id,
            "benchmark_name": "gsm8k",
            "sample_index": 0,
            "model_name": "qwen3.5:0.8b",
            "provider": "ollama",
            "prompt": "Question: 20 + 22",
            "response_text": "Final answer: 42",
            "expected_answer": "42",
            "parsed_prediction": "42",
            "is_correct": True,
            "error": None,
            "started_at": "2026-04-01T10:00:01+00:00",
            "ended_at": "2026-04-01T10:00:02+00:00",
            "latency_sec": 1.25,
            "prompt_tokens": 22,
            "completion_tokens": 20,
            "total_tokens": 42,
            "tokens_per_second": 18.4,
            "question": "20 + 22",
        }

        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (benchmark_dir / "qwen3.5_0.8b.json").write_text(json.dumps(summary["evaluations"][0], indent=2), encoding="utf-8")
        (sample_dir / "qwen3.5_0.8b.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
        (self.artifacts_dir / "latest_run.txt").write_text(self.run_id + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_exporter_embeds_samples_and_syncs_latest_session(self) -> None:
        exporter = WebsiteExporter(
            artifacts_dir=str(self.artifacts_dir),
            output_dir=str(self.output_dir),
            sync_dir=str(self.sync_dir),
        )

        exported = exporter.export_run()

        session_path = self.output_dir / "latest" / "session.json"
        self.assertTrue(session_path.exists())
        self.assertIn("session.json", exported)
        self.assertIn("sync/latest-session.json", exported)

        session = json.loads(session_path.read_text(encoding="utf-8"))
        self.assertEqual(session["schema_version"], "3.0")
        self.assertEqual(session["run"]["run_id"], self.run_id)
        self.assertEqual(session["summary"]["totals"]["samples"], 1)
        self.assertEqual(session["catalog"]["selected_benchmarks"][0]["key"], "gsm8k")

        evaluation = session["evaluations"][0]
        self.assertEqual(evaluation["evaluation_id"], "gsm8k__qwen3.5_0.8b")
        self.assertEqual(evaluation["sample_count_embedded"], 1)
        self.assertEqual(evaluation["samples"][0]["sample_id"], "gsm8k__qwen3.5_0.8b::0")
        self.assertEqual(evaluation["samples"][0]["prompt_chars"], len("Question: 20 + 22"))
        self.assertEqual(evaluation["metrics"]["total_prompt_chars"], len("Question: 20 + 22"))
        self.assertTrue(evaluation["samples"][0]["is_correct"])

        synced_session = self.sync_dir / "latest-session.json"
        self.assertTrue(synced_session.exists())
        synced_payload = json.loads(synced_session.read_text(encoding="utf-8"))
        self.assertEqual(synced_payload["run"]["run_id"], self.run_id)


if __name__ == "__main__":
    unittest.main()
