"""Tests for run-level reproducibility guardrails."""

from __future__ import annotations

import unittest

from src.pipeline.orchestrator import LocalBenchmarkOrchestrator


class OrchestratorReproducibilityTests(unittest.TestCase):
    """Validate offline preflight behavior before expensive model execution."""

    def test_offline_preflight_rejects_partial_cache(self) -> None:
        orchestrator = LocalBenchmarkOrchestrator()

        with self.assertRaisesRegex(RuntimeError, "benchmark cache is incomplete"):
            orchestrator._assert_offline_cache_ready(
                [
                    {
                        "benchmark": "piqa",
                        "cached_rows": 1,
                        "requested_samples": 2,
                        "ready": False,
                    }
                ]
            )

    def test_selected_model_inventory_preserves_order_and_missing_models(self) -> None:
        orchestrator = LocalBenchmarkOrchestrator()

        inventory = orchestrator._selected_model_inventory(
            {"ollama": [{"name": "qwen3.5:latest", "provider": "ollama", "digest": "sha256:abc"}]},
            ["missing:latest", "qwen3.5:latest"],
        )

        self.assertEqual([item["name"] for item in inventory], ["missing:latest", "qwen3.5:latest"])
        self.assertFalse(inventory[0]["available"])
        self.assertEqual(inventory[1]["digest"], "sha256:abc")


if __name__ == "__main__":
    unittest.main()
