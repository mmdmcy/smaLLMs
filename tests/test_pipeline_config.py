"""Tests for shared pipeline defaults and config loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipeline.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_LOCAL_SAMPLE_COUNT,
    DEFAULT_WEBSITE_EXPORT_DIR,
    DEFAULT_WEBSITE_SYNC_DIR,
    config_fingerprint,
    load_pipeline_config,
    local_benchmark_settings,
    redact_sensitive_config,
)


class PipelineConfigTests(unittest.TestCase):
    """Validate that modern pipeline defaults stay centralized."""

    def test_missing_config_uses_shared_defaults(self) -> None:
        config = load_pipeline_config("config/does-not-exist.yaml", ["gsm8k", "mmlu"])
        settings = local_benchmark_settings(config, ["gsm8k", "mmlu"])

        self.assertEqual(settings["artifacts_dir"], DEFAULT_ARTIFACTS_DIR)
        self.assertEqual(settings["website_export_dir"], DEFAULT_WEBSITE_EXPORT_DIR)
        self.assertEqual(settings["website_sync_dir"], DEFAULT_WEBSITE_SYNC_DIR)
        self.assertEqual(settings["default_samples"], DEFAULT_LOCAL_SAMPLE_COUNT)
        self.assertEqual(settings["default_benchmarks"], ["gsm8k", "mmlu"])

    def test_partial_config_merges_without_dropping_standard_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            config_path.write_text(
                "local_benchmarks:\n"
                "  default_samples: 7\n"
                "  website_sync_dir: null\n",
                encoding="utf-8",
            )

            config = load_pipeline_config(str(config_path), ["gsm8k"])
            settings = local_benchmark_settings(config, ["gsm8k"])

        self.assertEqual(settings["default_samples"], 7)
        self.assertEqual(settings["artifacts_dir"], DEFAULT_ARTIFACTS_DIR)
        self.assertEqual(settings["website_export_dir"], DEFAULT_WEBSITE_EXPORT_DIR)
        self.assertIsNone(settings["website_sync_dir"])
        self.assertEqual(settings["default_benchmarks"], ["gsm8k"])

    def test_config_snapshot_redacts_secrets_before_fingerprinting(self) -> None:
        config = {
            "huggingface": {"token": "hf_real_token"},
            "service": {"api_key": "abc123", "safe_value": "visible"},
        }

        redacted = redact_sensitive_config(config)

        self.assertEqual(redacted["huggingface"]["token"], "<redacted>")
        self.assertEqual(redacted["service"]["api_key"], "<redacted>")
        self.assertEqual(redacted["service"]["safe_value"], "visible")
        self.assertEqual(config_fingerprint(config), config_fingerprint(redacted))


if __name__ == "__main__":
    unittest.main()
