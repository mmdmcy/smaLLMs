"""Tests for Codex reasoning-effort sweep orchestration and exports."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_local_benchmarks import build_parser
from src.pipeline.reasoning_sweep import _compact_variant, run_reasoning_effort_sweep


class ReasoningSweepTests(unittest.TestCase):
    """Validate the CLI contract and a minimal provider-free sweep."""

    def test_cli_parser_accepts_reasoning_sweep_options(self) -> None:
        args = build_parser().parse_args(
            [
                "reasoning-sweep",
                "--models",
                "gpt-5.6-sol",
                "--efforts",
                "low",
                "--tasks",
                "median_bugfix",
                "--dry-run",
                "--sync-dir",
                "../websmaLLMs/public/data",
            ]
        )

        self.assertEqual(args.command, "reasoning-sweep")
        self.assertEqual(args.models, ["gpt-5.6-sol"])
        self.assertEqual(args.efforts, ["low"])
        self.assertEqual(args.tasks, ["median_bugfix"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.sync_dir, "../websmaLLMs/public/data")

    def test_dry_run_writes_one_canonical_reasoning_effort_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = run_reasoning_effort_sweep(
                models=["gpt-5.6-sol"],
                efforts=["low"],
                tasks=["median_bugfix"],
                artifacts_dir=root / "artifacts",
                timeout_seconds=30,
                dry_run=True,
                sync_dir=root / "web-data",
            )

            self.assertEqual(payload["schema_version"], "reasoning_effort.web.v1")
            self.assertEqual(payload["selected_models"], ["gpt-5.6-sol"])
            self.assertEqual(payload["selected_reasoning_efforts"], ["low"])
            self.assertIsNone(payload["sampling_temperature"])
            self.assertEqual(payload["totals"]["variants"], 1)
            self.assertEqual(payload["totals"]["rows"], 1)
            self.assertEqual(payload["totals"]["dry_run"], 1)
            self.assertEqual(payload["variants"][0]["model"], "gpt-5.6-sol")
            self.assertEqual(payload["variants"][0]["reasoning_effort"], "low")
            self.assertEqual(payload["results"][0]["status"], "dry_run")

            run_id = payload["run_id"]
            export_root = root / "web-data" / "reasoning-efforts"
            latest_path = export_root / "latest.json"
            history_path = export_root / "runs" / f"{run_id}.json"
            meta_path = export_root / "latest.meta.json"
            pointer_path = export_root / "latest-run.txt"

            self.assertEqual(latest_path.read_bytes(), history_path.read_bytes())
            self.assertEqual(json.loads(latest_path.read_text(encoding="utf-8"))["run_id"], run_id)
            self.assertEqual(json.loads(meta_path.read_text(encoding="utf-8"))["run_id"], run_id)
            self.assertEqual(pointer_path.read_text(encoding="utf-8").strip(), run_id)

    def test_compact_variant_expands_portable_home_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            web_path = root / "web_summary.json"
            web_path.write_text(
                json.dumps({"harnesses": [{}], "results": []}),
                encoding="utf-8",
            )
            result = {
                "run_id": "agent-harness-test",
                "summary_path": "~/summary.json",
                "web_summary_path": "~/web_summary.json",
                "run_dir": "~/run",
            }

            with patch.dict(
                os.environ,
                {"HOME": str(root), "USERPROFILE": str(root)},
            ):
                compact = _compact_variant(result, "gpt-5.6-sol", "low")

            self.assertEqual(compact["run_id"], "agent-harness-test")

    def test_unknown_reasoning_effort_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_reasoning_effort_sweep(
                    models=["gpt-5.6-sol"],
                    efforts=["unsupported"],
                    artifacts_dir=Path(tmpdir),
                    dry_run=True,
                    sync_dir=None,
                )


if __name__ == "__main__":
    unittest.main()
