"""Tests for coding-agent harness eval support."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.agent_harness import (
    DEFAULT_HARNESSES,
    DEFAULT_TASKS,
    list_agent_harness_tasks,
    list_agent_harnesses,
    run_agent_harness_eval,
)


class AgentHarnessTests(unittest.TestCase):
    """Validate local fixture and dry-run behavior."""

    def test_catalogs_include_default_harnesses_and_tasks(self) -> None:
        harnesses = {entry["key"] for entry in list_agent_harnesses()}
        tasks = {entry["key"] for entry in list_agent_harness_tasks()}

        self.assertIn("pi", harnesses)
        self.assertIn("opencode", harnesses)
        self.assertIn("codex", harnesses)
        self.assertIn("median_bugfix", tasks)
        self.assertIn("cli_feature", tasks)
        self.assertIn("path_safety", tasks)

    def test_codex_command_uses_workspace_and_xhigh_reasoning(self) -> None:
        workspace = Path("/tmp/smallms-workspace")
        command = DEFAULT_HARNESSES["codex"].command_args(workspace, DEFAULT_TASKS["median_bugfix"].prompt)

        self.assertIn("--cd", command)
        self.assertIn(str(workspace), command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        self.assertIn("gpt-5.5", command)

    def test_dry_run_writes_artifacts_without_agent_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            events = []
            summary = run_agent_harness_eval(
                harnesses=["pi"],
                tasks=["median_bugfix"],
                artifacts_dir=Path(tmpdir),
                timeout_seconds=30,
                dry_run=True,
                progress_callback=events.append,
            )

            self.assertEqual(summary["mode"], "dry_run")
            self.assertEqual(summary["totals"]["rows"], 1)
            self.assertEqual(summary["totals"]["dry_run"], 1)
            self.assertEqual(summary["results"][0]["status"], "dry_run")
            self.assertIn("usage", summary["results"][0])
            self.assertIn("context", summary["results"][0])
            self.assertIn("resource_usage", summary["results"][0])
            self.assertEqual([event["event"] for event in events], ["agent_harness_row_started", "agent_harness_row_completed"])

            run_dir = Path(tmpdir) / "runs" / summary["run_id"]
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "RESULTS.md").exists())
            self.assertTrue((run_dir / "web_summary.json").exists())
            self.assertTrue((Path(tmpdir) / "latest-web-summary.json").exists())
            self.assertTrue((run_dir / "workspaces" / "pi" / "median_bugfix" / "calcstats" / "stats.py").exists())

    def test_codex_model_and_reasoning_overrides_are_run_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_agent_harness_eval(
                harnesses=["codex"],
                tasks=["median_bugfix"],
                artifacts_dir=Path(tmpdir),
                timeout_seconds=30,
                dry_run=True,
                model_override="gpt-test-model",
                reasoning_override="low",
            )

            row = summary["results"][0]
            self.assertEqual(row["model"], "gpt-test-model")
            self.assertEqual(row["reasoning"], "low")
            self.assertIn("gpt-test-model", row["command"])
            self.assertIn('model_reasoning_effort="low"', row["command"])

            catalog_entry = next(entry for entry in summary["available_harnesses"] if entry["key"] == "codex")
            self.assertEqual(catalog_entry["model"], "gpt-test-model")
            self.assertEqual(catalog_entry["reasoning"], "low")
            self.assertEqual(DEFAULT_HARNESSES["codex"].model, "gpt-5.5")
            self.assertEqual(DEFAULT_HARNESSES["codex"].reasoning, "xhigh")

    def test_agent_harness_sync_writes_compact_website_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir) / "artifacts"
            sync_dir = Path(tmpdir) / "web-data"
            summary = run_agent_harness_eval(
                harnesses=["pi"],
                tasks=["median_bugfix"],
                artifacts_dir=artifacts_dir,
                timeout_seconds=30,
                dry_run=True,
                sync_dir=sync_dir,
            )

            latest_path = sync_dir / "agent-harness" / "latest.json"
            history_path = sync_dir / "agent-harness" / "runs" / f"{summary['run_id']}.json"
            self.assertTrue(latest_path.exists())
            self.assertTrue(history_path.exists())

            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "agent_harness.web.v1")
            self.assertEqual(payload["run_id"], summary["run_id"])
            self.assertEqual(payload["totals"]["rows"], 1)
            self.assertEqual(payload["results"][0]["usage"]["source"], "not_reported")

    def test_unknown_harness_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_agent_harness_eval(
                    harnesses=["missing"],
                    artifacts_dir=Path(tmpdir),
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
