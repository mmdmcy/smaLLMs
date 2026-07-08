"""
Local coding-agent harness evaluation.

This module compares agent harness CLIs such as Pi, OpenCode, and Codex on
small deterministic code-edit fixtures. It is intentionally separate from the
model benchmark runner: these tasks evaluate the harness loop around a model,
not a model served through Ollama or LM Studio.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.pipeline.artifacts import portable_path, safe_slug, utcnow_iso
from src.pipeline.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_WEBSITE_SYNC_DIR


DEFAULT_AGENT_HARNESS_DIR = Path(DEFAULT_ARTIFACTS_DIR) / "agent_harness"
DEFAULT_AGENT_HARNESS_WEBSITE_SYNC_DIR = DEFAULT_WEBSITE_SYNC_DIR
AGENT_HARNESS_WEB_SCHEMA_VERSION = "agent_harness.web.v1"
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


@dataclass(frozen=True)
class AgentHarness:
    """Command metadata for one coding-agent harness."""

    key: str
    display_name: str
    command: str
    model: str
    reasoning: str

    def command_args(self, workspace: Path, prompt: str) -> List[str]:
        """Build the non-interactive command for this harness."""
        if self.key == "pi":
            return [
                self.command,
                "--model",
                self.model,
                "--thinking",
                self.reasoning,
                "--no-session",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "--no-context-files",
                "-p",
                prompt,
            ]

        if self.key == "opencode":
            return [
                self.command,
                "run",
                "--pure",
                "--dir",
                str(workspace),
                "--model",
                self.model,
                "--variant",
                self.reasoning,
                prompt,
            ]

        if self.key == "codex":
            return [
                self.command,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--sandbox",
                "workspace-write",
                "--model",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning}"',
                "--cd",
                str(workspace),
                prompt,
            ]

        raise ValueError(f"Unsupported agent harness: {self.key}")


@dataclass(frozen=True)
class HarnessTask:
    """One deterministic code-edit task for harness comparison."""

    key: str
    display_name: str
    prompt: str
    test_command: Sequence[str]
    expected_files: Sequence[str]


DEFAULT_HARNESSES: Dict[str, AgentHarness] = {
    "pi": AgentHarness(
        key="pi",
        display_name="Pi",
        command="pi",
        model="openai-codex/gpt-5.5",
        reasoning="xhigh",
    ),
    "opencode": AgentHarness(
        key="opencode",
        display_name="OpenCode",
        command="opencode",
        model="openai/gpt-5.5",
        reasoning="xhigh",
    ),
    "codex": AgentHarness(
        key="codex",
        display_name="Codex CLI",
        command="codex",
        model="gpt-5.5",
        reasoning="xhigh",
    ),
}


DEFAULT_TASKS: Dict[str, HarnessTask] = {
    "median_bugfix": HarnessTask(
        key="median_bugfix",
        display_name="Median bugfix",
        prompt=(
            "You are in a small Python repository. Fix the failing median behavior "
            "in calcstats/stats.py with the smallest sensible change. Run: "
            "python3 -m unittest tests.test_stats. Do not change unrelated files. "
            "Summarize what changed and the test result."
        ),
        test_command=("python3", "-m", "unittest", "tests.test_stats"),
        expected_files=("calcstats/stats.py",),
    ),
    "cli_feature": HarnessTask(
        key="cli_feature",
        display_name="CLI feature",
        prompt=(
            "You are in a small Python repository. Implement the missing CLI so "
            "these commands work: python3 -m calcstats median 1 2 3 and "
            "python3 -m calcstats percentile 75 10 20 30 40. Run: "
            "python3 -m unittest tests.test_cli. Do not change unrelated behavior. "
            "Summarize what changed and the test result."
        ),
        test_command=("python3", "-m", "unittest", "tests.test_cli"),
        expected_files=("calcstats/cli.py",),
    ),
    "path_safety": HarnessTask(
        key="path_safety",
        display_name="Path safety",
        prompt=(
            "You are in a small Python repository. Fix calcstats/files.py so "
            "report_path only accepts safe report names and rejects path traversal "
            "or absolute paths. Run: python3 -m unittest tests.test_files. Do not "
            "change unrelated behavior. Summarize what changed and the test result."
        ),
        test_command=("python3", "-m", "unittest", "tests.test_files"),
        expected_files=("calcstats/files.py",),
    ),
}


FIXTURE_FILES: Dict[str, str] = {
    ".gitignore": "__pycache__/\n*.py[cod]\n",
    "README.md": (
        "# calcstats Fixture\n\n"
        "Small deterministic repo used to compare coding-agent harness behavior.\n"
    ),
    "pyproject.toml": (
        "[project]\n"
        'name = "calcstats"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.10"\n'
    ),
    "calcstats/__init__.py": 'from .stats import median, percentile\n\n__all__ = ["median", "percentile"]\n',
    "calcstats/__main__.py": "from .cli import main\n\nraise SystemExit(main())\n",
    "calcstats/files.py": (
        "from pathlib import Path\n\n\n"
        "def report_path(name: str) -> Path:\n"
        '    return Path("reports") / f"{name}.txt"\n'
    ),
    "calcstats/stats.py": (
        "def median(values):\n"
        "    if not values:\n"
        '        raise ValueError("median requires at least one value")\n'
        "    ordered = sorted(values)\n"
        "    return ordered[len(ordered) // 2]\n\n\n"
        "def percentile(values, percentile_value):\n"
        "    if not values:\n"
        '        raise ValueError("percentile requires at least one value")\n'
        "    if not 0 <= percentile_value <= 100:\n"
        '        raise ValueError("percentile must be between 0 and 100")\n\n'
        "    ordered = sorted(values)\n"
        "    index = round((percentile_value / 100) * (len(ordered) - 1))\n"
        "    return ordered[index]\n\n"
    ),
    "tests/__init__.py": "",
    "tests/test_stats.py": (
        "import unittest\n\n"
        "from calcstats.stats import median, percentile\n\n\n"
        "class TestStats(unittest.TestCase):\n"
        "    def test_median_odd_length(self):\n"
        "        self.assertEqual(median([9, 1, 5]), 5)\n\n"
        "    def test_median_even_length(self):\n"
        "        self.assertEqual(median([1, 2, 3, 4]), 2.5)\n\n"
        "    def test_empty_median_rejected(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            median([])\n\n"
        "    def test_percentile(self):\n"
        "        self.assertEqual(percentile([10, 20, 30, 40], 75), 30)\n\n\n"
        'if __name__ == "__main__":\n'
        "    unittest.main()\n"
    ),
    "tests/test_cli.py": (
        "import subprocess\n"
        "import sys\n"
        "import unittest\n\n\n"
        "class TestCli(unittest.TestCase):\n"
        "    def run_cli(self, *args):\n"
        "        return subprocess.run(\n"
        '            [sys.executable, "-m", "calcstats", *args],\n'
        "            check=False,\n"
        "            stdout=subprocess.PIPE,\n"
        "            stderr=subprocess.PIPE,\n"
        "            text=True,\n"
        "        )\n\n"
        "    def test_median_command(self):\n"
        '        result = self.run_cli("median", "1", "2", "3")\n'
        "        self.assertEqual(result.returncode, 0, result.stderr)\n"
        '        self.assertEqual(result.stdout.strip(), "2")\n\n'
        "    def test_percentile_command(self):\n"
        '        result = self.run_cli("percentile", "75", "10", "20", "30", "40")\n'
        "        self.assertEqual(result.returncode, 0, result.stderr)\n"
        '        self.assertEqual(result.stdout.strip(), "30")\n\n'
        "    def test_unknown_command_fails(self):\n"
        '        result = self.run_cli("mode", "1", "2", "2")\n'
        "        self.assertNotEqual(result.returncode, 0)\n"
        '        self.assertIn("usage:", result.stderr.lower())\n\n\n'
        'if __name__ == "__main__":\n'
        "    unittest.main()\n"
    ),
    "tests/test_files.py": (
        "from pathlib import Path\n"
        "import unittest\n\n"
        "from calcstats.files import report_path\n\n\n"
        "class TestFiles(unittest.TestCase):\n"
        "    def test_simple_report_name(self):\n"
        '        self.assertEqual(report_path("weekly-1"), Path("reports/weekly-1.txt"))\n\n'
        "    def test_rejects_path_traversal(self):\n"
        "        with self.assertRaises(ValueError):\n"
        '            report_path("../secrets")\n\n'
        "    def test_rejects_absolute_path(self):\n"
        "        with self.assertRaises(ValueError):\n"
        '            report_path("/tmp/report")\n\n'
        "    def test_rejects_separator(self):\n"
        "        with self.assertRaises(ValueError):\n"
        '            report_path("team/report")\n\n\n'
        'if __name__ == "__main__":\n'
        "    unittest.main()\n"
    ),
}


def list_agent_harnesses() -> List[Dict[str, Any]]:
    """Return default harness metadata and host availability."""
    result = []
    for harness in DEFAULT_HARNESSES.values():
        resolved = shutil.which(harness.command)
        result.append(
            {
                "key": harness.key,
                "display_name": harness.display_name,
                "command": harness.command,
                "path": _portable_executable_path(resolved) if resolved else None,
                "available": bool(resolved),
                "model": harness.model,
                "reasoning": harness.reasoning,
                "version": _command_version(harness.command) if resolved else None,
            }
        )
    return result


def list_agent_harness_tasks() -> List[Dict[str, Any]]:
    """Return task metadata for the local harness eval."""
    return [
        {
            "key": task.key,
            "display_name": task.display_name,
            "test_command": list(task.test_command),
            "expected_files": list(task.expected_files),
        }
        for task in DEFAULT_TASKS.values()
    ]


def _select_keys(requested: Optional[Sequence[str]], available: Dict[str, Any], label: str) -> List[str]:
    if not requested:
        return list(available)

    selected: List[str] = []
    for raw in requested:
        for part in raw.split(","):
            key = part.strip()
            if not key:
                continue
            if key not in available:
                raise ValueError(f"Unknown {label}: {key}. Available: {', '.join(available)}")
            selected.append(key)
    return selected


def _portable_executable_path(value: str) -> str:
    """Return an executable path without resolving user-local symlink targets."""
    path = Path(value)
    try:
        return (Path("~") / path.relative_to(Path.home())).as_posix()
    except ValueError:
        return portable_path(path)


def _command_version(command: str) -> Optional[str]:
    """Return a short CLI version string when the harness reports one."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    text = "\n".join(part for part in [result.stdout, result.stderr] if part)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _write_fixture(workspace: Path) -> None:
    """Create a fresh deterministic fixture workspace."""
    for relative_path, content in FIXTURE_FILES.items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _run(
    args: Sequence[str],
    cwd: Path,
    timeout_seconds: int,
    resource_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    started_at = utcnow_iso()
    command_args = list(args)
    resource_metrics = {"source": "not_collected"}
    if resource_log_path and _gnu_time_path():
        resource_log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_resource_log_path = resource_log_path.resolve(strict=False)
        command_args = [
            str(_gnu_time_path()),
            "-v",
            "-o",
            str(resolved_resource_log_path),
            *command_args,
        ]
        resource_metrics = {
            "source": "gnu_time_v",
            "log_path": portable_path(resource_log_path),
        }

    try:
        result = subprocess.run(
            command_args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        ended_at = utcnow_iso()
        resource_metrics.update(_read_resource_metrics(resource_log_path))
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "timed_out": False,
            "started_at": started_at,
            "ended_at": ended_at,
            "resource_metrics": resource_metrics,
        }
    except subprocess.TimeoutExpired as exc:
        ended_at = utcnow_iso()
        resource_metrics.update(_read_resource_metrics(resource_log_path))
        return {
            "returncode": None,
            "stdout": _coerce_output(exc.stdout),
            "stderr": _coerce_output(exc.stderr),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "timed_out": True,
            "started_at": started_at,
            "ended_at": ended_at,
            "resource_metrics": resource_metrics,
        }


def _gnu_time_path() -> Optional[Path]:
    path = Path("/usr/bin/time")
    return path if path.exists() and os.access(path, os.X_OK) else None


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_resource_metrics(resource_log_path: Optional[Path]) -> Dict[str, Any]:
    if not resource_log_path or not resource_log_path.exists():
        return {}

    text = resource_log_path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_gnu_time_metrics(text)
    return {
        "metrics": parsed,
        "log_bytes": resource_log_path.stat().st_size,
    }


def _parse_gnu_time_metrics(text: str) -> Dict[str, Any]:
    """Parse GNU time -v output into stable machine-readable keys."""
    key_map = {
        "User time (seconds)": "user_time_seconds",
        "System time (seconds)": "system_time_seconds",
        "Percent of CPU this job got": "cpu_percent",
        "Maximum resident set size (kbytes)": "max_resident_set_kb",
        "Average resident set size (kbytes)": "avg_resident_set_kb",
        "Major (requiring I/O) page faults": "major_page_faults",
        "Minor (reclaiming a frame) page faults": "minor_page_faults",
        "Voluntary context switches": "voluntary_context_switches",
        "Involuntary context switches": "involuntary_context_switches",
        "File system inputs": "file_system_inputs",
        "File system outputs": "file_system_outputs",
        "Socket messages sent": "socket_messages_sent",
        "Socket messages received": "socket_messages_received",
        "Signals delivered": "signals_delivered",
        "Page size (bytes)": "page_size_bytes",
        "Exit status": "exit_status",
    }
    parsed: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Elapsed (wall clock) time"):
            parsed["elapsed_wall_clock"] = line.split("):", 1)[-1].strip()
            continue
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = key_map.get(raw_key.strip())
        if not key:
            continue
        parsed[key] = _parse_metric_value(raw_value.strip())
    return parsed


def _parse_metric_value(value: str) -> Any:
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith("%"):
        return _parse_metric_value(cleaned[:-1])
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        return float(cleaned)
    return value


def _redact_text(value: str, workspace: Path) -> str:
    """Remove local machine paths from persisted command output."""
    if not value:
        return value

    redacted = value
    for candidate, replacement in [
        (str(workspace.resolve(strict=False)), "<workspace>"),
        (str(workspace), "<workspace>"),
        (str(Path.home()), "~"),
    ]:
        if candidate:
            redacted = redacted.replace(candidate, replacement)
    return redacted


def _portable_command_args(args: Sequence[str], workspace: Path) -> List[str]:
    """Return command args without embedding the absolute workspace path."""
    workspace_text = str(workspace)
    resolved_workspace_text = str(workspace.resolve(strict=False))
    return [
        "<workspace>" if item in {workspace_text, resolved_workspace_text} else item
        for item in args
    ]


def _init_baseline(workspace: Path) -> None:
    """Initialize a local git baseline when git is available."""
    if not shutil.which("git"):
        return

    _run(["git", "init", "-q"], workspace, timeout_seconds=10)
    _run(["git", "add", "."], workspace, timeout_seconds=10)
    _run(
        [
            "git",
            "-c",
            "user.name=smaLLMs Agent Harness",
            "-c",
            "user.email=smallms-agent-harness@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        workspace,
        timeout_seconds=10,
    )


def _changed_files(workspace: Path) -> List[str]:
    if not shutil.which("git") or not (workspace / ".git").exists():
        return []

    result = _run(["git", "status", "--short"], workspace, timeout_seconds=10)
    files: List[str] = []
    for line in result["stdout"].splitlines():
        if not line.strip():
            continue
        files.append(line[3:].strip())
    return files


def _diff_stat(workspace: Path) -> str:
    if not shutil.which("git") or not (workspace / ".git").exists():
        return ""
    return _run(["git", "diff", "--stat"], workspace, timeout_seconds=10)["stdout"]


def _diff_patch(workspace: Path) -> str:
    if not shutil.which("git") or not (workspace / ".git").exists():
        return ""
    return _run(["git", "diff", "--binary"], workspace, timeout_seconds=10)["stdout"]


def _text_metrics(value: str) -> Dict[str, int]:
    encoded = value.encode("utf-8", errors="replace")
    return {
        "bytes": len(encoded),
        "chars": len(value),
        "lines": len(value.splitlines()),
    }


def _file_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": portable_path(path), "bytes": 0, "sha256": None}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _extract_reported_usage(harness_key: str, stdout: str, stderr: str) -> Dict[str, Any]:
    usage = {
        "source": "not_reported",
        "total_tokens": None,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "cached_input_tokens": None,
        "cost_usd": None,
        "raw": None,
    }
    combined = "\n".join(part for part in [stdout, stderr] if part)
    if harness_key == "codex":
        match = re.search(r"(?im)^tokens used\s*\n\s*([0-9][0-9,]*)\s*$", combined)
        if match:
            usage.update(
                {
                    "source": "codex_cli_log",
                    "total_tokens": int(match.group(1).replace(",", "")),
                    "raw": f"tokens used\n{match.group(1)}",
                }
            )
        else:
            usage["source"] = "codex_cli_log_missing"
    return usage


def _context_metrics() -> Dict[str, Any]:
    return {
        "source": "not_reported",
        "context_window_tokens": None,
        "context_used_tokens": None,
        "remaining_context_tokens": None,
    }


def _quality_signals(task: HarnessTask, changed_files: Sequence[str], agent_result: Dict[str, Any], test_result: Dict[str, Any]) -> Dict[str, Any]:
    expected = set(task.expected_files)
    changed = set(changed_files)
    return {
        "expected_files": list(task.expected_files),
        "expected_files_modified": sorted(expected & changed),
        "missing_expected_files": sorted(expected - changed),
        "unexpected_changed_files": sorted(changed - expected),
        "changed_file_count": len(changed_files),
        "agent_exited_cleanly": agent_result["returncode"] == 0,
        "tests_passed": test_result["returncode"] == 0,
        "agent_timed_out": bool(agent_result["timed_out"]),
        "test_timed_out": bool(test_result["timed_out"]),
    }


def _system_metadata() -> Dict[str, Any]:
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }


def _format_optional_number(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:,}"


def _format_seconds(value: float) -> str:
    return f"{value:.1f}s"


def _format_mb_from_kb(value: float) -> str:
    return f"{value / 1024:.1f} MB"


def _agent_harness_findings(harnesses: Sequence[Dict[str, Any]], totals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build concise public-facing findings from aggregate harness metrics."""
    completed = [harness for harness in harnesses if harness.get("completed")]
    if not completed:
        return []

    findings: List[Dict[str, Any]] = []
    all_passed = all(harness.get("failed") == 0 and harness.get("passed") == harness.get("rows") for harness in completed)
    no_unexpected_changes = all(harness.get("unexpected_change_rows") == 0 for harness in completed)
    if all_passed and no_unexpected_changes:
        findings.append(
            {
                "kind": "quality",
                "title": "Quality tied on this run",
                "summary": (
                    f"All {len(completed)} harnesses passed their rows and changed only expected files. "
                    "The quality result is therefore a tie on this fixture set."
                ),
            }
        )
    else:
        best_quality = max(completed, key=lambda harness: (harness.get("pass_rate") or 0.0, -(harness.get("unexpected_change_rows") or 0)))
        findings.append(
            {
                "kind": "quality",
                "title": f"Best quality: {best_quality.get('display_name', best_quality['key'])}",
                "summary": (
                    f"{best_quality.get('display_name', best_quality['key'])} had the strongest pass-rate and file-scope result "
                    "on this fixture set."
                ),
            }
        )

    timed = [harness for harness in completed if isinstance(harness.get("avg_duration_seconds"), (int, float))]
    if timed:
        fastest = min(timed, key=lambda harness: float(harness["avg_duration_seconds"]))
        slowest = max(timed, key=lambda harness: float(harness["avg_duration_seconds"]))
        delta = float(slowest["avg_duration_seconds"]) - float(fastest["avg_duration_seconds"])
        findings.append(
            {
                "kind": "speed",
                "title": f"Fastest average: {fastest.get('display_name', fastest['key'])}",
                "summary": (
                    f"{fastest.get('display_name', fastest['key'])} averaged {_format_seconds(float(fastest['avg_duration_seconds']))} per row. "
                    f"The slowest average was {slowest.get('display_name', slowest['key'])} at "
                    f"{_format_seconds(float(slowest['avg_duration_seconds']))}, a {_format_seconds(delta)} gap."
                ),
            }
        )

    memory_rows = [harness for harness in completed if isinstance(harness.get("max_agent_rss_kb"), (int, float))]
    if memory_rows:
        lowest = min(memory_rows, key=lambda harness: float(harness["max_agent_rss_kb"]))
        highest = max(memory_rows, key=lambda harness: float(harness["max_agent_rss_kb"]))
        findings.append(
            {
                "kind": "memory",
                "title": f"Lowest local peak RSS: {lowest.get('display_name', lowest['key'])}",
                "summary": (
                    f"{lowest.get('display_name', lowest['key'])} peaked at {_format_mb_from_kb(float(lowest['max_agent_rss_kb']))}. "
                    f"The highest local peak RSS was {highest.get('display_name', highest['key'])} at "
                    f"{_format_mb_from_kb(float(highest['max_agent_rss_kb']))}."
                ),
            }
        )

    token_reporting = [
        harness
        for harness in completed
        if isinstance(harness.get("reported_token_rows"), int) and harness.get("reported_token_rows", 0) > 0
    ]
    if token_reporting:
        names = ", ".join(str(harness.get("display_name", harness["key"])) for harness in token_reporting)
        findings.append(
            {
                "kind": "tokens",
                "title": "Token reporting is partial",
                "summary": (
                    f"Only {names} reported per-run token usage in captured CLI output. "
                    f"The run recorded {totals.get('reported_total_tokens'):,} reported tokens across "
                    f"{totals.get('reported_token_rows')} rows."
                ),
            }
        )
    else:
        findings.append(
            {
                "kind": "tokens",
                "title": "Token reporting unavailable",
                "summary": "No selected harness reported comparable token usage in captured CLI output.",
            }
        )

    return findings


def _write_result_files(run_dir: Path, base_dir: Path, summary: Dict[str, Any], web_summary: Dict[str, Any]) -> None:
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    web_path = run_dir / "web_summary.json"
    web_path.write_text(json.dumps(web_summary, indent=2), encoding="utf-8")
    latest_web_path = base_dir / "latest-web-summary.json"
    latest_web_path.write_text(json.dumps(web_summary, indent=2), encoding="utf-8")

    rows = summary["results"]
    lines = [
        "# Agent Harness Eval",
        "",
        f"Run ID: `{summary['run_id']}`",
        "",
        "| Harness | Task | Status | Test | Seconds | Tokens | Peak RSS MB | Changed files |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        changed = ", ".join(row["changed_files"]) or "-"
        max_rss_kb = (
            row.get("resource_usage", {})
            .get("agent", {})
            .get("metrics", {})
            .get("max_resident_set_kb")
        )
        max_rss_mb = round(max_rss_kb / 1024, 1) if isinstance(max_rss_kb, (int, float)) else None
        lines.append(
            f"| {row['harness']} | {row['task']} | {row['status']} | "
            f"{row['test_status']} | {row['duration_seconds']:.3f} | "
            f"{_format_optional_number(row.get('usage', {}).get('total_tokens'))} | "
            f"{_format_optional_number(max_rss_mb)} | {changed} |"
        )
    lines.extend(
        [""] if not any(row["status"] == "dry_run" for row in rows)
        else [
            "",
            "Dry-run rows validate fixture setup and command construction without calling model providers.",
        ]
    )
    findings = web_summary.get("findings", [])
    if findings:
        lines.extend(["", "Findings:", ""])
        for finding in findings:
            lines.append(f"- {finding['title']}: {finding['summary']}")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Token and context fields are reported only when a harness prints them in captured CLI output.",
            "- Peak RSS uses GNU `time -v` when available and is best-effort process memory for the invoked harness command.",
            f"- Web summary: `{portable_path(web_path)}`",
        ]
    )
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_results_for_web(summary: Dict[str, Any]) -> Dict[str, Any]:
    results = summary["results"]
    harnesses = []
    for harness_key in summary["harnesses"]:
        rows = [row for row in results if row["harness"] == harness_key]
        completed = [row for row in rows if row["status"] in {"passed", "failed"}]
        token_values = [
            row.get("usage", {}).get("total_tokens")
            for row in rows
            if isinstance(row.get("usage", {}).get("total_tokens"), int)
        ]
        rss_values = [
            row.get("resource_usage", {}).get("agent", {}).get("metrics", {}).get("max_resident_set_kb")
            for row in rows
        ]
        numeric_rss = [value for value in rss_values if isinstance(value, (int, float))]
        harness_meta = next(
            (entry for entry in summary["available_harnesses"] if entry["key"] == harness_key),
            {},
        )
        duration_total = round(sum(float(row.get("duration_seconds") or 0.0) for row in completed), 3)
        harnesses.append(
            {
                "key": harness_key,
                "display_name": harness_meta.get("display_name", harness_key),
                "version": harness_meta.get("version"),
                "model": harness_meta.get("model"),
                "reasoning": harness_meta.get("reasoning"),
                "rows": len(rows),
                "completed": len(completed),
                "passed": sum(1 for row in rows if row["status"] == "passed"),
                "failed": sum(1 for row in rows if row["status"] == "failed"),
                "pass_rate": (sum(1 for row in completed if row["status"] == "passed") / len(completed)) if completed else None,
                "duration_seconds_total": duration_total,
                "avg_duration_seconds": round(duration_total / len(completed), 3) if completed else None,
                "reported_total_tokens": sum(token_values) if token_values else None,
                "reported_token_rows": len(token_values),
                "max_agent_rss_kb": max(numeric_rss) if numeric_rss else None,
                "unexpected_change_rows": sum(
                    1
                    for row in rows
                    if row.get("quality_signals", {}).get("unexpected_changed_files")
                ),
            }
        )

    compact_rows = []
    for row in results:
        agent_resource = row.get("resource_usage", {}).get("agent", {})
        compact_rows.append(
            {
                "harness": row["harness"],
                "harness_display_name": row["harness_display_name"],
                "task": row["task"],
                "task_display_name": row["task_display_name"],
                "status": row["status"],
                "test_status": row["test_status"],
                "model": row["model"],
                "reasoning": row["reasoning"],
                "duration_seconds": row["duration_seconds"],
                "started_at": row.get("started_at"),
                "ended_at": row.get("ended_at"),
                "agent_returncode": row.get("agent_returncode"),
                "agent_timed_out": row.get("agent_timed_out"),
                "test_returncode": row.get("test_returncode"),
                "changed_files": row.get("changed_files", []),
                "changed_file_count": row.get("quality_signals", {}).get("changed_file_count", len(row.get("changed_files", []))),
                "diff_stat": row.get("diff_stat", ""),
                "usage": row.get("usage", {"source": "not_reported"}),
                "context": row.get("context", _context_metrics()),
                "resource_usage": {
                    "agent": {
                        "source": agent_resource.get("source", "not_collected"),
                        "metrics": agent_resource.get("metrics", {}),
                    }
                },
                "output_metrics": row.get("output_metrics", {}),
                "quality_signals": row.get("quality_signals", {}),
                "artifact_paths": {
                    "log_dir": row.get("log_dir"),
                    "patch": row.get("patch", {}).get("path"),
                },
            }
        )

    totals = dict(summary["totals"])
    totals["duration_seconds_total"] = round(
        sum(float(row.get("duration_seconds") or 0.0) for row in results if row["status"] in {"passed", "failed"}),
        3,
    )
    reported_tokens = [
        row.get("usage", {}).get("total_tokens")
        for row in results
        if isinstance(row.get("usage", {}).get("total_tokens"), int)
    ]
    totals["reported_total_tokens"] = sum(reported_tokens) if reported_tokens else None
    totals["reported_token_rows"] = len(reported_tokens)
    findings = _agent_harness_findings(harnesses, totals)

    return {
        "schema_version": AGENT_HARNESS_WEB_SCHEMA_VERSION,
        "exported_at": utcnow_iso(),
        "run_id": summary["run_id"],
        "created_at": summary["created_at"],
        "mode": summary["mode"],
        "timeout_seconds": summary["timeout_seconds"],
        "source": {
            "summary_path": summary.get("summary_path"),
            "run_dir": summary.get("run_dir"),
            "payload_profile": "compact_agent_harness_public",
        },
        "system": summary.get("system", {}),
        "totals": totals,
        "harnesses": harnesses,
        "findings": findings,
        "tasks": summary["available_tasks"],
        "results": compact_rows,
        "limitations": [
            "Token, cost, and context metrics are nullable because Pi and OpenCode did not report comparable values in captured CLI output.",
            "Codex CLI reported a single total token count; prompt/completion/reasoning token splits were not available in the log.",
            "Peak RSS is measured with GNU time -v when available and should be treated as subprocess memory telemetry, not full provider-side memory.",
        ],
    }


def _agent_harness_web_sync_paths(sync_dir: str | Path, run_id: str) -> Dict[str, str]:
    root = Path(sync_dir) / "agent-harness"
    latest_path = root / "latest.json"
    run_path = root / "runs" / f"{safe_slug(run_id)}.json"
    meta_path = root / "latest.meta.json"
    pointer_path = root / "latest-run.txt"
    return {
        "sync/agent-harness/latest.json": portable_path(latest_path),
        "sync/agent-harness/latest.meta.json": portable_path(meta_path),
        "sync/agent-harness/latest-run.txt": portable_path(pointer_path),
        f"sync/agent-harness/runs/{safe_slug(run_id)}.json": portable_path(run_path),
    }


def _sync_agent_harness_web_summary(web_summary: Dict[str, Any], sync_dir: str | Path) -> Dict[str, str]:
    root = Path(sync_dir) / "agent-harness"
    run_id = str(web_summary["run_id"])
    latest_path = root / "latest.json"
    run_path = root / "runs" / f"{safe_slug(run_id)}.json"
    meta_path = root / "latest.meta.json"
    pointer_path = root / "latest-run.txt"
    root.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)

    latest_path.write_text(json.dumps(web_summary, indent=2), encoding="utf-8")
    run_path.write_text(json.dumps(web_summary, indent=2), encoding="utf-8")
    pointer_path.write_text(run_id + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": AGENT_HARNESS_WEB_SCHEMA_VERSION,
                "run_id": run_id,
                "exported_at": web_summary["exported_at"],
                "latest_path": "agent-harness/latest.json",
                "run_path": f"agent-harness/runs/{safe_slug(run_id)}.json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return _agent_harness_web_sync_paths(sync_dir, run_id)


def run_agent_harness_eval(
    harnesses: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    artifacts_dir: str | Path = DEFAULT_AGENT_HARNESS_DIR,
    timeout_seconds: int = 900,
    dry_run: bool = False,
    sync_dir: Optional[str | Path] = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """Run or dry-run the local agent-harness comparison."""
    selected_harnesses = _select_keys(harnesses, DEFAULT_HARNESSES, "harness")
    selected_tasks = _select_keys(tasks, DEFAULT_TASKS, "task")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    created_at = utcnow_iso()
    run_stamp = created_at.replace("+00:00", "Z").replace(":", "").replace(".", "-")
    run_id = "agent-harness-" + run_stamp
    base_dir = Path(artifacts_dir)
    run_dir = base_dir / "runs" / safe_slug(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    available_harnesses = list_agent_harnesses()
    results: List[Dict[str, Any]] = []
    for harness_key in selected_harnesses:
        harness = DEFAULT_HARNESSES[harness_key]
        harness_path = shutil.which(harness.command)
        if not harness_path and not dry_run:
            raise FileNotFoundError(f"{harness.command} not found on PATH")

        for task_key in selected_tasks:
            task = DEFAULT_TASKS[task_key]
            workspace = run_dir / "workspaces" / harness.key / task.key
            if progress_callback:
                progress_callback(
                    {
                        "event": "agent_harness_row_started",
                        "harness": harness.key,
                        "task": task.key,
                        "dry_run": dry_run,
                    }
                )
            _write_fixture(workspace)
            _init_baseline(workspace)

            command_args = harness.command_args(workspace, task.prompt)
            harness_meta = next(
                (entry for entry in available_harnesses if entry["key"] == harness.key),
                {},
            )
            row: Dict[str, Any] = {
                "harness": harness.key,
                "harness_display_name": harness.display_name,
                "harness_version": harness_meta.get("version"),
                "task": task.key,
                "task_display_name": task.display_name,
                "model": harness.model,
                "reasoning": harness.reasoning,
                "command": _portable_command_args(command_args, workspace),
                "workspace": portable_path(workspace),
                "status": "dry_run" if dry_run else "pending",
                "test_status": "not_run",
                "duration_seconds": 0.0,
                "changed_files": [],
                "diff_stat": "",
                "usage": _extract_reported_usage(harness.key, "", ""),
                "context": _context_metrics(),
            }

            if dry_run:
                setup_result = _run(list(task.test_command), workspace, timeout_seconds=timeout_seconds)
                row["baseline_test_returncode"] = setup_result["returncode"]
                row["baseline_test_stdout"] = _redact_text(setup_result["stdout"], workspace)
                row["baseline_test_stderr"] = _redact_text(setup_result["stderr"], workspace)
                row["baseline_test_started_at"] = setup_result["started_at"]
                row["baseline_test_ended_at"] = setup_result["ended_at"]
                row["resource_usage"] = {"baseline_test": setup_result["resource_metrics"]}
                results.append(row)
                if progress_callback:
                    progress_callback(
                        {
                            "event": "agent_harness_row_completed",
                            "harness": harness.key,
                            "task": task.key,
                            "status": row["status"],
                            "test_status": row["test_status"],
                            "duration_seconds": row["duration_seconds"],
                        }
                )
                continue

            cwd = workspace if harness.key == "pi" else Path.cwd()
            log_dir = run_dir / "logs" / harness.key / task.key
            log_dir.mkdir(parents=True, exist_ok=True)
            agent_result = _run(
                command_args,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                resource_log_path=log_dir / "agent.time.txt",
            )
            test_result = _run(
                list(task.test_command),
                workspace,
                timeout_seconds=timeout_seconds,
                resource_log_path=log_dir / "test.time.txt",
            )
            agent_stdout = _redact_text(agent_result["stdout"], workspace)
            agent_stderr = _redact_text(agent_result["stderr"], workspace)
            test_stdout = _redact_text(test_result["stdout"], workspace)
            test_stderr = _redact_text(test_result["stderr"], workspace)
            log_paths = {
                "agent_stdout": log_dir / "agent.stdout.txt",
                "agent_stderr": log_dir / "agent.stderr.txt",
                "test_stdout": log_dir / "test.stdout.txt",
                "test_stderr": log_dir / "test.stderr.txt",
            }
            log_paths["agent_stdout"].write_text(agent_stdout, encoding="utf-8")
            log_paths["agent_stderr"].write_text(agent_stderr, encoding="utf-8")
            log_paths["test_stdout"].write_text(test_stdout, encoding="utf-8")
            log_paths["test_stderr"].write_text(test_stderr, encoding="utf-8")

            changed_files = _changed_files(workspace)
            diff_patch = _redact_text(_diff_patch(workspace), workspace)
            patch_path = log_dir / "workspace.patch"
            patch_path.write_text(diff_patch, encoding="utf-8")
            test_passed = test_result["returncode"] == 0
            row.update(
                {
                    "status": "passed" if test_passed and agent_result["returncode"] == 0 else "failed",
                    "agent_returncode": agent_result["returncode"],
                    "agent_timed_out": agent_result["timed_out"],
                    "started_at": agent_result["started_at"],
                    "ended_at": agent_result["ended_at"],
                    "test_returncode": test_result["returncode"],
                    "test_timed_out": test_result["timed_out"],
                    "test_started_at": test_result["started_at"],
                    "test_ended_at": test_result["ended_at"],
                    "test_status": "passed" if test_passed else "failed",
                    "duration_seconds": agent_result["duration_seconds"],
                    "changed_files": changed_files,
                    "diff_stat": _diff_stat(workspace),
                    "log_dir": portable_path(log_dir),
                    "log_files": {name: portable_path(path) for name, path in log_paths.items()},
                    "log_file_metrics": {name: _file_metrics(path) for name, path in log_paths.items()},
                    "output_metrics": {
                        "agent_stdout": _text_metrics(agent_stdout),
                        "agent_stderr": _text_metrics(agent_stderr),
                        "test_stdout": _text_metrics(test_stdout),
                        "test_stderr": _text_metrics(test_stderr),
                    },
                    "patch": _file_metrics(patch_path),
                    "usage": _extract_reported_usage(harness.key, agent_stdout, agent_stderr),
                    "context": _context_metrics(),
                    "resource_usage": {
                        "agent": agent_result["resource_metrics"],
                        "test": test_result["resource_metrics"],
                    },
                    "quality_signals": _quality_signals(task, changed_files, agent_result, test_result),
                }
            )
            results.append(row)
            if progress_callback:
                progress_callback(
                    {
                        "event": "agent_harness_row_completed",
                        "harness": harness.key,
                        "task": task.key,
                        "status": row["status"],
                        "test_status": row["test_status"],
                        "duration_seconds": row["duration_seconds"],
                    }
                )

    completed = [row for row in results if row["status"] in {"passed", "failed"}]
    summary = {
        "run_id": safe_slug(run_id),
        "created_at": created_at,
        "mode": "dry_run" if dry_run else "agent_harness_eval",
        "run_dir": portable_path(run_dir),
        "summary_path": portable_path(run_dir / "summary.json"),
        "web_summary_path": portable_path(run_dir / "web_summary.json"),
        "harnesses": selected_harnesses,
        "tasks": selected_tasks,
        "timeout_seconds": timeout_seconds,
        "totals": {
            "rows": len(results),
            "completed": len(completed),
            "passed": sum(1 for row in results if row["status"] == "passed"),
            "failed": sum(1 for row in results if row["status"] == "failed"),
            "dry_run": sum(1 for row in results if row["status"] == "dry_run"),
            "duration_seconds_total": round(sum(float(row.get("duration_seconds") or 0.0) for row in completed), 3),
        },
        "available_harnesses": available_harnesses,
        "available_tasks": list_agent_harness_tasks(),
        "system": _system_metadata(),
        "web_sync_dir": portable_path(Path(sync_dir)) if sync_dir else None,
        "results": results,
    }
    web_summary = _summarize_results_for_web(summary)
    if sync_dir:
        sync_paths = _agent_harness_web_sync_paths(sync_dir, summary["run_id"])
        summary["web_sync"] = sync_paths
        web_summary["source"]["synced"] = True
        web_summary["source"]["latest_path"] = "agent-harness/latest.json"
        web_summary["source"]["history_path"] = f"agent-harness/runs/{safe_slug(summary['run_id'])}.json"
        _sync_agent_harness_web_summary(web_summary, sync_dir)
    _write_result_files(run_dir, base_dir, summary, web_summary)
    return summary
