#!/usr/bin/env python3
"""
smaLLMs - CLI-first local benchmarking for local model runtimes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.append(str(Path(__file__).parent))

from src.pipeline.benchmarks import (
    DEFAULT_BENCHMARKS,
    dataset_runtime_info,
    expand_benchmark_selection,
    list_benchmark_suites,
    list_supported_benchmarks,
    warm_benchmark_cache,
)
from src.pipeline.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOCAL_PROVIDER,
    DEFAULT_LOCAL_SAMPLE_COUNT,
    DEFAULT_WEBSITE_EXPORT_DIR,
    load_pipeline_config,
    local_benchmark_settings,
)
from src.cli.setup_checks import build_setup_report_lines, collect_setup_report


def _parse_list(value: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize repeated or comma-separated CLI values."""
    if not value:
        return None

    items: List[str] = []
    for chunk in value:
        items.extend(part.strip() for part in chunk.split(",") if part.strip())
    return items or None


@dataclass
class ProgressRow:
    """One row in the live terminal table."""

    target: str
    tests_done: int
    total_tests: int
    correct: int = 0
    errors: int = 0
    avg_latency: float = 0.0
    slowest: float = 0.0
    avg_tps: float = 0.0
    running: bool = False
    failed: bool = False

    @property
    def accuracy(self) -> float:
        return (self.correct / self.tests_done) if self.tests_done else 0.0


class BeautifulSmaLLMsTerminal:
    """Beautiful, CLI-first terminal UI for local benchmark runs."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GRAY = "\033[90m"

    def __init__(self) -> None:
        self.is_tty = sys.stdout.isatty()
        self.rows: Dict[str, ProgressRow] = {}
        self.model_order: List[str] = []
        self.benchmark_order: List[str] = []
        self.run_id: str = ""
        self.samples_per_benchmark: int = 0
        self.render_started = False
        self.start_time = time.time()

    def clear_screen(self) -> None:
        """Clear the screen only for interactive terminals."""
        if self.is_tty:
            print("\033[2J\033[H", end="")

    def format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds <= 0:
            return "-"
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)}m{remainder:.0f}s"

    def _row_key(self, model_name: str, benchmark_name: str) -> str:
        return f"{model_name}::{benchmark_name}"

    def start_run(self, run_id: str, models: Sequence[str], benchmarks: Sequence[str], samples: int) -> None:
        """Initialize terminal state for a run."""
        self.run_id = run_id
        self.samples_per_benchmark = samples
        self.model_order = list(models)
        self.benchmark_order = list(benchmarks)
        self.rows = {}
        self.start_time = time.time()
        self.render_started = True

        for model_name in models:
            for benchmark_name in benchmarks:
                target = f"{model_name} / {benchmark_name}"
                self.rows[self._row_key(model_name, benchmark_name)] = ProgressRow(
                    target=target,
                    tests_done=0,
                    total_tests=samples,
                )

        if self.is_tty:
            self.render()
        else:
            print(f"Run ID: {run_id}")
            print(f"Models: {', '.join(models)}")
            print(f"Benchmarks: {', '.join(benchmarks)}")
            print(f"Samples per benchmark: {samples}")

    def handle_event(self, event: Dict[str, Any]) -> None:
        """Apply one progress event from the orchestrator."""
        event_type = event.get("event")
        model_name = event.get("model_name")
        benchmark_name = event.get("benchmark_name")
        row = self.rows.get(self._row_key(model_name, benchmark_name)) if model_name and benchmark_name else None

        if event_type == "benchmark_started" and row:
            row.running = True
        elif event_type == "sample_completed" and row:
            row.tests_done = int(event.get("sample_index", row.tests_done))
            row.correct = int(event.get("correct_count", row.correct))
            row.errors = int(event.get("error_count", row.errors))
            row.running = True
            sample = event.get("sample", {})
            latency = float(sample.get("latency_sec") or 0.0)
            row.avg_latency = (
                ((row.avg_latency * max(row.tests_done - 1, 0)) + latency) / row.tests_done
                if row.tests_done
                else 0.0
            )
            row.slowest = max(row.slowest, latency)
            tps = float(sample.get("tokens_per_second") or 0.0)
            if tps > 0:
                row.avg_tps = (((row.avg_tps * max(row.tests_done - 1, 0)) + tps) / row.tests_done)
        elif event_type == "benchmark_completed" and row:
            metrics = event.get("benchmark_result", {}).get("metrics", {})
            row.tests_done = int(metrics.get("sample_count", row.tests_done))
            row.correct = int(metrics.get("correct_count", row.correct))
            row.errors = int(metrics.get("error_count", row.errors))
            row.avg_latency = float(metrics.get("avg_latency_sec", row.avg_latency))
            row.slowest = float(metrics.get("max_latency_sec", row.slowest))
            row.avg_tps = float(metrics.get("avg_tokens_per_second", row.avg_tps))
            row.running = False
        elif event_type == "benchmark_failed" and row:
            row.running = False
            row.failed = True
            row.errors += 1

        if self.is_tty and self.render_started:
            self.render()
        elif not self.is_tty:
            self._print_line_mode_event(event)

    def _print_line_mode_event(self, event: Dict[str, Any]) -> None:
        """Print concise progress updates when stdout is not a TTY."""
        event_type = event.get("event")
        model_name = event.get("model_name")
        benchmark_name = event.get("benchmark_name")

        if event_type == "benchmark_started":
            print(f"Starting {model_name} on {benchmark_name}...")
        elif event_type == "benchmark_completed":
            metrics = event.get("benchmark_result", {}).get("metrics", {})
            print(
                f"Completed {model_name} on {benchmark_name}: "
                f"{metrics.get('accuracy', 0.0):.4f} accuracy over {metrics.get('sample_count', 0)} samples"
            )
        elif event_type == "benchmark_failed":
            print(f"Failed {model_name} on {benchmark_name}: {event.get('error', 'unknown error')}")

    def render(self) -> None:
        """Render the live progress table."""
        self.clear_screen()
        elapsed = time.time() - self.start_time
        completed = sum(1 for row in self.rows.values() if not row.running and row.tests_done == row.total_tests)
        total = len(self.rows)
        total_samples_done = sum(row.tests_done for row in self.rows.values())
        total_samples = sum(row.total_tests for row in self.rows.values())
        total_correct = sum(row.correct for row in self.rows.values())
        accuracy = (total_correct / total_samples_done) if total_samples_done else 0.0

        print(
            f"\n{self.CYAN}->{self.RESET}  {self.BOLD}smaLLMs{self.RESET} "
            f"{self.GRAY}local model benchmarking{self.RESET} "
            f"{self.RED}x{self.RESET} {self.WHITE}CLI-first leaderboard runner{self.RESET}"
        )
        print(f"{self.GRAY}Run {self.run_id} | elapsed {self.format_duration(elapsed)}{self.RESET}\n")
        print(
            f"{self.BOLD}{'Target':<38} {'Done':<9} {'% Right':<9} {'Errors':<8} "
            f"{'Avg Lat':<10} {'Slowest':<10} {'Tok/s':<10} {'State':<10}{self.RESET}"
        )
        print(f"{self.GRAY}{'-' * 108}{self.RESET}")

        for model_name in self.model_order:
            for benchmark_name in self.benchmark_order:
                row = self.rows[self._row_key(model_name, benchmark_name)]
                target = row.target[:37]
                accuracy_pct = row.accuracy * 100
                accuracy_color = self.GREEN if accuracy_pct >= 70 else self.YELLOW if accuracy_pct > 0 else self.GRAY
                state = "running" if row.running else "failed" if row.failed else "done" if row.tests_done else "-"
                state_color = self.CYAN if row.running else self.RED if row.failed else self.GRAY
                print(
                    f"{target:<38} {row.tests_done}/{row.total_tests:<6} "
                    f"{accuracy_color}{accuracy_pct:>7.0f}%{self.RESET} "
                    f"{row.errors:<8} {self.format_duration(row.avg_latency):<10} "
                    f"{self.format_duration(row.slowest):<10} "
                    f"{row.avg_tps if row.avg_tps else 0.0:<10.2f} "
                    f"{state_color}{state:<10}{self.RESET}"
                )

        progress = (completed / total) if total else 0.0
        filled = int(40 * progress)
        bar = f"{self.GREEN}{'█' * filled}{self.GRAY}{'░' * (40 - filled)}{self.RESET}"
        print(f"\n{bar} {progress * 100:.0f}% ({completed}/{total} completed)")
        print(
            f"{self.BOLD}Overall:{self.RESET} {total_samples_done}/{total_samples} samples • "
            f"{accuracy * 100:.1f}% correct • {sum(row.errors for row in self.rows.values())} errors"
        )
        sys.stdout.flush()


class SmaLLMsCLI:
    """Main local CLI application."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.local_settings = local_benchmark_settings(self.config, DEFAULT_BENCHMARKS)
        self.terminal = BeautifulSmaLLMsTerminal()
        self.orchestrator = None

        from src.pipeline.artifacts import ArtifactStore
        from src.pipeline.exporter import WebsiteExporter

        self.artifact_store = ArtifactStore(self.local_settings.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR))
        self.exporter = WebsiteExporter(
            artifacts_dir=self.local_settings.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR),
            output_dir=self.local_settings.get("website_export_dir", DEFAULT_WEBSITE_EXPORT_DIR),
            sync_dir=self.local_settings.get("website_sync_dir"),
        )

        Path(self.local_settings.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR)).mkdir(parents=True, exist_ok=True)
        Path(self.local_settings.get("website_export_dir", DEFAULT_WEBSITE_EXPORT_DIR)).mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config from YAML with shared pipeline defaults."""
        return load_pipeline_config(config_path, DEFAULT_BENCHMARKS)

    def _get_orchestrator(self) -> Any:
        """Initialize the heavy runner only when needed."""
        if self.orchestrator is None:
            from src.pipeline.orchestrator import LocalBenchmarkOrchestrator

            self.orchestrator = LocalBenchmarkOrchestrator(config_path=self.config_path)
        return self.orchestrator

    def print_welcome(self) -> None:
        """Print the main interactive welcome screen."""
        self.terminal.clear_screen()
        default_benchmarks = ", ".join(self.local_settings.get("default_benchmarks", DEFAULT_BENCHMARKS))
        print(
            f"""
{self.terminal.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                          {self.terminal.BOLD}smaLLMs Local CLI{self.terminal.RESET}{self.terminal.CYAN}                                   ║
║                  {self.terminal.GRAY}Cross-platform local LLM benchmark runner{self.terminal.RESET}{self.terminal.CYAN}                   ║
╚══════════════════════════════════════════════════════════════════════════════╝{self.terminal.RESET}

{self.terminal.BOLD}Commands:{self.terminal.RESET}
  {self.terminal.GREEN}doctor{self.terminal.RESET}     - Check Ollama / LM Studio status and next steps
  {self.terminal.GREEN}discover{self.terminal.RESET}   - Find local models across supported providers
  {self.terminal.GREEN}benchmarks{self.terminal.RESET} - Show supported benchmark suites and benchmarks
  {self.terminal.GREEN}run{self.terminal.RESET}        - Start a benchmark run
  {self.terminal.GREEN}quick{self.terminal.RESET}      - Run the default core suite on all discovered models
  {self.terminal.GREEN}status{self.terminal.RESET}     - Show the latest run summary
  {self.terminal.GREEN}export{self.terminal.RESET}     - Export the latest run for the website
  {self.terminal.GREEN}clear{self.terminal.RESET}      - Clear the screen
  {self.terminal.GREEN}exit{self.terminal.RESET}       - Quit

{self.terminal.BOLD}Interactive Mode:{self.terminal.RESET} arrow keys, space to toggle, enter to confirm
{self.terminal.BOLD}Local Models:{self.terminal.RESET} if Ollama already has models installed, smaLLMs will reuse them automatically
{self.terminal.BOLD}Default Core Suite:{self.terminal.RESET} {default_benchmarks}
"""
        )

    async def discover_models(self, json_output: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Discover local models across supported providers."""
        orchestrator = self._get_orchestrator()
        discovered = await orchestrator.discover_local_models()

        if json_output:
            print(json.dumps(discovered, indent=2))
            return discovered

        print(f"\n{self.terminal.BOLD}Local Model Discovery{self.terminal.RESET}")
        total_models = 0
        for provider in ["ollama", "lm_studio"]:
            models = discovered.get(provider, [])
            total_models += len(models)
            print(f"\n{self.terminal.CYAN}{provider}{self.terminal.RESET} ({len(models)})")
            if not models:
                print(f"  {self.terminal.GRAY}No models found.{self.terminal.RESET}")
                continue

            for index, model in enumerate(models, start=1):
                details = [
                    f"{float(model.get('size_gb') or 0.0):.1f}GB" if model.get("size_gb") else None,
                    model.get("parameters"),
                    model.get("family"),
                    model.get("quantization"),
                ]
                details = [item for item in details if item and item != "unknown"]
                detail_text = " | ".join(details) if details else "metadata unavailable"
                print(f"  {index:>2}. {model['name']}  {self.terminal.GRAY}{detail_text}{self.terminal.RESET}")

        if total_models == 0:
            print(f"\n{self.terminal.YELLOW}No reachable local models found.{self.terminal.RESET}")
        return discovered

    def get_setup_report(self) -> Dict[str, Any]:
        """Collect a lightweight setup report for local runtimes."""
        return collect_setup_report().to_dict()

    def build_setup_status_lines(self) -> List[str]:
        """Build user-facing setup guidance lines."""
        return build_setup_report_lines(collect_setup_report())

    def show_setup_status(self, json_output: bool = False) -> None:
        """Show local runtime readiness and next steps."""
        self._get_orchestrator()
        report = collect_setup_report()
        dataset_runtime = dataset_runtime_info()
        if json_output:
            payload = report.to_dict()
            payload["dataset_runtime"] = dataset_runtime
            print(json.dumps(payload, indent=2))
            return

        print(f"\n{self.terminal.BOLD}Setup And Model Status{self.terminal.RESET}")
        for line in build_setup_report_lines(report):
            print(line)
        if dataset_runtime["allow_remote_dataset_downloads"]:
            print(f"Benchmark data cache: {dataset_runtime['cache_dir']} (filled automatically on first use)")
        else:
            print(f"Benchmark data cache: {dataset_runtime['cache_dir']} (offline-only mode)")

    def show_benchmarks(self, json_output: bool = False) -> None:
        """Print benchmark and suite metadata."""
        runnable = list_supported_benchmarks()
        suites = list_benchmark_suites()

        if json_output:
            print(
                json.dumps(
                    {
                        "suites": suites,
                        "benchmarks": runnable,
                    },
                    indent=2,
                )
            )
            return

        print(f"\n{self.terminal.BOLD}Supported Benchmark Suites{self.terminal.RESET}")
        for suite in suites:
            print(
                f"  {self.terminal.CYAN}{suite['key']}{self.terminal.RESET} - "
                f"{suite['display_name']}: {suite['description']}"
            )
            print(f"      {', '.join(suite['benchmarks'])}")

        print(f"\n{self.terminal.BOLD}Supported Benchmarks{self.terminal.RESET}")
        for benchmark in runnable:
            print(
                f"  {self.terminal.GREEN}{benchmark['key']}{self.terminal.RESET} "
                f"[{benchmark['category']}] - {benchmark['display_name']}: {benchmark['description']}"
            )

    def show_status(self, json_output: bool = False) -> None:
        """Show the latest run summary."""
        latest_run_id = self.artifact_store.latest_run_id()
        if not latest_run_id:
            print(f"{self.terminal.YELLOW}No runs found yet.{self.terminal.RESET}")
            return

        run_data = self.artifact_store.load_run(latest_run_id)
        summary = run_data.get("summary", {})
        totals = summary.get("totals", {})
        leaderboard = summary.get("leaderboard", [])

        if json_output:
            print(json.dumps(run_data, indent=2))
            return

        print(f"\n{self.terminal.BOLD}Latest Run{self.terminal.RESET} {latest_run_id}")
        print(f"  Models: {totals.get('models', 0)}")
        print(f"  Benchmarks: {totals.get('benchmarks', 0)}")
        print(f"  Evaluations: {totals.get('evaluations', 0)}")
        print(f"  Samples: {totals.get('samples', 0)}")
        print(f"  Accuracy: {totals.get('accuracy', 0.0):.4f}")
        print(f"  Total tokens: {totals.get('total_tokens', 0)}")
        print(f"  Artifacts: {run_data.get('run_dir')}")

        if leaderboard:
            print(f"\n{self.terminal.BOLD}Leaderboard{self.terminal.RESET}")
            for row in leaderboard[:10]:
                print(
                    f"  {row.get('rank', '-'):>2}. {row['model_name']} | "
                    f"acc {row.get('overall_accuracy', 0.0):.4f} | "
                    f"samples {row.get('total_samples', 0)} | "
                    f"lat {row.get('avg_latency_sec', 0.0):.2f}s"
                )

    def export_results(self, run_id: Optional[str] = None, sync_dir: Optional[str] = None) -> None:
        """Export website-friendly files from artifacts."""
        if sync_dir:
            from src.pipeline.exporter import WebsiteExporter

            exporter = WebsiteExporter(
                artifacts_dir=self.local_settings.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR),
                output_dir=self.local_settings.get("website_export_dir", DEFAULT_WEBSITE_EXPORT_DIR),
                sync_dir=sync_dir,
            )
            exported = exporter.export_run(run_id)
        else:
            exported = self.exporter.export_run(run_id)
        print(f"\n{self.terminal.BOLD}Exported Website Bundle{self.terminal.RESET}")
        for name, path in exported.items():
            print(f"  {name}: {path}")

    def warm_dataset_cache(
        self,
        benchmarks: Optional[List[str]] = None,
        samples: Optional[int] = None,
        json_output: bool = False,
    ) -> List[Dict[str, Any]]:
        """Warm the benchmark sample cache outside the repository."""
        self._get_orchestrator()
        benchmark_selection = benchmarks or list(self.local_settings.get("default_benchmarks", DEFAULT_BENCHMARKS))
        sample_count = samples if samples is not None else int(
            self.local_settings.get("default_samples", DEFAULT_LOCAL_SAMPLE_COUNT)
        )
        prepared = warm_benchmark_cache(benchmark_selection, sample_count)
        dataset_runtime = dataset_runtime_info()

        if json_output:
            print(json.dumps({"dataset_runtime": dataset_runtime, "prepared": prepared}, indent=2))
            return prepared

        print(f"\n{self.terminal.BOLD}Benchmark Dataset Cache{self.terminal.RESET}")
        print(f"  Cache dir: {dataset_runtime['cache_dir']}")
        remote_status = "enabled" if dataset_runtime["allow_remote_dataset_downloads"] else "disabled"
        print(f"  Remote downloads: {remote_status}")
        for entry in prepared:
            state = "ready" if entry["ready"] else "partial"
            print(
                f"  {entry['benchmark']}: cached {entry['cached_rows']}/{entry['requested_samples']} rows "
                f"({state})"
            )
        return prepared

    def _prompt_model_selection(self, models: Sequence[Dict[str, Any]]) -> List[str]:
        """Prompt for model selection in interactive mode."""
        print(f"\n{self.terminal.BOLD}Model Selection{self.terminal.RESET}")
        print("Press Enter for all models, or type comma-separated numbers/names.")
        raw = input(f"{self.terminal.CYAN}Models:{self.terminal.RESET} ").strip()
        if not raw or raw.lower() in {"all", "*"}:
            return [model["name"] for model in models]

        chosen: List[str] = []
        options = {str(index): model["name"] for index, model in enumerate(models, start=1)}
        for part in [piece.strip() for piece in raw.split(",") if piece.strip()]:
            if part in options:
                chosen.append(options[part])
            else:
                chosen.append(part)
        return chosen

    def _prompt_benchmark_selection(self) -> List[str]:
        """Prompt for benchmark or suite selection."""
        suites = list_benchmark_suites()
        benchmarks = list_supported_benchmarks()

        print(f"\n{self.terminal.BOLD}Benchmark Selection{self.terminal.RESET}")
        print("Press Enter for the default core suite.")

        option_map: Dict[str, str] = {}
        counter = 1
        for suite in suites:
            option_map[str(counter)] = suite["key"]
            print(f"  {counter:>2}. {suite['key']} - {suite['description']}")
            counter += 1

        print(f"\n{self.terminal.GRAY}Individual benchmarks:{self.terminal.RESET}")
        for benchmark in benchmarks:
            option_map[str(counter)] = benchmark["key"]
            print(f"  {counter:>2}. {benchmark['key']} - {benchmark['display_name']}")
            counter += 1

        raw = input(f"{self.terminal.CYAN}Benchmarks or suites:{self.terminal.RESET} ").strip()
        if not raw:
            return list(DEFAULT_BENCHMARKS)

        selected: List[str] = []
        for part in [piece.strip() for piece in raw.split(",") if piece.strip()]:
            selected.append(option_map.get(part, part))
        return selected

    def _prompt_sample_count(self) -> int:
        """Prompt for sample count."""
        default_samples = int(self.local_settings.get("default_samples", DEFAULT_LOCAL_SAMPLE_COUNT))
        print(f"\n{self.terminal.BOLD}Sample Count{self.terminal.RESET}")
        print("Suggested: 3 quick, 10 useful, 25 stable, 50+ serious.")
        raw = input(f"{self.terminal.CYAN}Samples per benchmark [{default_samples}]:{self.terminal.RESET} ").strip()
        if not raw:
            return default_samples
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("Sample count must be an integer.") from exc
        if value <= 0:
            raise ValueError("Sample count must be positive.")
        return value

    async def run_benchmarks(
        self,
        models: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        samples: Optional[int] = None,
        export_after_run: Optional[bool] = None,
        all_local: bool = False,
    ) -> Dict[str, Any]:
        """Run benchmarks with the live terminal renderer attached."""
        benchmark_selection = benchmarks or list(self.local_settings.get("default_benchmarks", DEFAULT_BENCHMARKS))
        expanded_benchmarks = expand_benchmark_selection(benchmark_selection)
        orchestrator = self._get_orchestrator()
        default_provider = self.local_settings.get("default_provider", DEFAULT_LOCAL_PROVIDER)

        if not models:
            discovered = await orchestrator.discover_local_models()
            if all_local:
                models = [entry["name"] for entries in discovered.values() for entry in entries]
            else:
                models = [entry["name"] for entry in discovered.get(default_provider, [])]
        if not models:
            raise RuntimeError("No local models found. Start Ollama or LM Studio, or pass --models explicitly.")

        sample_count = samples if samples is not None else int(
            self.local_settings.get("default_samples", DEFAULT_LOCAL_SAMPLE_COUNT)
        )
        run_id = f"pending_{int(time.time())}"
        self.terminal.start_run(run_id, models, expanded_benchmarks, sample_count)

        result = await orchestrator.run(
            models=models,
            benchmarks=benchmark_selection,
            samples=sample_count,
            provider=default_provider,
            all_local=all_local,
            export_after_run=export_after_run,
            progress_callback=self.terminal.handle_event,
        )

        if self.terminal.is_tty:
            self.terminal.run_id = result["run_id"]
            self.terminal.render()

        totals = result.get("summary", {}).get("totals", {})
        print(f"\n{self.terminal.BOLD}Run Complete{self.terminal.RESET}")
        print(f"  Run ID: {result['run_id']}")
        print(f"  Accuracy: {totals.get('accuracy', 0.0):.4f}")
        print(f"  Samples: {totals.get('samples', 0)}")
        print(f"  Artifacts: {result['run_dir']}")
        website_dir = self.local_settings.get("website_export_dir", DEFAULT_WEBSITE_EXPORT_DIR)
        sync_dir = self.local_settings.get("website_sync_dir")
        if export_after_run is not False:
            print(f"  Website export: {website_dir}/latest")
            if sync_dir:
                print(f"  Website sync: {sync_dir}/latest-session.json")
        return result

    async def run_interactive_benchmark(self) -> None:
        """Run the interactive benchmark workflow."""
        discovered = await self._get_orchestrator().discover_local_models()
        models = discovered.get("ollama", []) + discovered.get("lm_studio", [])
        if not models:
            print(f"{self.terminal.YELLOW}No local models found.{self.terminal.RESET}")
            print("")
            for line in self.build_setup_status_lines():
                print(line)
            return

        selected_models = self._prompt_model_selection(models)
        selected_benchmarks = self._prompt_benchmark_selection()
        sample_count = self._prompt_sample_count()
        await self.run_benchmarks(
            models=selected_models,
            benchmarks=selected_benchmarks,
            samples=sample_count,
            export_after_run=True,
        )

    async def run_quick(self, samples: int = 3) -> Dict[str, Any]:
        """Run the default core suite with a small sample count."""
        return await self.run_benchmarks(
            models=None,
            benchmarks=list(self.local_settings.get("default_benchmarks", DEFAULT_BENCHMARKS)),
            samples=samples,
            export_after_run=True,
            all_local=True,
        )

    def run_interactive(self) -> None:
        """Run the default interactive UI."""
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                from src.cli.terminal_menu import TerminalMenuApp

                TerminalMenuApp(self).run()
                return
            except Exception as exc:
                print(f"{self.terminal.YELLOW}Arrow-key mode unavailable: {exc}{self.terminal.RESET}")

        self._run_command_shell()

    def _run_command_shell(self) -> None:
        """Fallback interactive command loop for non-TTY environments."""
        self.print_welcome()

        while True:
            try:
                command = input(f"\n{self.terminal.BOLD}smaLLMs{self.terminal.RESET} {self.terminal.GRAY}${self.terminal.RESET} ").strip().lower()
                if command in {"exit", "quit", "q"}:
                    break
                if command in {"help", "h", ""}:
                    self.print_welcome()
                elif command == "clear":
                    self.print_welcome()
                elif command in {"doctor", "setup"}:
                    self.show_setup_status()
                elif command == "discover":
                    asyncio.run(self.discover_models())
                elif command == "benchmarks":
                    self.show_benchmarks()
                elif command == "cache":
                    self.warm_dataset_cache()
                elif command == "run":
                    asyncio.run(self.run_interactive_benchmark())
                elif command == "quick":
                    asyncio.run(self.run_quick())
                elif command == "status":
                    self.show_status()
                elif command == "export":
                    self.export_results()
                else:
                    print(f"{self.terminal.YELLOW}Unknown command: {command}{self.terminal.RESET}")
            except KeyboardInterrupt:
                print()
            except Exception as exc:
                print(f"{self.terminal.RED}Error: {exc}{self.terminal.RESET}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser."""
    parser = argparse.ArgumentParser(description="smaLLMs local CLI for benchmarking local LLMs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        choices=[
            "interactive",
            "menu",
            "doctor",
            "setup",
            "discover",
            "discover-local",
            "benchmarks",
            "cache",
            "run",
            "local-run",
            "quick",
            "status",
            "export",
        ],
        help="Command to run.",
    )
    parser.add_argument("--models", nargs="*", help="Model names. Supports repeated flags or comma-separated values.")
    parser.add_argument("--benchmarks", nargs="*", help="Benchmark or suite names. Supports repeated flags or comma-separated values.")
    parser.add_argument("--samples", type=int, help="Samples per benchmark.")
    parser.add_argument("--all-local", action="store_true", help="Run across all discovered local models.")
    parser.add_argument("--no-export", action="store_true", help="Skip website export after a run.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for doctor/discover/benchmarks/status.")
    parser.add_argument("--run-id", help="Run id for export; defaults to the latest run.")
    parser.add_argument("--sync-dir", help="Optional website public/data directory to mirror the exported session bundle into.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    app = SmaLLMsCLI(config_path=args.config)
    command = args.command
    models = _parse_list(args.models)
    benchmarks = _parse_list(args.benchmarks)

    try:
        if command in {"interactive", "menu"}:
            app.run_interactive()
        elif command in {"doctor", "setup"}:
            app.show_setup_status(json_output=args.json)
        elif command in {"discover", "discover-local"}:
            asyncio.run(app.discover_models(json_output=args.json))
        elif command == "benchmarks":
            app.show_benchmarks(json_output=args.json)
        elif command == "cache":
            app.warm_dataset_cache(benchmarks=benchmarks, samples=args.samples, json_output=args.json)
        elif command in {"run", "local-run"}:
            asyncio.run(
                app.run_benchmarks(
                    models=models,
                    benchmarks=benchmarks,
                    samples=args.samples,
                    export_after_run=not args.no_export,
                    all_local=args.all_local,
                )
            )
        elif command == "quick":
            asyncio.run(app.run_quick(samples=args.samples or 3))
        elif command == "status":
            app.show_status(json_output=args.json)
        elif command == "export":
            app.export_results(run_id=args.run_id, sync_dir=args.sync_dir)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency '{exc.name}'. Install the requirements before running this command.")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
