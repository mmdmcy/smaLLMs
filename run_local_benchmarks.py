#!/usr/bin/env python3
"""
CLI for the modern Ollama-first local benchmark pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from typing import List, Optional

from src.pipeline.config import DEFAULT_CONFIG_PATH


def _parse_list(value: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize repeated or comma-separated CLI values."""
    if not value:
        return None

    items: List[str] = []
    for chunk in value:
        items.extend(part.strip() for part in chunk.split(",") if part.strip())
    return items or None


async def _handle_run(args: argparse.Namespace) -> None:
    from src.pipeline.orchestrator import LocalBenchmarkOrchestrator

    orchestrator = LocalBenchmarkOrchestrator(config_path=args.config)
    result = await orchestrator.run(
        models=_parse_list(args.models),
        benchmarks=_parse_list(args.benchmarks),
        samples=args.samples,
        provider=args.provider,
        all_local=args.all_local,
        export_after_run=not args.no_export,
        offline=args.offline,
    )

    summary = result["summary"]
    totals = summary.get("totals", {})
    print(f"Run ID: {result['run_id']}")
    print(f"Models: {totals.get('models', 0)}")
    print(f"Benchmarks: {totals.get('benchmarks', 0)}")
    print(f"Evaluations: {totals.get('evaluations', 0)}")
    print(f"Samples: {totals.get('samples', 0)}")
    print(f"Accuracy: {totals.get('accuracy', 0.0):.4f}")
    print(f"Artifacts: {result['run_dir']}")


async def _handle_discover(args: argparse.Namespace) -> None:
    from src.pipeline.orchestrator import LocalBenchmarkOrchestrator

    orchestrator = LocalBenchmarkOrchestrator(config_path=args.config)
    discovered = await orchestrator.discover_local_models()
    print(json.dumps(discovered, indent=2))


def _handle_benchmarks(_: argparse.Namespace) -> None:
    from src.pipeline.benchmarks import list_benchmark_suites, list_supported_benchmarks

    print(json.dumps({"suites": list_benchmark_suites(), "benchmarks": list_supported_benchmarks()}, indent=2))


def _handle_cache(args: argparse.Namespace) -> None:
    from src.pipeline.benchmarks import configure_dataset_runtime, dataset_runtime_info, warm_benchmark_cache
    from src.pipeline.orchestrator import LocalBenchmarkOrchestrator

    orchestrator = LocalBenchmarkOrchestrator(config_path=args.config)
    if args.offline:
        orchestrator.local_settings["allow_remote_dataset_downloads"] = False
    runtime = configure_dataset_runtime(orchestrator.local_settings)
    orchestrator.local_settings["allow_remote_dataset_downloads"] = runtime["allow_remote_dataset_downloads"]
    sample_count = args.samples if args.samples is not None else int(orchestrator.local_settings["default_samples"])
    prepared = warm_benchmark_cache(_parse_list(args.benchmarks), sample_count)
    print(json.dumps({"dataset_runtime": dataset_runtime_info(), "prepared": prepared}, indent=2))


def _handle_export(args: argparse.Namespace) -> None:
    from src.pipeline.benchmarks import DEFAULT_BENCHMARKS
    from src.pipeline.config import load_pipeline_config, local_benchmark_settings
    from src.pipeline.exporter import WebsiteExporter

    config = load_pipeline_config(args.config, DEFAULT_BENCHMARKS)
    settings = local_benchmark_settings(config, DEFAULT_BENCHMARKS)
    exporter = WebsiteExporter(
        artifacts_dir=args.artifacts_dir or settings["artifacts_dir"],
        output_dir=args.output_dir or settings["website_export_dir"],
        sync_dir=args.sync_dir if args.sync_dir is not None else settings.get("website_sync_dir"),
    )
    exported = exporter.export_run(args.run_id)
    print(json.dumps(exported, indent=2))


def _handle_agent_harness(args: argparse.Namespace) -> None:
    from src.pipeline.agent_harness import DEFAULT_AGENT_HARNESS_WEBSITE_SYNC_DIR, run_agent_harness_eval

    summary = run_agent_harness_eval(
        harnesses=_parse_list(args.harnesses),
        tasks=_parse_list(args.tasks),
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
        sync_dir=args.sync_dir if args.sync_dir is not None else DEFAULT_AGENT_HARNESS_WEBSITE_SYNC_DIR,
    )
    print(json.dumps(summary, indent=2))


def _handle_reasoning_sweep(args: argparse.Namespace) -> None:
    from src.pipeline.agent_harness import DEFAULT_AGENT_HARNESS_WEBSITE_SYNC_DIR
    from src.pipeline.reasoning_sweep import run_reasoning_effort_sweep

    def progress(event: dict) -> None:
        if event.get("event") in {"reasoning_variant_started", "reasoning_variant_completed"}:
            print(
                f"{event['event']}: {event.get('model')} {event.get('reasoning_effort')}",
                file=sys.stderr,
                flush=True,
            )
        elif event.get("event") == "agent_harness_row_completed":
            print(
                f"row: {event.get('model')} {event.get('reasoning_effort')} {event.get('task')} {event.get('status')}",
                file=sys.stderr,
                flush=True,
            )

    payload = run_reasoning_effort_sweep(
        models=_parse_list(args.models),
        efforts=_parse_list(args.efforts),
        tasks=_parse_list(args.tasks),
        artifacts_dir=args.artifacts_dir,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
        sync_dir=args.sync_dir if args.sync_dir is not None else DEFAULT_AGENT_HARNESS_WEBSITE_SYNC_DIR,
        progress_callback=progress,
    )
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(description="Run local smaLLMs benchmarks and export website data.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run local benchmarks.")
    run_parser.add_argument("--models", nargs="*", help="Model names. Supports repeated flags or comma-separated values.")
    run_parser.add_argument("--benchmarks", nargs="*", help="Benchmark keys or suite names. Defaults to local_benchmarks.default_benchmarks.")
    run_parser.add_argument("--samples", type=int, help="Samples per benchmark. Defaults to local_benchmarks.default_samples.")
    run_parser.add_argument("--provider", choices=["ollama", "lm_studio"], help="Discovery provider when --models is omitted.")
    run_parser.add_argument("--all-local", action="store_true", help="Run across all discovered local models.")
    run_parser.add_argument("--no-export", action="store_true", help="Skip website export after the run.")
    run_parser.add_argument("--offline", action="store_true", help="Disallow remote dataset downloads; require warmed benchmark cache.")
    run_parser.set_defaults(handler=_handle_run)

    discover_parser = subparsers.add_parser("discover", help="Print discovered local models.")
    discover_parser.set_defaults(handler=_handle_discover)

    benchmarks_parser = subparsers.add_parser("benchmarks", help="List supported benchmark definitions.")
    benchmarks_parser.set_defaults(handler=_handle_benchmarks)

    cache_parser = subparsers.add_parser("cache", help="Warm the local benchmark dataset cache.")
    cache_parser.add_argument("--benchmarks", nargs="*", help="Benchmark keys or suite names to cache.")
    cache_parser.add_argument("--samples", type=int, help="Rows to cache per benchmark. Defaults to local_benchmarks.default_samples.")
    cache_parser.add_argument("--offline", action="store_true", help="Validate local cache readiness without downloading datasets.")
    cache_parser.set_defaults(handler=_handle_cache)

    export_parser = subparsers.add_parser("export", help="Export website files for a run.")
    export_parser.add_argument("--run-id", help="Run id to export. Defaults to the latest run.")
    export_parser.add_argument("--artifacts-dir", help="Artifact root directory. Defaults to local_benchmarks.artifacts_dir.")
    export_parser.add_argument("--output-dir", help="Website export directory. Defaults to local_benchmarks.website_export_dir.")
    export_parser.add_argument("--sync-dir", help="Optional website public/data directory. Defaults to local_benchmarks.website_sync_dir.")
    export_parser.set_defaults(handler=_handle_export)

    harness_parser = subparsers.add_parser("agent-harness", help="Run or dry-run coding-agent harness evals.")
    harness_parser.add_argument("--harnesses", nargs="*", help="Harness keys: pi, opencode, codex.")
    harness_parser.add_argument("--tasks", nargs="*", help="Task keys. Defaults to all tasks.")
    harness_parser.add_argument("--timeout-seconds", type=int, default=900, help="Per-agent timeout in seconds.")
    harness_parser.add_argument("--dry-run", action="store_true", help="Prepare fixtures without calling model providers.")
    harness_parser.add_argument("--sync-dir", help="Optional website public/data directory for agent-harness JSON.")
    harness_parser.set_defaults(handler=_handle_agent_harness)

    sweep_parser = subparsers.add_parser(
        "reasoning-sweep",
        help="Run GPT-5.6 Codex reasoning-effort variants and export a separate website JSON feed.",
    )
    sweep_parser.add_argument("--models", nargs="*", help="GPT-5.6 model slugs. Defaults to Sol, Terra, and Luna.")
    sweep_parser.add_argument(
        "--efforts",
        nargs="*",
        help="Reasoning efforts such as low, medium, high, xhigh, max, or ultra. Defaults to each model's supported levels.",
    )
    sweep_parser.add_argument("--tasks", nargs="*", help="Fixture task keys. Defaults to all agent-harness tasks.")
    sweep_parser.add_argument("--artifacts-dir", default="artifacts/reasoning_sweep", help="Root directory for sweep artifacts.")
    sweep_parser.add_argument("--timeout-seconds", type=int, default=900, help="Per-agent timeout in seconds.")
    sweep_parser.add_argument("--dry-run", action="store_true", help="Prepare every variant without calling model providers.")
    sweep_parser.add_argument("--sync-dir", help="Optional website public/data directory for reasoning-effort JSON.")
    sweep_parser.set_defaults(handler=_handle_reasoning_sweep)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    handler = args.handler
    try:
        if inspect.iscoroutinefunction(handler):
            asyncio.run(handler(args))
        else:
            handler(args)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency '{exc.name}'. Install the requirements before running this command.")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
