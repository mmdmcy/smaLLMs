#!/usr/bin/env python3
"""
CLI for the modern Ollama-first local benchmark pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import List, Optional


def _parse_list(value: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize repeated or comma-separated CLI values."""
    if not value:
        return None

    items: List[str] = []
    for chunk in value:
        items.extend(part.strip() for part in chunk.split(",") if part.strip())
    return items or None


async def _handle_run(args: argparse.Namespace) -> None:
    from src.pipeline.benchmarks import DEFAULT_BENCHMARKS
    from src.pipeline.orchestrator import LocalBenchmarkOrchestrator

    orchestrator = LocalBenchmarkOrchestrator(config_path=args.config)
    result = await orchestrator.run(
        models=_parse_list(args.models),
        benchmarks=_parse_list(args.benchmarks) or DEFAULT_BENCHMARKS,
        samples=args.samples,
        provider=args.provider,
        all_local=args.all_local,
        export_after_run=not args.no_export,
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


def _handle_export(args: argparse.Namespace) -> None:
    from src.pipeline.exporter import WebsiteExporter

    exporter = WebsiteExporter(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        sync_dir=args.sync_dir,
    )
    exported = exporter.export_run(args.run_id)
    print(json.dumps(exported, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(description="Run local smaLLMs benchmarks and export website data.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run local benchmarks.")
    run_parser.add_argument("--models", nargs="*", help="Model names. Supports repeated flags or comma-separated values.")
    run_parser.add_argument("--benchmarks", nargs="*", help="Benchmark keys. Defaults to the supported local suite.")
    run_parser.add_argument("--samples", type=int, default=25, help="Samples per benchmark.")
    run_parser.add_argument("--provider", default="ollama", choices=["ollama"], help="Discovery provider when --models is omitted.")
    run_parser.add_argument("--all-local", action="store_true", help="Run across all discovered local models.")
    run_parser.add_argument("--no-export", action="store_true", help="Skip website export after the run.")
    run_parser.set_defaults(handler=_handle_run)

    discover_parser = subparsers.add_parser("discover", help="Print discovered local models.")
    discover_parser.set_defaults(handler=_handle_discover)

    benchmarks_parser = subparsers.add_parser("benchmarks", help="List supported benchmark definitions.")
    benchmarks_parser.set_defaults(handler=_handle_benchmarks)

    export_parser = subparsers.add_parser("export", help="Export website files for a run.")
    export_parser.add_argument("--run-id", help="Run id to export. Defaults to the latest run.")
    export_parser.add_argument("--artifacts-dir", default="artifacts", help="Artifact root directory.")
    export_parser.add_argument("--output-dir", default="website_exports", help="Website export directory.")
    export_parser.add_argument("--sync-dir", help="Optional website public/data directory to mirror the session bundle into.")
    export_parser.set_defaults(handler=_handle_export)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    handler = args.handler
    try:
        if asyncio.iscoroutinefunction(handler):
            asyncio.run(handler(args))
        else:
            handler(args)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency '{exc.name}'. Install the requirements before running this command.")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
