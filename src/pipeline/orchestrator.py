"""
Ollama-first local benchmark orchestration.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from src.models.model_manager import ModelInfo, ModelManager
from src.pipeline.artifacts import (
    ArtifactStore,
    RunPaths,
    collect_repository_metadata,
    collect_system_metadata,
    safe_slug,
    utcnow_iso,
)
from src.pipeline.benchmarks import (
    DEFAULT_BENCHMARKS,
    expand_benchmark_selection,
    get_supported_benchmark,
    list_benchmark_suites,
    list_supported_benchmarks,
)


LOGGER = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


DEFAULT_CONFIG: Dict[str, Any] = {
    "evaluation_mode": {
        "default": "local",
        "prefer_local": True,
        "auto_discover_models": True,
        "include_vision_models": False,
    },
    "local_benchmarks": {
        "artifacts_dir": "artifacts",
        "website_export_dir": "website_exports",
        "website_sync_dir": "../websmaLLMs/public/data",
        "default_provider": "ollama",
        "default_samples": 25,
        "default_temperature": 0.0,
        "default_benchmarks": DEFAULT_BENCHMARKS,
        "export_after_run": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge config dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config with defaults."""
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def _model_info_to_dict(info: ModelInfo) -> Dict[str, Any]:
    """Serialize model info for artifacts."""
    return {
        "name": info.name,
        "provider": info.provider,
        "size_gb": info.size_gb,
        "parameters": info.parameters,
        "architecture": info.architecture,
        "license": info.license,
        "max_context": info.max_context,
        "supports_vision": info.supports_vision,
        "model_type": info.model_type,
        "family": info.family,
        "quantization": info.quantization,
    }


class LocalBenchmarkOrchestrator:
    """Run supported local benchmarks and persist structured artifacts."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = _load_config(config_path)
        self.model_manager = ModelManager(self.config)
        artifacts_dir = self.config.get("local_benchmarks", {}).get("artifacts_dir", "artifacts")
        self.artifact_store = ArtifactStore(artifacts_dir)

    async def discover_local_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover locally available models."""
        return await self.model_manager.discover_local_models()

    async def resolve_models(
        self,
        models: Optional[List[str]] = None,
        provider: Optional[str] = None,
        all_local: bool = False,
    ) -> List[str]:
        """Resolve the target model list from arguments or discovery."""
        if models:
            return models

        provider = provider or self.config.get("local_benchmarks", {}).get("default_provider", "ollama")
        discovered = await self.discover_local_models()

        if all_local:
            return [entry["name"] for entries in discovered.values() for entry in entries]

        return [entry["name"] for entry in discovered.get(provider, [])]

    async def run(
        self,
        models: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        samples: Optional[int] = None,
        provider: Optional[str] = None,
        all_local: bool = False,
        export_after_run: Optional[bool] = None,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        """Run local benchmarks and optionally export the latest website bundle."""
        requested_benchmarks = benchmarks or list(
            self.config.get("local_benchmarks", {}).get("default_benchmarks", DEFAULT_BENCHMARKS)
        )
        benchmark_names = expand_benchmark_selection(requested_benchmarks)
        sample_count = samples or int(self.config.get("local_benchmarks", {}).get("default_samples", 25))
        temperature = float(self.config.get("local_benchmarks", {}).get("default_temperature", 0.0))
        export_after_run = (
            self.config.get("local_benchmarks", {}).get("export_after_run", True)
            if export_after_run is None
            else export_after_run
        )

        resolved_models = await self.resolve_models(models=models, provider=provider, all_local=all_local)
        if not resolved_models:
            raise RuntimeError("No local models resolved. Pull an Ollama model or pass --models explicitly.")

        created_at = utcnow_iso()
        manifest = {
            "created_at": created_at,
            "mode": "local_benchmarks",
            "models": resolved_models,
            "benchmarks": benchmark_names,
            "requested_benchmarks": requested_benchmarks,
            "samples_per_benchmark": sample_count,
            "temperature": temperature,
            "supported_benchmarks": list_supported_benchmarks(),
            "benchmark_suites": list_benchmark_suites(),
            "system": collect_system_metadata(),
            "repository": collect_repository_metadata("."),
            "config_path": self.config_path,
        }
        paths = self.artifact_store.create_run(manifest)

        benchmark_results: List[Dict[str, Any]] = []

        try:
            for model_name in resolved_models:
                for benchmark_name in benchmark_names:
                    LOGGER.info("Running %s on %s", model_name, benchmark_name)
                    if progress_callback:
                        progress_callback(
                            {
                                "event": "benchmark_started",
                                "run_id": paths.run_id,
                                "model_name": model_name,
                                "benchmark_name": benchmark_name,
                                "sample_count": sample_count,
                            }
                        )

                    benchmark_result = await self._run_single_benchmark(
                        paths=paths,
                        model_name=model_name,
                        benchmark_name=benchmark_name,
                        sample_count=sample_count,
                        temperature=temperature,
                        progress_callback=progress_callback,
                    )
                    benchmark_results.append(benchmark_result)
            summary = self._build_run_summary(paths, benchmark_results)
            self.artifact_store.write_summary(paths, summary)
        finally:
            await self.model_manager.cleanup()

        if export_after_run:
            from src.pipeline.exporter import WebsiteExporter

            exporter = WebsiteExporter(
                artifacts_dir=self.config.get("local_benchmarks", {}).get("artifacts_dir", "artifacts"),
                output_dir=self.config.get("local_benchmarks", {}).get("website_export_dir", "website_exports"),
                sync_dir=self.config.get("local_benchmarks", {}).get("website_sync_dir"),
            )
            exporter.export_run(paths.run_id)

        return self.artifact_store.load_run(paths.run_id)

    async def _run_single_benchmark(
        self,
        paths: RunPaths,
        model_name: str,
        benchmark_name: str,
        sample_count: int,
        temperature: float,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        """Run one benchmark against one model and persist its artifacts."""
        benchmark = get_supported_benchmark(benchmark_name)
        model = await self.model_manager.get_model(model_name)
        model_info = _model_info_to_dict(model.get_model_info())

        try:
            execution = await benchmark.evaluate(
                model=model,
                model_info=model_info,
                run_id=paths.run_id,
                num_samples=sample_count,
                temperature=temperature,
                progress_callback=progress_callback,
            )
            benchmark_result = execution.benchmark_result
            sample_file = self.artifact_store.write_sample_results(paths, benchmark_name, model_name, execution.samples)
            benchmark_result["artifact_paths"] = {
                "samples_jsonl": str(sample_file),
            }
            benchmark_file = self.artifact_store.write_benchmark_result(paths, benchmark_name, model_name, benchmark_result)
            benchmark_result["artifact_paths"]["benchmark_json"] = str(benchmark_file)
            self.artifact_store.write_benchmark_result(paths, benchmark_name, model_name, benchmark_result)
            return benchmark_result
        except Exception as exc:
            failure = {
                "schema_version": "2.0",
                "run_id": paths.run_id,
                "benchmark_name": benchmark_name,
                "benchmark_display_name": benchmark.display_name,
                "description": benchmark.description,
                "model": model_info,
                "status": "failed",
                "error": str(exc),
                "metrics": {
                    "sample_count": 0,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "error_count": 1,
                    "avg_latency_sec": 0.0,
                    "max_latency_sec": 0.0,
                    "min_latency_sec": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "avg_tokens_per_second": 0.0,
                    "local_cost_estimate": 0.0,
                },
            }
            if progress_callback:
                progress_callback(
                    {
                        "event": "benchmark_failed",
                        "run_id": paths.run_id,
                        "model_name": model_name,
                        "benchmark_name": benchmark_name,
                        "error": str(exc),
                    }
                )
            failure_file = self.artifact_store.write_benchmark_result(paths, benchmark_name, model_name, failure)
            failure["artifact_paths"] = {"benchmark_json": str(failure_file)}
            self.artifact_store.write_benchmark_result(paths, benchmark_name, model_name, failure)
            return failure
        finally:
            await self.model_manager.cleanup()

    def _build_run_summary(self, paths: RunPaths, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an aggregate run summary plus per-model leaderboard."""
        completed = [result for result in benchmark_results if result.get("status") == "completed"]
        leaderboard: Dict[str, Dict[str, Any]] = {}

        total_samples = 0
        total_correct = 0
        total_errors = 0
        total_tokens = 0
        total_duration = 0.0

        for result in completed:
            metrics = result.get("metrics", {})
            model_name = result["model"]["name"]
            total_samples += int(metrics.get("sample_count", 0))
            total_correct += int(metrics.get("correct_count", 0))
            total_errors += int(metrics.get("error_count", 0))
            total_tokens += int(metrics.get("total_tokens", 0))
            total_duration += float(metrics.get("avg_latency_sec", 0.0)) * int(metrics.get("sample_count", 0))

            if model_name not in leaderboard:
                leaderboard[model_name] = {
                    "model_name": model_name,
                    "provider": result["model"]["provider"],
                    "size_gb": result["model"].get("size_gb", 0.0),
                    "parameters": result["model"].get("parameters", "unknown"),
                    "family": result["model"].get("family", "unknown"),
                    "quantization": result["model"].get("quantization", "unknown"),
                    "benchmarks": {},
                    "total_samples": 0,
                    "correct_count": 0,
                    "error_count": 0,
                    "total_tokens": 0,
                    "total_duration_sec": 0.0,
                }

            entry = leaderboard[model_name]
            entry["benchmarks"][result["benchmark_name"]] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "sample_count": metrics.get("sample_count", 0),
                "avg_latency_sec": metrics.get("avg_latency_sec", 0.0),
                "avg_tokens_per_second": metrics.get("avg_tokens_per_second", 0.0),
            }
            entry["total_samples"] += int(metrics.get("sample_count", 0))
            entry["correct_count"] += int(metrics.get("correct_count", 0))
            entry["error_count"] += int(metrics.get("error_count", 0))
            entry["total_tokens"] += int(metrics.get("total_tokens", 0))
            entry["total_duration_sec"] += float(metrics.get("avg_latency_sec", 0.0)) * int(metrics.get("sample_count", 0))

        leaderboard_rows: List[Dict[str, Any]] = []
        for entry in leaderboard.values():
            total_samples_for_model = entry["total_samples"]
            overall_accuracy = entry["correct_count"] / total_samples_for_model if total_samples_for_model else 0.0
            avg_latency_sec = entry["total_duration_sec"] / total_samples_for_model if total_samples_for_model else 0.0
            row = {
                "model_name": entry["model_name"],
                "provider": entry["provider"],
                "size_gb": entry["size_gb"],
                "parameters": entry["parameters"],
                "family": entry["family"],
                "quantization": entry["quantization"],
                "overall_accuracy": round(overall_accuracy, 4),
                "benchmarks_run": len(entry["benchmarks"]),
                "total_samples": total_samples_for_model,
                "correct_count": entry["correct_count"],
                "error_count": entry["error_count"],
                "avg_latency_sec": round(avg_latency_sec, 4),
                "total_tokens": entry["total_tokens"],
                "benchmarks": entry["benchmarks"],
            }
            leaderboard_rows.append(row)

        leaderboard_rows.sort(key=lambda row: (row["overall_accuracy"], -row["avg_latency_sec"]), reverse=True)
        for index, row in enumerate(leaderboard_rows, start=1):
            row["rank"] = index

        return {
            "schema_version": "2.0",
            "run_id": paths.run_id,
            "generated_at": utcnow_iso(),
            "manifest_path": str(paths.manifest_path),
            "totals": {
                "models": len({result["model"]["name"] for result in benchmark_results}),
                "benchmarks": len({result["benchmark_name"] for result in benchmark_results}),
                "evaluations": len(benchmark_results),
                "completed_evaluations": len(completed),
                "failed_evaluations": len(benchmark_results) - len(completed),
                "samples": total_samples,
                "correct": total_correct,
                "accuracy": round(total_correct / total_samples, 4) if total_samples else 0.0,
                "errors": total_errors,
                "total_tokens": total_tokens,
                "total_duration_sec": round(total_duration, 4),
            },
            "leaderboard": leaderboard_rows,
            "evaluations": benchmark_results,
        }


def run_local_benchmarks_sync(**kwargs: Any) -> Dict[str, Any]:
    """Synchronous helper for CLI entrypoints."""
    orchestrator = LocalBenchmarkOrchestrator(config_path=kwargs.pop("config_path", "config/config.yaml"))
    return asyncio.run(orchestrator.run(**kwargs))
