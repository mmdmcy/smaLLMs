"""
Ollama-first local benchmark orchestration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from src.models.model_manager import ModelInfo, ModelManager
from src.pipeline.artifacts import (
    ArtifactStore,
    RunPaths,
    collect_repository_metadata,
    collect_system_metadata,
    portable_path,
    utcnow_iso,
)
from src.pipeline.benchmarks import (
    DEFAULT_BENCHMARKS,
    benchmark_cache_status,
    configure_dataset_runtime,
    dataset_runtime_info,
    expand_benchmark_selection,
    get_supported_benchmark,
    list_benchmark_suites,
    list_supported_benchmarks,
    wilson_interval,
)
from src.pipeline.config import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXPORT_AFTER_RUN,
    DEFAULT_LOCAL_PROVIDER,
    DEFAULT_LOCAL_SAMPLE_COUNT,
    DEFAULT_LOCAL_TEMPERATURE,
    DEFAULT_WEBSITE_EXPORT_DIR,
    config_fingerprint,
    load_pipeline_config,
    local_benchmark_settings,
    redact_sensitive_config,
)


LOGGER = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


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
        "digest": info.digest,
        "modified_at": info.modified_at,
    }


class LocalBenchmarkOrchestrator:
    """Run supported local benchmarks and persist structured artifacts."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = load_pipeline_config(config_path, DEFAULT_BENCHMARKS)
        self.local_settings = local_benchmark_settings(self.config, DEFAULT_BENCHMARKS)
        configure_dataset_runtime(self.local_settings)
        self.model_manager = ModelManager(self.config)
        self.artifact_store = ArtifactStore(self.local_settings.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR))

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

        provider = provider or self.local_settings.get("default_provider", DEFAULT_LOCAL_PROVIDER)
        discovered = await self.discover_local_models()

        if all_local:
            return [entry["name"] for entries in discovered.values() for entry in entries]

        return [entry["name"] for entry in discovered.get(provider, [])]

    def _configure_run_dataset_runtime(self, offline: Optional[bool] = None) -> Dict[str, Any]:
        """Apply per-run dataset network policy and return the active runtime."""
        if offline is not None:
            self.local_settings["allow_remote_dataset_downloads"] = not offline
        runtime = configure_dataset_runtime(self.local_settings)
        self.local_settings["allow_remote_dataset_downloads"] = runtime["allow_remote_dataset_downloads"]
        return runtime

    def _portable_cache_status(self, statuses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize cache status paths before persisting them in run manifests."""
        normalized: List[Dict[str, Any]] = []
        for status in statuses:
            item = dict(status)
            if item.get("rows_path"):
                item["rows_path"] = portable_path(item["rows_path"])
            if item.get("meta_path"):
                item["meta_path"] = portable_path(item["meta_path"])
            metadata = dict(item.get("metadata") or {})
            if metadata.get("rows_path"):
                metadata["rows_path"] = portable_path(metadata["rows_path"])
            item["metadata"] = metadata
            normalized.append(item)
        return normalized

    def _assert_offline_cache_ready(self, cache_statuses: List[Dict[str, Any]]) -> None:
        """Fail early when an offline run would need uncached benchmark rows."""
        missing = [status for status in cache_statuses if not status.get("ready")]
        if not missing:
            return

        details = ", ".join(
            f"{item['benchmark']} ({item['cached_rows']}/{item['requested_samples']} cached)"
            for item in missing
        )
        raise RuntimeError(
            "Offline run requested, but the benchmark cache is incomplete: "
            f"{details}. Run `smaLLMs.py cache --benchmarks ... --samples ...` while online, "
            "then rerun with `--offline`."
        )

    def _selected_model_inventory(
        self,
        discovered: Dict[str, List[Dict[str, Any]]],
        resolved_models: List[str],
    ) -> List[Dict[str, Any]]:
        """Return discovery metadata for selected models, preserving requested order."""
        by_name: Dict[str, Dict[str, Any]] = {}
        for provider, models in discovered.items():
            for model in models:
                entry = dict(model)
                entry.setdefault("provider", provider)
                by_name[entry.get("name", "")] = entry

        inventory: List[Dict[str, Any]] = []
        for name in resolved_models:
            if name in by_name:
                inventory.append(by_name[name])
            else:
                inventory.append(
                    {
                        "name": name,
                        "provider": self.model_manager._detect_provider(name),
                        "available": False,
                        "metadata_warning": "not_returned_by_local_discovery",
                    }
                )
        return inventory

    async def run(
        self,
        models: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        samples: Optional[int] = None,
        provider: Optional[str] = None,
        all_local: bool = False,
        export_after_run: Optional[bool] = None,
        offline: Optional[bool] = None,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        """Run local benchmarks and optionally export the latest website bundle."""
        dataset_runtime = self._configure_run_dataset_runtime(offline=offline)
        requested_benchmarks = benchmarks or list(self.local_settings.get("default_benchmarks", DEFAULT_BENCHMARKS))
        benchmark_names = expand_benchmark_selection(requested_benchmarks)
        sample_count = samples if samples is not None else int(
            self.local_settings.get("default_samples", DEFAULT_LOCAL_SAMPLE_COUNT)
        )
        temperature = float(self.local_settings.get("default_temperature", DEFAULT_LOCAL_TEMPERATURE))
        export_after_run = (
            bool(self.local_settings.get("export_after_run", DEFAULT_EXPORT_AFTER_RUN))
            if export_after_run is None
            else export_after_run
        )

        resolved_models = await self.resolve_models(models=models, provider=provider, all_local=all_local)
        if not resolved_models:
            raise RuntimeError("No local models resolved. Pull an Ollama model or pass --models explicitly.")

        discovered_models = await self.discover_local_models()
        cache_status = benchmark_cache_status(benchmark_names, sample_count)
        if not dataset_runtime["allow_remote_dataset_downloads"]:
            self._assert_offline_cache_ready(cache_status)

        effective_config = dict(self.config)
        effective_config["local_benchmarks"] = dict(self.local_settings)
        redacted_config = redact_sensitive_config(effective_config)
        manifest_dataset_runtime = dict(dataset_runtime)
        manifest_dataset_runtime["cache_dir"] = portable_path(manifest_dataset_runtime["cache_dir"])
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
            "execution_policy": {
                "offline": not dataset_runtime["allow_remote_dataset_downloads"],
                "remote_dataset_downloads_allowed": dataset_runtime["allow_remote_dataset_downloads"],
                "network_scope": (
                    "local_model_endpoints_and_cached_datasets"
                    if not dataset_runtime["allow_remote_dataset_downloads"]
                    else "local_model_endpoints_plus_dataset_cache_warmup"
                ),
            },
            "dataset_runtime": manifest_dataset_runtime,
            "dataset_cache": self._portable_cache_status(cache_status),
            "model_inventory": self._selected_model_inventory(discovered_models, resolved_models),
            "system": collect_system_metadata(),
            "repository": collect_repository_metadata("."),
            "config": {
                "path": self.config_path,
                "sha256": config_fingerprint(effective_config),
                "snapshot": redacted_config,
            },
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
            run_card_path = self.artifact_store.write_run_card(paths, self._build_run_card(paths, manifest, summary))
            summary["run_card_path"] = portable_path(run_card_path)
            self.artifact_store.write_summary(paths, summary)
        finally:
            await self.model_manager.cleanup()

        if export_after_run:
            from src.pipeline.exporter import WebsiteExporter

            exporter = WebsiteExporter(
                artifacts_dir=self.local_settings.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR),
                output_dir=self.local_settings.get("website_export_dir", DEFAULT_WEBSITE_EXPORT_DIR),
                sync_dir=self.local_settings.get("website_sync_dir"),
            )
            exporter.export_run(paths.run_id)

        return self.artifact_store.load_run(paths.run_id)

    def _build_run_card(self, paths: RunPaths, manifest: Dict[str, Any], summary: Dict[str, Any]) -> str:
        """Create a compact human-readable report for one benchmark run."""
        totals = summary.get("totals", {})
        policy = manifest.get("execution_policy", {})
        repository = manifest.get("repository", {})
        system = manifest.get("system", {})
        config = manifest.get("config", {})
        cache_rows = manifest.get("dataset_cache", [])
        leaderboard = summary.get("leaderboard", [])

        lines = [
            f"# smaLLMs Run Card: {paths.run_id}",
            "",
            "## Result",
            "",
            f"- Accuracy: {totals.get('accuracy', 0.0):.4f} "
            f"(95% CI {totals.get('accuracy_ci95_low', 0.0):.4f}-{totals.get('accuracy_ci95_high', 0.0):.4f})",
            f"- Samples: {totals.get('samples', 0)}",
            f"- Models: {totals.get('models', 0)}",
            f"- Benchmarks: {totals.get('benchmarks', 0)}",
            f"- Response rate: {totals.get('response_rate', 0.0):.4f}",
            f"- Invalid prediction rate: {totals.get('invalid_prediction_rate', 0.0):.4f}",
            f"- Raw fallback rate: {totals.get('raw_fallback_rate', 0.0):.4f}",
            "",
            "## Execution Policy",
            "",
            f"- Offline: {bool(policy.get('offline'))}",
            f"- Remote dataset downloads allowed: {bool(policy.get('remote_dataset_downloads_allowed'))}",
            f"- Network scope: {policy.get('network_scope', 'unknown')}",
            "",
            "## Reproducibility",
            "",
            f"- Git SHA: {repository.get('git_sha', '')}",
            f"- Git branch: {repository.get('git_branch', '')}",
            f"- Git dirty: {repository.get('git_dirty')}",
            f"- Config SHA-256: {config.get('sha256', '')}",
            f"- Python: {system.get('python_version', '')}",
            f"- Ollama: {system.get('ollama_version', '')}",
            "",
            "## Leaderboard",
            "",
            "| Rank | Model | Accuracy | 95% CI | Invalid | Raw Fallback | Avg Latency |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
        ]

        for row in leaderboard:
            lines.append(
                "| {rank} | {model} | {acc:.4f} | {low:.4f}-{high:.4f} | {invalid:.4f} | {raw:.4f} | {lat:.4f}s |".format(
                    rank=row.get("rank", ""),
                    model=row.get("model_name", ""),
                    acc=float(row.get("overall_accuracy", 0.0)),
                    low=float(row.get("overall_accuracy_ci95_low", 0.0)),
                    high=float(row.get("overall_accuracy_ci95_high", 0.0)),
                    invalid=float(row.get("invalid_prediction_rate", 0.0)),
                    raw=float(row.get("raw_fallback_rate", 0.0)),
                    lat=float(row.get("avg_latency_sec", 0.0)),
                )
            )

        lines.extend(["", "## Dataset Cache", ""])
        if cache_rows:
            lines.extend(
                [
                    "| Benchmark | Ready | Cached Rows | Requested | Rows SHA-256 |",
                    "| --- | ---: | ---: | ---: | --- |",
                ]
            )
            for item in cache_rows:
                lines.append(
                    "| {benchmark} | {ready} | {cached} | {requested} | `{sha}` |".format(
                        benchmark=item.get("benchmark", ""),
                        ready=str(bool(item.get("ready"))).lower(),
                        cached=item.get("cached_rows", 0),
                        requested=item.get("requested_samples", 0),
                        sha=str(item.get("rows_sha256") or "")[:16],
                    )
                )
        else:
            lines.append("No dataset cache metadata recorded.")

        lines.extend(["", "## Artifact Pointers", ""])
        lines.append(f"- Manifest: {portable_path(paths.manifest_path)}")
        lines.append(f"- Summary: {portable_path(paths.summary_path)}")
        lines.append(f"- Samples: {portable_path(paths.sample_dir)}")
        lines.append("")
        return "\n".join(lines)

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
                "samples_jsonl": portable_path(sample_file),
            }
            benchmark_file = self.artifact_store.write_benchmark_result(paths, benchmark_name, model_name, benchmark_result)
            benchmark_result["artifact_paths"]["benchmark_json"] = portable_path(benchmark_file)
            self.artifact_store.write_benchmark_result(paths, benchmark_name, model_name, benchmark_result)
            return benchmark_result
        except Exception as exc:
            failure = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
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
                    "success_count": 0,
                    "success_rate": 0.0,
                    "responded_count": 0,
                    "response_rate": 0.0,
                    "valid_prediction_count": 0,
                    "invalid_prediction_count": 0,
                    "invalid_prediction_rate": 0.0,
                    "invalid_prediction_rate_ci95_low": 0.0,
                    "invalid_prediction_rate_ci95_high": 0.0,
                    "accuracy_ci95_low": 0.0,
                    "accuracy_ci95_high": 0.0,
                    "raw_fallback_count": 0,
                    "raw_fallback_rate": 0.0,
                    "raw_fallback_attempted_count": 0,
                    "raw_fallback_attempted_rate": 0.0,
                    "error_count": 1,
                    "total_latency_sec": 0.0,
                    "avg_latency_sec": 0.0,
                    "max_latency_sec": 0.0,
                    "min_latency_sec": 0.0,
                    "total_load_duration_sec": 0.0,
                    "avg_load_duration_sec": 0.0,
                    "max_load_duration_sec": 0.0,
                    "min_load_duration_sec": 0.0,
                    "total_prompt_eval_duration_sec": 0.0,
                    "avg_prompt_eval_duration_sec": 0.0,
                    "max_prompt_eval_duration_sec": 0.0,
                    "min_prompt_eval_duration_sec": 0.0,
                    "total_eval_duration_sec": 0.0,
                    "avg_eval_duration_sec": 0.0,
                    "max_eval_duration_sec": 0.0,
                    "min_eval_duration_sec": 0.0,
                    "total_total_duration_sec": 0.0,
                    "avg_total_duration_sec": 0.0,
                    "max_total_duration_sec": 0.0,
                    "min_total_duration_sec": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "avg_prompt_tokens": 0.0,
                    "max_prompt_tokens": 0.0,
                    "min_prompt_tokens": 0.0,
                    "avg_completion_tokens": 0.0,
                    "max_completion_tokens": 0.0,
                    "min_completion_tokens": 0.0,
                    "avg_total_tokens": 0.0,
                    "max_total_tokens": 0.0,
                    "min_total_tokens": 0.0,
                    "total_prompt_chars": 0,
                    "total_response_chars": 0,
                    "total_expected_answer_chars": 0,
                    "total_parsed_prediction_chars": 0,
                    "avg_prompt_chars": 0.0,
                    "avg_response_chars": 0.0,
                    "avg_expected_answer_chars": 0.0,
                    "avg_parsed_prediction_chars": 0.0,
                    "overall_tokens_per_second": 0.0,
                    "avg_tokens_per_second": 0.0,
                    "max_tokens_per_second": 0.0,
                    "min_tokens_per_second": 0.0,
                    "overall_eval_tokens_per_second": 0.0,
                    "avg_eval_tokens_per_second": 0.0,
                    "max_eval_tokens_per_second": 0.0,
                    "min_eval_tokens_per_second": 0.0,
                    "overall_prompt_tokens_per_second": 0.0,
                    "avg_prompt_tokens_per_second": 0.0,
                    "max_prompt_tokens_per_second": 0.0,
                    "min_prompt_tokens_per_second": 0.0,
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
            failure["artifact_paths"] = {"benchmark_json": portable_path(failure_file)}
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
        total_success = 0
        total_responded = 0
        total_valid_predictions = 0
        total_invalid_predictions = 0
        total_raw_fallback_count = 0
        total_raw_fallback_attempted_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_latency_sec = 0.0
        total_load_duration = 0.0
        total_prompt_eval_duration = 0.0
        total_eval_duration = 0.0
        total_provider_duration = 0.0
        total_prompt_chars = 0
        total_response_chars = 0

        for result in completed:
            metrics = result.get("metrics", {})
            model_name = result["model"]["name"]
            sample_count = int(metrics.get("sample_count", 0))
            evaluation_total_latency = float(metrics.get("total_latency_sec", 0.0)) or (
                float(metrics.get("avg_latency_sec", 0.0)) * sample_count
            )
            evaluation_total_load_duration = float(metrics.get("total_load_duration_sec", 0.0)) or (
                float(metrics.get("avg_load_duration_sec", 0.0)) * sample_count
            )
            evaluation_total_prompt_eval_duration = float(metrics.get("total_prompt_eval_duration_sec", 0.0)) or (
                float(metrics.get("avg_prompt_eval_duration_sec", 0.0)) * sample_count
            )
            evaluation_total_eval_duration = float(metrics.get("total_eval_duration_sec", 0.0)) or (
                float(metrics.get("avg_eval_duration_sec", 0.0)) * sample_count
            )
            evaluation_total_provider_duration = float(metrics.get("total_total_duration_sec", 0.0)) or (
                float(metrics.get("avg_total_duration_sec", 0.0)) * sample_count
            )
            total_samples += int(metrics.get("sample_count", 0))
            total_correct += int(metrics.get("correct_count", 0))
            total_errors += int(metrics.get("error_count", 0))
            total_success += int(metrics.get("success_count", 0))
            total_responded += int(metrics.get("responded_count", 0))
            total_valid_predictions += int(metrics.get("valid_prediction_count", 0))
            total_invalid_predictions += int(metrics.get("invalid_prediction_count", 0))
            total_raw_fallback_count += int(metrics.get("raw_fallback_count", 0))
            total_raw_fallback_attempted_count += int(metrics.get("raw_fallback_attempted_count", 0))
            total_prompt_tokens += int(metrics.get("total_prompt_tokens", 0))
            total_completion_tokens += int(metrics.get("total_completion_tokens", 0))
            total_tokens += int(metrics.get("total_tokens", 0))
            total_latency_sec += evaluation_total_latency
            total_load_duration += evaluation_total_load_duration
            total_prompt_eval_duration += evaluation_total_prompt_eval_duration
            total_eval_duration += evaluation_total_eval_duration
            total_provider_duration += evaluation_total_provider_duration
            total_prompt_chars += int(metrics.get("total_prompt_chars", 0))
            total_response_chars += int(metrics.get("total_response_chars", 0))

            if model_name not in leaderboard:
                leaderboard[model_name] = {
                    "model_name": model_name,
                    "provider": result["model"]["provider"],
                    "size_gb": result["model"].get("size_gb", 0.0),
                    "parameters": result["model"].get("parameters", "unknown"),
                    "architecture": result["model"].get("architecture", "unknown"),
                    "license": result["model"].get("license", "unknown"),
                    "max_context": result["model"].get("max_context", 0),
                    "supports_vision": result["model"].get("supports_vision", False),
                    "model_type": result["model"].get("model_type", "text"),
                    "family": result["model"].get("family", "unknown"),
                    "quantization": result["model"].get("quantization", "unknown"),
                    "digest": result["model"].get("digest", ""),
                    "modified_at": result["model"].get("modified_at", ""),
                    "benchmarks": {},
                    "total_samples": 0,
                    "correct_count": 0,
                    "error_count": 0,
                    "success_count": 0,
                    "responded_count": 0,
                    "valid_prediction_count": 0,
                    "invalid_prediction_count": 0,
                    "raw_fallback_count": 0,
                    "raw_fallback_attempted_count": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "total_latency_sec": 0.0,
                    "total_load_duration_sec": 0.0,
                    "total_prompt_eval_duration_sec": 0.0,
                    "total_eval_duration_sec": 0.0,
                    "total_total_duration_sec": 0.0,
                    "total_prompt_chars": 0,
                    "total_response_chars": 0,
                    "total_expected_answer_chars": 0,
                    "total_parsed_prediction_chars": 0,
                    "tokens_per_second_weighted_sum": 0.0,
                    "eval_tokens_per_second_weighted_sum": 0.0,
                    "prompt_tokens_per_second_weighted_sum": 0.0,
                }

            entry = leaderboard[model_name]
            entry["benchmarks"][result["benchmark_name"]] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "sample_count": metrics.get("sample_count", 0),
                "correct_count": metrics.get("correct_count", 0),
                "success_count": metrics.get("success_count", 0),
                "responded_count": metrics.get("responded_count", 0),
                "valid_prediction_count": metrics.get("valid_prediction_count", 0),
                "invalid_prediction_count": metrics.get("invalid_prediction_count", 0),
                "invalid_prediction_rate": metrics.get("invalid_prediction_rate", 0.0),
                "raw_fallback_count": metrics.get("raw_fallback_count", 0),
                "raw_fallback_rate": metrics.get("raw_fallback_rate", 0.0),
                "raw_fallback_attempted_count": metrics.get("raw_fallback_attempted_count", 0),
                "raw_fallback_attempted_rate": metrics.get("raw_fallback_attempted_rate", 0.0),
                "total_latency_sec": metrics.get("total_latency_sec", 0.0),
                "avg_latency_sec": metrics.get("avg_latency_sec", 0.0),
                "max_latency_sec": metrics.get("max_latency_sec", 0.0),
                "min_latency_sec": metrics.get("min_latency_sec", 0.0),
                "total_load_duration_sec": metrics.get("total_load_duration_sec", 0.0),
                "avg_load_duration_sec": metrics.get("avg_load_duration_sec", 0.0),
                "total_prompt_eval_duration_sec": metrics.get("total_prompt_eval_duration_sec", 0.0),
                "avg_prompt_eval_duration_sec": metrics.get("avg_prompt_eval_duration_sec", 0.0),
                "total_eval_duration_sec": metrics.get("total_eval_duration_sec", 0.0),
                "avg_eval_duration_sec": metrics.get("avg_eval_duration_sec", 0.0),
                "total_total_duration_sec": metrics.get("total_total_duration_sec", 0.0),
                "avg_total_duration_sec": metrics.get("avg_total_duration_sec", 0.0),
                "total_prompt_tokens": metrics.get("total_prompt_tokens", 0),
                "total_completion_tokens": metrics.get("total_completion_tokens", 0),
                "total_tokens": metrics.get("total_tokens", 0),
                "avg_prompt_tokens": metrics.get("avg_prompt_tokens", 0.0),
                "avg_completion_tokens": metrics.get("avg_completion_tokens", 0.0),
                "avg_total_tokens": metrics.get("avg_total_tokens", 0.0),
                "success_rate": metrics.get("success_rate", 0.0),
                "response_rate": metrics.get("response_rate", 0.0),
                "overall_tokens_per_second": metrics.get("overall_tokens_per_second", 0.0),
                "avg_tokens_per_second": metrics.get("avg_tokens_per_second", 0.0),
                "max_tokens_per_second": metrics.get("max_tokens_per_second", 0.0),
                "min_tokens_per_second": metrics.get("min_tokens_per_second", 0.0),
                "overall_eval_tokens_per_second": metrics.get("overall_eval_tokens_per_second", 0.0),
                "avg_eval_tokens_per_second": metrics.get("avg_eval_tokens_per_second", 0.0),
                "max_eval_tokens_per_second": metrics.get("max_eval_tokens_per_second", 0.0),
                "min_eval_tokens_per_second": metrics.get("min_eval_tokens_per_second", 0.0),
                "overall_prompt_tokens_per_second": metrics.get("overall_prompt_tokens_per_second", 0.0),
                "avg_prompt_tokens_per_second": metrics.get("avg_prompt_tokens_per_second", 0.0),
                "max_prompt_tokens_per_second": metrics.get("max_prompt_tokens_per_second", 0.0),
                "min_prompt_tokens_per_second": metrics.get("min_prompt_tokens_per_second", 0.0),
                "total_prompt_chars": metrics.get("total_prompt_chars", 0),
                "total_response_chars": metrics.get("total_response_chars", 0),
                "avg_prompt_chars": metrics.get("avg_prompt_chars", 0.0),
                "avg_response_chars": metrics.get("avg_response_chars", 0.0),
            }
            entry["total_samples"] += int(metrics.get("sample_count", 0))
            entry["correct_count"] += int(metrics.get("correct_count", 0))
            entry["error_count"] += int(metrics.get("error_count", 0))
            entry["success_count"] += int(metrics.get("success_count", 0))
            entry["responded_count"] += int(metrics.get("responded_count", 0))
            entry["valid_prediction_count"] += int(metrics.get("valid_prediction_count", 0))
            entry["invalid_prediction_count"] += int(metrics.get("invalid_prediction_count", 0))
            entry["raw_fallback_count"] += int(metrics.get("raw_fallback_count", 0))
            entry["raw_fallback_attempted_count"] += int(metrics.get("raw_fallback_attempted_count", 0))
            entry["total_prompt_tokens"] += int(metrics.get("total_prompt_tokens", 0))
            entry["total_completion_tokens"] += int(metrics.get("total_completion_tokens", 0))
            entry["total_tokens"] += int(metrics.get("total_tokens", 0))
            entry["total_latency_sec"] += evaluation_total_latency
            entry["total_load_duration_sec"] += evaluation_total_load_duration
            entry["total_prompt_eval_duration_sec"] += evaluation_total_prompt_eval_duration
            entry["total_eval_duration_sec"] += evaluation_total_eval_duration
            entry["total_total_duration_sec"] += evaluation_total_provider_duration
            entry["total_prompt_chars"] += int(metrics.get("total_prompt_chars", 0))
            entry["total_response_chars"] += int(metrics.get("total_response_chars", 0))
            entry["total_expected_answer_chars"] += int(metrics.get("total_expected_answer_chars", 0))
            entry["total_parsed_prediction_chars"] += int(metrics.get("total_parsed_prediction_chars", 0))
            entry["tokens_per_second_weighted_sum"] += float(metrics.get("avg_tokens_per_second", 0.0)) * int(
                metrics.get("sample_count", 0)
            )
            entry["eval_tokens_per_second_weighted_sum"] += float(metrics.get("avg_eval_tokens_per_second", 0.0)) * int(
                metrics.get("sample_count", 0)
            )
            entry["prompt_tokens_per_second_weighted_sum"] += float(
                metrics.get("avg_prompt_tokens_per_second", 0.0)
            ) * int(metrics.get("sample_count", 0))

        leaderboard_rows: List[Dict[str, Any]] = []
        for entry in leaderboard.values():
            total_samples_for_model = entry["total_samples"]
            overall_accuracy = entry["correct_count"] / total_samples_for_model if total_samples_for_model else 0.0
            avg_latency_sec = entry["total_latency_sec"] / total_samples_for_model if total_samples_for_model else 0.0
            model_accuracy_ci_low, model_accuracy_ci_high = wilson_interval(
                entry["correct_count"],
                total_samples_for_model,
            )
            model_invalid_ci_low, model_invalid_ci_high = wilson_interval(
                entry["invalid_prediction_count"],
                total_samples_for_model,
            )
            row = {
                "model_name": entry["model_name"],
                "provider": entry["provider"],
                "size_gb": entry["size_gb"],
                "parameters": entry["parameters"],
                "architecture": entry["architecture"],
                "license": entry["license"],
                "max_context": entry["max_context"],
                "supports_vision": entry["supports_vision"],
                "model_type": entry["model_type"],
                "family": entry["family"],
                "quantization": entry["quantization"],
                "digest": entry["digest"],
                "modified_at": entry["modified_at"],
                "overall_accuracy": round(overall_accuracy, 4),
                "overall_accuracy_ci95_low": model_accuracy_ci_low,
                "overall_accuracy_ci95_high": model_accuracy_ci_high,
                "success_rate": round(entry["success_count"] / total_samples_for_model, 4) if total_samples_for_model else 0.0,
                "response_rate": round(entry["responded_count"] / total_samples_for_model, 4) if total_samples_for_model else 0.0,
                "benchmarks_run": len(entry["benchmarks"]),
                "total_samples": total_samples_for_model,
                "correct_count": entry["correct_count"],
                "error_count": entry["error_count"],
                "success_count": entry["success_count"],
                "responded_count": entry["responded_count"],
                "valid_prediction_count": entry["valid_prediction_count"],
                "invalid_prediction_count": entry["invalid_prediction_count"],
                "invalid_prediction_rate": round(entry["invalid_prediction_count"] / total_samples_for_model, 4)
                if total_samples_for_model
                else 0.0,
                "invalid_prediction_rate_ci95_low": model_invalid_ci_low,
                "invalid_prediction_rate_ci95_high": model_invalid_ci_high,
                "raw_fallback_count": entry["raw_fallback_count"],
                "raw_fallback_rate": round(entry["raw_fallback_count"] / total_samples_for_model, 4)
                if total_samples_for_model
                else 0.0,
                "raw_fallback_attempted_count": entry["raw_fallback_attempted_count"],
                "raw_fallback_attempted_rate": round(entry["raw_fallback_attempted_count"] / total_samples_for_model, 4)
                if total_samples_for_model
                else 0.0,
                "total_latency_sec": round(entry["total_latency_sec"], 4),
                "avg_latency_sec": round(avg_latency_sec, 4),
                "avg_load_duration_sec": round(entry["total_load_duration_sec"] / total_samples_for_model, 4)
                if total_samples_for_model
                else 0.0,
                "avg_prompt_eval_duration_sec": round(
                    entry["total_prompt_eval_duration_sec"] / total_samples_for_model, 4
                )
                if total_samples_for_model
                else 0.0,
                "avg_eval_duration_sec": round(entry["total_eval_duration_sec"] / total_samples_for_model, 4)
                if total_samples_for_model
                else 0.0,
                "avg_total_duration_sec": round(entry["total_total_duration_sec"] / total_samples_for_model, 4)
                if total_samples_for_model
                else 0.0,
                "total_prompt_tokens": entry["total_prompt_tokens"],
                "total_completion_tokens": entry["total_completion_tokens"],
                "total_tokens": entry["total_tokens"],
                "overall_tokens_per_second": round(
                    entry["total_completion_tokens"] / entry["total_latency_sec"], 4
                )
                if entry["total_latency_sec"] > 0
                else 0.0,
                "avg_tokens_per_second": round(
                    entry["tokens_per_second_weighted_sum"] / total_samples_for_model, 4
                )
                if total_samples_for_model
                else 0.0,
                "overall_eval_tokens_per_second": round(
                    entry["total_completion_tokens"] / entry["total_eval_duration_sec"], 4
                )
                if entry["total_eval_duration_sec"] > 0
                else 0.0,
                "avg_eval_tokens_per_second": round(
                    entry["eval_tokens_per_second_weighted_sum"] / total_samples_for_model, 4
                )
                if total_samples_for_model
                else 0.0,
                "overall_prompt_tokens_per_second": round(
                    entry["total_prompt_tokens"] / entry["total_prompt_eval_duration_sec"], 4
                )
                if entry["total_prompt_eval_duration_sec"] > 0
                else 0.0,
                "avg_prompt_tokens_per_second": round(
                    entry["prompt_tokens_per_second_weighted_sum"] / total_samples_for_model, 4
                )
                if total_samples_for_model
                else 0.0,
                "total_prompt_chars": entry["total_prompt_chars"],
                "total_response_chars": entry["total_response_chars"],
                "total_expected_answer_chars": entry["total_expected_answer_chars"],
                "total_parsed_prediction_chars": entry["total_parsed_prediction_chars"],
                "benchmarks": entry["benchmarks"],
            }
            leaderboard_rows.append(row)

        leaderboard_rows.sort(key=lambda row: (row["overall_accuracy"], -row["avg_latency_sec"]), reverse=True)
        for index, row in enumerate(leaderboard_rows, start=1):
            row["rank"] = index

        total_accuracy_ci_low, total_accuracy_ci_high = wilson_interval(total_correct, total_samples)
        total_invalid_ci_low, total_invalid_ci_high = wilson_interval(total_invalid_predictions, total_samples)
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": paths.run_id,
            "generated_at": utcnow_iso(),
            "manifest_path": portable_path(paths.manifest_path),
            "totals": {
                "models": len({result["model"]["name"] for result in benchmark_results}),
                "benchmarks": len({result["benchmark_name"] for result in benchmark_results}),
                "evaluations": len(benchmark_results),
                "completed_evaluations": len(completed),
                "failed_evaluations": len(benchmark_results) - len(completed),
                "samples": total_samples,
                "correct": total_correct,
                "accuracy": round(total_correct / total_samples, 4) if total_samples else 0.0,
                "accuracy_ci95_low": total_accuracy_ci_low,
                "accuracy_ci95_high": total_accuracy_ci_high,
                "success": total_success,
                "success_rate": round(total_success / total_samples, 4) if total_samples else 0.0,
                "responded": total_responded,
                "response_rate": round(total_responded / total_samples, 4) if total_samples else 0.0,
                "valid_predictions": total_valid_predictions,
                "invalid_predictions": total_invalid_predictions,
                "invalid_prediction_rate": round(total_invalid_predictions / total_samples, 4) if total_samples else 0.0,
                "invalid_prediction_rate_ci95_low": total_invalid_ci_low,
                "invalid_prediction_rate_ci95_high": total_invalid_ci_high,
                "raw_fallback_count": total_raw_fallback_count,
                "raw_fallback_rate": round(total_raw_fallback_count / total_samples, 4) if total_samples else 0.0,
                "raw_fallback_attempted_count": total_raw_fallback_attempted_count,
                "raw_fallback_attempted_rate": round(total_raw_fallback_attempted_count / total_samples, 4)
                if total_samples
                else 0.0,
                "errors": total_errors,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "avg_latency_sec": round(total_latency_sec / total_samples, 4) if total_samples else 0.0,
                "total_latency_sec": round(total_latency_sec, 4),
                "total_duration_sec": round(total_latency_sec, 4),
                "avg_load_duration_sec": round(total_load_duration / total_samples, 4) if total_samples else 0.0,
                "avg_prompt_eval_duration_sec": round(total_prompt_eval_duration / total_samples, 4)
                if total_samples
                else 0.0,
                "avg_eval_duration_sec": round(total_eval_duration / total_samples, 4) if total_samples else 0.0,
                "total_load_duration_sec": round(total_load_duration, 4),
                "total_prompt_eval_duration_sec": round(total_prompt_eval_duration, 4),
                "total_eval_duration_sec": round(total_eval_duration, 4),
                "total_provider_duration_sec": round(total_provider_duration, 4),
                "overall_tokens_per_second": round(total_completion_tokens / total_latency_sec, 4)
                if total_latency_sec > 0
                else 0.0,
                "overall_eval_tokens_per_second": round(total_completion_tokens / total_eval_duration, 4)
                if total_eval_duration > 0
                else 0.0,
                "overall_prompt_tokens_per_second": round(total_prompt_tokens / total_prompt_eval_duration, 4)
                if total_prompt_eval_duration > 0
                else 0.0,
                "total_prompt_chars": total_prompt_chars,
                "total_response_chars": total_response_chars,
            },
            "leaderboard": leaderboard_rows,
            "evaluations": benchmark_results,
        }


def run_local_benchmarks_sync(**kwargs: Any) -> Dict[str, Any]:
    """Synchronous helper for CLI entrypoints."""
    orchestrator = LocalBenchmarkOrchestrator(config_path=kwargs.pop("config_path", DEFAULT_CONFIG_PATH))
    return asyncio.run(orchestrator.run(**kwargs))
