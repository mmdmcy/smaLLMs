"""
Website export generation for structured benchmark artifacts.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.artifacts import ArtifactStore, portable_path, safe_slug, sanitize_system_metadata, utcnow_iso
from src.pipeline.benchmarks import summarize_samples
from src.pipeline.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_WEBSITE_EXPORT_DIR,
    WEBSITE_EXPORT_SCHEMA_VERSION,
)


class WebsiteExporter:
    """Export benchmark runs to website-friendly bundles."""

    def __init__(
        self,
        artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
        output_dir: str = DEFAULT_WEBSITE_EXPORT_DIR,
        sync_dir: Optional[str] = None,
    ):
        self.artifacts = ArtifactStore(artifacts_dir)
        self.output_dir = Path(output_dir)
        self.repo_root = self.artifacts.base_dir.resolve().parent
        self.sync_dir = self._resolve_sync_dir(sync_dir)

    def export_run(self, run_id: Optional[str] = None) -> Dict[str, str]:
        """Export one run to a stable website bundle."""
        run_data = self.artifacts.load_run(run_id)
        run_id = run_data["run_id"]
        sanitized_manifest = self._sanitize_manifest(run_data["manifest"])
        schema_version = WEBSITE_EXPORT_SCHEMA_VERSION

        latest_dir = self.output_dir / "latest"
        run_dir = self.output_dir / "runs" / run_id
        latest_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = run_data.get("summary", {})
        evaluations = self._build_evaluations_bundle(summary.get("evaluations", []))
        leaderboard = self._enrich_leaderboard(summary.get("leaderboard", []), evaluations)
        session_summary = self._make_session_summary(summary, evaluations)
        exported_summary = dict(summary)
        exported_summary["manifest_path"] = portable_path(exported_summary.get("manifest_path", "")) if exported_summary.get("manifest_path") else None
        exported_summary["evaluations"] = [self._sanitize_evaluation_record(item) for item in summary.get("evaluations", [])]
        exported_summary["leaderboard"] = leaderboard
        exported_summary["totals"] = session_summary.get("totals", {})
        evaluation_briefs = [self._make_evaluation_brief(evaluation) for evaluation in evaluations]
        models_bundle = self._build_models_bundle(leaderboard, evaluation_briefs)
        benchmarks_bundle = self._build_benchmarks_bundle(evaluation_briefs)
        catalog_bundle = self._build_catalog_bundle(run_data["manifest"])

        session_payload = {
            "schema_version": schema_version,
            "exported_at": utcnow_iso(),
            "source": {
                "artifacts_dir": portable_path(self.artifacts.base_dir),
                "output_dir": portable_path(self.output_dir),
                "sync_dir": portable_path(self.sync_dir) if self.sync_dir else None,
                "payload_profile": "full",
                "sample_payload_mode": "complete",
            },
            "run": {
                "run_id": run_id,
                "run_dir": portable_path(run_data["run_dir"]),
                "manifest": sanitized_manifest,
            },
            "summary": session_summary,
            "catalog": catalog_bundle,
            "leaderboard": leaderboard,
            "models": models_bundle,
            "benchmarks": benchmarks_bundle,
            "evaluations": evaluations,
        }

        exported_files: Dict[str, str] = {}

        files_to_write = {
            "manifest.json": sanitized_manifest,
            "summary.json": exported_summary,
            "leaderboard.json": leaderboard,
            "models.json": models_bundle,
            "benchmarks.json": benchmarks_bundle,
            "session.json": session_payload,
            f"run_{run_id}.json": session_payload,
        }

        for filename, payload in files_to_write.items():
            latest_path = latest_dir / filename
            run_path = run_dir / filename
            self._write_json(latest_path, payload)
            self._write_json(run_path, payload)
            exported_files[filename] = portable_path(latest_path)

        csv_path = latest_dir / "leaderboard.csv"
        self._write_leaderboard_csv(csv_path, leaderboard)
        self._write_leaderboard_csv(run_dir / "leaderboard.csv", leaderboard)
        exported_files["leaderboard.csv"] = portable_path(csv_path)

        html_path = latest_dir / "index.html"
        self._write_html_preview(html_path, session_summary, leaderboard)
        self._write_html_preview(run_dir / "index.html", session_summary, leaderboard)
        exported_files["index.html"] = portable_path(html_path)

        latest_pointer = self.output_dir / "latest_run.txt"
        latest_pointer.write_text(run_id + "\n", encoding="utf-8")

        if self.sync_dir is not None:
            exported_files.update(
                self._sync_session_bundle(
                    run_id,
                    self._build_sync_session_payload(session_payload),
                    session_summary,
                    full_payload_size_bytes=self._payload_size_bytes(session_payload),
                )
            )

        return exported_files

    def _resolve_sync_dir(self, sync_dir: Optional[str]) -> Optional[Path]:
        """Resolve the optional website sync directory."""
        if sync_dir:
            return Path(sync_dir)

        candidate = self.repo_root.parent / "websmaLLMs" / "public" / "data"
        website_root = candidate.parent.parent
        if website_root.exists():
            return candidate
        return None

    def _sanitize_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Drop host-identifying fields and normalize any path-like values."""
        sanitized = dict(manifest)
        if "system" in sanitized and isinstance(sanitized["system"], dict):
            sanitized["system"] = sanitize_system_metadata(sanitized["system"])
        if sanitized.get("config_path"):
            sanitized["config_path"] = portable_path(sanitized["config_path"])
        return sanitized

    def _sanitize_artifact_paths(self, artifact_paths: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize evaluation artifact paths for export."""
        sanitized: Dict[str, Any] = {}
        for key, value in (artifact_paths or {}).items():
            sanitized[key] = portable_path(value) if isinstance(value, str) and value else value
        return sanitized

    def _sanitize_evaluation_record(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize one evaluation record before writing exported JSON."""
        sanitized = dict(evaluation)
        sanitized["artifact_paths"] = self._sanitize_artifact_paths(evaluation.get("artifact_paths", {}))
        return sanitized

    def _build_evaluations_bundle(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build full evaluation records with embedded samples."""
        bundled: List[Dict[str, Any]] = []

        for evaluation in evaluations:
            model_name = evaluation.get("model", {}).get("name", "unknown")
            benchmark_name = evaluation.get("benchmark_name", "unknown")
            evaluation_id = f"{safe_slug(benchmark_name)}__{safe_slug(model_name)}"
            sample_path = evaluation.get("artifact_paths", {}).get("samples_jsonl")
            samples = self._load_samples(sample_path=sample_path, evaluation_id=evaluation_id)

            bundled_evaluation = self._sanitize_evaluation_record(evaluation)
            bundled_evaluation["evaluation_id"] = evaluation_id
            bundled_evaluation["samples"] = samples
            bundled_evaluation["sample_count_embedded"] = len(samples)
            if samples:
                bundled_evaluation["metrics"] = summarize_samples(samples)
            bundled.append(bundled_evaluation)

        return bundled

    def _load_samples(self, sample_path: Optional[str], evaluation_id: str) -> List[Dict[str, Any]]:
        """Load per-sample JSONL data for one evaluation."""
        resolved_path = self._resolve_artifact_path(sample_path)
        if resolved_path is None or not resolved_path.exists():
            return []

        samples: List[Dict[str, Any]] = []
        with open(resolved_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue

                sample = json.loads(stripped)
                sample_index = sample.get("sample_index", line_number)
                sample["prompt_chars"] = int(sample.get("prompt_chars") or len(str(sample.get("prompt") or "")))
                sample["response_chars"] = int(sample.get("response_chars") or len(str(sample.get("response_text") or "")))
                sample["expected_answer_chars"] = int(
                    sample.get("expected_answer_chars") or len(str(sample.get("expected_answer") or ""))
                )
                sample["parsed_prediction_chars"] = int(
                    sample.get("parsed_prediction_chars") or len(str(sample.get("parsed_prediction") or ""))
                )
                sample["evaluation_id"] = evaluation_id
                sample["sample_id"] = f"{evaluation_id}::{sample_index}"
                samples.append(sample)

        return samples

    def _resolve_artifact_path(self, raw_path: Optional[str]) -> Optional[Path]:
        """Resolve an artifact path relative to the repo root when needed."""
        if not raw_path:
            return None

        normalized = raw_path.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute():
            return path

        direct = path.resolve()
        if direct.exists():
            return direct

        repo_relative = (self.repo_root / path).resolve()
        if repo_relative.exists():
            return repo_relative

        return repo_relative

    def _make_evaluation_brief(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Strip sample-heavy data out of an evaluation for secondary bundles."""
        return {
            "evaluation_id": evaluation.get("evaluation_id"),
            "benchmark_name": evaluation.get("benchmark_name"),
            "benchmark_display_name": evaluation.get("benchmark_display_name"),
            "description": evaluation.get("description", ""),
            "dataset": evaluation.get("dataset", {}),
            "model": evaluation.get("model", {}),
            "metrics": evaluation.get("metrics", {}),
            "status": evaluation.get("status"),
            "error": evaluation.get("error"),
            "artifact_paths": self._sanitize_artifact_paths(evaluation.get("artifact_paths", {})),
            "sample_count_embedded": evaluation.get("sample_count_embedded", 0),
        }

    def _build_models_bundle(self, leaderboard: List[Dict[str, Any]], evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build per-model website bundles without duplicating sample payloads."""
        evaluation_map: Dict[str, List[Dict[str, Any]]] = {}
        for evaluation in evaluations:
            model_name = evaluation.get("model", {}).get("name", "unknown")
            evaluation_map.setdefault(model_name, []).append(evaluation)

        models_bundle: List[Dict[str, Any]] = []
        for row in leaderboard:
            model_name = row["model_name"]
            model_evaluations = sorted(
                evaluation_map.get(model_name, []),
                key=lambda item: str(item.get("benchmark_name", "")),
            )
            models_bundle.append(
                {
                    "model_name": model_name,
                    "slug": safe_slug(model_name),
                    "leaderboard": row,
                    "evaluation_ids": [item.get("evaluation_id") for item in model_evaluations],
                    "evaluations": model_evaluations,
                }
            )

        return models_bundle

    def _build_benchmarks_bundle(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build per-benchmark website bundles without duplicating sample payloads."""
        grouped: Dict[str, Dict[str, Any]] = {}
        for evaluation in evaluations:
            benchmark_name = str(evaluation.get("benchmark_name", "unknown"))
            grouped.setdefault(
                benchmark_name,
                {
                    "benchmark_name": benchmark_name,
                    "display_name": evaluation.get("benchmark_display_name", benchmark_name),
                    "description": evaluation.get("description", ""),
                    "dataset": evaluation.get("dataset", {}),
                    "results": [],
                },
            )
            grouped[benchmark_name]["results"].append(evaluation)

        bundles = list(grouped.values())
        for bundle in bundles:
            bundle["results"].sort(
                key=lambda item: (
                    float(item.get("metrics", {}).get("accuracy", 0.0)),
                    -float(item.get("metrics", {}).get("avg_latency_sec", 0.0)),
                ),
                reverse=True,
            )

        bundles.sort(key=lambda item: item["benchmark_name"])
        return bundles

    def _build_catalog_bundle(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Expose selected benchmark metadata for the website."""
        selected = set(manifest.get("benchmarks", []))
        supported = [
            benchmark
            for benchmark in manifest.get("supported_benchmarks", [])
            if benchmark.get("key") in selected
        ]

        return {
            "selected_benchmarks": supported,
            "benchmark_suites": manifest.get("benchmark_suites", []),
        }

    def _enrich_leaderboard(
        self,
        leaderboard: List[Dict[str, Any]],
        evaluations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rebuild leaderboard rows from evaluation metrics so legacy runs stay rich."""
        existing_by_model = {row.get("model_name"): dict(row) for row in leaderboard}
        rebuilt: Dict[str, Dict[str, Any]] = {}

        for evaluation in evaluations:
            model = evaluation.get("model", {})
            metrics = evaluation.get("metrics", {})
            model_name = str(model.get("name", "unknown"))
            sample_count = int(metrics.get("sample_count", 0))
            entry = rebuilt.setdefault(
                model_name,
                {
                    **existing_by_model.get(model_name, {}),
                    "model_name": model_name,
                    "provider": model.get("provider", existing_by_model.get(model_name, {}).get("provider", "unknown")),
                    "size_gb": model.get("size_gb", existing_by_model.get(model_name, {}).get("size_gb", 0.0)),
                    "parameters": model.get("parameters", existing_by_model.get(model_name, {}).get("parameters", "unknown")),
                    "architecture": model.get("architecture", existing_by_model.get(model_name, {}).get("architecture", "unknown")),
                    "license": model.get("license", existing_by_model.get(model_name, {}).get("license", "unknown")),
                    "max_context": model.get("max_context", existing_by_model.get(model_name, {}).get("max_context", 0)),
                    "supports_vision": model.get(
                        "supports_vision", existing_by_model.get(model_name, {}).get("supports_vision", False)
                    ),
                    "model_type": model.get("model_type", existing_by_model.get(model_name, {}).get("model_type", "text")),
                    "family": model.get("family", existing_by_model.get(model_name, {}).get("family", "unknown")),
                    "quantization": model.get(
                        "quantization", existing_by_model.get(model_name, {}).get("quantization", "unknown")
                    ),
                    "benchmarks": {},
                    "benchmarks_run": 0,
                    "total_samples": 0,
                    "correct_count": 0,
                    "success_count": 0,
                    "responded_count": 0,
                    "error_count": 0,
                    "raw_fallback_count": 0,
                    "raw_fallback_attempted_count": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "total_prompt_chars": 0,
                    "total_response_chars": 0,
                    "total_expected_answer_chars": 0,
                    "total_parsed_prediction_chars": 0,
                    "total_latency_sec": 0.0,
                    "total_load_duration_sec": 0.0,
                    "total_prompt_eval_duration_sec": 0.0,
                    "total_eval_duration_sec": 0.0,
                    "total_total_duration_sec": 0.0,
                    "_tps_weighted_sum": 0.0,
                    "_eval_tps_weighted_sum": 0.0,
                    "_prompt_tps_weighted_sum": 0.0,
                },
            )

            entry["benchmarks"][str(evaluation.get("benchmark_name", "unknown"))] = dict(metrics)
            entry["benchmarks_run"] = len(entry["benchmarks"])
            entry["total_samples"] += sample_count
            entry["correct_count"] += int(metrics.get("correct_count", 0))
            entry["success_count"] += int(metrics.get("success_count", 0))
            entry["responded_count"] += int(metrics.get("responded_count", 0))
            entry["error_count"] += int(metrics.get("error_count", 0))
            entry["raw_fallback_count"] += int(metrics.get("raw_fallback_count", 0))
            entry["raw_fallback_attempted_count"] += int(metrics.get("raw_fallback_attempted_count", 0))
            entry["total_prompt_tokens"] += int(metrics.get("total_prompt_tokens", 0))
            entry["total_completion_tokens"] += int(metrics.get("total_completion_tokens", 0))
            entry["total_tokens"] += int(metrics.get("total_tokens", 0))
            entry["total_prompt_chars"] += int(metrics.get("total_prompt_chars", 0))
            entry["total_response_chars"] += int(metrics.get("total_response_chars", 0))
            entry["total_expected_answer_chars"] += int(metrics.get("total_expected_answer_chars", 0))
            entry["total_parsed_prediction_chars"] += int(metrics.get("total_parsed_prediction_chars", 0))
            entry["total_latency_sec"] += float(metrics.get("total_latency_sec", 0.0)) or (
                float(metrics.get("avg_latency_sec", 0.0)) * sample_count
            )
            entry["total_load_duration_sec"] += float(metrics.get("total_load_duration_sec", 0.0)) or (
                float(metrics.get("avg_load_duration_sec", 0.0)) * sample_count
            )
            entry["total_prompt_eval_duration_sec"] += float(metrics.get("total_prompt_eval_duration_sec", 0.0)) or (
                float(metrics.get("avg_prompt_eval_duration_sec", 0.0)) * sample_count
            )
            entry["total_eval_duration_sec"] += float(metrics.get("total_eval_duration_sec", 0.0)) or (
                float(metrics.get("avg_eval_duration_sec", 0.0)) * sample_count
            )
            entry["total_total_duration_sec"] += float(metrics.get("total_total_duration_sec", 0.0)) or (
                float(metrics.get("avg_total_duration_sec", 0.0)) * sample_count
            )
            entry["_tps_weighted_sum"] += float(metrics.get("avg_tokens_per_second", 0.0)) * sample_count
            entry["_eval_tps_weighted_sum"] += float(metrics.get("avg_eval_tokens_per_second", 0.0)) * sample_count
            entry["_prompt_tps_weighted_sum"] += float(metrics.get("avg_prompt_tokens_per_second", 0.0)) * sample_count

        rows: List[Dict[str, Any]] = []
        for entry in rebuilt.values():
            sample_count = int(entry.get("total_samples", 0))
            row = dict(entry)
            row["overall_accuracy"] = round(row["correct_count"] / sample_count, 4) if sample_count else 0.0
            row["success_rate"] = round(row["success_count"] / sample_count, 4) if sample_count else 0.0
            row["response_rate"] = round(row["responded_count"] / sample_count, 4) if sample_count else 0.0
            row["raw_fallback_rate"] = round(row["raw_fallback_count"] / sample_count, 4) if sample_count else 0.0
            row["raw_fallback_attempted_rate"] = (
                round(row["raw_fallback_attempted_count"] / sample_count, 4) if sample_count else 0.0
            )
            row["avg_latency_sec"] = round(row["total_latency_sec"] / sample_count, 4) if sample_count else 0.0
            row["avg_load_duration_sec"] = (
                round(row["total_load_duration_sec"] / sample_count, 4) if sample_count else 0.0
            )
            row["avg_prompt_eval_duration_sec"] = (
                round(row["total_prompt_eval_duration_sec"] / sample_count, 4) if sample_count else 0.0
            )
            row["avg_eval_duration_sec"] = (
                round(row["total_eval_duration_sec"] / sample_count, 4) if sample_count else 0.0
            )
            row["avg_total_duration_sec"] = (
                round(row["total_total_duration_sec"] / sample_count, 4) if sample_count else 0.0
            )
            row["overall_tokens_per_second"] = (
                round(row["total_completion_tokens"] / row["total_latency_sec"], 4) if row["total_latency_sec"] else 0.0
            )
            row["avg_tokens_per_second"] = (
                round(row.pop("_tps_weighted_sum", 0.0) / sample_count, 4) if sample_count else 0.0
            )
            row["overall_eval_tokens_per_second"] = (
                round(row["total_completion_tokens"] / row["total_eval_duration_sec"], 4)
                if row["total_eval_duration_sec"]
                else 0.0
            )
            row["avg_eval_tokens_per_second"] = (
                round(row.pop("_eval_tps_weighted_sum", 0.0) / sample_count, 4) if sample_count else 0.0
            )
            row["overall_prompt_tokens_per_second"] = (
                round(row["total_prompt_tokens"] / row["total_prompt_eval_duration_sec"], 4)
                if row["total_prompt_eval_duration_sec"]
                else 0.0
            )
            row["avg_prompt_tokens_per_second"] = (
                round(row.pop("_prompt_tps_weighted_sum", 0.0) / sample_count, 4) if sample_count else 0.0
            )
            rows.append(row)

        rows.sort(key=lambda item: (float(item.get("overall_accuracy", 0.0)), -float(item.get("avg_latency_sec", 0.0))), reverse=True)
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def _make_session_summary(self, summary: Dict[str, Any], evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trim the full summary down for the website session bundle."""
        totals = dict(summary.get("totals", {}))
        if evaluations:
            sample_count = sum(int(item.get("metrics", {}).get("sample_count", 0)) for item in evaluations)
            correct_count = sum(int(item.get("metrics", {}).get("correct_count", 0)) for item in evaluations)
            success_count = sum(int(item.get("metrics", {}).get("success_count", 0)) for item in evaluations)
            responded_count = sum(int(item.get("metrics", {}).get("responded_count", 0)) for item in evaluations)
            error_count = sum(int(item.get("metrics", {}).get("error_count", 0)) for item in evaluations)
            raw_fallback_count = sum(int(item.get("metrics", {}).get("raw_fallback_count", 0)) for item in evaluations)
            raw_fallback_attempted_count = sum(
                int(item.get("metrics", {}).get("raw_fallback_attempted_count", 0)) for item in evaluations
            )
            total_prompt_tokens = sum(int(item.get("metrics", {}).get("total_prompt_tokens", 0)) for item in evaluations)
            total_completion_tokens = sum(
                int(item.get("metrics", {}).get("total_completion_tokens", 0)) for item in evaluations
            )
            total_tokens = sum(int(item.get("metrics", {}).get("total_tokens", 0)) for item in evaluations)
            total_prompt_chars = sum(int(item.get("metrics", {}).get("total_prompt_chars", 0)) for item in evaluations)
            total_response_chars = sum(int(item.get("metrics", {}).get("total_response_chars", 0)) for item in evaluations)
            total_latency_sec = sum(
                float(item.get("metrics", {}).get("total_latency_sec", 0.0))
                or (
                    float(item.get("metrics", {}).get("avg_latency_sec", 0.0))
                    * int(item.get("metrics", {}).get("sample_count", 0))
                )
                for item in evaluations
            )
            total_load_duration_sec = sum(
                float(item.get("metrics", {}).get("total_load_duration_sec", 0.0))
                or (
                    float(item.get("metrics", {}).get("avg_load_duration_sec", 0.0))
                    * int(item.get("metrics", {}).get("sample_count", 0))
                )
                for item in evaluations
            )
            total_prompt_eval_duration_sec = sum(
                float(item.get("metrics", {}).get("total_prompt_eval_duration_sec", 0.0))
                or (
                    float(item.get("metrics", {}).get("avg_prompt_eval_duration_sec", 0.0))
                    * int(item.get("metrics", {}).get("sample_count", 0))
                )
                for item in evaluations
            )
            total_eval_duration_sec = sum(
                float(item.get("metrics", {}).get("total_eval_duration_sec", 0.0))
                or (
                    float(item.get("metrics", {}).get("avg_eval_duration_sec", 0.0))
                    * int(item.get("metrics", {}).get("sample_count", 0))
                )
                for item in evaluations
            )

            totals.update(
                {
                    "models": len({item.get("model", {}).get("name") for item in evaluations}),
                    "benchmarks": len({item.get("benchmark_name") for item in evaluations}),
                    "evaluations": len(evaluations),
                    "completed_evaluations": len([item for item in evaluations if item.get("status") == "completed"]),
                    "failed_evaluations": len([item for item in evaluations if item.get("status") != "completed"]),
                    "samples": sample_count,
                    "correct": correct_count,
                    "accuracy": round(correct_count / sample_count, 4) if sample_count else 0.0,
                    "success": success_count,
                    "success_rate": round(success_count / sample_count, 4) if sample_count else 0.0,
                    "responded": responded_count,
                    "response_rate": round(responded_count / sample_count, 4) if sample_count else 0.0,
                    "raw_fallback_count": raw_fallback_count,
                    "raw_fallback_rate": round(raw_fallback_count / sample_count, 4) if sample_count else 0.0,
                    "raw_fallback_attempted_count": raw_fallback_attempted_count,
                    "raw_fallback_attempted_rate": round(raw_fallback_attempted_count / sample_count, 4)
                    if sample_count
                    else 0.0,
                    "errors": error_count,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                    "avg_latency_sec": round(total_latency_sec / sample_count, 4) if sample_count else 0.0,
                    "total_latency_sec": round(total_latency_sec, 4),
                    "total_duration_sec": round(total_latency_sec, 4),
                    "avg_load_duration_sec": round(total_load_duration_sec / sample_count, 4) if sample_count else 0.0,
                    "total_load_duration_sec": round(total_load_duration_sec, 4),
                    "avg_prompt_eval_duration_sec": round(total_prompt_eval_duration_sec / sample_count, 4)
                    if sample_count
                    else 0.0,
                    "total_prompt_eval_duration_sec": round(total_prompt_eval_duration_sec, 4),
                    "avg_eval_duration_sec": round(total_eval_duration_sec / sample_count, 4) if sample_count else 0.0,
                    "total_eval_duration_sec": round(total_eval_duration_sec, 4),
                    "overall_tokens_per_second": round(total_completion_tokens / total_latency_sec, 4)
                    if total_latency_sec > 0
                    else 0.0,
                    "overall_eval_tokens_per_second": round(total_completion_tokens / total_eval_duration_sec, 4)
                    if total_eval_duration_sec > 0
                    else 0.0,
                    "overall_prompt_tokens_per_second": round(total_prompt_tokens / total_prompt_eval_duration_sec, 4)
                    if total_prompt_eval_duration_sec > 0
                    else 0.0,
                    "total_prompt_chars": total_prompt_chars,
                    "total_response_chars": total_response_chars,
                }
            )

        return {
            "run_id": summary.get("run_id"),
            "generated_at": summary.get("generated_at"),
            "manifest_path": portable_path(summary.get("manifest_path")) if summary.get("manifest_path") else None,
            "totals": totals,
        }

    def _build_sync_session_payload(self, session_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compact website payload while keeping full local exports intact."""
        compact_payload = deepcopy(session_payload)
        source = compact_payload.get("source", {})
        if isinstance(source, dict):
            source["payload_profile"] = "compact"
            source["sample_payload_mode"] = "trimmed_raw_metrics"

        for evaluation in compact_payload.get("evaluations", []):
            if not isinstance(evaluation, dict):
                continue
            for sample in evaluation.get("samples", []):
                if not isinstance(sample, dict):
                    continue
                sample["raw_provider_metrics"] = self._compact_raw_provider_metrics(sample.get("raw_provider_metrics"))

        return compact_payload

    def _compact_raw_provider_metrics(self, raw_metrics: Any) -> Any:
        """Keep only the raw-provider fields that help interpret website results."""
        if not isinstance(raw_metrics, dict):
            return raw_metrics

        keep_keys = [
            "model",
            "created_at",
            "done",
            "done_reason",
            "error",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "attempt_count",
            "transport",
            "raw_fallback_attempted",
            "used_raw_fallback",
        ]
        attempt_keep_keys = [
            "transport",
            "raw_mode",
            "error",
            "response_chars",
            "thinking_chars",
            "prompt_eval_count",
            "eval_count",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
            "total_duration",
        ]

        compact = {key: raw_metrics[key] for key in keep_keys if key in raw_metrics}
        attempts = raw_metrics.get("attempts")
        if isinstance(attempts, list):
            compact_attempts = []
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                compact_attempt = {key: attempt[key] for key in attempt_keep_keys if key in attempt}
                if compact_attempt:
                    compact_attempts.append(compact_attempt)
            if compact_attempts:
                compact["attempts"] = compact_attempts

        if compact:
            compact["trimmed"] = True
            return compact
        return None

    def _payload_size_bytes(self, payload: Any) -> int:
        """Return the UTF-8 payload size for compact JSON serialization."""
        return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def _sync_session_bundle(
        self,
        run_id: str,
        session_payload: Dict[str, Any],
        summary: Dict[str, Any],
        full_payload_size_bytes: Optional[int] = None,
    ) -> Dict[str, str]:
        """Mirror the latest session bundle into the website repo."""
        if self.sync_dir is None:
            return {}

        latest_path = self.sync_dir / "latest-session.json"
        meta_path = self.sync_dir / "latest-session.meta.json"
        history_path = self.sync_dir / "runs" / f"{run_id}.json"
        latest_pointer = self.sync_dir / "latest-run.txt"
        payload_size_bytes = self._payload_size_bytes(session_payload)

        self._write_json(latest_path, session_payload, compact=True)
        self._write_json(
            meta_path,
            {
                "schema_version": session_payload.get("schema_version", WEBSITE_EXPORT_SCHEMA_VERSION),
                "run_id": run_id,
                "updated_at": utcnow_iso(),
                "totals": summary.get("totals", {}),
                "payload_profile": session_payload.get("source", {}).get("payload_profile"),
                "sample_payload_mode": session_payload.get("source", {}).get("sample_payload_mode"),
                "payload_size_bytes": payload_size_bytes,
                "payload_size_mb": round(payload_size_bytes / (1024 * 1024), 2),
                "full_payload_size_bytes": full_payload_size_bytes,
                "full_payload_size_mb": round(full_payload_size_bytes / (1024 * 1024), 2)
                if full_payload_size_bytes is not None
                else None,
            },
            compact=True,
        )
        self._write_json(history_path, session_payload, compact=True)
        latest_pointer.parent.mkdir(parents=True, exist_ok=True)
        latest_pointer.write_text(run_id + "\n", encoding="utf-8")

        return {
            "sync/latest-session.json": portable_path(latest_path),
            "sync/latest-session.meta.json": portable_path(meta_path),
            f"sync/runs/{run_id}.json": portable_path(history_path),
        }

    def _write_json(self, path: Path, payload: Any, compact: bool = False) -> None:
        """Write a JSON payload to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            if compact:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _write_leaderboard_csv(self, path: Path, leaderboard: List[Dict[str, Any]]) -> None:
        """Write a flattened leaderboard CSV."""
        benchmark_names = sorted(
            {
                benchmark_name
                for row in leaderboard
                for benchmark_name in row.get("benchmarks", {})
            }
        )

        fieldnames = [
            "rank",
            "model_name",
            "provider",
            "size_gb",
            "parameters",
            "architecture",
            "license",
            "max_context",
            "supports_vision",
            "model_type",
            "family",
            "quantization",
            "overall_accuracy",
            "success_rate",
            "response_rate",
            "benchmarks_run",
            "total_samples",
            "correct_count",
            "error_count",
            "raw_fallback_rate",
            "avg_latency_sec",
            "avg_load_duration_sec",
            "avg_prompt_eval_duration_sec",
            "avg_eval_duration_sec",
            "overall_tokens_per_second",
            "avg_tokens_per_second",
            "overall_eval_tokens_per_second",
            "avg_eval_tokens_per_second",
            "overall_prompt_tokens_per_second",
            "avg_prompt_tokens_per_second",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
        ] + [f"{name}_accuracy" for name in benchmark_names]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in leaderboard:
                output = {key: row.get(key) for key in fieldnames if key in row}
                for benchmark_name in benchmark_names:
                    output[f"{benchmark_name}_accuracy"] = row.get("benchmarks", {}).get(benchmark_name, {}).get("accuracy")
                writer.writerow(output)

    def _write_html_preview(self, path: Path, summary: Dict[str, Any], leaderboard: List[Dict[str, Any]]) -> None:
        """Write a lightweight HTML preview for manual inspection."""
        rows = []
        for row in leaderboard:
            rows.append(
                "<tr>"
                f"<td>{row.get('rank', '')}</td>"
                f"<td>{row['model_name']}</td>"
                f"<td>{row['provider']}</td>"
                f"<td>{row['overall_accuracy']:.4f}</td>"
                f"<td>{row['benchmarks_run']}</td>"
                f"<td>{row['avg_latency_sec']:.4f}</td>"
                f"<td>{row['total_samples']}</td>"
                f"<td>{row['error_count']}</td>"
                "</tr>"
            )

        totals = summary.get("totals", {})
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>smaLLMs Local Benchmark Results</title>
  <style>
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      margin: 32px;
      background: #f7f8fa;
      color: #15171a;
    }}
    h1, h2 {{
      margin-bottom: 0.25rem;
    }}
    .meta {{
      color: #59636e;
      margin-bottom: 1.5rem;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d8dde3;
      border-radius: 10px;
      padding: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid #d8dde3;
      border-radius: 10px;
      overflow: hidden;
    }}
    th, td {{
      text-align: left;
      padding: 12px;
      border-bottom: 1px solid #e7ebef;
    }}
    th {{
      background: #0f1720;
      color: #fff;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <h1>smaLLMs Local Benchmark Results</h1>
  <p class="meta">Run ID: {summary.get('run_id', '')}</p>

  <div class="stats">
    <div class="card"><strong>Models</strong><br />{totals.get('models', 0)}</div>
    <div class="card"><strong>Benchmarks</strong><br />{totals.get('benchmarks', 0)}</div>
    <div class="card"><strong>Evaluations</strong><br />{totals.get('evaluations', 0)}</div>
    <div class="card"><strong>Samples</strong><br />{totals.get('samples', 0)}</div>
    <div class="card"><strong>Accuracy</strong><br />{totals.get('accuracy', 0.0):.4f}</div>
    <div class="card"><strong>Total tokens</strong><br />{totals.get('total_tokens', 0)}</div>
    <div class="card"><strong>Failed evals</strong><br />{totals.get('failed_evaluations', 0)}</div>
    <div class="card"><strong>Duration (s)</strong><br />{totals.get('total_duration_sec', 0.0):.2f}</div>
  </div>

  <h2>Leaderboard</h2>
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Model</th>
        <th>Provider</th>
        <th>Accuracy</th>
        <th>Benchmarks</th>
        <th>Avg latency (s)</th>
        <th>Samples</th>
        <th>Errors</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
