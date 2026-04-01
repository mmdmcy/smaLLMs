"""
Website export generation for structured benchmark artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.artifacts import ArtifactStore, safe_slug, utcnow_iso


class WebsiteExporter:
    """Export benchmark runs to website-friendly bundles."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        output_dir: str = "website_exports",
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

        latest_dir = self.output_dir / "latest"
        run_dir = self.output_dir / "runs" / run_id
        latest_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = run_data.get("summary", {})
        leaderboard = summary.get("leaderboard", [])
        evaluations = self._build_evaluations_bundle(summary.get("evaluations", []))
        evaluation_briefs = [self._make_evaluation_brief(evaluation) for evaluation in evaluations]
        models_bundle = self._build_models_bundle(leaderboard, evaluation_briefs)
        benchmarks_bundle = self._build_benchmarks_bundle(evaluation_briefs)
        catalog_bundle = self._build_catalog_bundle(run_data["manifest"])

        session_payload = {
            "schema_version": "3.0",
            "exported_at": utcnow_iso(),
            "source": {
                "artifacts_dir": str(self.artifacts.base_dir),
                "output_dir": str(self.output_dir),
                "sync_dir": str(self.sync_dir) if self.sync_dir else None,
            },
            "run": {
                "run_id": run_id,
                "run_dir": run_data["run_dir"],
                "manifest": run_data["manifest"],
            },
            "summary": self._make_session_summary(summary),
            "catalog": catalog_bundle,
            "leaderboard": leaderboard,
            "models": models_bundle,
            "benchmarks": benchmarks_bundle,
            "evaluations": evaluations,
        }

        exported_files: Dict[str, str] = {}

        files_to_write = {
            "manifest.json": run_data["manifest"],
            "summary.json": summary,
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
            exported_files[filename] = str(latest_path)

        csv_path = latest_dir / "leaderboard.csv"
        self._write_leaderboard_csv(csv_path, leaderboard)
        self._write_leaderboard_csv(run_dir / "leaderboard.csv", leaderboard)
        exported_files["leaderboard.csv"] = str(csv_path)

        html_path = latest_dir / "index.html"
        self._write_html_preview(html_path, summary, leaderboard)
        self._write_html_preview(run_dir / "index.html", summary, leaderboard)
        exported_files["index.html"] = str(html_path)

        latest_pointer = self.output_dir / "latest_run.txt"
        latest_pointer.write_text(run_id + "\n", encoding="utf-8")

        if self.sync_dir is not None:
            exported_files.update(self._sync_session_bundle(run_id, session_payload, summary))

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

    def _build_evaluations_bundle(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build full evaluation records with embedded samples."""
        bundled: List[Dict[str, Any]] = []

        for evaluation in evaluations:
            model_name = evaluation.get("model", {}).get("name", "unknown")
            benchmark_name = evaluation.get("benchmark_name", "unknown")
            evaluation_id = f"{safe_slug(benchmark_name)}__{safe_slug(model_name)}"
            sample_path = evaluation.get("artifact_paths", {}).get("samples_jsonl")
            samples = self._load_samples(sample_path=sample_path, evaluation_id=evaluation_id)

            bundled_evaluation = dict(evaluation)
            bundled_evaluation["evaluation_id"] = evaluation_id
            bundled_evaluation["samples"] = samples
            bundled_evaluation["sample_count_embedded"] = len(samples)
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
            "artifact_paths": evaluation.get("artifact_paths", {}),
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

    def _make_session_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Trim the full summary down for the website session bundle."""
        return {
            "run_id": summary.get("run_id"),
            "generated_at": summary.get("generated_at"),
            "manifest_path": summary.get("manifest_path"),
            "totals": summary.get("totals", {}),
        }

    def _sync_session_bundle(
        self,
        run_id: str,
        session_payload: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        """Mirror the latest session bundle into the website repo."""
        if self.sync_dir is None:
            return {}

        latest_path = self.sync_dir / "latest-session.json"
        meta_path = self.sync_dir / "latest-session.meta.json"
        history_path = self.sync_dir / "runs" / f"{run_id}.json"
        latest_pointer = self.sync_dir / "latest-run.txt"

        self._write_json(latest_path, session_payload)
        self._write_json(
            meta_path,
            {
                "schema_version": "3.0",
                "run_id": run_id,
                "updated_at": utcnow_iso(),
                "totals": summary.get("totals", {}),
            },
        )
        self._write_json(history_path, session_payload)
        latest_pointer.parent.mkdir(parents=True, exist_ok=True)
        latest_pointer.write_text(run_id + "\n", encoding="utf-8")

        return {
            "sync/latest-session.json": str(latest_path),
            "sync/latest-session.meta.json": str(meta_path),
            "sync/runs.json": str(history_path),
        }

    def _write_json(self, path: Path, payload: Any) -> None:
        """Write a JSON payload to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

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
            "family",
            "quantization",
            "overall_accuracy",
            "benchmarks_run",
            "total_samples",
            "correct_count",
            "error_count",
            "avg_latency_sec",
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
