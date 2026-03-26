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
    """Export the latest benchmark run to website-friendly JSON, CSV, and HTML."""

    def __init__(self, artifacts_dir: str = "artifacts", output_dir: str = "website_exports"):
        self.artifacts = ArtifactStore(artifacts_dir)
        self.output_dir = Path(output_dir)

    def export_run(self, run_id: Optional[str] = None) -> Dict[str, str]:
        """Export one run to a stable website bundle."""
        run_data = self.artifacts.load_run(run_id)
        run_id = run_data["run_id"]

        latest_dir = self.output_dir / "latest"
        run_dir = self.output_dir / "runs" / run_id
        latest_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        leaderboard = run_data["summary"].get("leaderboard", [])
        evaluations = run_data["summary"].get("evaluations", [])
        models_bundle = self._build_models_bundle(leaderboard, evaluations)
        benchmarks_bundle = self._build_benchmarks_bundle(evaluations)

        export_payload = {
            "schema_version": "2.0",
            "generated_at": utcnow_iso(),
            "run_id": run_id,
            "manifest": run_data["manifest"],
            "summary": run_data["summary"],
            "leaderboard": leaderboard,
            "models": models_bundle,
            "benchmarks": benchmarks_bundle,
            "sample_files": run_data["sample_files"],
        }

        exported_files: Dict[str, str] = {}

        files_to_write = {
            "manifest.json": run_data["manifest"],
            "summary.json": run_data["summary"],
            "leaderboard.json": leaderboard,
            "models.json": models_bundle,
            "benchmarks.json": benchmarks_bundle,
            f"run_{run_id}.json": export_payload,
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
        self._write_html_preview(html_path, run_data["summary"], leaderboard)
        self._write_html_preview(run_dir / "index.html", run_data["summary"], leaderboard)
        exported_files["index.html"] = str(html_path)

        latest_pointer = self.output_dir / "latest_run.txt"
        latest_pointer.write_text(run_id + "\n", encoding="utf-8")

        return exported_files

    def _build_models_bundle(self, leaderboard: List[Dict[str, Any]], evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build per-model website bundles."""
        evaluation_map: Dict[str, List[Dict[str, Any]]] = {}
        for evaluation in evaluations:
            model_name = evaluation["model"]["name"]
            evaluation_map.setdefault(model_name, []).append(evaluation)

        models_bundle: List[Dict[str, Any]] = []
        for row in leaderboard:
            model_name = row["model_name"]
            models_bundle.append(
                {
                    "model_name": model_name,
                    "slug": safe_slug(model_name),
                    "leaderboard": row,
                    "evaluations": evaluation_map.get(model_name, []),
                }
            )
        return models_bundle

    def _build_benchmarks_bundle(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build per-benchmark website bundles."""
        grouped: Dict[str, Dict[str, Any]] = {}
        for evaluation in evaluations:
            benchmark_name = evaluation["benchmark_name"]
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
        bundles.sort(key=lambda item: item["benchmark_name"])
        return bundles

    def _write_json(self, path: Path, payload: Any) -> None:
        """Write a JSON payload to disk."""
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
    <div class="card"><strong>Total Tokens</strong><br />{totals.get('total_tokens', 0)}</div>
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
        <th>Avg Latency (s)</th>
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
        path.write_text(html, encoding="utf-8")
