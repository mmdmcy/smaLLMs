"""Codex reasoning-effort sweeps and their separate website export feed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.pipeline.agent_harness import (
    DEFAULT_AGENT_HARNESS_WEBSITE_SYNC_DIR,
    list_agent_harness_tasks,
    run_agent_harness_eval,
)
from src.pipeline.artifacts import portable_path, safe_slug, utcnow_iso


REASONING_SWEEP_WEB_SCHEMA_VERSION = "reasoning_effort.web.v1"
DEFAULT_REASONING_SWEEP_ARTIFACTS_DIR = Path("artifacts") / "reasoning_sweep"
REASONING_EFFORT_ORDER = ["low", "medium", "high", "xhigh", "max", "ultra"]

DEFAULT_REASONING_SWEEP_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-5.6-sol": {
        "display_name": "GPT-5.6 Sol",
        "description": "Frontier agentic coding model.",
        "efforts": ["low", "medium", "high", "xhigh", "max", "ultra"],
    },
    "gpt-5.6-terra": {
        "display_name": "GPT-5.6 Terra",
        "description": "Balanced agentic coding model.",
        "efforts": ["low", "medium", "high", "xhigh", "max", "ultra"],
    },
    "gpt-5.6-luna": {
        "display_name": "GPT-5.6 Luna",
        "description": "Fast and efficient agentic coding model.",
        "efforts": ["low", "medium", "high", "xhigh", "max"],
    },
}


def _select_requested(values: Optional[Sequence[str]], available: Sequence[str], label: str) -> List[str]:
    if not values:
        return list(available)

    selected: List[str] = []
    for raw in values:
        for part in raw.split(","):
            value = part.strip()
            if not value:
                continue
            if value not in available:
                raise ValueError(f"Unknown {label}: {value}. Available: {', '.join(available)}")
            if value not in selected:
                selected.append(value)
    return selected


def _variant_id(model: str, effort: str) -> str:
    return safe_slug(f"{model}-{effort}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sync_paths(sync_dir: str | Path, run_id: str) -> Dict[str, str]:
    root = Path(sync_dir) / "reasoning-efforts"
    latest = root / "latest.json"
    run = root / "runs" / f"{safe_slug(run_id)}.json"
    meta = root / "latest.meta.json"
    pointer = root / "latest-run.txt"
    return {
        "latest_path": latest.as_posix(),
        "run_path": run.as_posix(),
        "meta_path": meta.as_posix(),
        "pointer_path": pointer.as_posix(),
    }


def _compact_variant(result: Dict[str, Any], model: str, effort: str) -> Dict[str, Any]:
    web_path = Path(str(result["web_summary_path"])).expanduser()
    web_summary = json.loads(web_path.read_text(encoding="utf-8"))
    aggregate = (web_summary.get("harnesses") or [{}])[0]
    variant_id = _variant_id(model, effort)
    return {
        "variant_id": variant_id,
        "model": model,
        "model_display_name": DEFAULT_REASONING_SWEEP_MODELS[model]["display_name"],
        "reasoning_effort": effort,
        "rows": aggregate.get("rows", 0),
        "completed": aggregate.get("completed", 0),
        "passed": aggregate.get("passed", 0),
        "failed": aggregate.get("failed", 0),
        "pass_rate": aggregate.get("pass_rate"),
        "duration_seconds_total": aggregate.get("duration_seconds_total", 0.0),
        "avg_duration_seconds": aggregate.get("avg_duration_seconds"),
        "reported_total_tokens": aggregate.get("reported_total_tokens"),
        "reported_token_rows": aggregate.get("reported_token_rows", 0),
        "max_agent_rss_kb": aggregate.get("max_agent_rss_kb"),
        "unexpected_change_rows": aggregate.get("unexpected_change_rows", 0),
        "run_id": result["run_id"],
        "summary_path": result["summary_path"],
        "web_summary_path": result["web_summary_path"],
        "artifact_dir": result["run_dir"],
        "results": [
            {
                **row,
                "variant_id": variant_id,
                "model": model,
                "reasoning_effort": effort,
            }
            for row in web_summary.get("results", [])
        ],
    }


def _build_findings(variants: Sequence[Dict[str, Any]], totals: Dict[str, Any]) -> List[Dict[str, str]]:
    completed = [variant for variant in variants if variant.get("completed")]
    findings: List[Dict[str, str]] = []
    if completed and all(variant.get("failed", 0) == 0 for variant in completed):
        findings.append({
            "kind": "quality",
            "title": "All completed variants passed",
            "summary": f"{len(completed)} reasoning-effort variants completed without a failing fixture row.",
        })

    timed = [variant for variant in completed if isinstance(variant.get("avg_duration_seconds"), (int, float))]
    if timed:
        fastest = min(timed, key=lambda variant: float(variant["avg_duration_seconds"]))
        findings.append({
            "kind": "speed",
            "title": f"Fastest: {fastest['model_display_name']} {fastest['reasoning_effort']}",
            "summary": f"This variant averaged {float(fastest['avg_duration_seconds']):.3f}s per fixture row.",
        })

    tokenized = [variant for variant in completed if isinstance(variant.get("reported_total_tokens"), int)]
    if tokenized:
        efficient = min(tokenized, key=lambda variant: int(variant["reported_total_tokens"]))
        findings.append({
            "kind": "tokens",
            "title": f"Fewest reported tokens: {efficient['model_display_name']} {efficient['reasoning_effort']}",
            "summary": f"This variant reported {int(efficient['reported_total_tokens']):,} CLI tokens across its rows.",
        })

    findings.append({
        "kind": "scope",
        "title": "Codex effort sweep, not temperature sweep",
        "summary": "Codex CLI exposes reasoning effort for this workflow; sampling temperature is not part of this feed.",
    })
    return findings


def run_reasoning_effort_sweep(
    models: Optional[Sequence[str]] = None,
    efforts: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    artifacts_dir: str | Path = DEFAULT_REASONING_SWEEP_ARTIFACTS_DIR,
    timeout_seconds: int = 900,
    dry_run: bool = False,
    sync_dir: Optional[str | Path] = DEFAULT_AGENT_HARNESS_WEBSITE_SYNC_DIR,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run every selected GPT-5.6 model/effort pair and export one separate feed."""
    selected_models = _select_requested(models, list(DEFAULT_REASONING_SWEEP_MODELS), "model")
    selected_efforts = _select_requested(
        efforts,
        [
            effort
            for effort in REASONING_EFFORT_ORDER
            if any(effort in DEFAULT_REASONING_SWEEP_MODELS[model]["efforts"] for model in selected_models)
        ],
        "reasoning effort",
    )

    variants_to_run = [
        (model, effort)
        for model in selected_models
        for effort in selected_efforts
        if effort in DEFAULT_REASONING_SWEEP_MODELS[model]["efforts"]
    ]
    if not variants_to_run:
        raise ValueError("No selected model supports the requested reasoning efforts.")

    created_at = utcnow_iso()
    sweep_id = safe_slug(f"reasoning-sweep-{created_at.replace('+00:00', 'Z').replace(':', '').replace('.', '-')}")
    base_dir = Path(artifacts_dir)
    sweep_dir = base_dir / "runs" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=False)

    compact_variants: List[Dict[str, Any]] = []
    for model, effort in variants_to_run:
        variant_id = _variant_id(model, effort)
        if progress_callback:
            progress_callback({"event": "reasoning_variant_started", "variant_id": variant_id, "model": model, "reasoning_effort": effort})

        result = run_agent_harness_eval(
            harnesses=["codex"],
            tasks=tasks,
            artifacts_dir=sweep_dir / "variants" / variant_id,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            sync_dir=None,
            model_override=model,
            reasoning_override=effort,
            progress_callback=(
                lambda event, model=model, effort=effort: progress_callback(
                    {**event, "model": model, "reasoning_effort": effort, "variant_id": _variant_id(model, effort)}
                )
                if progress_callback
                else None
            ),
        )
        compact_variants.append(_compact_variant(result, model, effort))
        if progress_callback:
            progress_callback({"event": "reasoning_variant_completed", "variant_id": variant_id, "model": model, "reasoning_effort": effort})

    flat_results = [row for variant in compact_variants for row in variant.pop("results")]
    completed_rows = [row for row in flat_results if row.get("status") in {"passed", "failed"}]
    token_values = [row.get("usage", {}).get("total_tokens") for row in flat_results if isinstance(row.get("usage", {}).get("total_tokens"), int)]
    totals = {
        "variants": len(compact_variants),
        "rows": len(flat_results),
        "completed": len(completed_rows),
        "passed": sum(1 for row in flat_results if row.get("status") == "passed"),
        "failed": sum(1 for row in flat_results if row.get("status") == "failed"),
        "dry_run": sum(1 for row in flat_results if row.get("status") == "dry_run"),
        "duration_seconds_total": round(sum(float(row.get("duration_seconds") or 0.0) for row in completed_rows), 3),
        "reported_total_tokens": sum(token_values) if token_values else None,
        "reported_token_rows": len(token_values),
    }
    payload: Dict[str, Any] = {
        "schema_version": REASONING_SWEEP_WEB_SCHEMA_VERSION,
        "exported_at": utcnow_iso(),
        "run_id": sweep_id,
        "created_at": created_at,
        "mode": "reasoning_effort_sweep",
        "sweep_kind": "codex_reasoning_effort",
        "sampling_temperature": None,
        "temperature_note": "Codex CLI exposes reasoning effort for this sweep; no sampling temperature was set or measured.",
        "source": {
            "payload_profile": "compact_reasoning_effort_sweep",
            "artifacts_dir": portable_path(sweep_dir),
            "synced": bool(sync_dir),
        },
        "model_catalog": [
            {
                "model": model,
                "display_name": DEFAULT_REASONING_SWEEP_MODELS[model]["display_name"],
                "description": DEFAULT_REASONING_SWEEP_MODELS[model]["description"],
                "supported_reasoning_efforts": DEFAULT_REASONING_SWEEP_MODELS[model]["efforts"],
            }
            for model in selected_models
        ],
        "selected_models": selected_models,
        "selected_reasoning_efforts": selected_efforts,
        "totals": totals,
        "variants": compact_variants,
        "findings": _build_findings(compact_variants, totals),
        "tasks": list_agent_harness_tasks(),
        "results": flat_results,
        "limitations": [
            "This feed measures Codex reasoning effort, not sampling temperature.",
            "Reported token usage is the total emitted by Codex CLI; prompt/completion/reasoning splits are not available here.",
            "Fixture tasks are useful for workflow comparison but are not a broad intelligence benchmark.",
        ],
    }

    summary_path = sweep_dir / "summary.json"
    web_path = sweep_dir / "web_summary.json"
    _write_json(summary_path, {**payload, "results": flat_results})
    _write_json(web_path, payload)
    payload["source"]["summary_path"] = portable_path(summary_path)
    payload["source"]["web_summary_path"] = portable_path(web_path)

    if sync_dir:
        paths = _sync_paths(sync_dir, sweep_id)
        _write_json(Path(paths["latest_path"]), payload)
        _write_json(Path(paths["run_path"]), payload)
        _write_json(Path(paths["meta_path"]), {
            "schema_version": REASONING_SWEEP_WEB_SCHEMA_VERSION,
            "run_id": sweep_id,
            "exported_at": payload["exported_at"],
            "latest_path": "reasoning-efforts/latest.json",
            "run_path": f"reasoning-efforts/runs/{sweep_id}.json",
        })
        Path(paths["pointer_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(paths["pointer_path"]).write_text(f"{sweep_id}\n", encoding="utf-8")
        payload["source"]["sync_paths"] = paths

    return payload
