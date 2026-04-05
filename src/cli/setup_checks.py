"""Local runtime checks shared by the launcher and CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_LM_STUDIO_ENDPOINT = "http://localhost:1234"


def _trim_error(message: str, limit: int = 140) -> str:
    """Return a compact single-line error message."""
    cleaned = " ".join(str(message).split())
    return cleaned[: limit - 3] + "..." if len(cleaned) > limit else cleaned


def parse_ollama_list_output(stdout: str) -> List[str]:
    """Extract model names from `ollama list` output."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    return [line.split()[0] for line in lines[1:] if line.split()]


def _http_get_json(url: str, timeout: float = 3.0) -> Dict[str, Any]:
    """Fetch a JSON payload with the stdlib only."""
    with urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _parse_ollama_api_models(payload: Dict[str, Any]) -> List[str]:
    """Extract model names from Ollama's `/api/tags` response."""
    models = payload.get("models", [])
    return [str(model.get("name")).strip() for model in models if str(model.get("name", "")).strip()]


def _parse_lm_studio_models(payload: Dict[str, Any]) -> List[str]:
    """Extract model ids from LM Studio's OpenAI-compatible response."""
    models = payload.get("data", [])
    return [str(model.get("id")).strip() for model in models if str(model.get("id", "")).strip()]


@dataclass(frozen=True)
class OllamaStatus:
    """Ollama install and runtime status."""

    endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    cli_path: Optional[str] = None
    running: bool = False
    models: Tuple[str, ...] = ()
    source: str = ""
    detail: str = ""

    @property
    def installed(self) -> bool:
        return self.cli_path is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly shape."""
        return {
            "endpoint": self.endpoint,
            "installed": self.installed,
            "cli_path": self.cli_path,
            "running": self.running,
            "models": list(self.models),
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LMStudioStatus:
    """LM Studio runtime status."""

    endpoint: str = DEFAULT_LM_STUDIO_ENDPOINT
    running: bool = False
    models: Tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly shape."""
        return {
            "endpoint": self.endpoint,
            "running": self.running,
            "models": list(self.models),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SetupReport:
    """Combined local setup status."""

    ollama: OllamaStatus
    lm_studio: LMStudioStatus

    @property
    def has_local_models(self) -> bool:
        return bool(self.ollama.models or self.lm_studio.models)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly shape."""
        return {
            "has_local_models": self.has_local_models,
            "ollama": self.ollama.to_dict(),
            "lm_studio": self.lm_studio.to_dict(),
        }


def collect_setup_report(
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    lm_studio_endpoint: str = DEFAULT_LM_STUDIO_ENDPOINT,
) -> SetupReport:
    """Collect a lightweight runtime readiness report."""
    return SetupReport(
        ollama=check_ollama_status(ollama_endpoint),
        lm_studio=check_lm_studio_status(lm_studio_endpoint),
    )


def check_ollama_status(endpoint: str = DEFAULT_OLLAMA_ENDPOINT) -> OllamaStatus:
    """Inspect whether Ollama is installed, running, and already has models."""
    cli_path = shutil.which("ollama")
    api_error = ""
    cli_error = ""

    try:
        payload = _http_get_json(f"{endpoint}/api/tags")
        models = tuple(_parse_ollama_api_models(payload))
        return OllamaStatus(
            endpoint=endpoint,
            cli_path=cli_path,
            running=True,
            models=models,
            source="api",
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        api_error = _trim_error(exc)

    if cli_path:
        try:
            result = subprocess.run(
                [cli_path, "list"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return OllamaStatus(
                    endpoint=endpoint,
                    cli_path=cli_path,
                    running=True,
                    models=tuple(parse_ollama_list_output(result.stdout)),
                    source="cli",
                )
            cli_error = _trim_error(result.stderr or result.stdout or "Ollama command returned a non-zero exit code.")
        except (OSError, subprocess.SubprocessError) as exc:
            cli_error = _trim_error(exc)

    detail = cli_error or api_error
    if cli_path and not detail:
        detail = "Ollama is installed but the local service did not answer."
    if not cli_path and not detail:
        detail = "Ollama was not found in PATH."

    return OllamaStatus(
        endpoint=endpoint,
        cli_path=cli_path,
        running=False,
        models=(),
        source="",
        detail=detail,
    )


def check_lm_studio_status(endpoint: str = DEFAULT_LM_STUDIO_ENDPOINT) -> LMStudioStatus:
    """Inspect whether LM Studio's local server is responding."""
    try:
        payload = _http_get_json(f"{endpoint}/v1/models")
        models = tuple(_parse_lm_studio_models(payload))
        return LMStudioStatus(
            endpoint=endpoint,
            running=True,
            models=models,
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return LMStudioStatus(
            endpoint=endpoint,
            running=False,
            models=(),
            detail=_trim_error(exc),
        )


def build_setup_report_lines(report: SetupReport, model_limit: int = 5) -> List[str]:
    """Render the setup report as compact, user-facing lines."""
    lines: List[str] = []

    ollama = report.ollama
    if ollama.models:
        lines.append(f"Ollama: ready ({len(ollama.models)} model(s) detected)")
        lines.append(f"  Models: {_format_model_list(ollama.models, limit=model_limit)}")
        lines.append("  Existing Ollama models are reused automatically. You do not need to pull them again.")
    elif ollama.running:
        lines.append("Ollama: running, but no models are installed yet")
        lines.append("  Pull a model once with `ollama pull llama3.2`, then smaLLMs will pick it up automatically.")
    elif ollama.installed:
        lines.append(f"Ollama: installed, but not responding on {ollama.endpoint}")
        lines.append("  If you already pulled models before, just start Ollama again. You do not need to reinstall them.")
        if ollama.detail:
            lines.append(f"  Detail: {ollama.detail}")
    else:
        lines.append("Ollama: not installed or not available in PATH")
        lines.append("  Install Ollama if you want the simplest local setup path.")
        if ollama.detail:
            lines.append(f"  Detail: {ollama.detail}")

    lines.append("")

    lm_studio = report.lm_studio
    if lm_studio.models:
        lines.append(f"LM Studio: ready ({len(lm_studio.models)} model(s) loaded)")
        lines.append(f"  Models: {_format_model_list(lm_studio.models, limit=model_limit)}")
    elif lm_studio.running:
        lines.append(f"LM Studio: server reachable on {lm_studio.endpoint}, but no model is loaded")
        lines.append("  Load a model in LM Studio and keep the local server running if you want to use it here.")
    else:
        lines.append(f"LM Studio: not detected on {lm_studio.endpoint}")
        lines.append("  This is optional. smaLLMs works fine with Ollama only.")
        if lm_studio.detail:
            lines.append(f"  Detail: {lm_studio.detail}")

    lines.append("")

    if report.has_local_models:
        lines.append("smaLLMs can run right now with the local models above.")
    else:
        lines.append("No local models were detected yet.")
        lines.append("Fastest fix: start Ollama, run `ollama pull llama3.2` once, then relaunch the menu.")

    return lines


def _format_model_list(models: Sequence[str], limit: int = 5) -> str:
    """Format a compact preview of detected model names."""
    visible = [str(model) for model in models[:limit]]
    remaining = max(0, len(models) - len(visible))
    suffix = f" (+{remaining} more)" if remaining else ""
    return ", ".join(visible) + suffix
