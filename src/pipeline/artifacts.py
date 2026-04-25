"""
Structured artifact storage for local benchmark runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import platform

from src.pipeline.config import ARTIFACT_SCHEMA_VERSION, DEFAULT_ARTIFACTS_DIR


def utcnow_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    """Make a filesystem-safe slug."""
    cleaned = value.replace("/", "__").replace(":", "_").replace(" ", "_")
    return "".join(char for char in cleaned if char.isalnum() or char in {"_", "-", "."})


def _run_command(args: List[str]) -> str:
    """Run a best-effort command for metadata collection."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_command() -> Optional[str]:
    """Locate Git across common Windows and POSIX installs."""
    discovered = shutil.which("git")
    if discovered:
        return discovered

    for candidate in [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\bin\git.exe",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def _ollama_command() -> Optional[str]:
    """Locate Ollama for version metadata collection."""
    discovered = shutil.which("ollama")
    if discovered:
        return discovered

    for candidate in [
        Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        Path(r"C:\Program Files\Ollama\ollama.exe"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def portable_path(value: str | Path) -> str:
    """Return a path string without embedding user-specific absolute locations."""
    candidate = Path(value)
    if not candidate.is_absolute():
        return candidate.as_posix()

    try:
        resolved = candidate.resolve(strict=False)
    except Exception:
        resolved = candidate

    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        pass

    try:
        relative = os.path.relpath(resolved, cwd)
        if relative and relative != ".":
            return Path(relative).as_posix()
    except ValueError:
        pass

    home = Path.home().resolve()
    try:
        return (Path("~") / resolved.relative_to(home)).as_posix()
    except ValueError:
        pass

    filtered_parts = [part for part in resolved.parts if part not in {"", resolved.anchor, os.sep}]
    tail = filtered_parts[-3:] if filtered_parts else [resolved.name or "path"]
    return Path("<external>", *tail).as_posix()


def sanitize_system_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Drop host-identifying fields from persisted system metadata."""
    sanitized = dict(metadata)
    sanitized.pop("hostname", None)
    return sanitized


@dataclass
class RunPaths:
    """Filesystem locations for a benchmark run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    summary_path: Path
    benchmark_dir: Path
    sample_dir: Path


class ArtifactStore:
    """Write and load structured benchmark artifacts."""

    def __init__(self, base_dir: str = DEFAULT_ARTIFACTS_DIR):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, manifest: Dict[str, Any]) -> RunPaths:
        """Create folders for a new run and persist the manifest."""
        created_at = manifest.get("created_at") or utcnow_iso()
        run_id = manifest.get("run_id") or self._make_run_id(created_at)
        run_dir = self.runs_dir / run_id
        benchmark_dir = run_dir / "benchmarks"
        sample_dir = run_dir / "samples"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"

        manifest = dict(manifest)
        manifest["run_id"] = run_id
        manifest["created_at"] = created_at
        manifest["schema_version"] = ARTIFACT_SCHEMA_VERSION

        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        self._write_latest_run_id(run_id)
        return RunPaths(
            run_id=run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            summary_path=summary_path,
            benchmark_dir=benchmark_dir,
            sample_dir=sample_dir,
        )

    def write_benchmark_result(self, paths: RunPaths, benchmark_name: str, model_name: str, data: Dict[str, Any]) -> Path:
        """Persist a model x benchmark summary."""
        target_dir = paths.benchmark_dir / safe_slug(benchmark_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{safe_slug(model_name)}.json"
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return file_path

    def write_sample_results(
        self,
        paths: RunPaths,
        benchmark_name: str,
        model_name: str,
        samples: Iterable[Dict[str, Any]],
    ) -> Path:
        """Persist per-sample results as JSONL."""
        target_dir = paths.sample_dir / safe_slug(benchmark_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{safe_slug(model_name)}.jsonl"
        with open(file_path, "w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
        return file_path

    def write_summary(self, paths: RunPaths, summary: Dict[str, Any]) -> Path:
        """Persist a run summary."""
        with open(paths.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._write_latest_run_id(paths.run_id)
        return paths.summary_path

    def latest_run_id(self) -> Optional[str]:
        """Return the most recently completed run id."""
        latest_file = self.base_dir / "latest_run.txt"
        if latest_file.exists():
            run_id = latest_file.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id

        run_dirs = sorted([path.name for path in self.runs_dir.iterdir() if path.is_dir()])
        return run_dirs[-1] if run_dirs else None

    def load_run(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Load manifest, summary, benchmark files, and sample file paths for a run."""
        run_id = run_id or self.latest_run_id()
        if not run_id:
            raise FileNotFoundError("No benchmark runs found")

        run_dir = self.runs_dir / run_id
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        benchmark_dir = run_dir / "benchmarks"
        sample_dir = run_dir / "samples"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run manifest: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        benchmark_results: List[Dict[str, Any]] = []
        if benchmark_dir.exists():
            for file_path in sorted(benchmark_dir.rglob("*.json")):
                benchmark_results.append(json.loads(file_path.read_text(encoding="utf-8")))

        sample_files = sorted(str(file_path) for file_path in sample_dir.rglob("*.jsonl")) if sample_dir.exists() else []

        return {
            "run_id": run_id,
            "run_dir": portable_path(run_dir),
            "manifest": manifest,
            "summary": summary,
            "benchmark_results": benchmark_results,
            "sample_files": [portable_path(file_path) for file_path in sample_files],
        }

    def _write_latest_run_id(self, run_id: str) -> None:
        """Track the latest run for export convenience."""
        latest_file = self.base_dir / "latest_run.txt"
        latest_file.write_text(run_id + "\n", encoding="utf-8")

    def _make_run_id(self, created_at: str) -> str:
        """Generate a stable, sortable run id."""
        compact = created_at.replace("-", "").replace(":", "").replace(".", "").replace("+00:00", "Z")
        compact = compact.replace("T", "_")
        return f"run_{compact}"


def collect_system_metadata() -> Dict[str, Any]:
    """Collect non-identifying system metadata for reproducible benchmark runs."""
    ollama = _ollama_command()
    metadata = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "ollama_version": _run_command([ollama, "--version"]) if ollama else "",
    }

    try:
        import psutil

        memory = psutil.virtual_memory()
        metadata.update(
            {
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "memory_total_mb": round(memory.total / (1024 * 1024), 2),
            }
        )
    except ModuleNotFoundError:
        metadata.update(
            {
                "cpu_count_logical": os.cpu_count(),
                "cpu_count_physical": None,
                "memory_total_mb": None,
                "system_metadata_warning": "psutil_not_installed",
            }
        )

    return sanitize_system_metadata(metadata)


def collect_repository_metadata(repo_root: str = ".") -> Dict[str, Any]:
    """Collect git metadata for reproducibility."""
    root = Path(repo_root)
    git = _git_command()
    if not git:
        return {
            "git_sha": "",
            "git_branch": "",
            "git_dirty": None,
            "git_warning": "git_not_found",
        }

    git_sha = _run_command([git, "-C", str(root), "rev-parse", "HEAD"])
    git_branch = _run_command([git, "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])
    git_status = _run_command([git, "-C", str(root), "status", "--short"])
    return {
        "git_sha": git_sha,
        "git_branch": git_branch,
        "git_dirty": bool(git_status),
    }
