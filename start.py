#!/usr/bin/env python3
"""Friendly one-command launcher for smaLLMs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REQUIRED_MODULES = ("yaml", "aiohttp", "datasets")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _running_inside_virtualenv() -> bool:
    return bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_runtime_python(repo_root: Path) -> Path:
    """Use the current venv if present, otherwise bootstrap a local `.venv`."""
    if _running_inside_virtualenv():
        return Path(sys.executable)

    venv_dir = repo_root / ".venv"
    venv_python = _venv_python(venv_dir)
    if venv_python.exists():
        return venv_python

    print("Creating local virtual environment in .venv ...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=repo_root, check=True)
    return venv_python


def _missing_modules_for_current_python() -> List[str]:
    return [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def _missing_modules_for_python(python_executable: Path) -> List[str]:
    if Path(sys.executable).resolve() == python_executable.resolve():
        return _missing_modules_for_current_python()

    script = (
        "import importlib.util\n"
        f"modules = {list(REQUIRED_MODULES)!r}\n"
        "missing = [name for name in modules if importlib.util.find_spec(name) is None]\n"
        "print('\\n'.join(missing))\n"
    )
    result = subprocess.run(
        [str(python_executable), "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _install_runtime_requirements(python_executable: Path, repo_root: Path, missing_modules: Iterable[str]) -> None:
    """Install the standard runtime dependencies when needed."""
    missing = list(missing_modules)
    if not missing:
        return

    print(f"Installing smaLLMs runtime dependencies ({', '.join(missing)}) ...")
    subprocess.run(
        [str(python_executable), "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=repo_root,
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the launcher parser."""
    parser = argparse.ArgumentParser(description="Bootstrap smaLLMs and open the arrow-key menu.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print setup status after bootstrapping dependencies, then exit without opening the menu.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from src.cli.setup_checks import build_setup_report_lines, collect_setup_report

    print("smaLLMs launcher")
    print("----------------")

    python_executable = _ensure_runtime_python(repo_root)
    missing_modules = _missing_modules_for_python(python_executable)
    _install_runtime_requirements(python_executable, repo_root, missing_modules)

    report = collect_setup_report()
    print("")
    for line in build_setup_report_lines(report):
        print(line)

    if args.check_only:
        return 0

    print("")
    print("Opening the arrow-key menu ...")
    result = subprocess.run(
        [str(python_executable), str(repo_root / "smaLLMs.py"), "menu"],
        cwd=repo_root,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
