#!/usr/bin/env python3
"""Friendly one-command launcher for smaLLMs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _running_inside_virtualenv() -> bool:
    return bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _current_python_matches(python_executable: Path) -> bool:
    """Compare the invoked interpreter paths without resolving symlinks."""
    return os.path.abspath(sys.executable) == os.path.abspath(str(python_executable))


def _ensure_runtime_python(repo_root: Path) -> Path:
    """Use the current venv if present, otherwise bootstrap a local `.venv`.

    smaLLMs needs only Python's standard library, so nothing is installed into it;
    the `.venv` keeps one stable interpreter path for the launcher and PortUI actions.
    """
    if _running_inside_virtualenv():
        return Path(sys.executable)

    venv_dir = repo_root / ".venv"
    venv_python = _venv_python(venv_dir)
    if venv_python.exists():
        return venv_python

    print("Creating local virtual environment in .venv ...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=repo_root, check=True)
    return venv_python


def build_parser() -> argparse.ArgumentParser:
    """Construct the launcher parser."""
    parser = argparse.ArgumentParser(description="Bootstrap smaLLMs and open the arrow-key menu.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print setup status after preparing the local Python, then exit without opening the menu.",
    )
    parser.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="Optional smaLLMs.py command to run after bootstrapping. Use '-- <command> ...'.",
    )
    return parser


def _run_smaLLMs_subprocess(python_executable: Path, repo_root: Path, cli_args: List[str]) -> int:
    """Run smaLLMs.py and return a clean cancellation code on Ctrl+C."""
    try:
        result = subprocess.run(
            [str(python_executable), str(repo_root / "smaLLMs.py"), *cli_args],
            cwd=repo_root,
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from src.cli.setup_checks import build_setup_report_lines, collect_setup_report

    print("smaLLMs launcher")
    print("----------------")
    sys.stdout.flush()

    python_executable = _ensure_runtime_python(repo_root)

    report = collect_setup_report()
    print("")
    for line in build_setup_report_lines(report):
        print(line)
    sys.stdout.flush()

    cli_args = list(args.cli_args)
    if cli_args[:1] == ["--"]:
        cli_args = cli_args[1:]

    if args.check_only:
        return 0

    if cli_args:
        print("")
        print(f"Running smaLLMs.py {' '.join(cli_args)} ...")
        sys.stdout.flush()
        return _run_smaLLMs_subprocess(python_executable, repo_root, cli_args)

    print("")
    print("Opening the arrow-key menu ...")
    sys.stdout.flush()
    return _run_smaLLMs_subprocess(python_executable, repo_root, ["menu"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
