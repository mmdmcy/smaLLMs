#!/usr/bin/env python3
"""Friendly one-command launcher for smaLLMs."""

from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REQUIRED_MODULES = ("yaml", "aiohttp", "datasets")
MIN_MODULE_VERSIONS = {"datasets": "4.8.0"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _running_inside_virtualenv() -> bool:
    return bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _version_tuple(value: str) -> tuple[int, ...]:
    """Best-effort tuple conversion for simple dotted versions."""
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts)


def _current_python_matches(python_executable: Path) -> bool:
    """Compare the invoked interpreter paths without resolving symlinks."""
    return os.path.abspath(sys.executable) == os.path.abspath(str(python_executable))


def _module_is_outdated(module: str) -> bool:
    """Return True when a module is present but below the required minimum version."""
    minimum = MIN_MODULE_VERSIONS.get(module)
    if not minimum:
        return False

    try:
        installed = metadata.version(module)
    except metadata.PackageNotFoundError:
        return True

    return _version_tuple(installed) < _version_tuple(minimum)


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
    missing_or_outdated: List[str] = []
    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None or _module_is_outdated(module):
            minimum = MIN_MODULE_VERSIONS.get(module)
            missing_or_outdated.append(f"{module}>={minimum}" if minimum else module)
    return missing_or_outdated


def _missing_modules_for_python(python_executable: Path) -> List[str]:
    if _current_python_matches(python_executable):
        return _missing_modules_for_current_python()

    script = (
        "import importlib.util\n"
        "import re\n"
        "from importlib import metadata\n"
        f"modules = {list(REQUIRED_MODULES)!r}\n"
        f"minimums = {MIN_MODULE_VERSIONS!r}\n"
        "def version_tuple(value):\n"
        "    return tuple(int(part) for part in re.findall(r'\\d+', value))\n"
        "issues = []\n"
        "for name in modules:\n"
        "    if importlib.util.find_spec(name) is None:\n"
        "        minimum = minimums.get(name)\n"
        "        issues.append(f'{name}>={minimum}' if minimum else name)\n"
        "        continue\n"
        "    minimum = minimums.get(name)\n"
        "    if not minimum:\n"
        "        continue\n"
        "    try:\n"
        "        installed = metadata.version(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        issues.append(f'{name}>={minimum}')\n"
        "        continue\n"
        "    if version_tuple(installed) < version_tuple(minimum):\n"
        "        issues.append(f'{name}>={minimum}')\n"
        "print('\\n'.join(issues))\n"
    )
    result = subprocess.run(
        [str(python_executable), "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _install_runtime_requirements(python_executable: Path, repo_root: Path, missing_modules: Iterable[str]) -> None:
    """Install or upgrade the standard runtime dependencies when needed."""
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
    parser.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="Optional smaLLMs.py command to run after bootstrapping. Use '-- <command> ...'.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from src.cli.setup_checks import build_setup_report_lines, collect_setup_report

    print("smaLLMs launcher")
    print("----------------")
    sys.stdout.flush()

    python_executable = _ensure_runtime_python(repo_root)
    missing_modules = _missing_modules_for_python(python_executable)
    _install_runtime_requirements(python_executable, repo_root, missing_modules)

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
        result = subprocess.run(
            [str(python_executable), str(repo_root / "smaLLMs.py"), *cli_args],
            cwd=repo_root,
            check=False,
        )
        return result.returncode

    print("")
    print("Opening the arrow-key menu ...")
    sys.stdout.flush()
    result = subprocess.run(
        [str(python_executable), str(repo_root / "smaLLMs.py"), "menu"],
        cwd=repo_root,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
