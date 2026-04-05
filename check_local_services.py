#!/usr/bin/env python3
"""Compatibility diagnostic wrapper for local runtime checks."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    from src.cli.setup_checks import build_setup_report_lines, collect_setup_report

    print("smaLLMs local setup check")
    print("-------------------------")
    print("")

    for line in build_setup_report_lines(collect_setup_report()):
        print(line)

    print("")
    print("Launch smaLLMs with `start.py` or `start.bat` to open the arrow-key menu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
