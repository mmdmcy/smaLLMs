#!/usr/bin/env python3
"""Canonical test entrypoint for the smaLLMs repository."""

from __future__ import annotations

import pathlib
import sys
import unittest


def main() -> int:
    """Run every test under ./tests from the repository root."""
    root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
