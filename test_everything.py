#!/usr/bin/env python3
"""Smoke-test entrypoint for the current smaLLMs CLI."""

from __future__ import annotations

import pathlib
import sys
import unittest


def main() -> int:
    """Run the test suite under ./tests."""
    root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
