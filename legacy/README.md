# Legacy Code

This directory preserves the older experimental smaLLMs evaluation stack for
reference. It is not the maintained product surface.

The maintained path is the CLI-first local benchmark pipeline:

- `smaLLMs.py` for the interactive and non-interactive CLI
- `src/pipeline/` for benchmark orchestration, artifacts, exports, and config
- `src/models/` for local model provider integration
- `src/cli/` for setup checks and terminal UI
- `tests/` for the canonical regression suite

Keeping the older stack here makes the repository history honest without letting
placeholder-era code dilute the public signal of the current runner.
