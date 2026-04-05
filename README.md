# smaLLMs

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CLI Only](https://img.shields.io/badge/interface-terminal-black)](https://github.com/mmdmcy/smaLLMs)
[![Local First](https://img.shields.io/badge/runtime-local%20first-green)](https://github.com/mmdmcy/smaLLMs)

CLI-only local LLM benchmarking with a real benchmark catalog, live terminal progress, and structured artifacts for local leaderboards.

smaLLMs is built for:
- local models running through Ollama or LM Studio
- cross-platform terminal use on macOS, Linux, Windows 11, and WSL
- reproducible benchmark runs with raw sample artifacts
- honest reporting about which benchmarks are runnable today versus only tracked for future support

## What smaLLMs is trying to be

Not a toy wrapper around a few prompts.

smaLLMs is meant to become a serious open benchmarking CLI for local models. That means:
- the default interface is the terminal, not a web app
- the interactive mode uses arrow keys and multi-select menus
- benchmark runs emit machine-readable artifacts for downstream sites and leaderboards
- the benchmark catalog distinguishes runnable local evals from frontier evals that still require specialized harnesses

## Current runtime support

Supported local providers:
- Ollama
- LM Studio

Platform targets:
- macOS
- Linux
- Windows 11
- WSL with Windows-hosted Ollama fallback support

## Benchmark model

### Runnable local benchmarks

These ship in the local runner today:

- `gsm8k`
- `mmlu`
- `mmlu_pro`
- `math`
- `arc_challenge`
- `arc_easy`
- `hellaswag`
- `winogrande`
- `boolq`
- `commonsense_qa`
- `piqa`
- `social_iqa`
- `openbookqa`
- `truthfulqa_mc1`
- `bbh_boolean_expressions`

### Tracked frontier benchmarks

These are included in the catalog because major labs report them, but smaLLMs does not pretend they are locally runnable yet:

- `aime_2024`
- `aime_2025`
- `gpqa_diamond`
- `swe_bench_verified`
- `codeforces`
- `mmmu`
- `video_mme`
- `healthbench`
- `healthbench_hard`
- `tau_bench`
- `humanitys_last_exam`
- `multi_challenge`

This distinction matters. Benchmarks like SWE-bench Verified, TauBench, MMMU, or HealthBench need agentic harnesses, multimodal graders, tool environments, or private/specialized evaluation pipelines. smaLLMs tracks them explicitly instead of faking support.

## Quick start

### 1. Run the launcher

Windows 11:

```powershell
py -3 start.py
```

or double-click:

```powershell
.\start.bat
```

macOS / Linux / WSL:

```bash
python3 start.py
```

The launcher does the user-friendly path automatically:
- creates `.venv` if you are not already in a virtual environment
- installs the standard local runtime from `requirements.txt`
- checks Ollama and LM Studio status
- tells you whether existing Ollama models were already found
- opens the arrow-key terminal UI

### 2. If you already have Ollama models installed

You do not need to pull them again.

smaLLMs automatically reuses whatever `ollama list` already shows on your machine. The only thing you need is for Ollama itself to be running.

### 3. If you do not have a local model yet

The fastest path is:

```bash
ollama pull llama3.2
```

You can also use LM Studio instead; just load a model there and keep its local server enabled.

### 4. Use the arrow-key terminal UI

The default interface is terminal-native:
- arrow keys to move
- `space` to toggle multi-select items
- `enter` to confirm
- `q` or `esc` to go back

### Dependency files

- `requirements.txt` is the standard local install for normal users.
- `requirements-local.txt` still exists only as a backwards-compatible alias.
- `requirements-dev.txt` is the broader development/legacy environment.

## Non-interactive CLI

The advanced commands are still available, but they are optional now.

Discover local models:

```bash
python3 smaLLMs.py doctor
python3 smaLLMs.py discover
```

Inspect suites and the benchmark catalog:

```bash
python3 smaLLMs.py benchmarks
```

Run the default core suite on every discovered local model:

```bash
python3 smaLLMs.py quick --samples 3
```

Run a specific suite:

```bash
python3 smaLLMs.py run --models llama3.2 qwen2.5:0.5b --benchmarks frontier_report_suite --samples 10
```

Export the latest artifacts:

```bash
python3 smaLLMs.py export
```

## Artifact outputs

Each run writes structured artifacts to:

- `artifacts/runs/<run_id>/`

Website-friendly bundles are exported to:

- `website_exports/latest/`
- `website_exports/runs/<run_id>/`

Useful files:

- `manifest.json`
- `summary.json`
- `leaderboard.json`
- `leaderboard.csv`
- `session.json` for the website
- per-model/per-benchmark sample JSONL artifacts

## Website workflow

The exporter now writes a single self-contained website bundle:

- `website_exports/latest/session.json`

That file includes:

- run metadata
- leaderboard rows
- per-benchmark evaluation summaries
- embedded sample records with prompts, responses, parsed answers, token counts, latency, and raw provider metadata

When the sibling repo exists at `../websmaLLMs`, the exporter also mirrors the latest session into:

- `../websmaLLMs/public/data/latest-session.json`

That means the website can either:

- auto-load the mirrored session on startup
- import any exported `session.json` manually through the UI

If you want to override the mirror location, use:

```bash
python3 run_local_benchmarks.py export --sync-dir /path/to/websmaLLMs/public/data
```

## Benchmark suites

Runnable suites currently include:

- `quick_suite`
- `core_suite`
- `knowledge_suite`
- `commonsense_suite`
- `reasoning_suite`
- `frontier_report_suite`
- `serious_suite`
- `all_benchmarks`

## Project principles

- Local-first: evaluate models on the user’s own hardware.
- CLI-only: the terminal is the primary product surface.
- Honest benchmark accounting: runnable versus tracked is explicit.
- Open artifacts: every run should be exportable and inspectable.
- Cross-platform pragmatism: no platform should be treated as second-class.

## Current limitations

- Some dependencies are required even for local discovery, including `aiohttp`.
- The standard local install is intentionally small; use `requirements-dev.txt` only if you need the broader development environment.
- Frontier agentic benchmarks are tracked in the catalog but not yet executed by the local runner.
- Results are only as comparable as the local runtime settings and hardware conditions you keep consistent.

## Short roadmap

- add more locally runnable public benchmarks where apples-to-apples evaluation is possible
- add dedicated harnesses for code and tool-use benchmarks
- improve benchmark normalization and run metadata for publishable leaderboard workflows
- keep the terminal experience first-class instead of bolting on a web UI
