# smaLLMs Evaluation Methodology

smaLLMs is designed for local, open-weight model evaluation on a single machine.
It is intentionally narrower than hosted benchmark platforms: the goal is to make
small local model runs inspectable, repeatable, and easy to export.

The optional agent-harness runner is a separate comparison mode. It evaluates
coding-agent harnesses such as Pi, OpenCode, and Codex CLI on deterministic
local code-edit fixtures. Those runs measure the tool loop around a model, not
the raw capability of a local Ollama or LM Studio model.

## Scope

The maintained runner evaluates local models served through Ollama or LM Studio.
It focuses on benchmark suites that are realistic for small local models and
excludes frontier-scale context bands that are not practical for ordinary local
runs.

## Data Access

Benchmark datasets are loaded through the Hugging Face `datasets` ecosystem on
first use. Rows are cached outside the git repository in a per-user cache
directory. After the cache is warm, runs can be executed with `--offline`, which
fails before model execution if any selected benchmark lacks enough cached rows.

This separates local model inference from benchmark data acquisition:

- local model calls go to Ollama or LM Studio
- benchmark rows come from the local cache during offline runs
- generated artifacts stay under `artifacts/` and are ignored by git by default

## Reproducibility Metadata

Each run manifest records:

- selected models and discovered provider metadata
- benchmark names and sample counts
- dataset cache readiness and cache file hashes
- git SHA, branch, and dirty-worktree state
- redacted effective config plus a config SHA-256
- operating system, Python version, memory, CPU, and Ollama version when available
- prompt template IDs, prompt template hashes, prompt hashes, and stable sample IDs

Host-identifying details such as hostnames are intentionally excluded from
persisted system metadata.

## Metrics

Run summaries report:

- accuracy and Wilson 95% confidence intervals
- response rate
- invalid-prediction rate
- raw fallback rate for models that fail to produce a normal answer channel
- latency, token, and provider timing fields when the local runtime exposes them

Small sample counts are useful for smoke tests, not model claims. Publishable
comparisons should use enough samples to make confidence intervals meaningful.

## Publishing Run Cards

Use run cards as the human-readable evidence layer for serious comparisons.

Recommended workflow:

```bash
python smaLLMs.py cache --benchmarks serious_suite --samples 25
python smaLLMs.py run --benchmarks serious_suite --samples 25 --offline
```

Before committing a run card, rerun from a clean git tree so the card records
`Git dirty: False`. Copy only the run card or a small curated summary into
`docs/run_cards/`; keep raw generated artifacts in `artifacts/`.

Do not edit run cards to improve results. If a run has high invalid-prediction or
raw-fallback rates, treat that as part of the result.

## Agent-Harness Runs

Use the agent-harness mode for local workflow comparisons between coding-agent
CLIs:

```bash
python smaLLMs.py agent-harness --dry-run
python smaLLMs.py agent-harness --harnesses pi opencode codex
```

Dry runs validate fixture generation and command construction without spending
model calls. Real runs invoke the selected external harnesses and then verify
the result with deterministic `unittest` commands. Raw logs and workspaces are
written under `artifacts/agent_harness/`, which is ignored by git.

Each real row records:

- agent and verification-test start/end timestamps, return codes, timeout flags, and duration
- changed files, unexpected changed files, diff stat, and a SHA-256 for the captured patch
- redacted stdout/stderr log paths, byte counts, character counts, and line counts
- best-effort process telemetry from GNU `time -v`, including peak resident set size when available
- harness-reported token usage, cost, and context fields when the CLI prints them

Missing telemetry is represented as `null` with a `source` value such as `not_reported`. In the
current CLI output, Codex CLI reports a single total token count, while Pi and OpenCode do not report
comparable per-run token or context accounting in captured stdout/stderr. Treat memory values as
local subprocess telemetry, not provider-side memory.

The detailed audit artifact remains:

```bash
artifacts/agent_harness/runs/<run_id>/summary.json
```

The website-facing payload is intentionally smaller and separate from the local Ollama benchmark
session schema:

```bash
artifacts/agent_harness/runs/<run_id>/web_summary.json
../websmaLLMs/public/data/agent-harness/latest.json
../websmaLLMs/public/data/agent-harness/runs/<run_id>.json
```

## Codex Reasoning-Effort Runs

Reasoning-effort sweeps are a third, separate data surface. They compare Codex CLI model/effort
pairs, not Ollama/LM Studio inference and not different harness implementations:

```bash
python3 run_local_benchmarks.py reasoning-sweep --tasks median_bugfix
```

The current Codex catalog exposes Sol and Terra at `low`, `medium`, `high`, `xhigh`, `max`, and
`ultra`, and Luna at `low`, `medium`, `high`, `xhigh`, and `max`. This surface does not set or
measure sampling temperature; it records `sampling_temperature: null` and names the control
`reasoning_effort` so the displayed result is not misinterpreted.

Sweep artifacts are stored under `artifacts/reasoning_sweep/`. The compact website feed is kept
separate from the other two feeds:

```text
../websmaLLMs/public/data/latest-session.json
../websmaLLMs/public/data/agent-harness/latest.json
../websmaLLMs/public/data/reasoning-efforts/latest.json
```

An interrupted sweep may leave completed per-variant artifacts without a sweep-level export. Do
not publish incomplete variants as if they were finished; mark a partial export explicitly.
