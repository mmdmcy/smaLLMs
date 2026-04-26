# smaLLMs Evaluation Methodology

smaLLMs is designed for local, open-weight model evaluation on a single machine.
It is intentionally narrower than hosted benchmark platforms: the goal is to make
small local model runs inspectable, repeatable, and easy to export.

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
