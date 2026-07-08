# Agent Harness Findings

Date: 2026-07-08

This note records the local harness comparison run produced by `smaLLMs.py agent-harness`.
It compares Pi, OpenCode, and Codex CLI as agent harnesses around GPT-5.5/xhigh defaults on the
same deterministic Python fixture tasks.

## Latest Measured Run

- Run ID: `agent-harness-2026-07-08T054337-221146Z`
- Detailed artifact: `artifacts/agent_harness/runs/agent-harness-2026-07-08T054337-221146Z/summary.json`
- Human report: `artifacts/agent_harness/runs/agent-harness-2026-07-08T054337-221146Z/RESULTS.md`
- Web payload: `../websmaLLMs/public/data/agent-harness/latest.json`
- Rows: 9
- Passed: 9
- Failed: 0

## Aggregate Results

| Harness | Passed | Avg seconds | Total seconds | Reported tokens | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pi | 3/3 | 63.628 | 190.883 | not reported | 175.6 MB |
| OpenCode | 3/3 | 78.196 | 234.587 | not reported | 634.9 MB |
| Codex CLI | 3/3 | 74.643 | 223.929 | 38,289 | 133.5 MB |

## Task Results

| Harness | Task | Status | Seconds | Reported tokens | Peak RSS | Changed files |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Pi | median_bugfix | passed | 17.411 | not reported | 169.7 MB | `calcstats/stats.py` |
| Pi | cli_feature | passed | 57.216 | not reported | 175.6 MB | `calcstats/cli.py` |
| Pi | path_safety | passed | 116.256 | not reported | 169.9 MB | `calcstats/files.py` |
| OpenCode | median_bugfix | passed | 70.971 | not reported | 605.0 MB | `calcstats/stats.py` |
| OpenCode | cli_feature | passed | 74.402 | not reported | 634.9 MB | `calcstats/cli.py` |
| OpenCode | path_safety | passed | 89.214 | not reported | 610.6 MB | `calcstats/files.py` |
| Codex CLI | median_bugfix | passed | 85.310 | 14,317 | 133.4 MB | `calcstats/stats.py` |
| Codex CLI | cli_feature | passed | 79.096 | 14,533 | 133.5 MB | `calcstats/cli.py` |
| Codex CLI | path_safety | passed | 59.523 | 9,439 | 133.5 MB | `calcstats/files.py` |

## Interpretation

Quality was a tie on this small fixture set: all three harnesses passed all targeted tests and
changed only the expected file for each task.

Pi was the fastest overall in this measured run. Its average row time was about 14.6 seconds faster
than OpenCode and about 11.0 seconds faster than Codex CLI. The difference is noticeable but not a
quality difference on these tasks.

Codex CLI was not the quality loser here. It was second on total time, fastest on `path_safety`, and
had the lowest measured local peak RSS. It is also the only harness in this run that reported token
usage in captured CLI output.

OpenCode was the slowest overall and used the most local memory in this run. It still passed every
task cleanly, so the negative signal is speed/resource usage rather than result quality.

## Measurement Caveats

- Token, cost, and context fields are nullable because Pi and OpenCode did not report comparable
  per-run usage in captured stdout/stderr.
- Codex CLI reported a single total token count; prompt/completion/reasoning token splits were not
  available from the captured log.
- Peak RSS comes from GNU `time -v` around the local subprocess. It is local harness process memory,
  not provider-side inference memory.
- The fixture suite is intentionally small. It is good for workflow smoke testing and harness
  comparison, not a broad claim about all coding-agent behavior.
