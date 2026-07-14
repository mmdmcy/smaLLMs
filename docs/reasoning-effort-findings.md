# Codex Reasoning-Effort Findings

Date: 2026-07-10

This note records the first reasoning-effort sweep. It was intentionally stopped after the Sol
matrix to conserve usage; the public payload is marked `partial`.

## Completed scope

- Model: `gpt-5.6-sol`
- Efforts: `low`, `medium`, `high`, `xhigh`, `max`, `ultra`
- Task: `median_bugfix`
- Rows: 6
- Passed: 6
- Failed: 0
- Reported CLI tokens: 92,247
- Total agent time: 349.828 seconds
- Export: `../websmaLLMs/public/data/reasoning-efforts/latest.json`
- Run artifact: `artifacts/reasoning_sweep/runs/reasoning-sweep-2026-07-10T162945-599873Z/`

## Results

| Model | Effort | Result | Seconds | Reported tokens |
| --- | --- | ---: | ---: | ---: |
| GPT-5.6 Sol | low | 1/1 | 17.936 | 6,486 |
| GPT-5.6 Sol | medium | 1/1 | 53.407 | 17,224 |
| GPT-5.6 Sol | high | 1/1 | 75.566 | 13,270 |
| GPT-5.6 Sol | xhigh | 1/1 | 69.668 | 16,528 |
| GPT-5.6 Sol | max | 1/1 | 77.425 | 24,526 |
| GPT-5.6 Sol | ultra | 1/1 | 55.826 | 14,213 |

## Interpretation

- Every completed Sol setting passed the deterministic fixture and changed only the expected file.
- `low` was fastest and used the fewest reported tokens in this one-task sample.
- The non-monotonic time/token curve means effort labels should be measured, not treated as a
  strict linear cost ladder.
- This does not establish that Sol is better than Terra or Luna, or that `max`/`ultra` improves
  quality on broader coding work. Terra and Luna were not completed in this sweep.

## Export contract

The reasoning-effort payload is intentionally separate from:

- `latest-session.json` for ordinary Ollama/LM Studio model runs;
- `agent-harness/latest.json` for Pi/OpenCode/Codex harness comparisons;
- `reasoning-efforts/latest.json` for Codex model/effort sweeps.

Codex CLI does not expose sampling temperature in this path, so the export records
`sampling_temperature: null` and uses `reasoning_effort` as the measured control.
