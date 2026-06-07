# TCL-v0 Extended Validation Final Decision Note

Generated at: `2026-06-07T16:48:25.069722+00:00`

## Decision

- Status: `ready`
- Aggregator decision: `mixed_continue_cautiously`
- Required stopping-rule choice: `continue TCL-v0 research`

The completed gate shows mixed but useful TCL-v0 signal. Continue TCL-v0 research cautiously rather than moving directly to TCL-v1.

## Gate Summary

- Matrix complete: True
- Completed runs: 4
- Manual-review completed runs: 4
- Aggregate review changes: 24
- Benchmarks: squad, triviaqa
- Models: Qwen/Qwen2.5-0.5B-Instruct, microsoft/Phi-3.5-mini-instruct
- Primary supported runs: 2
- Primary mixed runs: 2
- Primary failed runs: 0
- High-confidence regressions: 0

## Run Table

| Run | Model | Benchmark | Records | Test | Review | Conservative primary W/L/T |
|---|---|---|---:|---:|---|---:|
| `freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean` | `Qwen/Qwen2.5-0.5B-Instruct` | `squad` | 1000 | 200 | `complete` | 2/2/0 |
| `freegpu-triviaqa1000-Qwen_Qwen2.5-0.5B-Instruct-answermean` | `Qwen/Qwen2.5-0.5B-Instruct` | `triviaqa` | 1000 | 200 | `complete` | 4/0/0 |
| `freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean` | `microsoft/Phi-3.5-mini-instruct` | `squad` | 1000 | 200 | `complete` | 2/2/0 |
| `freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean` | `microsoft/Phi-3.5-mini-instruct` | `triviaqa` | 1000 | 200 | `complete` | 4/0/0 |

## Claim Boundary

This note concerns TCL-v0 only: frozen LLM hidden states, a small probe, and confidence calibration for answer correctness. It does not validate the full Truth Calibration Layer theory, does not prove model truthfulness, and does not establish deployment readiness.
