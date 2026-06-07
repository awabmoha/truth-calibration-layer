# TCL-v0 Extended Validation Decision Note

Decision: `incomplete`

The tested matrix does not yet include at least two distinct models and two benchmark types.

## Matrix Coverage

- Completed runs: 2
- Benchmark types: squad, triviaqa
- Models: Qwen/Qwen2.5-0.5B-Instruct
- Model families: qwen
- Matrix complete: False

## Run-Level Results

### freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Benchmark: `squad`
- Records/test: 1000 / 200
- Analysis dir: `runs\freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean\analysis`
- Targeted manual review: `missing`
- Conservative TCL-v0 primary metrics: 4 wins, 0 losses, 0 ties vs raw
  - brier: raw=0.207867, conservative=0.193888, winner=candidate
  - ece: raw=0.154464, conservative=0.14757, winner=candidate
  - wrong_conf_ge_0_8: raw=31, conservative=11, winner=candidate
  - wrong_conf_ge_0_9: raw=11, conservative=2, winner=candidate

### freegpu-triviaqa1000-Qwen_Qwen2.5-0.5B-Instruct-answermean

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Benchmark: `triviaqa`
- Records/test: 1000 / 200
- Analysis dir: `runs\freegpu-triviaqa1000-Qwen_Qwen2.5-0.5B-Instruct-answermean\analysis`
- Targeted manual review: `missing`
- Conservative TCL-v0 primary metrics: 4 wins, 0 losses, 0 ties vs raw
  - brier: raw=0.254576, conservative=0.1312, winner=candidate
  - ece: raw=0.354028, conservative=0.10974, winner=candidate
  - wrong_conf_ge_0_8: raw=9, conservative=1, winner=candidate
  - wrong_conf_ge_0_9: raw=3, conservative=0, winner=candidate

## Claim Boundary

This decision note only evaluates TCL-v0 confidence diagnostics. It does not validate the full Truth Calibration Layer theory, does not prove truthfulness, and does not show deployment readiness.
