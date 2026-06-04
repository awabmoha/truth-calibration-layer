# TCL-v0 Cross-Model Comparison

Status: early diagnostics, not full TCL validation

This comparison uses the same 500-example TriviaQA `rc.nocontext` validation subset and the same split file for two small CPU-runnable models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

Both runs use:

- hidden-state method: `answer_mean`
- probe: logistic regression on frozen hidden states
- conservative confidence: `min(raw_generation_confidence, tcl_v0_probe_confidence)`
- metrics on the 100-example held-out test split

## Test Metric Summary

### Qwen 500, Strict Labels

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2793 | 0.2107 | 0.3537 | 0.6300 | 0 | 0 |
| TCL-v0 probe confidence | 0.1252 | 0.1380 | 0.3601 | 0.8400 | 4 | 3 |
| Conservative TCL-v0 | 0.1088 | 0.1268 | 0.3601 | 0.8400 | 0 | 0 |

### SmolLM2 500, Strict Labels

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.4519 | 0.2953 | 0.9212 | 0.4800 | 0.5714 | 6 | 3 |
| TCL-v0 probe confidence | 0.0961 | 0.0824 | 0.8522 | 0.9000 | 0.9339 | 5 | 2 |
| Conservative TCL-v0 | 0.0661 | 0.0597 | 0.6442 | 0.9300 | 0.9309 | 0 | 0 |

## Interpretation

The same qualitative pattern appears in both models:

- Frozen hidden states contain useful correctness/calibration signal.
- The plain probe can still produce high-confidence wrong answers.
- Conservative TCL-v0 is safer than the plain probe.
- Conservative TCL-v0 improves calibration metrics over raw generation confidence.
- Conservative TCL-v0 avoids high-confidence wrong answers in both held-out test splits reviewed so far.
- The stricter label rule reduced accepted-answer substring false positives and should be used for future runs.

## Claim Boundary

Allowed:

- TCL-v0 has now been tested on two small CPU-runnable LLMs using the same TriviaQA subset.
- Conservative TCL-v0 is the strongest current diagnostic variant across these runs.
- The results support continuing TCL-v0 experiments.

Not allowed:

- TCL is validated.
- TCL solves hallucination.
- TCL-v0 generalizes broadly.
- The full TCL trust vector has been implemented.

## Next Step

The next research step should either:

- test a second benchmark source, or
- improve the correctness-labeling rule before scaling to more data.

The labeling-rule improvement has now been added as `strict_answer_segment_match_v1`. The next step is a second benchmark source.
