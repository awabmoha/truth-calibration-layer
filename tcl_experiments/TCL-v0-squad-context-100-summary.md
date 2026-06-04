# TCL-v0 SQuAD Context 100 Diagnostic

Status: context-grounded benchmark diagnostic, not broad validation

This run tested TCL-v0 on a cleaner short-answer benchmark after NQ-Open proved too sparse for the small CPU-runnable models.

Protocol:

- dataset: `rajpurkar/squad`
- config: `plain_text`
- source split: `validation`
- prepared subset: 100 examples
- split: 65 train, 15 validation, 20 test
- prompt: `chat_short_factual_answer_v1` with context included
- records preserve `context`, `raw_model_output`, and cleaned `model_answer`
- correctness method: `strict_answer_segment_match_v1`
- hidden-state method: `answer_mean`

## Runs

- Qwen: `runs/benchmark-squad100-qwen-answermean-20260604T1214Z/`
- SmolLM2: `runs/benchmark-squad100-smollm2-360m-answermean-20260604T1632Z/`

## Label Summary

| Model | All Correct | All Incorrect | Test Correct | Test Incorrect |
|---|---:|---:|---:|---:|
| Qwen | 85 | 15 | 14 | 6 |
| SmolLM2 | 61 | 39 | 12 | 8 |

## Test Metrics

| Model | Signal | ECE | Brier | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | Raw generation confidence | 0.1755 | 0.2183 | 0.7000 | 0.6667 | 3 | 2 |
| Qwen | TCL-v0 probe confidence | 0.1790 | 0.1696 | 0.8000 | 0.8452 | 2 | 2 |
| Qwen | Validation-calibrated TCL-v0 | 0.1782 | 0.1739 | 0.8000 | 0.8452 | 1 | 1 |
| Qwen | Conservative TCL-v0 | 0.2203 | 0.1401 | 0.8000 | 0.9286 | 0 | 0 |
| SmolLM2 | Raw generation confidence | 0.2101 | 0.2196 | 0.6500 | 0.7500 | 3 | 0 |
| SmolLM2 | TCL-v0 probe confidence | 0.2837 | 0.2695 | 0.7000 | 0.7500 | 2 | 2 |
| SmolLM2 | Validation-calibrated TCL-v0 | 0.1994 | 0.2211 | 0.7000 | 0.7500 | 1 | 0 |
| SmolLM2 | Conservative TCL-v0 | 0.2550 | 0.2512 | 0.7000 | 0.7708 | 1 | 0 |

## Interpretation

SQuAD is a better local benchmark for this stage than NQ-Open because it gives the small models enough correct and incorrect examples for a meaningful first probe diagnostic.

The Qwen run supports the TCL-v0 direction:

- the plain probe improves Brier score, threshold accuracy, and AUC over raw confidence
- conservative TCL-v0 has the best Brier score and AUC
- conservative TCL-v0 removes all wrong test examples with confidence >= 0.8

The SmolLM2 run is mixed:

- threshold accuracy improves from 0.65 to 0.70
- validation-calibrated TCL-v0 slightly improves ECE over raw confidence
- plain and conservative probe scores do not beat raw confidence on Brier score

This means the SQuAD result is not a simple "TCL-v0 always wins" story. It is more useful than that: it shows the probe can help on one model, but the calibration behavior depends on model, dataset, and score construction.

## Label Audit

The Qwen high-risk test cases were inspected against their contexts. The most important wrong answers appear to be genuine model errors, not obvious label mistakes:

- answered `San Francisco Bay Area` when the question asked for the city and the accepted answer was `Santa Clara`
- answered `Golden Anniversary` when the answer should be `Super Bowl L` or `L`
- answered `Coldplay` when the question asked for the two guest artists, `Beyonce and Bruno Mars`

Review CSVs:

- `runs/benchmark-squad100-qwen-answermean-20260604T1214Z/review_all.csv`
- `runs/benchmark-squad100-smollm2-360m-answermean-20260604T1632Z/review_all.csv`

## Claim Boundary

These are 100-example diagnostics with only 20 held-out test examples per model. They support continuing the TCL-v0 research direction and show SQuAD is a cleaner next benchmark, but they do not validate the full TCL framework.

## Next Step

Scale the context-grounded benchmark once the protocol is stable, likely SQuAD 500, then repeat the same label-audit and cross-model comparison.
