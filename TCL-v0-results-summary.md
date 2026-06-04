# TCL-v0 Results Summary

Status: early diagnostics, not full TCL validation

This summary collects the practical TCL-v0 work completed so far. The full Truth Calibration Layer remains a theoretical framework. TCL-v0 is a narrow confidence-only experiment:

```text
frozen LLM hidden states -> small probe -> correctness confidence
```

## Current Best Variant

The current best diagnostic variant is conservative TCL-v0:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

This rule was introduced because the plain hidden-state probe improved average calibration but produced high-confidence wrong answers. The conservative rule prevents the probe from increasing confidence above raw generation confidence.

## Local System Constraint

The local machine can run small CPU-only models, but not large 7B/8B models comfortably.

Observed practical setup:

- Model used: `Qwen/Qwen2.5-0.5B-Instruct`
- Hardware: AMD Ryzen 7 5825U, about 6 GB RAM, integrated AMD graphics, no CUDA
- Suitable local mode: small CPU inference

## Smoke Test

First smoke test:

- Model: `sshleifer/tiny-gpt2`
- Dataset: 20 seed questions
- Result: pipeline ran, but model got 0/20 correct
- Outcome: useful plumbing check only; not evidence

Second smoke test:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: 20 seed questions
- Result: mixed labels and probe training worked
- Outcome: confirmed the full TCL-v0 record -> probe -> metrics path

## Local 100-Question Diagnostic

Dataset:

- `tcl_experiments/data/diagnostic_questions_100.csv`

Model:

- `Qwen/Qwen2.5-0.5B-Instruct`

Result:

- Correct labels: 91/100
- `answer_mean` beat `answer_last` on ECE, Brier score, and MCE
- `answer_last` had higher AUC, so it may still matter for ranking/selective prediction

Method comparison report:

- `tcl_experiments/TCL-v0-method-comparison-qwen100.md`

Decision from this stage:

- Use `answer_mean` as the default hidden-state method for calibration-focused TCL-v0.

## TriviaQA 200-Example Benchmark Diagnostic

Dataset:

- Source: `mandarjoshi/trivia_qa`
- Config: `rc.nocontext`
- Source split: `validation`
- Prepared subset: 200 examples
- Recorded splits: 130 train, 30 validation, 40 test

Run:

- `tcl_experiments/runs/benchmark-triviaqa200-qwen-answermean-20260603T1700Z/`

Model:

- `Qwen/Qwen2.5-0.5B-Instruct`

Label summary:

- Correct labels: 36/200
- Incorrect labels: 164/200
- Test labels: 7 correct, 33 incorrect

## Benchmark Metric Summary

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.3292 | 0.1923 | 0.4544 | 0.6750 | 0.9048 | 0 | 0 |
| TCL-v0 probe confidence | 0.2086 | 0.1850 | 0.9567 | 0.7750 | 0.7576 | 4 | 3 |
| Validation-calibrated TCL-v0 | 0.2742 | 0.2270 | 0.9742 | 0.6750 | 0.7576 | 4 | 1 |
| Conservative TCL-v0 | 0.1347 | 0.1257 | 0.6784 | 0.8250 | 0.7922 | 0 | 0 |

## What The Results Suggest

The plain hidden-state probe showed useful signal:

- It improved ECE over raw confidence.
- It improved Brier score over raw confidence.
- It improved threshold accuracy over raw confidence.

But it also exposed a serious failure:

- It produced high-confidence wrong answers.
- Four incorrect test examples had probe confidence >= 0.8.
- Three incorrect test examples had probe confidence >= 0.9.

Validation-score calibration was not enough:

- It reduced wrong >= 0.9 from 3 to 1.
- But it worsened ECE, Brier score, and threshold accuracy.

The conservative rule worked best on this split:

- Best ECE.
- Best Brier score.
- Best threshold accuracy.
- Zero wrong test examples >= 0.8 confidence.
- Zero wrong test examples >= 0.9 confidence.

## Failure Analysis

The high-confidence plain-probe errors were manually inspected. They appear to be genuine model errors, not obvious label mistakes.

Examples:

- TAAG country: model said Taiwan; accepted answer was Angola.
- Armenia 1988 disaster: model said famine; accepted answer was earthquake.
- Israel head of state in 1993: model said Yitzhak Rabin; accepted answer was Ezer Weizman.
- 1984 LA Olympics Eastern Bloc country: model said Poland; accepted answer was Romania.

Failure report:

- `tcl_experiments/runs/benchmark-triviaqa200-qwen-answermean-20260603T1700Z/FAILURE_ANALYSIS.md`

## Held-Out Test Manual Review

The 40 held-out test examples from the first TriviaQA benchmark diagnostic were manually reviewed.

Review result:

- Reviewed examples: 40
- Auto/manual label agreement: 40/40
- Label changes: 0
- Manual correct labels: 7
- Manual incorrect labels: 33

Review artifacts:

- `tcl_experiments/runs/benchmark-triviaqa200-qwen-answermean-20260603T1700Z/manual_review_test_completed.csv`
- `tcl_experiments/runs/benchmark-triviaqa200-qwen-answermean-20260603T1700Z/MANUAL_REVIEW_TEST_REPORT.md`

Because no held-out test labels changed, the existing benchmark metrics do not need to be recomputed for this run.

## TriviaQA 500-Example Benchmark Diagnostic

The next diagnostic scaled the same setup to 500 TriviaQA examples.

Run:

- `tcl_experiments/runs/benchmark-triviaqa500-qwen-answermean-20260604T0932Z/`

Dataset:

- Source: `mandarjoshi/trivia_qa`
- Config: `rc.nocontext`
- Source split: `validation`
- Prepared subset: 500 examples
- Recorded splits: 325 train, 75 validation, 100 test

Label summary:

- Correct labels: 110/500
- Incorrect labels: 390/500
- Test labels: 26 correct, 74 incorrect

Metric summary:

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2668 | 0.2049 | 0.3537 | 0.6400 | 0.8430 | 0 | 0 |
| TCL-v0 probe confidence | 0.1768 | 0.1739 | 0.4511 | 0.8200 | 0.8020 | 5 | 5 |
| Validation-calibrated TCL-v0 | 0.2039 | 0.1877 | 0.3383 | 0.7200 | 0.8020 | 2 | 0 |
| Conservative TCL-v0 | 0.1282 | 0.1541 | 0.2679 | 0.8200 | 0.8124 | 0 | 0 |

High-risk review:

- Reviewed high-confidence wrong plain-probe cases: 5
- Label changes: 0
- All 5 reviewed cases appear to be genuine model errors.

Extended held-out review:

- Reviewed all 26 automatic positive labels in the 100-example test split.
- Reviewed the 25 top raw-vs-probe disagreement cases.
- Found 3 automatic false positives caused by accepted-answer substring matching.
- Reviewed test positives: 23/100.

Reviewed-label metric summary:

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2793 | 0.2107 | 0.3537 | 0.6300 | 0 | 0 |
| TCL-v0 probe confidence | 0.1568 | 0.1622 | 0.4511 | 0.8300 | 5 | 5 |
| Validation-calibrated TCL-v0 | 0.2339 | 0.1910 | 0.4098 | 0.7100 | 2 | 0 |
| Conservative TCL-v0 | 0.1021 | 0.1413 | 0.2427 | 0.8300 | 0 | 0 |

Reports:

- `tcl_experiments/runs/benchmark-triviaqa500-qwen-answermean-20260604T0932Z/RUN_REPORT.md`
- `tcl_experiments/runs/benchmark-triviaqa500-qwen-answermean-20260604T0932Z/HIGH_RISK_REVIEW_REPORT.md`
- `tcl_experiments/runs/benchmark-triviaqa500-qwen-answermean-20260604T0932Z/EXTENDED_MANUAL_REVIEW_REPORT.md`

Result:

The 500-example run repeats the main pattern from the 200-example run. After reviewed-label correction, conservative TCL-v0 remains the strongest current diagnostic score because it improves ECE, Brier score, MCE, and threshold accuracy over raw confidence while avoiding high-confidence wrong answers on the held-out test split.

## Second-Model Diagnostic

The same 500-example TriviaQA setup was then run on a second small CPU-runnable model:

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Run: `tcl_experiments/runs/benchmark-triviaqa500-smollm2-360m-answermean-20260604T1018Z/`
- Test labels: 8 correct, 92 incorrect

Metric summary:

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.4419 | 0.2924 | 0.9212 | 0.4900 | 0.5951 | 6 | 3 |
| TCL-v0 probe confidence | 0.0714 | 0.0676 | 0.8545 | 0.9200 | 0.9416 | 4 | 2 |
| Validation-calibrated TCL-v0 | 0.1763 | 0.1068 | 0.7577 | 0.8600 | 0.9416 | 6 | 4 |
| Conservative TCL-v0 | 0.0493 | 0.0478 | 0.3404 | 0.9400 | 0.9457 | 0 | 0 |

Manual review:

- High-risk wrong cases reviewed: 12
- Automatic positive labels reviewed: 8
- Label changes: 0

Reports:

- `tcl_experiments/runs/benchmark-triviaqa500-smollm2-360m-answermean-20260604T1018Z/RUN_REPORT.md`
- `tcl_experiments/runs/benchmark-triviaqa500-smollm2-360m-answermean-20260604T1018Z/HIGH_RISK_REVIEW_REPORT.md`
- `tcl_experiments/TCL-v0-cross-model-comparison.md`

Result:

The conservative TCL-v0 pattern now appears across two small models on the same TriviaQA subset. This supports continuing the research direction, but it is still not broad validation.

## Stricter Labeling Rule

The automatic correctness rule was improved after manual review exposed false positives from accepted-answer substring matching.

New method:

```text
strict_answer_segment_match_v1
```

Relabeling existing records only, with no model reruns:

| Run | Old Positives | Strict Positives | Label Changes |
|---|---:|---:|---:|
| Qwen 500 | 110 | 100 | 14 |
| SmolLM2 500 | 46 | 35 | 13 |

Strict-label metric summary:

| Model | Signal | ECE | Brier | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|
| Qwen 500 | Raw generation confidence | 0.2793 | 0.2107 | 0.6300 | 0 | 0 |
| Qwen 500 | Conservative TCL-v0 | 0.1088 | 0.1268 | 0.8400 | 0 | 0 |
| SmolLM2 500 | Raw generation confidence | 0.4519 | 0.2953 | 0.4800 | 6 | 3 |
| SmolLM2 500 | Conservative TCL-v0 | 0.0661 | 0.0597 | 0.9300 | 0 | 0 |

Report:

- `tcl_experiments/TCL-v0-labeling-rule-update.md`

## NQ-Open Second Benchmark

The stricter label rule was then used on a second benchmark source:

- Dataset: `google-research-datasets/nq_open`
- Config: `nq_open`
- Split: `validation`
- Prepared subset: 500 examples
- Splits: 325 train, 75 validation, 100 test

Runs:

- Qwen: `tcl_experiments/runs/benchmark-nqopen500-qwen-answermean-20260604T1049Z-retry2/`
- SmolLM2: `tcl_experiments/runs/benchmark-nqopen500-smollm2-360m-answermean-20260604T1118Z/`

Reviewed-label results:

| Model | Signal | ECE | Brier | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|
| Qwen NQ-Open | Raw generation confidence | 0.4883 | 0.2757 | 0.4800 | 0 | 0 |
| Qwen NQ-Open | Conservative TCL-v0 | 0.1018 | 0.0667 | 0.9200 | 0 | 0 |
| SmolLM2 NQ-Open | Raw generation confidence | 0.4416 | 0.2376 | 0.6200 | 3 | 0 |
| SmolLM2 NQ-Open | TCL-v0 probe confidence | 0.0135 | 0.0087 | 0.9900 | 0 | 0 |
| SmolLM2 NQ-Open | Conservative TCL-v0 | 0.0164 | 0.0100 | 0.9900 | 0 | 0 |

Report:

- `tcl_experiments/TCL-v0-nq-open-benchmark-summary.md`

Interpretation:

NQ-Open is a harder and sparser stress test under the current short-answer setup. The reviewed test splits have very few positives, so these metrics must be treated cautiously. Still, the second benchmark supports the central TCL-v0 direction: hidden-state probe confidence improves calibration over raw confidence, and conservative TCL-v0 remains a strong safer variant.

## Research Interpretation

The strongest current statement is:

TCL-v0 shows preliminary evidence that frozen hidden states contain useful calibration signal, but the signal must be constrained because a plain probe can become overconfident on fluent wrong answers.

The refined TCL-v0 hypothesis is:

```text
Hidden-state probe confidence can improve calibration when used as a constrained companion to raw generation confidence.
```

## Claim Boundaries

Allowed:

- TCL-v0 has a working diagnostic pipeline.
- `answer_mean` is the current default hidden-state method for calibration diagnostics.
- Conservative TCL-v0 performed best on the 200-example and 500-example TriviaQA diagnostics, including a second-model run.
- The current results support continuing the research direction.

Not allowed:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 generalizes beyond the tested model/dataset.
- The four-dimensional TCL trust vector has been implemented or validated.

## Next Step

The next research step is to improve the benchmark protocol for sparse open-domain QA, either by scaling NQ-Open, improving answer extraction, or adding a third benchmark with cleaner short-answer labels.
