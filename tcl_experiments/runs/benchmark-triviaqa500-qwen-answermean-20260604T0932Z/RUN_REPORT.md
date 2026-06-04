# TCL-v0 TriviaQA 500-Example Benchmark Diagnostic Report

Run ID: `benchmark-triviaqa500-qwen-answermean-20260604T0932Z`

Status: larger benchmark diagnostic, not full TCL validation

## Purpose

This run scales the first TriviaQA diagnostic from 200 examples to 500 examples while keeping the same model, hidden-state method, and conservative TCL-v0 scoring rule. The goal is to test whether the earlier pattern remains visible on a larger held-out test split.

This run evaluates only the confidence-only TCL-v0 probe:

```text
frozen generated-answer hidden states -> logistic-regression probe -> correctness confidence
```

## Model and Data

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset source: `mandarjoshi/trivia_qa`
- Config: `rc.nocontext`
- Source split: `validation`
- Prepared subset size: 500 examples
- Prepared dataset file: `data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_500.csv`
- Split file: `data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_500_splits.csv`
- Hidden-state method: `answer_mean`
- Hidden-state layer: final layer (`-1`)
- Max new tokens: 12
- Raw confidence method: geometric mean generated-token probability
- Correctness method: normalized exact match or word-boundary accepted-answer match

## Label Summary

- Total records: 500
- Correct labels: 110
- Incorrect labels: 390
- Overall model correctness: 0.22
- Train split: 325 examples, 78 correct, 247 incorrect
- Validation split: 75 examples, 6 correct, 69 incorrect
- Test split: 100 examples, 26 correct, 74 incorrect

## Test Metrics

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2668 | 0.2049 | 0.3537 | 0.6400 | 0.8430 | 0 | 0 |
| TCL-v0 probe confidence | 0.1768 | 0.1739 | 0.4511 | 0.8200 | 0.8020 | 5 | 5 |
| Validation-calibrated TCL-v0 | 0.2039 | 0.1877 | 0.3383 | 0.7200 | 0.8020 | 2 | 0 |
| Conservative TCL-v0 | 0.1282 | 0.1541 | 0.2679 | 0.8200 | 0.8124 | 0 | 0 |

## Interpretation

The 500-example diagnostic repeats the main pattern from the 200-example run.

The plain hidden-state probe improved average calibration and threshold accuracy over raw generation confidence:

- ECE improved from 0.2668 to 0.1768.
- Brier score improved from 0.2049 to 0.1739.
- Accuracy at 0.5 improved from 0.6400 to 0.8200.

But the plain probe again produced dangerous high-confidence errors:

- 5 wrong test examples had probe confidence >= 0.8.
- The same 5 wrong test examples also had probe confidence >= 0.9.

The conservative confidence rule performed best overall:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

It had the best ECE, best Brier score, best MCE, tied-best 0.5-threshold accuracy, and zero wrong test examples at or above 0.8 confidence.

## High-Risk Manual Review

The 5 high-confidence wrong plain-probe cases were manually reviewed. No label changes were made; all 5 appear to be genuine model errors under the accepted answers.

Examples:

- Longest moon landing: model said Apollo 11; accepted answer was Apollo 17.
- Black Hills rivers: model said Missouri and Big Sioux; accepted answer was Belle Fourche and Cheyenne.
- Irish coffee ingredient: model described coffee beans and did not answer whiskey/whisky.
- `The Sign Of Four` author: model said Agatha Christie; accepted answer was Arthur Conan Doyle.
- Macbeth dynasty: model said House of Scots; accepted answer was House of Dunkeld/Canmore.

Review artifact:

- `high_risk_manual_review.csv`

Additional review target:

- `top_confidence_disagreements.csv`

## Extended Manual Review Update

After the high-risk review, the audit was expanded to:

- the 25 top raw-vs-probe confidence disagreement cases
- all 26 automatic positive labels in the 100-example held-out test split

This found 3 automatic false positives, all caused by accepted-answer substring matching. The reviewed held-out test split has 23 positives instead of 26.

Reviewed-label metric summary:

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2793 | 0.2107 | 0.3537 | 0.6300 | 0 | 0 |
| TCL-v0 probe confidence | 0.1568 | 0.1622 | 0.4511 | 0.8300 | 5 | 5 |
| Validation-calibrated TCL-v0 | 0.2339 | 0.1910 | 0.4098 | 0.7100 | 2 | 0 |
| Conservative TCL-v0 | 0.1021 | 0.1413 | 0.2427 | 0.8300 | 0 | 0 |

Extended review report:

- `EXTENDED_MANUAL_REVIEW_REPORT.md`

## Claim Boundary

Allowed diagnostic claim:

On a 500-example TriviaQA `rc.nocontext` validation subset using Qwen/Qwen2.5-0.5B-Instruct, conservative TCL-v0 improved ECE, Brier score, MCE, and 0.5-threshold accuracy over raw generation confidence while avoiding high-confidence wrong answers on the held-out test split.

Not allowed:

- This does not validate TCL as a full framework.
- This does not prove hidden states generally predict truth.
- This does not show TCL solves hallucination.
- This does not justify deployment use.
- This does not validate the four-dimensional trust vector.

## Recommended Next Steps

1. Manually review the top confidence-disagreement cases.
2. Review a small random sample from the 100-example held-out test split.
3. After review, commit the 500-example benchmark milestone to GitHub.
4. Consider a second model or dataset only after the current benchmark labels are audited.
