# TCL-v0 TriviaQA Benchmark Diagnostic Report

Run ID: `benchmark-triviaqa200-qwen-answermean-20260603T1700Z`

Status: first small benchmark diagnostic, not full TCL validation

## Purpose

This run tested TCL-v0 on a prepared 200-example subset of TriviaQA using a fixed train/test split. The goal was to move beyond the local hand-built diagnostic CSV and check whether frozen hidden-state probe confidence remains useful on a known factual QA benchmark source.

This run evaluates only the confidence-only TCL-v0 probe:

```text
frozen generated-answer hidden states -> logistic-regression probe -> correctness confidence
```

## Model and Data

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset source: `mandarjoshi/trivia_qa`
- Config: `rc.nocontext`
- Source split: `validation`
- Prepared subset size: 200 examples
- Prepared dataset file: `data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_200.csv`
- Split file: `data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_200_splits.csv`
- Hidden-state method: `answer_mean`
- Hidden-state layer: final layer (`-1`)
- Max new tokens: 12
- Raw confidence method: geometric mean generated-token probability
- Correctness method: normalized exact match or word-boundary accepted-answer match

## Output Files

- Run config: `records_answer_mean.config.json`
- Full records: `records_answer_mean.jsonl`
- Analysis summary: `analysis/summary.json`
- Method summary: `analysis/method_summary.csv`
- Per-method metrics: `analysis/answer_mean/metrics.json`
- Raw reliability bins: `analysis/answer_mean/raw_reliability_bins.csv`
- TCL-v0 reliability bins: `analysis/answer_mean/tcl_v0_reliability_bins.csv`
- Test predictions: `analysis/answer_mean/test_predictions.csv`
- Split record: `analysis/answer_mean/splits.csv`
- Manual review from records: `manual_review_from_records.csv`
- High-confidence probe errors: `high_confidence_probe_errors.csv`
- Failure analysis: `FAILURE_ANALYSIS.md`
- Calibrated analysis: `analysis_calibrated`
- Calibration report: `CALIBRATION_REPORT.md`
- Calibrated high-confidence errors: `calibrated_high_confidence_errors.csv`
- Conservative analysis: `analysis_conservative`
- Conservative confidence report: `CONSERVATIVE_CONFIDENCE_REPORT.md`

## Label Summary

- Total records: 200
- Correct labels: 36
- Incorrect labels: 164
- Overall model correctness: 0.18
- Recorded splits:
  - Train: 130
  - Validation: 30
  - Test: 40
- Train labels:
  - Correct: 25
  - Incorrect: 105
- Test labels:
  - Correct: 7
  - Incorrect: 33

## Diagnostic Metrics

Raw generation confidence:

- Accuracy at 0.5 threshold: 0.675
- ECE: 0.3291638602097115
- MCE: 0.4544312635253378
- Brier score: 0.19228698534478522
- AUC: 0.9047619047619048

TCL-v0 probe confidence:

- Accuracy at 0.5 threshold: 0.775
- ECE: 0.20856175511130853
- MCE: 0.956668612544136
- Brier score: 0.18499081724792069
- AUC: 0.7575757575757576

## Interpretation

This benchmark diagnostic gives a mixed but useful result.

TCL-v0 improved:

- ECE: 0.3292 -> 0.2086
- Brier score: 0.1923 -> 0.1850
- Accuracy at 0.5 threshold: 0.675 -> 0.775

Raw generation confidence remained better at ranking:

- Raw AUC: 0.9048
- TCL-v0 probe AUC: 0.7576

The reliability bins show a major caution: the TCL-v0 probe placed 3 test examples in the 0.9-1.0 confidence bin, and all 3 were incorrect. This caused a very high TCL-v0 MCE of 0.9567. A separate high-confidence error file also records 4 incorrect test examples with TCL-v0 probe confidence at or above 0.8. So although the probe improved average calibration metrics, it still produced dangerous high-confidence errors on this small benchmark split.

## Claim Boundary

Allowed diagnostic claim:

On a 200-example TriviaQA `rc.nocontext` validation subset using Qwen/Qwen2.5-0.5B-Instruct, the TCL-v0 hidden-state probe improved ECE, Brier score, and 0.5-threshold accuracy over raw generation confidence, but raw confidence had better AUC and the probe produced a severe high-confidence error bin.

Not allowed:

- This does not validate TCL as a full framework.
- This does not prove hidden states generally predict truth.
- This does not show TCL solves hallucination.
- This does not justify deployment use.
- This does not validate the four-dimensional trust vector.

## Recommended Next Steps

1. Generate a manual review CSV from this run by filling model answers and automatic labels into the existing review template.
2. Inspect the high-confidence incorrect TCL-v0 test examples. Completed in `FAILURE_ANALYSIS.md`.
3. Add validation-split calibration for the probe, such as Platt scaling or isotonic calibration.
   - Completed initial 1D logistic score calibration in `CALIBRATION_REPORT.md`.
4. Test a conservative confidence rule to reduce high-confidence probe failures.
   - Completed in `CONSERVATIVE_CONFIDENCE_REPORT.md`.
5. Repeat with a larger benchmark subset only after reviewing label quality and high-confidence failures.
