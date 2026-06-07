# TCL-v0 Run Artifact Verification

Run directory: `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean`
Status: `pass`
Records: `1000`
Test predictions: `200`

## Required Files

- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/records_answer_mean.jsonl`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/records_answer_mean.config.json`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/summary.json`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/method_summary.csv`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/answer_mean/metrics.json`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/answer_mean/test_predictions.csv`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/answer_mean/raw_reliability_bins.csv`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/answer_mean/tcl_v0_reliability_bins.csv`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/answer_mean/tcl_v0_conservative_reliability_bins.csv`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/analysis/answer_mean/tcl_v0_calibrated_reliability_bins.csv`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/manual_review_all.csv`: present
- `runs/freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean/targeted_manual_review_candidates.csv`: present

## Metric Comparison

### answer_mean
- tcl_v0_probe_confidence: 2 wins, 5 losses, 0 ties vs raw
- tcl_v0_conservative_confidence: 6 wins, 1 losses, 0 ties vs raw
- tcl_v0_calibrated_confidence: 3 wins, 4 losses, 0 ties vs raw

## Claim Boundary

This verifies a TCL-v0 run artifact only. It does not validate the full TCL theory.
