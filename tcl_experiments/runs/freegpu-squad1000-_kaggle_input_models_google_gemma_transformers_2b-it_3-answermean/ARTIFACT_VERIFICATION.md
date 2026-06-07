# TCL-v0 Run Artifact Verification

Run directory: `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean`
Status: `pass`
Records: `1000`
Test predictions: `200`

## Required Files

- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/records_answer_mean.jsonl`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/records_answer_mean.config.json`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/summary.json`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/method_summary.csv`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/answer_mean/metrics.json`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/answer_mean/test_predictions.csv`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/answer_mean/raw_reliability_bins.csv`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/answer_mean/tcl_v0_reliability_bins.csv`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/answer_mean/tcl_v0_conservative_reliability_bins.csv`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/analysis/answer_mean/tcl_v0_calibrated_reliability_bins.csv`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/manual_review_all.csv`: present
- `runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean/targeted_manual_review_candidates.csv`: present

## Metric Comparison

### answer_mean
- tcl_v0_probe_confidence: 6 wins, 1 losses, 0 ties vs raw
- tcl_v0_conservative_confidence: 6 wins, 1 losses, 0 ties vs raw
- tcl_v0_calibrated_confidence: 6 wins, 1 losses, 0 ties vs raw

## Claim Boundary

This verifies a TCL-v0 run artifact only. It does not validate the full TCL theory.
