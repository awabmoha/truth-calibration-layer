# TCL-v0 Reviewed Answer-Mean Ablation Checkpoint

Status: local CPU analysis completed on existing reviewed 1,000-example records.

This checkpoint runs the non-GPU parts of `TCL-v0-Ablation-Plan.md` on the six reviewed Qwen/Phi/Gemma SQuAD and TriviaQA runs. It uses the existing `answer_mean` final-layer records, so it tests raw-only baselines and feature ablations but does not yet test new layer or pooling choices.

## Completed Locally

- temperature scaling on raw confidence
- Platt/logistic calibration on raw confidence
- isotonic calibration on raw confidence
- raw-only conservative fusion
- hidden-only logistic probe
- hidden conservative fusion
- raw-plus-hidden logistic probe
- raw-plus-hidden conservative fusion
- probe-score-plus-raw logistic probe
- weighted and unweighted sensitivity analyses

## Not Yet Completed

Layer and pooling ablations require new hidden-state extraction on Kaggle GPU. The existing reviewed records contain only `answer_mean` at final layer (`hidden_state_layer = -1`).

## Critical Weighted Comparison

Lower is better for Brier, ECE, and wrong-count metrics. Higher is better for AUC.

| Run | Metric | Raw | Best raw-only | Hidden conservative | Raw+hidden | Probe-score+raw |
|---|---:|---:|---:|---:|---:|---:|
| Qwen SQuAD | brier | 0.2079 | 0.2002 | 0.1939 | 0.2017 | 0.2237 |
| Qwen SQuAD | ece | 0.1545 | 0.0836 | 0.1476 | 0.1758 | 0.2104 |
| Qwen SQuAD | auc | 0.7291 | 0.7293 | 0.7951 | 0.7594 | 0.7246 |
| Qwen SQuAD | wrong_conf_ge_0_8 | 31.0000 | 0.0000 | 11.0000 | 19.0000 | 24.0000 |
| Qwen TriviaQA | brier | 0.2546 | 0.1882 | 0.1312 | 0.1524 | 0.1587 |
| Qwen TriviaQA | ece | 0.3540 | 0.2385 | 0.1097 | 0.1370 | 0.1362 |
| Qwen TriviaQA | auc | 0.7655 | 0.7655 | 0.7917 | 0.8032 | 0.7833 |
| Qwen TriviaQA | wrong_conf_ge_0_8 | 9.0000 | 1.0000 | 1.0000 | 9.0000 | 10.0000 |
| Phi SQuAD | brier | 0.0540 | 0.1656 | 0.0515 | 0.0594 | 0.0573 |
| Phi SQuAD | ece | 0.0383 | 0.3074 | 0.0341 | 0.0547 | 0.0524 |
| Phi SQuAD | auc | 0.8301 | 0.8301 | 0.8767 | 0.8333 | 0.8429 |
| Phi SQuAD | wrong_conf_ge_0_8 | 11.0000 | 0.0000 | 7.0000 | 10.0000 | 8.0000 |
| Phi TriviaQA | brier | 0.2892 | 0.1705 | 0.2012 | 0.2141 | 0.2005 |
| Phi TriviaQA | ece | 0.3169 | 0.0635 | 0.1577 | 0.1957 | 0.1543 |
| Phi TriviaQA | auc | 0.8514 | 0.8514 | 0.8306 | 0.8135 | 0.8198 |
| Phi TriviaQA | wrong_conf_ge_0_8 | 44.0000 | 0.0000 | 11.0000 | 17.0000 | 15.0000 |
| Gemma SQuAD | brier | 0.2910 | 0.2444 | 0.1339 | 0.1368 | 0.1270 |
| Gemma SQuAD | ece | 0.2490 | 0.1307 | 0.1211 | 0.1429 | 0.1115 |
| Gemma SQuAD | auc | 0.5125 | 0.5313 | 0.8757 | 0.9024 | 0.9029 |
| Gemma SQuAD | wrong_conf_ge_0_8 | 64.0000 | 0.0000 | 11.0000 | 13.0000 | 12.0000 |
| Gemma TriviaQA | brier | 0.4923 | 0.2190 | 0.1884 | 0.2034 | 0.1960 |
| Gemma TriviaQA | ece | 0.5764 | 0.2447 | 0.1750 | 0.1911 | 0.1772 |
| Gemma TriviaQA | auc | 0.7291 | 0.7291 | 0.7709 | 0.7686 | 0.7685 |
| Gemma TriviaQA | wrong_conf_ge_0_8 | 80.0000 | 0.0000 | 10.0000 | 17.0000 | 15.0000 |

- Hidden conservative beats best raw-only on brier: 5 of 6 reviewed runs.
- Hidden conservative beats best raw-only on ece: 4 of 6 reviewed runs.
- Hidden conservative beats best raw-only on auc: 5 of 6 reviewed runs.
- Hidden conservative beats best raw-only on wrong_conf_ge_0_8: 0 of 6 reviewed runs.

## Interpretation

This checkpoint strengthens the baseline story but does not finish the full ablation plan. The local results answer whether existing final-layer `answer_mean` hidden states still add value against raw-only calibration and feature fusions. The remaining research question is whether the signal is stable across hidden layer choice and pooling method.

The report should not be read as full TCL validation. It remains a TCL-v0 diagnostic around frozen models and short-answer QA correctness labels.

## Files

- `combined_ablation_summary.csv`
- `best_signal_by_metric.csv`
- `critical_comparisons.csv`
