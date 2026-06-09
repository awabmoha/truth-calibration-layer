# TCL-v0 Reviewed Answer-Mean Ablation Checkpoint

Status: local CPU analysis completed on existing reviewed 1,000-example records.

Date: June 9, 2026.

This checkpoint executes the non-GPU parts of `TCL-v0-Ablation-Plan.md` on the six reviewed Qwen, Phi, and Gemma SQuAD/TriviaQA runs. It uses the already collected `answer_mean` final-layer records, so it tests raw-only calibration baselines and feature ablations. It does not yet test new hidden-layer or pooling choices because those require new hidden-state extraction on Kaggle GPU.

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

## Main Weighted Result

Lower is better for Brier, ECE, and wrong-count metrics. Higher is better for AUC.

| Run | Metric | Raw | Best raw-only | Hidden conservative | Raw+hidden | Probe-score+raw |
|---|---:|---:|---:|---:|---:|---:|
| Qwen SQuAD | Brier | 0.2079 | 0.2002 | 0.1939 | 0.2017 | 0.2237 |
| Qwen SQuAD | ECE | 0.1545 | 0.0836 | 0.1476 | 0.1758 | 0.2104 |
| Qwen SQuAD | AUC | 0.7291 | 0.7293 | 0.7951 | 0.7594 | 0.7246 |
| Qwen SQuAD | wrong >= 0.8 | 31 | 0 | 11 | 19 | 24 |
| Qwen TriviaQA | Brier | 0.2546 | 0.1882 | 0.1312 | 0.1524 | 0.1587 |
| Qwen TriviaQA | ECE | 0.3540 | 0.2385 | 0.1097 | 0.1370 | 0.1362 |
| Qwen TriviaQA | AUC | 0.7655 | 0.7655 | 0.7917 | 0.8032 | 0.7833 |
| Qwen TriviaQA | wrong >= 0.8 | 9 | 1 | 1 | 9 | 10 |
| Phi SQuAD | Brier | 0.0540 | 0.1656 | 0.0515 | 0.0594 | 0.0573 |
| Phi SQuAD | ECE | 0.0383 | 0.3074 | 0.0341 | 0.0547 | 0.0524 |
| Phi SQuAD | AUC | 0.8301 | 0.8301 | 0.8767 | 0.8333 | 0.8429 |
| Phi SQuAD | wrong >= 0.8 | 11 | 0 | 7 | 10 | 8 |
| Phi TriviaQA | Brier | 0.2892 | 0.1705 | 0.2012 | 0.2141 | 0.2005 |
| Phi TriviaQA | ECE | 0.3169 | 0.0635 | 0.1577 | 0.1957 | 0.1543 |
| Phi TriviaQA | AUC | 0.8514 | 0.8514 | 0.8306 | 0.8135 | 0.8198 |
| Phi TriviaQA | wrong >= 0.8 | 44 | 0 | 11 | 17 | 15 |
| Gemma SQuAD | Brier | 0.2910 | 0.2444 | 0.1339 | 0.1368 | 0.1270 |
| Gemma SQuAD | ECE | 0.2490 | 0.1307 | 0.1211 | 0.1429 | 0.1115 |
| Gemma SQuAD | AUC | 0.5125 | 0.5313 | 0.8757 | 0.9024 | 0.9029 |
| Gemma SQuAD | wrong >= 0.8 | 64 | 0 | 11 | 13 | 12 |
| Gemma TriviaQA | Brier | 0.4923 | 0.2190 | 0.1884 | 0.2034 | 0.1960 |
| Gemma TriviaQA | ECE | 0.5764 | 0.2447 | 0.1750 | 0.1911 | 0.1772 |
| Gemma TriviaQA | AUC | 0.7291 | 0.7291 | 0.7709 | 0.7686 | 0.7685 |
| Gemma TriviaQA | wrong >= 0.8 | 80 | 0 | 10 | 17 | 15 |

## Win Counts Against Best Raw-Only Baseline

Using the weighted comparison:

- hidden conservative beats best raw-only on Brier: 5 of 6 reviewed runs
- hidden conservative beats best raw-only on ECE: 4 of 6 reviewed runs
- hidden conservative beats best raw-only on AUC: 5 of 6 reviewed runs
- hidden conservative beats best raw-only on wrong answers with confidence >= 0.8: 0 of 6 reviewed runs

## Interpretation

The checkpoint supports a narrower and stronger claim than the earlier raw-confidence comparison:

```text
Final-layer answer-mean hidden states often improve Brier score, ECE, and AUC beyond raw-only calibration in the reviewed QA runs, but high-confidence wrong-answer reduction is partly explained by ordinary raw-only post-hoc calibration.
```

This is a meaningful result for TCL-v0, but it is not a complete ablation. The remaining critical question is whether the signal survives hidden-layer and pooling choices.

## Negative Result

Raw-only calibration is a stronger competitor than the original raw-confidence baseline. In particular, raw-only calibrated scores can drive high-confidence wrong-answer counts to zero by suppressing confidence. Therefore, high-confidence wrong-answer reduction alone should no longer be used as the main evidence for hidden-state-specific signal.

Phi TriviaQA remains the clearest negative case for hidden conservative TCL-v0 under the weighted comparison: best raw-only calibration is stronger on Brier, ECE, AUC, and high-confidence wrong-answer count.

## Remaining Kaggle Work

The following part of `TCL-v0-Ablation-Plan.md` still requires GPU execution:

- hidden layer choice: `early_middle`, `middle`, `final`
- pooling method: `answer_mean`, `answer_last`, `prompt_answer_mean`
- the same raw-only and hidden-state feature comparisons on the newly extracted records

The notebook protocol for this is:

```text
notebooks/tcl_v0_kaggle_ablation_smoke.ipynb
```

The runbook is:

```text
TCL-v0-Kaggle-Ablation-Runbook.md
```

## Generated Artifacts

Aggregate files:

```text
tcl_experiments/runs/ablation_reviewed_answermean_checkpoint/combined_ablation_summary.csv
tcl_experiments/runs/ablation_reviewed_answermean_checkpoint/best_signal_by_metric.csv
tcl_experiments/runs/ablation_reviewed_answermean_checkpoint/critical_comparisons.csv
tcl_experiments/runs/ablation_reviewed_answermean_checkpoint/TCL-v0-REVIEWED-ANSWERMEAN-ABLATION-CHECKPOINT.md
```

Per-run weighted and unweighted ablation outputs are stored under each reviewed run folder as:

```text
ablation_analysis_weighted/
ablation_analysis_unweighted/
```

## Claim Boundary

This checkpoint is a TCL-v0 diagnostic around frozen models and short-answer QA correctness labels. It does not validate full TCL, a general truth detector, hallucination solving, or deployment readiness.
