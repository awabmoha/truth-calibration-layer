# TCL-v0 Baseline Checkpoint

Status: CPU-only baseline analysis on existing reviewed run artifacts.

Generated after the June 7, 2026 extended-validation checkpoint.

## Why This Checkpoint Exists

The strongest criticism of TCL-v0 was that it had been compared mainly against raw generation confidence. Raw confidence is often badly calibrated, especially for small models. A stronger test is whether hidden-state TCL-v0 adds value beyond standard raw-only post-hoc calibration.

This checkpoint adds raw-only baselines to the existing reviewed Qwen, Phi, and Gemma 1,000-example runs:

- temperature scaling on raw confidence
- Platt/logistic calibration on raw confidence
- isotonic regression on raw confidence
- conservative raw-only fusion: `min(raw, raw_only_calibrated)`
- hidden-state conservative TCL-v0: `min(raw, hidden_state_probe)`

The raw-only calibrators were fit on the recorded validation split when it had both classes. All six reviewed runs used the validation split. Metrics were evaluated on the reviewed test predictions.

## Important Implementation Notes

The original hidden-state probe already used:

```text
sklearn.linear_model.LogisticRegression(max_iter=1000, class_weight="balanced")
```

That matters because class imbalance was a valid concern. Future `train_probe.py` outputs now also record the probe configuration, including `C`, `class_weight`, solver, feature count, and whether `p > n_train`.

The new raw-only baseline script is:

```text
tcl_experiments/scripts/run_raw_calibration_baselines.py
```

It writes per-run outputs under:

```text
baseline_calibration/
baseline_calibration_unweighted/
```

The weighted baseline uses balanced weighting. The unweighted baseline is included as a sensitivity check.

## Key Result

The critique is partly correct:

```text
High-confidence wrong-answer reduction is not enough by itself to prove hidden-state signal.
```

Raw-only calibrators often reduce or eliminate high-confidence wrong answers because they can push confidence downward. Therefore, the earlier high-confidence-error result cannot be interpreted alone as evidence that hidden states are doing the work.

The hidden-state claim now needs to rest on stricter comparisons:

- Brier score and ECE against raw-only calibration
- AUC/ranking against raw-only calibration
- comparison of `min(raw, hidden_probe)` vs `min(raw, raw_only_calibrated)`
- behavior under cross-dataset or prompt-shift tests

## Weighted Baseline Summary

Weighted raw-only calibrators use balanced class weighting where applicable.

| Run | Best Raw-Only Brier | Best Raw-Only ECE | Best Raw-Only Wrong >= 0.8 | Hidden Conservative Brier | Hidden Conservative ECE | Hidden Conservative Wrong >= 0.8 | Hidden AUC | Best Raw-Only AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen SQuAD | 0.195974 | 0.128586 | 0 | 0.195644 | 0.152570 | 10 | 0.783394 | 0.678534 |
| Qwen TriviaQA | 0.190504 | 0.228478 | 1 | 0.131989 | 0.099740 | 1 | 0.804905 | 0.747093 |
| Phi SQuAD | 0.163009 | 0.342367 | 0 | 0.046439 | 0.040488 | 4 | 0.806810 | 0.705774 |
| Phi TriviaQA | 0.174747 | 0.058779 | 0 | 0.198426 | 0.152008 | 9 | 0.828226 | 0.837561 |
| Gemma SQuAD | 0.243452 | 0.140702 | 0 | 0.134160 | 0.121125 | 10 | 0.883021 | 0.552640 |
| Gemma TriviaQA | 0.219489 | 0.214738 | 0 | 0.197273 | 0.176408 | 10 | 0.760785 | 0.711928 |

Weighted interpretation:

- Hidden-state conservative TCL-v0 has better Brier score in 5 of 6 runs.
- Hidden-state conservative TCL-v0 has better ECE in 4 of 6 runs.
- Raw-only calibration often wins high-confidence wrong-answer count by pushing confidence downward.
- Hidden-state conservative TCL-v0 has better AUC in 5 of 6 runs, suggesting useful ranking signal beyond raw-only score transformations.
- Phi TriviaQA is the clearest negative case for hidden-state conservative TCL-v0 under these baselines: raw-only calibration is stronger on Brier, ECE, and high-confidence error count.

## Unweighted Sensitivity Summary

Unweighted raw-only calibrators are included because standard post-hoc calibration is often reported without class-balanced weights.

| Run | Best Raw-Only Brier | Best Raw-Only ECE | Best Raw-Only Wrong >= 0.8 | Hidden Conservative Brier | Hidden Conservative ECE | Hidden Conservative Wrong >= 0.8 | Hidden AUC | Best Raw-Only AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen SQuAD | 0.177707 | 0.040205 | 9 | 0.195644 | 0.152570 | 10 | 0.783394 | 0.685356 |
| Qwen TriviaQA | 0.135091 | 0.055998 | 0 | 0.131989 | 0.099740 | 1 | 0.804905 | 0.747093 |
| Phi SQuAD | 0.029245 | 0.004091 | 4 | 0.046439 | 0.040488 | 4 | 0.806810 | 0.705774 |
| Phi TriviaQA | 0.174747 | 0.058779 | 0 | 0.198426 | 0.152008 | 9 | 0.828226 | 0.837561 |
| Gemma SQuAD | 0.222291 | 0.021661 | 0 | 0.134160 | 0.121125 | 10 | 0.883021 | 0.552640 |
| Gemma TriviaQA | 0.176507 | 0.070345 | 0 | 0.197273 | 0.176408 | 10 | 0.760785 | 0.711928 |

Unweighted interpretation:

- Unweighted raw-only calibration is a much stronger competitor on ECE and high-confidence error counts.
- Hidden-state TCL-v0 still shows stronger AUC in 5 of 6 runs.
- Gemma SQuAD remains the strongest hidden-state case: hidden conservative TCL-v0 is much better on Brier and AUC, even though raw-only calibration wins high-confidence error count.
- The result is no longer "TCL-v0 clearly beats raw confidence"; it is "hidden states may improve ranking and some calibration metrics beyond raw-only baselines, but the high-confidence-error story is partly explained by post-hoc calibration."

## Updated Research Position

The strongest honest position is now:

```text
TCL-v0 still shows evidence that hidden states carry answer-correctness signal, especially in AUC/ranking and several Brier comparisons, but raw-only calibration explains a meaningful part of the earlier high-confidence-error reductions.
```

This is weaker than the pre-baseline interpretation, but stronger scientifically. It makes the next experiment clearer.

## What This Means For The Paper

Do not claim:

- conservative TCL-v0 reduces high-confidence errors because of hidden states alone
- raw confidence is an adequate baseline
- high-confidence wrong-answer count is sufficient evidence for TCL-v0

Allowed claim:

- after adding raw-only baselines, hidden-state TCL-v0 remains competitive and often improves ranking/AUC, but the method needs stronger ablations before claiming hidden-state-specific calibration gains

## Next Required Experiment

The next run should be a predeclared baseline-and-ablation experiment, not a scale-only experiment.

Minimum design:

```text
model: Gemma 2B-it and one second model
benchmark: TriviaQA and SQuAD
size: 1,000 first; 3,000 only after baseline code is locked
signals: raw, temperature-scaled raw, Platt raw, isotonic raw, hidden probe, min(raw, hidden probe), min(raw, raw-only calibrated)
ablations: hidden layer, pooling method, raw-only vs hidden-only vs raw-plus-hidden
review: same targeted manual-review protocol
```

Decision rule:

```text
Advance TCL-v0 only if hidden-state methods beat raw-only baselines on ranking/AUC and at least one calibration metric without relying only on confidence suppression.
```
