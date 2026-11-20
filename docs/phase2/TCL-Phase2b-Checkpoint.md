# TCL Phase 2b Checkpoint: The Fix Works, the Transfer Claim Still Fails

Written November 20, 2026, after the single preregistered Phase 2b analysis run. Protocol: [TCL-Phase2b-Protocol.md](TCL-Phase2b-Protocol.md) (locked before any Phase 2b training, no amendments). Phase 2 results: [TCL-Phase2-Checkpoint.md](TCL-Phase2-Checkpoint.md).

## 1. Decision

| Hypothesis | Rule | Result |
|---|---|---|
| **H3-T** (primary) | `d(C, A)` CI above zero in **both** transfer directions, and C meets the generation-quality bar | **Fails.** C beats A from TriviaQA to SQuAD (+0.034), but is still significantly **worse** than A from SQuAD to TriviaQA (-0.038), and C fails the generation-quality bar. |
| **H3-J** (secondary) | `d(C, B)` CI above zero in both directions, and C meets the bar | **Fails.** C beats B from TriviaQA to SQuAD (+0.018) and is worse than B from SQuAD to TriviaQA (-0.022). |
| Tie reading | Applies when C ties B | Not applicable: C fails the generation-quality bar. |
| Attribution to LM fine-tuning | `d(C, D)` CI includes zero in both directions | Not met (C is worse than D from SQuAD to TriviaQA). In the positive direction, LM-only fine-tuning again matches C (D - A = +0.041). |

**What follows, as fixed in the protocol before training:** H3-T failed on the one allowed rerun, so **I drop the claim that joint training improves the transfer of trust estimates** and publish the negative result, Phase 2 and Phase 2b together. There is no Phase 2c. TCL continues under a revised Phase 3, which I will redesign around transfer before using any new data.

**In plain words:** the bug I found is fixed, and the fix did what it should: fine-tuned models answer in 1-2 words again. But the transfer result did not change. From SQuAD to TriviaQA, the jointly trained head is still clearly worse than a head on the untouched model, and where joint training does help (TriviaQA to SQuAD), plain language-model fine-tuning helps just as much.

## 2. What Ran

- **Exactly the Phase 2 setup** with one change: the LM target is the reference answer followed by the chat template's end-of-turn token (`<|im_end|>`, id 151645), recorded in every run's `metrics.json`.
- **Arms C and D rerun**, 3 seeds each per training dataset; **A and B reused** unchanged from Phase 2 (byte-identical copies in `runs/phase2b/`).
- **Lambda for C** (seed 0, validation AUROC): **1.0** for both datasets. TriviaQA: 0.833 / 0.834 / **0.835** for 0.1 / 0.5 / 1.0 (Phase 2 chose 0.5). SQuAD: 0.773 / 0.787 / **0.796**.
- **Training check:** all 16 trainings took 151-157 optimizer steps per epoch, with 0-6 skipped by fp16 loss scaling.
- **Compute:** four Kaggle T4 x2 sessions, about 6 session hours in all, with no failures.

## 3. The Fix Worked: Answer Length

Median (mean) words per regenerated test answer, pooled over 3 seeds. The base model answers in a median of 2 words.

| Trained on | Arm | Phase 2: TriviaQA | Phase 2: SQuAD | **Phase 2b: TriviaQA** | **Phase 2b: SQuAD** |
|---|---|---:|---:|---:|---:|
| TriviaQA | C | 10.5 (11.7) | 5 (9.9) | **1 (1.8)** | **1 (1.8)** |
| TriviaQA | D | 13 (13.0) | 6 (10.7) | **2 (1.9)** | **1 (1.9)** |
| SQuAD | C | 13 (12.9) | 17 (15.0) | **2 (1.9)** | **2 (2.5)** |
| SQuAD | D | 9 (11.0) | 19 (15.7) | **2 (1.9)** | **2 (2.5)** |

Arm B (reused) keeps its Phase 2 lengths: median 5 and 4 words when trained on TriviaQA, 2 and 2 when trained on SQuAD. B has no LM loss, so the fix does not apply to it.

## 4. Primary Result: Transfer

Test AUROC on the **other** dataset's 1,500 fixed base-model answers, mean over 3 seeds. Phase 2 values in brackets for the rerun arms.

| | TriviaQA -> SQuAD | SQuAD -> TriviaQA |
|---|---:|---:|
| A: frozen head (reused) | 0.557 | **0.645** |
| B: trust-only LoRA (reused) | 0.572 | 0.629 |
| C: joint LoRA (TCL-J) | 0.591 [0.578] | 0.606 [0.606] |
| D: LM-only LoRA, then head | **0.598** [0.588] | **0.647** [0.643] |
| *Raw token confidence (no training)* | *0.762* | *0.712* |
| *Semantic entropy (10 samples)* | *0.775* | *0.773* |

Paired differences (seed-averaged, 95% bootstrap CI, 1,000 resamples):

| Difference | TriviaQA -> SQuAD | SQuAD -> TriviaQA |
|---|---|---|
| C - A | **+0.034** [+0.022, +0.047] | **-0.038** [-0.061, -0.015] |
| C - B | **+0.018** [+0.007, +0.031] | **-0.022** [-0.036, -0.009] |
| C - D | -0.007 [-0.015, +0.002] | **-0.040** [-0.057, -0.023] |
| D - A | **+0.041** [+0.028, +0.055] | +0.002 [-0.016, +0.022] |

Bold = CI excludes zero. No base-model answer is a refusal, so the refusal-excluded results are identical.

The fix moved TriviaQA -> SQuAD up by about 0.01 for C and D, and left SQuAD -> TriviaQA where it was. The SQuAD-trained C is almost exactly as far below A as in Phase 2, with different trained weights (the score files differ), so this gap is not an artifact of the missing end-of-turn token. Every trained head still transfers far below raw token confidence and semantic entropy.

## 5. In-Distribution

| | TriviaQA | SQuAD |
|---|---:|---:|
| A (reused) | 0.830 | 0.762 |
| B (reused) | 0.828 | 0.793 |
| C | 0.837 [0.849] | **0.802** [0.803] |
| D | **0.840** [0.840] | 0.796 [0.797] |

| Difference | TriviaQA | SQuAD |
|---|---|---|
| C - A | +0.008 [-0.003, +0.019] | **+0.040** [+0.026, +0.054] |
| C - B | **+0.010** [+0.001, +0.019] | **+0.010** [+0.002, +0.017] |
| C - D | -0.003 [-0.012, +0.006] | +0.006 [-0.004, +0.017] |

With the fix, C's in-distribution edge over A on TriviaQA shrinks from +0.019 to a non-significant +0.008. On SQuAD it is unchanged. C is never significantly better than D.

## 6. Generation Quality

Accuracy of each model's own regenerated answers (rule v3) and WikiText-2 perplexity, mean over 3 seeds. Base model: TriviaQA 14.3%, SQuAD 69.2%, perplexity 21.93. The bar: accuracy drops at most 1 point on each test set, perplexity rises at most 3%.

| Trained on | Arm | TriviaQA acc. | SQuAD acc. | Perplexity rise | Bar |
|---|---|---:|---:|---:|---|
| TriviaQA | B (reused) | 16.9% | 64.2% | +1.7% | fails (SQuAD -5.0) |
| TriviaQA | C | 14.8% | 52.6% | +1.9% | fails (SQuAD -16.6) |
| TriviaQA | D | 15.0% | 53.2% | +2.6% | fails (SQuAD -16.0) |
| SQuAD | B (reused) | 15.3% | 66.1% | +7.4% | fails |
| SQuAD | C | 14.7% | 80.4% | +9.1% | fails (perplexity) |
| SQuAD | D | 14.3% | 80.0% | +7.3% | fails (perplexity) |

The fix brought perplexity within the bar for the TriviaQA-trained models (+5.6% -> +1.9% for C), but not for the SQuAD-trained ones. TriviaQA-trained C and D now answer SQuAD questions in 1 word, and their SQuAD accuracy falls 16 points: they have learned TriviaQA's answer style (a short name) and apply it to reading comprehension, where many answers are phrases. Part of that drop may be label error, since rule v3 misses correct short answers on SQuAD (Section 7), but those regenerated answers were not in the audit sample, so I cannot say how much.

## 7. Label Audit (Protocol Section 7)

Drawn before any Phase 2b metric was read: 50 regenerated test answers per dataset from arm C, seed 0, chosen lambda, sampled with seed 2029 from labels only. File: [`label_audit_reviewed.csv`](../../tcl_experiments/runs/phase2b_audit/label_audit_reviewed.csv).

| Sample | Rule errors | Type |
|---|---:|---|
| C trained on TriviaQA, TriviaQA answers | 0 / 50 | - |
| C trained on SQuAD, SQuAD answers | 5 / 50 | all correct marked wrong (e.g. "Miracle Day" vs gold "Torchwood: Miracle Day"; "pi* orbitals" vs gold "antibonding") |
| **Total** | **5 / 100** | 1 of the 5 is a judgment call, flagged in the file |

The Phase 2 audit found the opposite error on long answers (wrong answers marked correct). With short answers, the remaining errors are rule v3 missing correct answers that are shorter than, or a paraphrase of, the gold span on SQuAD. The transfer comparisons score every arm on the same fixed base-model answers and labels, so these errors do not affect them. As before, every label is my own decision, with no second reviewer.

The analysis script also writes its Phase 2 audit sample (seed 2028) to `runs/phase2b_analysis/label_audit_sample.csv`. That file is a by-product of reusing the script unmodified; the Phase 2b audit is the seed-2029 one above.

## 8. What This Means for TCL

- **The joint-transfer claim is dropped.** Across two preregistered attempts, training the trust head jointly with the model does not produce trust estimates that transfer better than a frozen head. In one direction it helps, by no more than LM-only fine-tuning; in the other it hurts, with or without the fix.
- **The in-distribution effect is small and not specific to trust training.** Joint training beats trust-only fine-tuning in-distribution by about 0.01 AUROC on both datasets, but it never beats LM-only fine-tuning.
- **Transfer is the central problem, and it is not specific to joint training.** On a new dataset, every head trained on one dataset's correctness labels ranks answers far below the model's raw token confidence (0.56-0.65 vs 0.71-0.76) and semantic entropy (0.77). This held in Phase 1 (frozen probes) and in both Phase 2 runs (every arm).
- **The negative result is the main finding to publish:** Phase 2 and 2b together, with the setup flaw, its fix, and the unchanged outcome.

## 9. Next Step

Under the plan, TCL continues with a revised Phase 3, designed around transfer before any new data is used. The starting point is what does transfer here: raw token confidence and semantic entropy, which are not trained on any dataset's labels. I will write that redesign as a change to the program plan with its own locked protocol.

## 10. Files

| Path | Contents |
|---|---|
| `tcl_experiments/runs/phase2b/<train dataset>/<arm>/seed<k>/` | New C and D runs (end-of-turn LM target) and unchanged copies of the Phase 2 A and B runs |
| `tcl_experiments/runs/phase2b_audit/label_audit_reviewed.csv` | The 100-row Phase 2b audit with my labels and reasons |
| `tcl_experiments/runs/phase2b_analysis/decisions.json` | H3-T, H3-J, tie reading, attribution, generation bar |
| `tcl_experiments/runs/phase2b_analysis/details.json` | Every difference with CI and per-seed values, secondary metrics, generation quality, reference baselines |
| `tcl_experiments/runs/phase2b_analysis/analysis_run.log` | Output of the single analysis run |
