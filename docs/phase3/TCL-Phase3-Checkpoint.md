# TCL Phase 3 Checkpoint: A Learned Head Does Not Beat Raw Confidence on New Data

Written November 22, 2026, after the single preregistered Phase 3b analysis run. Protocol: [TCL-Phase3-Protocol.md](TCL-Phase3-Protocol.md) (locked November 21, 2026, before any Phase 3 code or metric; one amendment recording implementation details, made before any run on real data). Earlier results: [Phase 2b checkpoint](../phase2/TCL-Phase2b-Checkpoint.md).

## 1. Decision

| Hypothesis | Rule | Result |
|---|---|---|
| **H4-T** | `d(M+R, R)` CI lower bound above zero on a majority of eligible pairs (4 of 6), including at least one PopQA pair | **Fails.** The CI is above zero on **1 of 6** pairs (Qwen on PopQA, +0.079). On 3 pairs the head is significantly **worse** than calibrated raw confidence; on 2 the CI includes zero. |

**What follows, as fixed in the program plan (Section 8) before any Phase 3 data:** no learned correctness head beats calibrated raw confidence out of distribution at this scale. **TCL's correctness dimension becomes calibrated raw confidence**, with semantic entropy where sampling is affordable, and the H4 trust dimensions will be judged on whether they add to it. I report this as a main result.

**In plain words:** I gave the head the two things that plausibly held it back in Phases 1 and 2: more varied training data (a mixture of datasets) and the raw confidence itself as an input. It still does not reliably beat raw confidence on a dataset it was not trained on. It wins clearly in one case (Qwen on PopQA) and loses clearly in three.

## 2. What Ran

- **Models:** Qwen2.5-0.5B-Instruct and Phi-3.5-mini-instruct, Phase 1 prompts, decoding, and answer spans.
- **3a, design selection** (laptop CPU, training and validation splits only, no test split read): 20 settings x 6 leave-one-dataset-out rotations x 5 seeds. Chosen by the locked rule: **MLP head, weight decay 1, equal weight per dataset**, mean validation AUROC of M+R 0.729 (best logistic setting: 0.691). File: `runs/phase3a/selection.json`.
- **PopQA inference** (Kaggle T4 x2): sample drawn with seed 2030 from the 14,267 PopQA test questions. Phi met the size rule on batch 1 (1,500 questions). Qwen did not (136 correct answers, below 150), so, as the protocol fixes, I extended it to batches 1-2 (3,000 questions), decided from labels alone. The extended run reproduced all 1,500 batch-1 answers exactly.
- **3b** (laptop CPU): every arm trained with the chosen setting, 5 seeds, and scored on the test splits; the analysis run once.

## 3. Data and Eligible Pairs

| Model | Held-out dataset | Test correct / incorrect | Refusals | Eligible |
|---|---|---:|---:|---|
| Qwen | TriviaQA | 216 / 1,284 | 0 | Yes |
| Qwen | SQuAD | 1,040 / 460 | 0 | Yes |
| Qwen | NQ-Open | 108 / 1,392 | 0 | No |
| Qwen | **PopQA** | 285 / 2,715 | 0 | Yes |
| Phi | TriviaQA | 779 / 721 | 1 | Yes |
| Phi | SQuAD | 1,405 / 95 | - | No |
| Phi | NQ-Open | 322 / 1,178 | 0 | Yes |
| Phi | **PopQA** | 278 / 1,222 | 24 | Yes |

## 4. Primary Result

Test AUROC on the held-out dataset, mean over 5 seeds. R: raw answer-span confidence, Platt-mapped on the training mixture's validation splits. M: hidden-state head on the mixture. M+R: hidden states plus raw confidence. S: mean of the single-dataset heads. SE: semantic entropy (reference).

| Pair | R | S (mean) | M | **M+R** | *SE* |
|---|---:|---:|---:|---:|---:|
| Qwen / TriviaQA | 0.712 | 0.707 | 0.740 | 0.744 | *0.773* |
| Qwen / SQuAD | **0.762** | 0.612 | 0.624 | 0.667 | *0.775* |
| Qwen / PopQA | 0.828 | 0.845 | **0.909** | **0.907** | *0.879* |
| Phi / TriviaQA | **0.836** | 0.714 | 0.795 | 0.795 | *0.842* |
| Phi / NQ-Open | 0.765 | 0.670 | 0.759 | 0.760 | *0.762* |
| Phi / PopQA | **0.864** | 0.783 | 0.815 | 0.809 | *0.881* |

The gated difference, seed-averaged, with its 95% paired bootstrap CI (1,000 resamples of test items, seed 2026):

| Pair | `d(M+R, R)` | Per seed (0-4) | CI above zero |
|---|---|---|---|
| Qwen / TriviaQA | +0.032 [-0.010, +0.077] | +0.038 +0.031 +0.032 +0.034 +0.024 | No |
| Qwen / SQuAD | **-0.096** [-0.131, -0.061] | -0.092 -0.102 -0.094 -0.090 -0.101 | No |
| Qwen / PopQA | **+0.079** [+0.052, +0.105] | +0.072 +0.075 +0.088 +0.078 +0.085 | **Yes** |
| Phi / TriviaQA | **-0.041** [-0.064, -0.017] | -0.031 -0.045 -0.042 -0.046 -0.039 | No |
| Phi / NQ-Open | -0.006 [-0.036, +0.024] | -0.003 -0.002 -0.011 -0.001 -0.011 | No |
| Phi / PopQA | **-0.055** [-0.080, -0.029] | -0.055 -0.091 -0.048 -0.019 -0.064 | No |

Bold = CI excludes zero. **1 of 6 pairs passes; the rule needs 4.** The direction is the same in every seed of every pair, so the result does not depend on the seeds. With refusals excluded the results are the same to within 0.003 (Phi / PopQA: -0.058 [-0.084, -0.032]).

## 5. Reported Comparisons (Not Gated)

| Pair | `d(M, S)` mixture vs single | `d(M+R, M)` adding raw confidence | `d(M+R, SE)` |
|---|---:|---|---|
| Qwen / TriviaQA | +0.033 | +0.004 [-0.002, +0.009] | -0.029 [-0.066, +0.008] |
| Qwen / SQuAD | +0.012 | **+0.043** [+0.036, +0.049] | **-0.109** [-0.142, -0.077] |
| Qwen / PopQA | +0.064 | -0.001 [-0.003, +0.000] | **+0.029** [+0.010, +0.048] |
| Phi / TriviaQA | +0.081 | +0.000 [-0.001, +0.002] | **-0.047** [-0.069, -0.025] |
| Phi / NQ-Open | +0.089 | +0.000 [-0.001, +0.002] | -0.002 [-0.033, +0.030] |
| Phi / PopQA | +0.033 | **-0.006** [-0.008, -0.004] | **-0.072** [-0.096, -0.048] |

`d(M, S)` is the mean over the single training datasets; per-dataset CIs are in `details.json`.

- **Mixture training helps.** The mixture head beats the average single-dataset head on every pair (+0.012 to +0.089). This confirms the first cause I named in the protocol (too little variety), but the gain is not enough to reach raw confidence.
- **Giving the head raw confidence mostly does nothing.** `d(M+R, M)` is within 0.006 of zero on 5 of 6 pairs. The head largely ignores its one raw-confidence input next to hundreds of hidden-state dimensions, so the second fix (start from raw confidence and learn a correction) did not happen in practice. Where it matters (Qwen / SQuAD, +0.043), it still leaves the head 0.096 below raw confidence alone.
- **Semantic entropy stays the strongest signal.** The head beats it only on Qwen / PopQA.

**Secondary metrics** (after a Platt map on mixture validation; all in `details.json`) mostly follow the AUROC ranking. On Qwen / PopQA the head also has lower ECE (0.021 vs 0.074) and Brier score (0.057 vs 0.077) than raw confidence; on Qwen / SQuAD it is far worse calibrated (ECE 0.533 vs 0.261), because the mixture's accuracy level does not match SQuAD's.

## 6. The One Win: Qwen on PopQA

The single passing pair is a real effect, not noise: every hidden-state head beats raw confidence there, including heads trained on one dataset (TriviaQA 0.883, NQ-Open 0.890 vs 0.828), and the mixture head beats semantic entropy. Qwen answers only 9.5% of PopQA correctly, mostly with confident guesses ("Martin Scorsese", "J.K. Rowling") that keep its token confidence high when it is wrong. Its hidden states seem to separate questions about entities it knows from ones it does not, which fits Mallen et al. (2023), who found that model accuracy on PopQA tracks entity popularity. I treat this as a lead for later work, not as a result: it holds for one model on one dataset, and Phi on PopQA shows the opposite.

## 7. Label Audit (Protocol Section 6)

Drawn and reviewed before any 3b metric was read: 100 PopQA answers per model, seed 2031. Files: `runs/phase3_audit/label_audit_reviewed_{qwen,phi}.csv`.

| Sample | Rule v3 errors | Flagged as judgment calls |
|---|---:|---:|
| Qwen | 1 / 100 (an answer naming a different person, matched because it repeats the question's subject) | 8 |
| Phi | 1 / 100 (a variant of the gold name, "Adelaide" for "Adela of Blois", marked wrong) | 4 |

PopQA's gold lists are long alias lists, so the rule rarely misses a correct answer. The bigger caveat is the questions themselves: many titles are ambiguous ("Who was the director of Annie?" has several films), so a model can give a valid answer for another entity and be marked wrong. This affects every arm equally, since all arms score the same answers.

## 8. What This Means for TCL

- **Claim 3 now has its answer at this scale.** Across Phases 1, 2, 2b, and 3, on three models, five datasets, frozen and fine-tuned heads, single datasets and mixtures, a head trained on correctness labels does not reliably beat the model's own calibrated token confidence on data it was not trained on. Phase 3 was the strongest version of the idea I could design, and it fails 5 of 6 times.
- **The correctness dimension of TCL is calibrated raw confidence** (plus semantic entropy where 10 samples per answer are affordable), as fixed before the run. The H4 dimensions will be tested on whether they add information beyond it, on held-out data.
- **Two leads, not claims:** mixture training consistently improves transfer over single-dataset training, and on long-tail entity questions a small model's hidden states can carry signal its token confidence misses (Qwen / PopQA).

## 9. Next Step

As decided after the Phase 2 desk rejection, I will write one combined paper on Phases 2 and 3 in ACL format: the joint-training negative result as the motivation, and Phase 3 as the main test of whether learned trust heads transfer. Before that, I will update the program plan's status for H4-T. The H4 protocol comes after the paper.

## 10. Files

| Path | Contents |
|---|---|
| `tcl_experiments/runs/phase3a/selection.json`, `selection_run.log` | 3a: every setting's validation AUROC per rotation, and the chosen setting |
| `tcl_experiments/data/phase3/popqa.csv` | The PopQA sample, 3 batches of 1,500 (seed 2030) |
| `tcl_experiments/runs/phase3_popqa/<model>/popqa/` | PopQA records and configs (hidden states kept locally) |
| `tcl_experiments/runs/phase3_audit/` | Audit samples and my reviewed labels |
| `tcl_experiments/runs/phase3b/` | Test scores of every arm and seed per pair, the manifest with label counts, and the scoring log |
| `tcl_experiments/runs/phase3b_analysis/decisions.json` | The H4-T decision |
| `tcl_experiments/runs/phase3b_analysis/details.json` | Every difference with CI and per-seed values, secondary metrics, refusal-excluded results |
| `tcl_experiments/runs/phase3b_analysis/analysis_run.log` | Output of the single analysis run |

## References

- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In *ACL*.
