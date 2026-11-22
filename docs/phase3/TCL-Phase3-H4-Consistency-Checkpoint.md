# TCL Phase 3 H4 Checkpoint: Self-Consistency Is Learnable From One Pass, but Adds to Raw Confidence on Only Half the Pairs

Written November 22, 2026, after the single preregistered H4 analysis run. Protocol: [TCL-Phase3-H4-Consistency-Protocol.md](TCL-Phase3-H4-Consistency-Protocol.md) (locked November 22, 2026, before any H4 code or metric; one amendment recording implementation details, made before any run on real data). Earlier result: [Phase 3 checkpoint (H4-T)](TCL-Phase3-Checkpoint.md).

## 1. Decision

| Hypothesis | Rule | Result |
|---|---|---|
| **H4 (self-consistency)** | A pair passes if the one-pass head's AUROC for the consistency label is at least 0.70, its Spearman correlation with calibrated raw confidence is below 0.7, and combining it with raw confidence lowers correctness AURC (CI above zero). H4 passes on a majority of eligible pairs (4 of 6), including one PopQA pair. | **Fails, by one pair.** 3 of 6 pairs pass (Qwen / TriviaQA, Qwen / PopQA, Phi / NQ-Open), including a PopQA pair; 4 were needed. |

The first two conditions hold on **every** pair: self-consistency is predictable from a single forward pass on data the head was not trained on (AUROC 0.73-0.95), and the prediction is not a copy of raw confidence (Spearman 0.26-0.62). What fails is the third condition, the gain in selective prediction over raw confidence, on 3 of 6 pairs.

**What follows, as fixed in the program plan (Section 8):** H4 has failed for its first dimension. The plan tests the next dimension (grounding) before concluding that trust is one-dimensional at this scale, and re-tests at Phase 4's larger scale before dropping multi-dimensionality.

**In plain words:** a head can learn, from one pass, how consistently the model would answer if asked again, and this transfers to new datasets. Adding that prediction to raw confidence clearly helps on half of the pairs and does not help clearly on the other half. The preregistered bar was a majority, so this is a fail.

## 2. What Ran

- **Data:** exactly the Phase 3 data. No new inference. The consistency label is the share of 10 sampled answers in the greedy answer's meaning cluster (NLI clustering), already stored for every test split.
- **Consistency head:** the Phase 3a design (MLP, weight decay 1, equal weight per dataset), not tuned; 5 seeds; trained on the other pool datasets' test splits (the only splits with sampled answers), never on the held-out dataset.
- **Combiner:** logistic regression on logit(raw confidence) and the head's output, fitted on the training datasets' validation splits.
- **Compute:** laptop CPU, a few minutes.

## 3. Primary Result

| Pair | Learnability AUROC (>= 0.70) | Spearman with raw conf. (< 0.7) | `d_AURC` = AURC(R) - AURC(R+C-hat) [95% CI] | Pass |
|---|---:|---:|---|---|
| Qwen / TriviaQA | 0.876 | 0.258 | **+0.021** [+0.010, +0.034] | **Yes** |
| Qwen / SQuAD | 0.734 | 0.257 | +0.003 [-0.001, +0.008] | No |
| Qwen / PopQA | 0.947 | 0.537 | **+0.020** [+0.015, +0.026] | **Yes** |
| Phi / TriviaQA | 0.825 | 0.592 | -0.008 [-0.018, +0.001] | No |
| Phi / NQ-Open | 0.812 | 0.611 | **+0.016** [+0.007, +0.026] | **Yes** |
| Phi / PopQA | 0.880 | 0.624 | +0.003 [-0.006, +0.011] | No |

Positive `d_AURC` means the combination ranks answers better for selective prediction. Each value is the mean over 5 seeds, and every seed has the same sign on the three passing pairs. With refusals excluded, every pair's decision is unchanged.

The closest miss is Qwen / SQuAD: all 5 seeds improve AURC (+0.003 to +0.004), but the CI's lower bound is -0.001.

## 4. Reported Comparisons (Not Gated)

Correctness AUROC on the held-out test split (mean over 5 seeds):

| Pair | R | **R+C-hat** (one pass) | `d(R+C-hat, R)` [95% CI] | R+C oracle (10 samples) | SE (10 samples) |
|---|---:|---:|---|---:|---:|
| Qwen / TriviaQA | 0.712 | **0.785** | **+0.073** [+0.044, +0.106] | 0.790 | 0.773 |
| Qwen / SQuAD | 0.762 | 0.774 | **+0.012** [+0.001, +0.024] | 0.807 | 0.775 |
| Qwen / PopQA | 0.828 | **0.895** | **+0.067** [+0.050, +0.084] | 0.872 | 0.879 |
| Phi / TriviaQA | 0.836 | 0.825 | -0.011 [-0.027, +0.006] | 0.853 | 0.842 |
| Phi / NQ-Open | 0.765 | 0.787 | **+0.022** [+0.005, +0.039] | 0.758 | 0.762 |
| Phi / PopQA | 0.864 | 0.862 | -0.002 [-0.021, +0.017] | 0.874 | 0.881 |

- **By AUROC, the one-pass combination beats raw confidence on 4 of 6 pairs.** I report this but do not count it: the gate was AURC, fixed before the run, and I do not switch metrics after seeing results.
- **On Qwen it is the strongest signal I have measured.** On TriviaQA and PopQA, one pass plus raw confidence matches or beats semantic entropy, which needs 10 extra generations.
- **On two pairs the one-pass prediction beats the real sampled label** (Qwen / PopQA, Phi / NQ-Open: the oracle is lower). The head's output carries information beyond the 10-sample estimate, most likely correctness signal from the hidden states themselves; the oracle only knows the 10 samples.
- **Phi gains little.** Phi's raw confidence is already strong (0.84-0.86 on TriviaQA and PopQA), so there is less left to add.
- **Calibration** after the combiner improves on 5 of 6 pairs (for example Phi / NQ-Open, ECE 0.312 to 0.201; Qwen / PopQA, 0.074 to 0.057), and worsens slightly on Phi / TriviaQA (0.031 to 0.043). All values are in `details.json`.

## 5. Label Audit (Protocol Section 6)

Drawn and reviewed before any H4 metric was read: 50 answers per model (seed 2032) from the held-out test splits. For each, I counted by hand how many of the 10 samples mean the same as the greedy answer and compared that with the NLI count. Files: `runs/phase3_h4_audit/consistency_audit_reviewed_{qwen,phi}.csv`.

| Sample | Count differs | Binary label (>= 5 of 10) differs | Mean absolute difference |
|---|---:|---:|---:|
| Qwen | 7 / 50 | 4 / 50 | 0.18 samples |
| Phi | 11 / 50 | 4 / 50 | 0.40 samples |

The disagreements are mostly by one sample, in both directions: the NLI model sometimes counts a near-miss as the same answer (Qwen: 5 samples matching "1870" where I count 4 exact ones) and sometimes splits paraphrases (Phi: "Miami is not the capital of any U.S. state" counted apart from "Miami is not the capital of any state or country"). Phi's longer answers give it more room to disagree. A label that is 92% right on the binary split weakens every comparison equally, so it is more likely to hide an effect than to create one.

## 6. What This Means for TCL

- **Self-consistency is a real, transferable trust signal.** It can be predicted from one pass on held-out datasets, for both models, and it is not redundant with raw confidence. This is the first trust dimension in this program that a head learns and transfers, in contrast to correctness heads (Phases 1-3).
- **Its added value is model-dependent and not yet established.** It clearly helps Qwen, the smaller model with the weaker raw confidence, and helps Phi on one dataset. Under my preregistered rule, that is not enough to call it a second dimension of TCL.
- **Next, per the plan:** grounding is the next H4 dimension (it needs new SQuAD 2.0 inference). A failed H4 is re-tested at Phase 4's larger scale before multi-dimensionality is dropped; Phase 4 can also test whether the Qwen pattern (a large gain where raw confidence is weak) holds at 7-8B.

## 7. Files

| Path | Contents |
|---|---|
| `tcl_experiments/runs/phase3_h4_audit/` | Audit samples and my hand counts |
| `tcl_experiments/runs/phase3_h4/` | Test scores per arm and seed for every pair, the manifest with label counts, and the scoring log |
| `tcl_experiments/runs/phase3_h4_analysis/decisions.json` | The H4 decision |
| `tcl_experiments/runs/phase3_h4_analysis/details.json` | Every condition per pair with per-seed values and CIs, correctness AUROC of every arm, secondary metrics, refusal-excluded results |
| `tcl_experiments/runs/phase3_h4_analysis/analysis_run.log` | Output of the single analysis run |
