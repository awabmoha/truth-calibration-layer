# TCL Phase 3 H4 Checkpoint: Grounding Is Learnable From One Pass, but Mostly Detects an Irrelevant Passage, and Adds to Raw Confidence on Only 2 of 7 Pairs

Written November 24, 2026, after the single preregistered H4 grounding analysis run. Protocol: [TCL-Phase3-H4-Grounding-Protocol.md](TCL-Phase3-H4-Grounding-Protocol.md) (locked November 22, 2026, before any grounding data or metric; one amendment recording implementation details, made before any model answered a grounding question). Earlier results: [Phase 3 checkpoint (H4-T)](TCL-Phase3-Checkpoint.md), [H4 self-consistency checkpoint](TCL-Phase3-H4-Consistency-Checkpoint.md).

## 1. Decision

| Hypothesis | Rule | Result |
|---|---|---|
| **H4 (grounding)** | A pair passes if the one-pass head's AUROC for the NLI grounding label is at least 0.70, its Spearman correlation with calibrated raw confidence is below 0.7, and combining it with raw confidence lowers correctness AURC (CI above zero). H4 passes on a majority of eligible pairs (4 of 7), including the RelationExtraction pair of at least one model. | **Fails.** 2 of 7 pairs pass (Qwen / SQuAD, Qwen / NQ); 4 were needed, and the only eligible RelationExtraction pair (Qwen) does not pass. |

As with self-consistency, the first two conditions mostly hold: the head predicts grounding on a held-out dataset with AUROC 0.70-0.84 on 6 of 7 pairs (Phi / HotpotQA: 0.695), and it is not a copy of raw confidence (Spearman 0.11-0.61 on every pair). What fails is the third condition, the gain in selective prediction, on 5 of 7 pairs.

**What follows, as fixed in the program plan (Section 8, row "H4 fails for all dimensions"):** self-consistency and grounding have both failed, and evidence density cannot be tested for transfer (its label exists only on PopQA). So trust is effectively one-dimensional at this scale. I re-test at Phase 4's larger scale before dropping multi-dimensionality. Under my publication plan, Phases 1-3 are not written up on their own; they go into the paper of the first phase that succeeds.

**In plain words:** a head can learn, from one pass, whether the model's answer is supported by the passage it was given, and this transfers to new datasets. Adding it to raw confidence ranks answers better on every Qwen pair, but the gain is large enough to be certain on only two, and Phi barely gains. Most of what the head adds is spotting answers given from an irrelevant passage; among answers given from the right passage, it adds nothing.

## 2. What Ran

- **Data:** 4,500 questions each from MRQA's SQuAD, Natural Questions, and HotpotQA (2,500 train, 500 validation, 1,500 test) and 1,500 test questions from RelationExtraction (the new held-out dataset). One third of the questions in every split got an unrelated passage from the same dataset instead of their own (distractors, seed 2034).
- **Inference:** Qwen2.5-0.5B and Phi-3.5-mini on Kaggle T4, with the passage in the prompt and one extra line allowing "unanswerable". 15,000 answers per model, about 2 hours (Qwen) and 3.5 hours (Phi).
- **Grounding label:** DeBERTa-large-MNLI, passage as premise and "question answer" as hypothesis; grounded if the entailment probability is above 0.5. Abstentions get no label and are left out of every arm.
- **Size rule:** Phi / RelationExtraction was dropped (125 incorrect test answers; 150 needed): Phi is right on 88% of the questions it answers there. No sample was extended. 7 pairs remain.
- **Grounding head:** the Phase 3a design (MLP, weight decay 1, equal weight per dataset), not tuned; 5 seeds; trained on the other pool datasets' training splits.
- **Combiner:** logistic regression on logit(raw confidence) and the head's output, fitted on the training datasets' validation splits.
- **Compute:** roughly 6 GPU hours on Kaggle, including the smoke tests; scoring and analysis on a laptop CPU in a few minutes.

## 3. Primary Result

| Pair | Learnability AUROC (>= 0.70) | Spearman with raw conf. (< 0.7) | `d_AURC` = AURC(R) - AURC(R+G-hat) [95% CI] | Pass |
|---|---:|---:|---|---|
| Qwen / SQuAD | 0.798 | 0.561 | **+0.009** [+0.001, +0.017] | **Yes** |
| Qwen / NQ | 0.765 | 0.564 | **+0.014** [+0.004, +0.024] | **Yes** |
| Qwen / HotpotQA | 0.715 | 0.527 | +0.009 [-0.000, +0.019] | No |
| Qwen / RelationExtraction | 0.838 | 0.605 | +0.004 [-0.002, +0.010] | No |
| Phi / SQuAD | 0.795 | 0.219 | +0.004 [-0.001, +0.010] | No |
| Phi / NQ | 0.741 | 0.112 | -0.011 [-0.026, +0.000] | No |
| Phi / HotpotQA | 0.695 | 0.163 | +0.001 [-0.011, +0.014] | No |

Positive `d_AURC` means the combination ranks answers better for selective prediction. Each value is the mean over 5 seeds. On every Qwen pair, all 5 seeds improve AURC; on Phi / NQ all 5 seeds make it worse. With refusals excluded, every pair's decision is unchanged.

The closest miss is Qwen / HotpotQA: every seed improves AURC (+0.006 to +0.012), but the CI's lower bound is -0.0003.

## 4. Reported Comparisons (Not Gated)

Correctness AUROC on the held-out test answers (mean over 5 seeds):

| Pair | R | **R+G-hat** | `d(R+G-hat, R)` [95% CI] | R+G oracle (NLI label) | P(True) |
|---|---:|---:|---|---:|---:|
| Qwen / SQuAD | 0.835 | **0.858** | **+0.023** [+0.011, +0.035] | 0.865 | 0.744 |
| Qwen / NQ | 0.780 | **0.804** | **+0.025** [+0.010, +0.040] | 0.785 | 0.726 |
| Qwen / HotpotQA | 0.745 | **0.770** | **+0.025** [+0.013, +0.038] | 0.765 | 0.744 |
| Qwen / RelationExtraction | 0.831 | **0.845** | **+0.014** [+0.003, +0.025] | 0.858 | 0.854 |
| Phi / SQuAD | 0.821 | **0.843** | **+0.023** [+0.002, +0.044] | 0.827 | 0.728 |
| Phi / NQ | 0.708 | 0.685 | -0.023 [-0.048, +0.004] | 0.689 | 0.574 |
| Phi / HotpotQA | 0.670 | 0.681 | +0.011 [-0.000, +0.022] | 0.664 | 0.658 |

- **By AUROC, the combination beats raw confidence on 5 of 7 pairs, including Qwen / RelationExtraction.** I report this but do not count it: the gate was AURC, fixed before the run, and I do not switch metrics after seeing results. This is the second time AUROC and AURC disagree in the same direction (self-consistency: 4 of 6 by AUROC, 3 of 6 by AURC): the gains sit mostly in the middle of the ranking, while AURC weighs the most confident answers most.
- **The gain comes from the distractor rows.** Split by passage type, the combination gains at most +0.003 AUROC over raw confidence among answers given from the right passage, on every pair (for example Qwen / SQuAD 0.766 vs 0.766; Qwen / RelationExtraction 0.757 vs 0.721). The pooled gain comes from ranking distractor-passage answers, which are almost all wrong (5-31 correct of 287-359 on Qwen), below right-passage answers. In other words, the head mainly detects "this passage is not about the question", not "this answer is unsupported by a relevant passage".
- **This explains why Phi gains little.** Phi already says "unanswerable" to most distractor passages (25-33% of its test answers are abstentions, against 10-16% for Qwen), so only 28-122 distractor answers remain in its evaluation sets, against 287-359 for Qwen. Abstaining does Phi's grounding work before the head sees anything.
- **Grounded and correct** (both labels 1, the target that matters when answering from documents): the combination beats raw confidence on all 7 pairs (+0.03 to +0.08 AUROC), and the head alone beats raw confidence on 6 of 7. Not gated.
- **The oracle is barely better than the head.** Adding the NLI label itself (with all its errors) lifts AUROC by at most 0.03 over raw confidence, and on Qwen / NQ, Qwen / HotpotQA, Phi / SQuAD, and Phi / HotpotQA it is below the one-pass head.
- **Calibration** (ECE) after the combiner improves slightly on 5 of 7 pairs and worsens on Qwen / HotpotQA (0.050 to 0.054) and Qwen / RelationExtraction (0.077 to 0.163). All values are in `details.json`.

## 5. Label Audit (Protocol Section 6)

Drawn and reviewed before any H4 metric was computed: 50 answers per model (seed 2035) from the held-out test splits of the eligible pairs, 25 from right-passage rows and 25 from distractor rows, abstentions left out. The same 100 answers serve both audits. Files: `runs/phase3_h4g_audit/grounding_audit_reviewed_{qwen,phi}.csv`.

| Sample | NLI grounding label wrong: right passage | NLI wrong: distractor | Correctness label wrong |
|---|---:|---:|---:|
| Qwen | 13 / 25 (11 supported answers missed, 2 false) | 2 / 25 (false support) | 5 / 50 (2 uncertain) |
| Phi | 6 / 25 (6 missed) | 2 / 25 (false support) | 6 / 50 (1 uncertain) |
| **Both** | **19 / 50** | **4 / 50** | **11 / 100** |

- **The NLI grounding label is wrong on 23 of 100 answers**, much worse than any label in this program so far. The main error is one-directional: on the right passage, it rejects answers the passage plainly states, especially short ones ("eight" for "How many rows of combs are there?", entailment 0.39; "Nalini" for a passage saying Ramarajan married Nalini, 0.24). The "question answer" hypothesis is a poor input for an NLI model. I saw this pattern in the 160-answer smoke test before the full runs and kept the locked rule; changing it then would have been a choice made after looking at labels.
- **Correctness labels (rule v3) are wrong on 11 of 100**, all correct answers marked wrong: paraphrases ("To provide jobs for young men and relieve families during the Great Depression"), shortened names ("Burnham" for Forbes Burnham), and name variants ("Columbus Crew SC"). Answers with a passage in the prompt are longer and closer to the passage's wording than the short closed-book answers rule v3 was built on.
- **One abstention was not detected:** Qwen answered "Unanswered." once, which the detector (matching "unanswerable") counts as a claim. It is labeled not grounded and incorrect, which is the right outcome for that answer.

What this means for the result: label noise pushes every comparison toward no effect. It probably lowered learnability (the head learns the NLI model's errors) and the oracle. But it cannot explain the main finding in Section 4. Among right-passage answers, where most NLI errors are, the head adds nothing; the gain comes from distractor rows, where the label is almost always right.

## 6. What This Means for TCL

- **Two candidate trust dimensions, the same pattern.** Self-consistency and grounding are both learnable from one forward pass on held-out data, and neither is redundant with raw confidence. Both add to raw confidence for Qwen, the smaller model with the weaker raw confidence, and barely for Phi. Under my preregistered rule, neither is a second dimension of TCL at this scale.
- **Grounding, as tested here, is mostly passage relevance.** The head learned to spot an irrelevant passage, which a model that abstains well (Phi) already does on its own. Whether a head can detect an unsupported answer given a relevant passage (the harder and more useful case) remains open, and needs a better grounding label than NLI on "question answer".
- **Next, per the plan:** H4 has failed for every dimension I can test for transfer. Trust is effectively one-dimensional at this scale. Phase 4 re-tests at 7-8B before I drop multi-dimensionality. The Qwen pattern (gains where raw confidence is weak) is the specific thing to check there, and Phase 4 should fix the grounding label first (a statement-form hypothesis or a stronger judge, validated on this audit set before use).

## 7. Files

| Path | Contents |
|---|---|
| `tcl_experiments/data/phase3_grounding/` | Question sets from MRQA with distractor passages, and metadata |
| `tcl_experiments/runs/phase3_grounding/<model>/<dataset>/` | Answers, raw confidence, P(True), and NLI grounding labels for every question (hidden states kept local, not in the repository) |
| `tcl_experiments/runs/phase3_h4g_audit/` | Audit samples and my labels |
| `tcl_experiments/runs/phase3_h4g/` | Test scores per arm and seed for every pair, the manifest with label and abstention counts, and the scoring log |
| `tcl_experiments/runs/phase3_h4g_analysis/decisions.json` | The H4 (grounding) decision |
| `tcl_experiments/runs/phase3_h4g_analysis/details.json` | Every condition per pair with per-seed values and CIs, correctness AUROC of every arm, grounded-and-correct, results by passage type, secondary metrics, refusal-excluded results |
| `tcl_experiments/runs/phase3_h4g_analysis/analysis_run.log` | Output of the single analysis run |
