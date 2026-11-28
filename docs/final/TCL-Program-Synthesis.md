# TCL Program Synthesis: What Six Preregistered Tests Show About Learned Trust Signals

Written November 28, 2026, when I closed the experimental program at the Phase 4b gate (`TCL-Research-Program.md`, Section 2 and changelog). This document pulls every phase's recorded result into one place, as the evidence base for the final paper. It adds no new test: every number below comes from a checkpoint written after a single preregistered analysis run, or from a reported, ungated comparison marked as such.

## 1. The Question and the Answer

**The question.** TCL proposed a "truth calibration layer": heads trained on a language model's hidden states that estimate how far to trust each answer, possibly along several dimensions (correctness, self-consistency, grounding), possibly trained jointly with the model. The question running through every phase was whether such learned signals beat the cheapest signal the model already gives: the probability it assigns to its own answer tokens, calibrated on held-out data.

**The answer.** On short-answer QA, from 0.5B to 7B parameters, they do not, once the comparison is fair: calibrated raw confidence as the baseline, a dataset the head never saw as the test, and the decision rule fixed in advance. What learned signals add is small, appears where raw confidence is weak, and shrinks as the model grows.

## 2. Every Preregistered Test

| Phase | Test | Models | What was compared | Result |
|---|---|---|---|---|
| 1 | **H1-R** (in-distribution) | Qwen2.5-0.5B, Phi-3.5-mini, Gemma-2B | Frozen hidden-state probe vs calibrated raw confidence, same dataset | **Passes, exactly at the threshold** (4 of 6 pairings) |
| 1 | **H2-R** (transfer) | same | Probe trained on one dataset, tested on another | **Fails** (retention >= 0.70 in 2 of 6 directions, both Gemma) |
| 2 | **H3-T, H3-J** | Qwen2.5-0.5B, LoRA | Joint LM + trust training vs frozen head vs trust-only LoRA, transfer | **Fail** |
| 2b | **H3-T, H3-J** (one allowed rerun, bug fixed) | same | same | **Fail again**; the joint-training claim is dropped |
| 3 | **H4-T** | Qwen2.5-0.5B, Phi-3.5-mini | Head trained on a mixture, given raw confidence, vs raw confidence on a held-out dataset | **Fails** (1 of 6 pairs) |
| 3 | **H4, self-consistency** | same | One-pass consistency head added to raw confidence (AURC gain) | **Fails by one pair** (3 of 6) |
| 3 | **H4, grounding** | same | One-pass grounding head added to raw confidence | **Fails** (2 of 7) |
| 4a | **H4-T at 7B** | Qwen2.5-7B | as Phase 3 | **Fails** (0 of 3) |
| 4a | **H4 self-consistency at 7B** | same | as Phase 3 | **Fails** (2 of 3, without the required PopQA pair; CI bound -0.0003) |
| 4a | **H4 grounding at 7B** | same | as Phase 3 | **Not testable** (the new-dataset pair fails the size rule) |

Phases 4b (GSM8K, joint training at 7B) and 5 (claim-level trust) were not run.

## 3. Five Findings That Hold Across Phases

### 3.1 Calibrated raw confidence is the baseline that matters, and nothing learned beats it on new data

Every test of transfer to a dataset the head did not see failed against calibrated raw confidence:

- **Phase 1 (frozen probe, one training dataset):** transferred probes lose to raw confidence in 3 of 4 Qwen and Phi directions (for example Qwen TriviaQA to SQuAD: 0.602 vs 0.772 AUROC) and tie in the fourth.
- **Phase 2 (trained heads, with and without LoRA):** on the other dataset, every trained arm ranks answers far below raw confidence (0.56-0.65 vs 0.71-0.76 AUROC). Semantic entropy is higher still (0.773-0.775).
- **Phase 3 (mixture training plus raw confidence as an input):** the head beats raw confidence on 1 of 6 pairs, is significantly worse on 3.
- **Phase 4a (7B):** adding hidden states to raw confidence lowers AUROC on all 3 pairs (by 0.004-0.022).

Mixture training does help relative to single-dataset heads (Phase 3, every pair), so the failure is not a training bug; it is that raw confidence is a strong, free baseline that most published probe comparisons do not calibrate or do not test out of distribution.

### 3.2 Learned signals help where raw confidence is weak

The clearest wins in the program sit where the model's raw confidence was poor:

| Where | Raw-confidence AUROC | Learned signal's gain |
|---|---:|---:|
| Gemma-2B, Phase 1 (in-distribution) | 0.527 / 0.620 | +0.306 / +0.250 |
| Gemma-2B, Phase 1 (transfer) | 0.527 / 0.620 | +0.247 / +0.207 |
| Qwen 0.5B on PopQA, Phase 3 | 0.828 | +0.079 |
| Qwen 7B on PopQA, Phase 4a | 0.865 | -0.022 |

Gemma is the only model whose frozen probe transferred, and it is the model whose raw token confidence was close to chance. Across the 27 eligible Phase 3 and 4a pairs, the correlation between raw-confidence AUROC and a head's gain is negative for the correctness head (Spearman -0.43) and the consistency head (-0.60), as this predicts. For grounding it is positive (+0.63), for a different reason (Section 3.4). These correlations are descriptive and decided nothing.

### 3.3 Joint training buys in-distribution accuracy, not transfer

Training the trust head jointly with the language model (LoRA, `L_LM + lambda * L_trust`) improved the head on its own dataset but not on a new one, in two preregistered runs. Where it helped in one direction (TriviaQA to SQuAD, +0.022 and +0.034), plain LM fine-tuning followed by a frozen head helped as much (+0.031, +0.041); in the other direction it was significantly worse than a frozen head (-0.038 both times). The jointly trained models also failed the generation-quality bar in both runs; Phase 2b fixed the answer-length bug behind it in Phase 2, but not the failure of the bar.

### 3.4 Of two extra trust dimensions, self-consistency is the only robust one

- **Self-consistency** is learnable from one forward pass on every held-out pair in the program (AUROC 0.73-0.95), is not a copy of raw confidence (Spearman 0.26-0.64), and, combined with raw confidence, matches or beats 10-sample semantic entropy on several pairs (Qwen 0.5B TriviaQA 0.785 vs 0.773; Qwen 7B TriviaQA 0.869 vs 0.864, NQ-Open 0.772 vs 0.747) at roughly a tenth of the cost. It lowered selective-prediction risk on every Qwen pair at both sizes, but passed the preregistered majority rule at neither (3 of 6 pooled with Phi; 2 of 3 at 7B). Within Qwen its gain shrinks from about +0.02 to +0.007-0.009 AURC.
- **Grounding** (is the answer supported by the passage the model saw) is learnable, but its gain came mostly from spotting answers given from an irrelevant passage. At 7B the model refused 1,996 of 2,000 irrelevant passages by itself, which removed that case. This is why its weak-confidence correlation reverses: grounding helped most on Qwen 0.5B, which both had decent raw confidence and answered from irrelevant passages most often.

### 3.5 Label quality decided at least one verdict

Every phase audited its labels before reading a metric. The correctness rule was reliable throughout (about 1 error per 100 answers). The NLI grounding label was not: 76-79% agreement with my labels, mostly calling supported short answers unsupported. Re-running the Phase 3 grounding analysis with the label fixed in Phase 4a turns its recorded 2 of 7 pairs into 5 of 7 (ungated; Phase 3's decision stands). The grounding failure in Phase 3 was therefore at least partly a labeling failure, and any conclusion about grounding heads trained on NLI labels should be read with that in mind.

## 4. What the Program Does Not Show

- **Other task families.** Everything here is short-answer QA. Long-form generation, math, and code were not tested (Phases 4b and 5 were not run).
- **Larger models.** The largest model is 7B. Raw confidence improved with scale here, so I expect the conclusion to hold or strengthen above 7B, but I did not test it.
- **Other probe designs.** Heads read one layer (50% depth) with mean pooling over the answer span; the design was selected once on validation splits (Phase 3a) and then frozen. Attention-based features, multiple layers, or token-level heads might do better.
- **A clean scale curve.** The three model sizes come from two families (Qwen and Phi), so scale and family are confounded; only the Qwen pair (0.5B and 7B) is a within-family comparison.
- **Statistical power for small gains.** Several tests missed by tiny margins (self-consistency at 7B on PopQA by a CI bound of -0.0003). The program can say these gains are small; it cannot say they are zero.

## 5. Candidate Contributions for the Paper

Pending the literature review (`reports/TCL novelty literature review.md`), the candidates are:

1. A preregistered, multi-phase test of learned trust heads against **calibrated** raw confidence **on held-out datasets**, across three model sizes, with every failure reported.
2. The **weak-confidence pattern**: learned signals help where raw confidence is poor (Gemma, small Qwen on PopQA) and stop helping as raw confidence improves with scale.
3. A **joint-training negative result**: LM + trust training improves in-distribution heads but not transfer, and plain LM fine-tuning explains its one positive direction.
4. **Self-consistency from one pass** as the most robust second signal: matching 10-sample semantic entropy when combined with raw confidence, but not passing a preregistered majority rule.
5. **Grounding heads mostly detect irrelevant passages**, a case that a 7B model handles by abstaining, and **NLI label noise** large enough to flip a preregistered verdict.

The literature review ([reports/TCL novelty literature review.md](reports/TCL%20novelty%20literature%20review.md)) found none of these to be a new phenomenon. Each is partially known: Orgad et al. (2024) for 1, Kapoor et al. (2024, the opposite result) for 3, Semantic Entropy Probes and CoCoA Light for 4, Dubanowska et al. (2025) and QA2D-style NLI work for 5, and Kadavath et al. (2022) for the scale premise of 2. What is new is the strictness of the test and a few diagnostics; the paper is framed that way ([TCL-Paper-Outline.md](TCL-Paper-Outline.md)).
