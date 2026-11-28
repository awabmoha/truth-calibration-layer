# TCL Final Analyses: Power, the Weak-Confidence Pattern, and the Choice of Baseline

Written November 28, 2026, after the program closed. These are the reporting analyses the literature review asked for before drafting the paper ([TCL-Paper-Outline.md](TCL-Paper-Outline.md)). They use saved scores and records only (`tcl_experiments/scripts/final_analyses.py`, outputs in `tcl_experiments/runs/final_analyses/`). **They are exploratory: none was preregistered, and none changes a recorded decision.**

## 1. How Large an Effect Could Each Test Detect?

For each gated difference I take the recorded 95% bootstrap CI, estimate its standard error as (upper - lower) / 3.92, and report the minimum detectable effect (MDE): the true effect a test of this precision would detect with 80% power under the "CI lower bound above zero" rule, MDE = 2.80 x SE. Full table: `runs/final_analyses/mde.csv`.

| Test | Metric | Typical MDE (range over pairs) | What the nulls rule out |
|---|---|---|---|
| H1-R (Phase 1, in-distribution) | AUROC difference | 0.03-0.07 | |
| H3-T, H3-J (Phases 2, 2b) | AUROC difference | 0.017-0.033 | A transfer gain of joint training above about 0.03 AUROC in the SQuAD-to-TriviaQA direction, where the observed effect is -0.038 in both runs |
| H4-T (Phase 3) | AUROC d(M+R, R) | 0.033-0.062 | |
| H4-T (Phase 4a, 7B) | AUROC d(M+R, R) | 0.030-0.045 | A head gain over answer probability of about 0.03-0.045 AUROC or more at 7B, on any of the 3 pairs |
| H4 self-consistency (3, 4a) | AURC gain | 0.006-0.018 | |
| H4 grounding (3, 4a) | AURC gain | 0.008-0.019 | |

**Reading.** The tests were sized for moderate effects, not tiny ones. At 7B, a learned correctness head that beat answer probability by 0.03-0.045 AUROC would most likely have been detected; a gain of 0.01 would not. The self-consistency and grounding tests are more precise (MDE about 0.01 AURC), so their misses are closer calls: self-consistency at 7B on PopQA observed +0.0069 with an MDE of 0.011. The honest summary is "no gain of practical size", not "no gain".

**Phase 2 is not the weakest test.** The outline expected the Phase 2 tests (2 training datasets x 3 seeds) to be underpowered. Their CIs are in fact among the narrowest (MDE about 0.02-0.03), because the bootstrap resamples 1,500 fixed test items and averages over seeds. What 3 seeds cannot capture is seed-to-seed variation larger than the seeds observed; the per-seed values are in each checkpoint.

## 2. The Weak-Confidence Pattern, Corrected

Phase 4a reported that gains are larger where raw confidence is weaker. The literature review pointed out that this correlation is partly built in: the gain is (head AUROC - raw AUROC), so noise in the raw AUROC enters both axes with opposite signs, and there is less room to gain near AUROC 1. Corrected versions (`runs/final_analyses/weak_confidence_corrected.json`; Spearman, 9 eligible pairs per head from Phases 3 and 4a):

| Head | Full sample | 95% CI (resampling pairs) | Headroom-normalized | Split-sample | Leave one model out |
|---|---:|---|---:|---:|---|
| H4-T (M+R) | -0.43 | [-0.95, +0.36] | -0.52 | -0.27 | -0.71 to -0.20 |
| Self-consistency (R+C-hat) | -0.60 | [-0.95, +0.14] | -0.48 | -0.53 | -0.71 to -0.54 |
| Grounding (R+G-hat) | +0.63 | [-0.16, +0.89] | +0.82 | +0.62 | +0.36 to +0.50 |
| All 27 pooled | -0.19 | | -0.16 | | |

**Reading.**
- **For self-consistency the pattern survives every correction** (-0.48 to -0.71), but its CI over 9 pairs still includes zero.
- **For the correctness head it weakens** once the shared noise is removed (split-sample -0.27), so part of the Phase 4a number was the artifact.
- **For grounding the sign is reversed throughout**, for the reason in the Phase 4a checkpoint: the gains came from irrelevant-passage answers, not from weak confidence.
- **Pooled, there is no pattern** (-0.19).

The clearest support for "learned signals help where confidence is weak" is therefore not this correlation but the Phase 1 Gemma case (raw AUROC 0.53-0.62, probe gains +0.25 to +0.31, the only model whose probe transferred). In the paper I present the pattern as a hypothesis suggested by Gemma and by self-consistency, not as an established regularity.

## 3. Does the Choice of Baseline Matter?

TCL's baseline is the answer-span probability: the geometric mean of the answer tokens' probabilities. Two other token-probability baselines cost nothing extra: the first answer token's probability, and the minimum token probability in the answer. AUROC on the H4-T held-out test answers (`runs/final_analyses/baseline_choice.csv`; head = M+R, mean over 5 seeds):

| Pair | Answer span (TCL) | First token | Minimum token | Head M+R | Head minus best baseline |
|---|---:|---:|---:|---:|---:|
| Qwen 0.5B / TriviaQA | 0.712 | 0.746 | **0.773** | 0.744 | -0.030 |
| Qwen 0.5B / SQuAD | **0.762** | 0.717 | 0.750 | 0.667 | -0.096 |
| Qwen 0.5B / PopQA | 0.828 | 0.884 | 0.896 | **0.907** | +0.012 |
| Phi / TriviaQA | **0.836** | 0.807 | 0.834 | 0.795 | -0.041 |
| Phi / NQ-Open | **0.765** | 0.708 | 0.749 | 0.760 | -0.006 |
| Phi / PopQA | 0.864 | 0.874 | **0.881** | 0.809 | -0.072 |
| Qwen 7B / TriviaQA | **0.854** | 0.826 | 0.853 | 0.833 | -0.021 |
| Qwen 7B / NQ-Open | **0.742** | 0.707 | 0.735 | 0.738 | -0.004 |
| Qwen 7B / PopQA | 0.865 | 0.879 | **0.882** | 0.844 | -0.038 |

**Reading.**
1. **The answer-span probability was not the strongest free baseline everywhere.** The minimum token probability beats it on 4 of 9 pairs, by up to +0.068 (Qwen 0.5B on PopQA).
2. **Against the best of the three baselines, the learned head wins on 1 of 9 pairs** (Qwen 0.5B on PopQA, by 0.012), and loses on 8. Phase 3's one clear H4-T win (+0.079 over the answer span) is mostly a weak-baseline effect: the minimum token probability gets 0.896 of the head's 0.907 for free.
3. **Against the first-token probability, the head looks better** (it wins on 4 of 9 pairs, by up to +0.051). A paper that used first-token probability as its baseline would have reported more probe wins from the same data. This is direct evidence for the review's point that the probe-versus-logit gap depends on which logit baseline is chosen, and it may explain part of the disagreement with work reporting that probes beat token probabilities.

This strengthens the program's conclusion, with one caution: the minimum-token comparison was not preregistered, and choosing the best of three baselines after seeing the data favors the baselines. The fair statement is that the preregistered baseline was already hard to beat, and a different free baseline would have been harder still.

## 4. 7B H4-T Losses, With Paired CIs

From `runs/phase4a_h4t_analysis/details.json` (seed-averaged; 95% paired bootstrap CI over 1,500 test items):

| Pair | d(M+R, R) | 95% CI | Seeds below zero |
|---|---:|---|---|
| TriviaQA | -0.021 | [-0.042, +0.000] | 5 of 5 |
| NQ-Open | -0.004 | [-0.035, +0.027] | 2 of 5 |
| PopQA | -0.022 | [-0.045, +0.005] | 5 of 5 |

No loss is significant on its own (each CI touches or crosses zero), but on TriviaQA and PopQA every seed is below zero; on NQ-Open the seeds split (2 below, 3 above). The accurate statement is that adding hidden states never helped and probably hurt slightly, not that it significantly hurt.

## 5. What Changes in the Paper

- Report MDEs next to every null.
- Present the weak-confidence pattern as a hypothesis (Gemma, self-consistency), not a finding; drop the pooled correlation as evidence.
- Add the baseline-choice table: it is the most useful new diagnostic, and it connects the program to the disagreement in the literature.
- Describe the 7B H4-T result as "no gain, probably a small loss".
