# TCL Phase 4a Checkpoint: At 7B, No Learned Head Beats Raw Confidence, Self-Consistency Adds to It on 2 of 3 Pairs but Misses on PopQA, and Grounding Is Not Testable

Written November 28, 2026, after the single preregistered run of each Phase 4a analysis. Protocol: [TCL-Phase4a-Protocol.md](TCL-Phase4a-Protocol.md) (locked November 24, 2026, before any Phase 4a code or 7B output; one amendment on the grounding audit's composition, made from abstention counts before any metric). Earlier results: [Phase 3 checkpoint (H4-T)](../phase3/TCL-Phase3-Checkpoint.md), [H4 self-consistency](../phase3/TCL-Phase3-H4-Consistency-Checkpoint.md), [H4 grounding](../phase3/TCL-Phase3-H4-Grounding-Checkpoint.md).

## 1. Decision

| Test | Rule (Phase 3, unchanged) | Eligible 7B pairs | Result |
|---|---|---|---|
| **H4-T** | `d(M+R, R)` CI lower bound above zero on a majority of eligible pairs, including PopQA | TriviaQA, NQ-Open, PopQA | **Fails.** 0 of 3. |
| **H4 self-consistency** | Learnability AUROC >= 0.70, Spearman with raw confidence < 0.7, `d_AURC` CI lower bound above zero, on a majority of eligible pairs including PopQA | TriviaQA, NQ-Open, PopQA | **Fails.** 2 of 3 pass (TriviaQA, NQ-Open), but the PopQA pair does not (CI lower bound -0.0003). |
| **H4 grounding** | The same, including RelationExtraction | NQ, HotpotQA | **Not testable** (Section 7 of the protocol): the RelationExtraction pair fails the size rule. Both eligible pairs fail anyway. |

**H4 at 7B fails:** self-consistency fails and grounding is not testable.

**What follows, as fixed in the protocol (Section 8, row "Both fail or are not testable"):** trust is effectively one-dimensional from 0.5B to 7B on QA. I drop the multi-dimensional claim for this program's scale and record it as a main result. The scope of Phase 4b is decided at this gate, and the program plan is revised before any 4b work. Under my publication plan this is not the first success, so Phases 1-4a are not written up yet.

**In plain words:** at 7B, the model's own confidence is already good, and no head trained on its hidden states improves on it. A head that predicts, from one pass, whether the model would give the same answer again adds a small, real gain on two of three held-out datasets, and misses the bar on the third by the smallest margin. Grounding could not be tested fairly: the 7B model refuses almost every irrelevant passage by itself and rarely answers wrongly when it does answer.

## 2. What Ran

- **Model:** Qwen2.5-7B-Instruct, float16 split across Kaggle's two T4s (`device_map="auto"`). The smoke test (50 questions per dataset) showed no non-finite value and no empty answer, so the 4-bit fallback was not needed.
- **Inference:** two Kaggle sessions at commit 5b40b30, about 11 GPU hours in total: TriviaQA, SQuAD, NQ-Open (all splits, semantic entropy on test), PopQA (the first 1,500 questions), and the four grounding sets with the passage and the abstention line. 29,110 answers, none empty.
- **Grounding label:** the template labeler, chosen before any 7B data (79 of 100 on the Phase 3 audit set).
- **Heads, arms, statistics:** the Phase 3 designs, unchanged: the Phase 3a MLP (weight decay 1, equal weighting, 50%-depth answer-mean feature), 5 seeds, 1,000 paired bootstrap resamples (seed 2026).
- **Analyses:** the Phase 3 scripts with `--runs runs/phase4a --models qwen7b`, run once each on a laptop CPU.

### Size rule

| Test | Pair | Smaller class on the test split | Eligible |
|---|---|---|---|
| H4-T, self-consistency | TriviaQA | 664 incorrect | Yes |
| | NQ-Open | 344 correct | Yes |
| | PopQA | 275 correct | Yes (no extension needed) |
| | SQuAD (Phase 1) | 86 incorrect | No, as the protocol expected; training data only |
| Grounding (answered questions only) | NQ | 226 incorrect | Yes |
| | HotpotQA | 155 not grounded | Yes |
| | SQuAD | 67 incorrect | No |
| | RelationExtraction | 75 incorrect | No: the new-dataset pair |

The protocol expected Phase 1 SQuAD to fail. It did not expect the grounding pairs to fail, and I did not foresee it either: when I first counted the labels, I counted incorrect answers including abstentions and reported all four grounding pairs as eligible. The rule counts only answered questions, and the locked script applied it that way, from labels alone, before computing any metric. Once the 7B model has abstained on what it cannot answer, it is wrong on only 7% of the SQuAD and 8% of the RelationExtraction questions it answers.

### The 7B model refuses irrelevant passages

With the abstention line, the 7B model said "unanswerable" on 1,996 of the 2,000 held-out distractor rows (passages unrelated to the question). Qwen 0.5B answered 287-359 distractor rows per held-out dataset and Phi 28-122. In Phase 3, almost all of the grounding head's gain came from ranking those distractor answers low. At 7B, the model's own abstention does that work, so grounding at 7B is tested on answers given from the right passage only. This is why the grounding audit took 4 distractor answers instead of 25 (amendment, protocol Section 11).

## 3. Label Audits (Before Any Metric)

| Audit | Sample | Result |
|---|---|---|
| PopQA correctness (rule v3) | 100 answers, seed 2040 | 1 rule error (a genre list including "war" for the gold "war film"); 7 flagged as uncertain (gold lists that are incomplete or refer to another work with the same title) |
| Self-consistency count | 50 held-out test answers, seed 2041 | NLI count equal to my count on 43 of 50; binary label (count >= 5) agrees on 47 of 50 |
| Grounding label | 50 answered held-out test answers (46 gold, 4 distractor), seed 2042 | Agrees with me on 38 of 50: 11 supported answers labeled unsupported, 1 unsupported answer labeled supported |
| Grounding audit, correctness | same 50 | Rule v3 agrees with me on 47 of 50 (all 3 errors mark a correct answer incorrect) |

The correctness and consistency labels are about as good as in Phase 3 (Phase 3: 1 PopQA error per model; consistency binary label 92 of 100). The grounding label is not: the labeler chosen on the Phase 3 audit set (79 of 100 there) agrees with me on 76% of 7B answers, and its errors run one way, calling supported answers unsupported. At 7B almost every answer comes from the right passage, so the "not grounded" class is made largely of labeler errors. A noisy target weakens the grounding head's learnability and its combination with raw confidence alike, so it is more likely to hide an effect than to create one. With RelationExtraction ineligible, the grounding test is not testable regardless.

## 4. Primary Results

### H4-T

| Pair | AUROC R | AUROC M | AUROC M+R | `d(M+R, R)` [95% CI] | Pass |
|---|---:|---:|---:|---|---|
| TriviaQA | 0.854 | 0.832 | 0.833 | -0.021 [-0.042, +0.000] | No |
| NQ-Open | 0.742 | 0.734 | 0.738 | -0.004 [-0.035, +0.027] | No |
| PopQA | 0.865 | 0.834 | 0.844 | -0.022 [-0.045, +0.005] | No |

Adding hidden states to calibrated raw confidence makes the ranking slightly worse on every pair (TriviaQA and PopQA on all 5 seeds). With refusals excluded, no decision changes.

### H4 self-consistency

| Pair | Learnability AUROC (>= 0.70) | Spearman with raw (< 0.7) | `d_AURC` = AURC(R) - AURC(R+C-hat) [95% CI] | Pass |
|---|---:|---:|---|---|
| TriviaQA | 0.864 | 0.614 | **+0.0085** [+0.0006, +0.0165] | **Yes** |
| NQ-Open | 0.834 | 0.573 | **+0.0202** [+0.0113, +0.0291] | **Yes** |
| PopQA | 0.881 | 0.636 | +0.0069 [-0.0003, +0.0150] | No |

All 5 seeds improve AURC on every pair (PopQA: +0.005 to +0.009); PopQA's interval misses zero by 0.0003. With refusals excluded (a reported secondary analysis), the PopQA pair passes too; the gate is on all answers. The rule requires the new-dataset pair, fixed before the run, and I do not relax it after seeing the result.

### H4 grounding (eligible pairs; the test is not testable)

| Pair | Learnability AUROC | Spearman with raw | `d_AURC` [95% CI] | Pass |
|---|---:|---:|---|---|
| NQ | 0.696 | 0.192 | -0.0099 [-0.0145, +0.0001] | No |
| HotpotQA | 0.713 | 0.036 | -0.0123 [-0.0204, -0.0032] | No |

## 5. Reported Comparisons (Not Gated)

**Correctness AUROC, self-consistency arms:**

| Pair | R | **R+C-hat** (one pass) | `d(R+C-hat, R)` [95% CI] | R+C oracle | Semantic entropy (10 samples) |
|---|---:|---:|---|---:|---:|
| TriviaQA | 0.854 | **0.869** | **+0.016** [+0.002, +0.029] | 0.879 | 0.864 |
| NQ-Open | 0.742 | **0.772** | **+0.030** [+0.014, +0.046] | 0.756 | 0.747 |
| PopQA | 0.865 | 0.878 | +0.012 [-0.002, +0.027] | 0.887 | 0.884 |

One forward pass plus raw confidence matches or beats semantic entropy, which needs 10 extra samples, on TriviaQA (0.869 vs 0.864) and NQ-Open (0.772 vs 0.747), and comes within 0.006 on PopQA. Phase 3 saw the same on Qwen 0.5B. On NQ-Open the one-pass head even beats the sampled label it imitates (0.772 vs 0.756).

**Semantic entropy against the H4-T head:** semantic entropy beats M+R on every pair (TriviaQA -0.030 [-0.050, -0.011], PopQA -0.040 [-0.065, -0.014], NQ-Open -0.009, CI spanning zero).

**Grounding:** correctness AUROC R 0.716 vs R+G-hat 0.712 (NQ) and 0.601 vs 0.595 (HotpotQA); the oracle with the NLI label adds nothing either (0.715, 0.601). For grounded-and-correct answers, R+G-hat beats raw confidence slightly (NQ 0.647 vs 0.634; HotpotQA 0.612 vs 0.605). Raw confidence itself is much weaker on these sets than on the closed-book ones, because the easy contrast (distractor answers) is gone.

## 6. Scale Comparison (Not Gated)

Each metric at 0.5B (Qwen2.5-0.5B), 3.8B (Phi-3.5-mini), and 7B (Qwen2.5-7B), on the held-out datasets all three share as eligible pairs (`runs/phase4a_ungated/scale_comparison.csv`). Grounding uses Phase 3's recorded labels.

| Test | Dataset | Gain over raw confidence (AUROC): 0.5B / 3.8B / 7B | Raw confidence AUROC: 0.5B / 3.8B / 7B |
|---|---|---|---|
| H4-T (M+R) | TriviaQA | +0.032 / -0.041 / -0.021 | 0.712 / 0.836 / 0.854 |
| | PopQA | +0.079 / -0.055 / -0.022 | 0.828 / 0.864 / 0.865 |
| Self-consistency (R+C-hat) | TriviaQA | +0.073 / -0.011 / +0.016 | |
| | PopQA | +0.067 / -0.002 / +0.012 | |
| Grounding (R+G-hat) | NQ | +0.025 / -0.023 / -0.004 | 0.780 / 0.708 / 0.716 |
| | HotpotQA | +0.025 / +0.011 / -0.006 | 0.745 / 0.670 / 0.601 |

- **Raw confidence gets better with scale on closed-book QA** (TriviaQA 0.712 to 0.854), and the room for any head shrinks with it.
- **Within the Qwen family, every head's gain shrinks from 0.5B to 7B.** Self-consistency's `d_AURC` falls from about +0.02 to +0.007-0.009 on TriviaQA and PopQA. Phi gains less than either Qwen model in 5 of these 6 comparisons, so the three points are not a clean scale curve: model family and scale are confounded.
- **Self-consistency is the one dimension that adds at every Qwen size.** At 0.5B it passed on Qwen's TriviaQA and PopQA pairs (Phase 3 failed on the pooled Qwen and Phi pairs, 3 of 6); at 7B it passes on TriviaQA and NQ-Open and misses narrowly on PopQA. I report this as an observation; it decides nothing.

## 7. Weak-Confidence Pattern (Not Gated)

Across every eligible pair from Phase 3 and Phase 4a, the correlation between raw confidence's AUROC and each head's AUROC gain over it (`runs/phase4a_ungated/weak_confidence.json`):

| Head | Pairs | Spearman | Pearson |
|---|---:|---:|---:|
| H4-T (M+R) | 9 | -0.43 | -0.16 |
| Self-consistency (R+C-hat) | 9 | -0.60 | -0.53 |
| Grounding (R+G-hat) | 9 | +0.63 | +0.63 |
| All pooled | 27 | -0.19 | -0.08 |

For the correctness and consistency heads, gains are larger where raw confidence is weaker, as Phase 3 suggested. For grounding the sign reverses: the gains were largest on Qwen 0.5B, whose raw confidence was strongest on those sets, because they came from distractor answers rather than from weak confidence. Nine points per head are few; this is descriptive.

## 8. Phase 3 Grounding With the Fixed Label (Not Gated)

I relabeled the Phase 3 grounding answers (Qwen 0.5B and Phi) with the template labeler and re-ran the Phase 3 grounding analysis unchanged (`runs/phase4a_relabel_h4g_analysis/`). This shows how much label noise mattered at Phase 3. **It does not change Phase 3's recorded decision**, which stands as a failure.

| Pair | Learnability AUROC | Spearman with raw | `d_AURC` [95% CI], fixed label | Pass (fixed label) | Pass (recorded, Phase 3) |
|---|---:|---:|---|---|---|
| Qwen / SQuAD | 0.863 | 0.604 | +0.0167 [+0.0086, +0.0247] | Yes | Yes |
| Qwen / NQ | 0.860 | 0.565 | +0.0126 [+0.0008, +0.0238] | Yes | Yes |
| Qwen / HotpotQA | 0.782 | 0.552 | +0.0199 [+0.0091, +0.0302] | Yes | No |
| Qwen / RelationExtraction | 0.875 | 0.607 | +0.0113 [+0.0052, +0.0178] | Yes | No |
| Phi / SQuAD | 0.833 | 0.303 | +0.0078 [+0.0027, +0.0149] | Yes | No |
| Phi / NQ | 0.805 | 0.196 | -0.0092 [-0.0267, +0.0033] | No | No |
| Phi / HotpotQA | 0.748 | 0.246 | +0.0065 [-0.0041, +0.0198] | No | No |

**With the fixed label, the Phase 3 grounding test would pass: 5 of 7 pairs, including Qwen / RelationExtraction** (recorded: 2 of 7). Learnability rises on every pair (by 0.04-0.10), and so does the gain over raw confidence on 6 of 7. The NLI label's errors at Phase 3, mostly supported answers called unsupported, were hiding most of the effect.

Three cautions keep this from being a result I can claim:
1. **It is ungated and after the fact.** The Phase 3 decision was fixed before the run, and the fixed label exists because Phase 3 failed.
2. **The labeler was chosen on 100 of these same answers** (the Phase 3 audit set). That is 100 of about 8,000 test answers, and the selection used my labels, not any result, but it is not independent.
3. **The gain still comes mostly from distractor rows.** Among answers given from the right passage, the fixed-label combination gains over raw confidence on SQuAD (Qwen +0.020, Phi +0.005) and HotpotQA (Qwen +0.010, Phi +0.012), not on NQ or RelationExtraction (-0.026 to +0.002). This is more than the recorded labels showed (at most +0.003), but most of the pooled gain is still the head spotting an irrelevant passage, which the 7B model does by itself.

For grounded-and-correct answers (both labels 1), the fixed-label combination beats raw confidence on all 7 pairs (+0.04 to +0.11 AUROC).

What this changes is the reading of Phase 3, not its decision: grounding at 0.5B and 3.8B looks like a real, learnable, partly non-redundant signal that a noisy label obscured, and at 7B the setting that made it useful (answering from an irrelevant passage) has disappeared.

## 9. What This Phase Shows

1. **At 7B, calibrated raw confidence is the correctness signal to beat, and nothing trained on hidden states beats it.** The learned correctness head (H4-T) is worse than raw confidence on every pair.
2. **Self-consistency is the most robust second signal in this program.** From one pass, it adds a small but real gain over raw confidence on two of three held-out datasets at 7B, and it matches 10-sample semantic entropy at a tenth of the cost. It failed the preregistered rule twice (pooled Qwen and Phi at Phase 3; PopQA at 7B), so I do not claim it.
3. **Larger models make passage grounding a non-problem in this setup.** Given a way to say "unanswerable", the 7B model refuses irrelevant passages by itself, which removes the case where a grounding head helped.
4. **The grounding label is the weakest part of the pipeline, and it mattered.** NLI labels agree with my labels on 76-79% of answers, and most errors call a supported answer unsupported. With the fixed label, the Phase 3 grounding test would have passed (5 of 7, ungated); Phase 3's recorded failure was at least partly a labeling failure.

## 10. Next

- Revise the program plan at the Phase 4b gate before any 4b work: decide whether 4b (GSM8K, joint training at 7B) still tests a live hypothesis, and how self-consistency's near-pass should shape it.
- Keep Phases 1-4a polished for the cumulative paper; under my publication plan, the paper waits for the first success.
