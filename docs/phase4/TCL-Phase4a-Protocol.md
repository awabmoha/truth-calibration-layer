# TCL Phase 4a Protocol: Do the Phase 3 Results Hold at 7B?

Status: **locked** on November 24, 2026, before any Phase 4a code was written and before Qwen2.5-7B answered any question. Changes are logged as amendments (Section 11) with the reason, and the decision rule in Section 7 cannot change after any Phase 4a test metric has been computed.

**Outcome** (added November 28, 2026; the protocol text above and below is unchanged): H4-T failed (0 of 3), self-consistency failed (2 of 3, without PopQA), grounding was not testable; every step in Section 9 was carried out, using about 12 Kaggle GPU hours ([checkpoint](TCL-Phase4a-Checkpoint.md)). The program closed afterwards.

This protocol carries out the program plan's Section 8 row **"H4 fails for all dimensions"**: *trust is effectively one-dimensional at this scale; I re-test at Phase 4 scale before dropping multi-dimensionality.* It re-runs Phase 3's three preregistered tests (H4-T, H4 self-consistency, H4 grounding) on a 7B model, with the same designs, thresholds, and statistics. It is the first part of Phase 4; Phase 4b (a non-QA task and joint training at scale) is planned after 4a's result.

## 1. Question

Phase 3 found, at 0.5B (Qwen2.5-0.5B) and 3.8B (Phi-3.5-mini), that:
- no correctness head trained on a mixture of datasets beats calibrated raw confidence on a held-out dataset (H4-T: 1 of 6 pairs);
- self-consistency and grounding are learnable from one pass and not redundant with raw confidence, but add to it on too few pairs (3 of 6, 2 of 7).

**Phase 4a asks whether the same holds for a 7B model**, the largest I can run on free compute. The gains in Phase 3 were largest where raw confidence was weak (Qwen 0.5B), and larger models usually have stronger raw confidence, so a pass is not expected; this is the test the plan requires before I drop the multi-dimensional claim.

## 2. Model

**Qwen2.5-7B-Instruct**, the same family as Phase 3's Qwen2.5-0.5B, so the comparison is mostly about scale (14 times the parameters) and not about a new tokenizer, chat template, or training recipe. Same prompts, decoding, answer spans, and features as Phases 1-3 (`run_inference_v2.py`): greedy answers, at most 32 new tokens, answer-span confidence, hidden states at 25/50/100% depth, P(True), and semantic entropy with 10 samples on test splits.

**Precision:** float16, split across Kaggle's two T4 GPUs (`device_map="auto"`). This keeps the model unquantized, so its confidence and hidden states are the model's own. If the smoke test (Section 8) shows numerical failures in float16 (NaN or inf logits, or empty answers on more than 5% of 50 questions), I switch to 4-bit NF4 on one T4, recorded as an amendment before any metric.

One model only. Pairs are therefore (Qwen2.5-7B, held-out dataset), and the majority rules below count datasets.

## 3. Data

Exactly the Phase 3 question sets and splits. No new sampling.

| Test | Pool (heads trained here) | Held out (test only) | Source |
|---|---|---|---|
| H4-T and H4 self-consistency | TriviaQA, SQuAD, NQ-Open (Phase 1 splits) | each pool dataset in turn, and PopQA | `data/phase1_kaggle/`, `data/phase3/popqa.csv` |
| H4 grounding | MRQA SQuAD, NQ, HotpotQA (2,500 / 500 / 1,500) | each pool dataset in turn, and RelationExtraction (1,500) | `data/phase3_grounding/` |

- **PopQA:** the first batch of 1,500 questions (seed 2030), extended by further batches of 1,500 up to 4,500 if the smaller class has fewer than 150 answers: the Phase 3 rule, decided from labels alone.
- **Correctness labels:** rule v3, as in Phase 3.
- **Grounding inference:** with the passage in the prompt and the abstention line (`--abstain-instruction`), semantic entropy skipped, as in Phase 3.
- **Size rule:** a pair counts only if its test split has at least 150 answers in each correctness class (and, for grounding, each grounding class among non-abstentions). No sample other than PopQA is extended. I expect the Phase 1 SQuAD pair to fail this rule: its questions come with their passage, and Phi already answered 94% of them correctly (95 incorrect). SQuAD then serves only as training data for H4-T and self-consistency, leaving 3 pairs (TriviaQA, NQ-Open, PopQA).

## 4. Grounding Label: One Fix, Chosen Before Any 7B Data

The Phase 3 audit found the NLI grounding label wrong on 23 of 100 answers, mostly supported short answers rejected because the hypothesis "question answer" reads poorly to an NLI model. Phase 4a fixes this once, with a rule set now:

- **Development set:** the 100 Phase 3 audit answers with my labels (`runs/phase3_h4g_audit/grounding_audit_reviewed_*.csv`). No 7B answer is used.
- **Candidates (premise = the passage the model saw, threshold 0.5):**
  1. the Phase 3 labeler (`microsoft/deberta-large-mnli`, hypothesis "question answer");
  2. the same model with a template hypothesis: *The answer to the question "{question}" is {answer}.*
  3. a stronger NLI model, `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, with hypothesis 2.
- **Selection rule:** the candidate with the most agreement with my labels on the 100 answers is used; ties go to the lower-numbered candidate. Its agreement on the development set is optimistic and is reported as such; the reported error rate comes from a fresh audit of 7B answers (Section 6).
- **Ungated check on Phase 3:** I relabel the Phase 3 Qwen and Phi grounding answers with the chosen labeler and rerun the Phase 3 grounding analysis as a secondary result. This shows how much the label noise mattered at 0.5B and 3.8B. It does not change Phase 3's recorded decision.

## 5. Heads and Arms

Fixed to the Phase 3 designs; nothing is re-selected.

- **Head:** the Phase 3a setting (`runs/phase3a/selection.json`): MLP (LayerNorm, 256 hidden units, GELU, dropout 0.1), weight decay 1, equal weight per dataset; input the 50%-depth answer-mean hidden state, standardized on the training mixture; 5 seeds (0-4).
- **H4-T arms:** R (Platt-mapped raw confidence), S, M, M+R, and SE (reference), exactly as in the Phase 3 protocol, Section 5.2.
- **H4 self-consistency arms:** R, C-hat (one-pass consistency head), R+C-hat (logistic combiner on validation splits), R+C oracle, and SE, exactly as in the consistency protocol.
- **H4 grounding arms:** R, G-hat (one-pass grounding head, trained on training splits), R+G-hat, R+G oracle, and P(True), exactly as in the grounding protocol. Abstentions are left out of every arm.

## 6. Evaluation

- **Primary metrics:** as in Phase 3. H4-T: AUROC of M+R against R. H4 dimensions: learnability AUROC, Spearman correlation with raw confidence, and `d_AURC` = AURC(R) - AURC(R+head).
- **Statistics:** 1,000 paired bootstrap resamples of test items (seed 2026), seed-averaged metrics, 95% percentile CIs; per-seed values reported.
- **Also reported:** AURC, accuracy at 50% and 80% coverage, ECE and Brier; refusal-excluded results; for grounding, results on gold and distractor rows separately and the grounded-and-correct target.
- **Label audits, before any metric:** 100 PopQA answers (seed 2040) for rule v3; 50 consistency labels (seed 2041) counted by hand; 50 grounding answers (seed 2042), half gold and half distractor, for both the grounding label and rule v3.

## 7. Decision Rule

Per test, a pair passes on the Phase 3 condition:
- **H4-T:** `d(M+R, R)` CI lower bound above zero.
- **H4 (each dimension):** learnability AUROC >= 0.70, Spearman with raw confidence < 0.7, and `d_AURC` CI lower bound above zero.

A test passes if **a majority of its eligible pairs** pass (with 4 eligible: at least 3; with 3: at least 2; with 2: both), **including its new-dataset pair** (PopQA for H4-T and self-consistency; RelationExtraction for grounding). A test with fewer than 2 eligible pairs, or without its new-dataset pair, is **not testable** at 7B and is recorded as such.

**H4 at 7B passes** if self-consistency or grounding passes. Two dimensions give two chances, as in Phase 3; I report this.

**Also reported, not gated:**
- **Scale comparison:** each metric at 0.5B, 3.8B, and 7B on the datasets all three share.
- **Weak-confidence pattern:** across every pair from Phase 3 and Phase 4a, the correlation between raw confidence's AUROC and each head's gain over it. This tests the Phase 3 observation that gains appear where raw confidence is weak; it is descriptive and decides nothing.

## 8. What Follows

| Outcome | Decision |
|---|---|
| H4-T or H4 passes at 7B | The multi-dimensional (or learned-correctness) claim holds at 7B but not at 0.5B and 3.8B. Under my publication plan this is the first success: I write up Phases 1-4a as one paper, with every earlier failure. Phase 4b tests whether it holds on GSM8K. |
| Both fail (or are not testable) | Trust is effectively one-dimensional from 0.5B to 7B on QA. Per the plan's Section 8, I drop the multi-dimensional claim for this program's scale and record it as a main result. The scope of Phase 4b (GSM8K, joint training at scale) is decided at that gate, and the program plan is revised before any 4b work. |

## 9. Order of Work

1. Lock this protocol and the plan revision (done, November 24, 2026). Commit.
2. Code: a `--device-map` option in `run_inference_v2.py` (float16 across two GPUs); a `--models` option in the Phase 3 heads, H4, and grounding scripts (default unchanged); the grounding-label candidates and selection script. Test on synthetic data and on the development set. Commit, with the chosen labeler.
3. Kaggle smoke test: 50 questions per dataset. Check numerics, timing, and labels. Then full inference: the three pool datasets, PopQA, and the four grounding datasets.
4. Count labels; apply the size rule (and the PopQA extension if needed). Draw and review the three audits. Commit.
5. Run the three analyses **once** each. Write the checkpoint.

## 10. Compute Estimate

Qwen2.5-7B in float16 on two T4s, about twice Phi-3.5-mini's time per question (Phase 1 Phi: 0.5-0.7 s per question greedy, 2.3-3.0 s with semantic entropy):

| Run | Questions | Estimate |
|---|---:|---:|
| Pool, train and validation (greedy) | 8,110 | ~3 h |
| Pool and PopQA test (with semantic entropy) | 6,000 | ~8-10 h |
| Grounding (greedy, with passage) | 15,000 | ~6-7 h |
| NLI labels, smoke tests, buffer | | ~4 h |
| **Total** | | **~20-25 Kaggle GPU hours**, within one to two weeks of free quota |

Kaggle sessions stop after about 12 hours, so the pool and grounding runs go in separate sessions (each dataset can run on its own).

## 11. Amendments

**2026-11-28: Grounding audit composition.** The protocol planned the grounding audit as 25 gold-passage and 25 distractor answers. With the passage and the abstention line, Qwen2.5-7B abstained on 1,996 of the 2,000 held-out distractor rows (SQuAD 498 of 500, NQ 500, HotpotQA 499, RelationExtraction 499), leaving 4 distractor answers. The audit therefore takes all 4, and gold rows fill the rest of the 50 (seed 2042, as planned). I decided this from abstention counts alone, before computing any metric. Nothing else changes: abstentions stay out of every arm, as fixed in Section 5. The grounding test at 7B is thus effectively a test on gold-passage answers, unlike Phase 3, where most of the head's gain came from distractor rows. I report this with the results.

## References

- Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.
- Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. *Political Analysis*, 32(1), 84-100.
- Protocols this one reuses: [Phase 3](../phase3/TCL-Phase3-Protocol.md), [H4 self-consistency](../phase3/TCL-Phase3-H4-Consistency-Protocol.md), [H4 grounding](../phase3/TCL-Phase3-H4-Grounding-Protocol.md).
