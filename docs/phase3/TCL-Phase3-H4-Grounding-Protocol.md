# TCL Phase 3 H4 Protocol: Does Predicted Grounding Add to Raw Confidence on New Data?

Status: **locked** on November 22, 2026, before any grounding data was drawn, any model answered a grounding question, or any metric existed. The only numbers seen are MRQA passage lengths and dataset sizes from public samples, used to choose datasets. Changes after this point are logged as amendments (Section 11) with the reason; the decision rule in Section 5 cannot change once any grounding test metric has been computed.

**Outcome** (added November 28, 2026; the protocol text above and below is unchanged): H4 failed for grounding (2 of 7 pairs) ([checkpoint](TCL-Phase3-H4-Grounding-Checkpoint.md)). An ungated re-run with the label fixed in Phase 4a would pass 5 of 7; the recorded decision stands.

This protocol tests **H4** from my program plan (`TCL-Research-Program.md`, Sections 4 and 7) for the second trust dimension in the plan's order, **grounding**. The first, self-consistency, failed by one pair (`TCL-Phase3-H4-Consistency-Checkpoint.md`).

## 1. Question

**H4 (grounding):** when the model answers a question with a passage in front of it, can a head predict from **one forward pass** whether the answer is **supported by the passage**, and does that prediction add information about correctness **beyond calibrated raw confidence**, on a dataset held out from training?

Grounding is different from correctness (theory paper v2, Section 5.3): an answer can be right but not in the passage (recalled from memory), or in the passage but wrong. It is the trust signal that matters most when a model answers from retrieved documents.

## 2. Change to the Plan's Data Source

The plan names **SQuAD 2.0** for grounding, using its unanswerable questions as negatives. SQuAD 2.0 is one dataset, and from Phase 3 on every trust result must hold on a held-out dataset, so I need the same kind of negatives in several datasets. I replace SQuAD 2.0's hand-written unanswerable questions with **distractor passages**, which work on any reading-comprehension dataset: for a fixed share of questions, the model sees a passage from a different question instead of its own. Any answer it then gives cannot be supported by that passage. This change is recorded in the plan's changelog when this protocol is locked.

## 3. Data

**Source: MRQA 2019** (Fisch et al., 2019; `mrqa-workshop/mrqa`), which provides many reading-comprehension datasets in one format with passages and accepted answers.

**Pool (training and leave-one-out testing):** the three MRQA datasets with short passages, which fit both models and the NLI model's input limit:

| Dataset | Median passage length (words, sampled) |
|---|---:|
| SQuAD | 114 |
| Natural Questions (short) | 99 |
| HotpotQA | 196 |

NewsQA, TriviaQA-web, and SearchQA (medians about 550-680 words) are left out for length.

**New held-out dataset: RelationExtraction**, one of MRQA's out-of-domain test datasets, never used for training or design (Section 10).

**Samples:** per pool dataset, 4,500 questions drawn with seed 2033 (as in Phase 1: 2,500 train, 500 validation, 1,500 test; train and validation from MRQA's training split, test from its validation split, so no passage is shared across my splits). For the new dataset, 1,500 test questions. Then, with seed 2034, **one third of the questions in every split get a distractor**: a passage drawn at random from another question in the same dataset and split.

**Inference:** `run_inference_v2.py` with the passage in the prompt (the existing context prompt: "Answer with only the answer phrase from the context"), plus the abstention line of Section 10; same decoding, answer spans, raw confidence, hidden states at 25/50/100% depth, and P(True) as in Phase 1. No semantic entropy.

**Labels:**
- **Correctness:** rule v3 against the question's accepted answers (for distractor rows too: an answer from memory can still be right).
- **Grounding:** `y_g = 1` if an NLI model (DeBERTa-large-MNLI, as used for self-consistency) judges that the passage entails the answer as a response to the question (premise: the passage; hypothesis: the question followed by the answer), with entailment probability above 0.5; else 0. An answer that appears verbatim in the passage is not counted as grounded on that basis alone (Section 10).
- **Abstentions** ("unanswerable", or a refusal by the shared `is_refusal` rule) are not claims: they get no grounding label and are counted separately.

**Size rule:** a (model, held-out dataset) pair counts only if its test split has at least 150 answers in each correctness class and each grounding class. If not, it is dropped and recorded; no sample is extended after inference.

## 4. Heads and Arms

Exactly the self-consistency design, with the grounding label in place of the consistency label:

- **G-hat:** reads the 50%-depth answer-span hidden state and predicts `y_g` (BCE). Design fixed to the Phase 3a setting (MLP, weight decay 1, equal weight per dataset), not tuned; 5 seeds. Unlike self-consistency, grounding labels exist on every split, so it is trained on the **training** splits of the other pool datasets (for the new dataset: all three) and early-stopped on their validation splits.
- **R:** raw confidence, Platt-mapped on the training mixture's validation splits.
- **R+G-hat:** logistic regression on logit(raw confidence) and the G-hat output, fitted on the same validation splits.
- *R+G (oracle):* the same with the NLI label (reference). *P(True):* reference.

## 5. Metrics and Decision Rule

Per eligible pair, on the held-out test split, each metric computed per seed and averaged:

1. **Learnability:** AUROC of G-hat against `y_g`, at least 0.70.
2. **Non-redundancy:** Spearman correlation of G-hat with calibrated raw confidence, below 0.7.
3. **Added value:** `d_AURC = AURC(R) - AURC(R+G-hat)` for correctness, with its 95% paired bootstrap CI (1,000 resamples, seed 2026) above zero.

A pair passes if all three hold (the plan's thresholds, unchanged). **H4 (grounding) passes** if a majority of eligible pairs pass (with 8: at least 5), including the new-dataset pair of at least one model.

**Also reported, not gated:** correctness AUROC of every arm; a "grounded and correct" target (both labels 1), which is what matters when answering from documents; abstention rates; results on gold-passage and distractor rows separately; refusal-excluded results; per-seed values.

**What follows:**
- **Passes:** grounding becomes TCL's second dimension, and under my publication plan Phases 1-3 are written up together.
- **Fails:** with self-consistency also failed, the plan's Section 8 row applies: trust is effectively one-dimensional at this scale, and I re-test at Phase 4's scale before dropping multi-dimensionality. (Evidence density cannot be tested for transfer: its label exists only on PopQA.)

## 6. Label Audit

Before any metric: 50 answers per model (seed 2035) from the test splits, half from gold-passage rows and half from distractor rows. I judge whether the passage supports the answer and report the NLI label's error rate. Correctness labels are audited the same way on 50 answers per model.

## 7. Order of Work

1. Lock this protocol (done, November 22, 2026). Commit.
2. Write the data script (MRQA sample and distractors), the inference change for decision 2, the NLI grounding labeler, the heads and analysis scripts; test on synthetic data. Commit.
3. Run inference and grounding labels on Kaggle. Count labels; apply the size rule.
4. Label audits. Commit.
5. Train, score, and run the analysis **once**. Write the checkpoint.

## 8. Code and Outputs (planned)

- `scripts/prepare_phase3_grounding.py`: MRQA samples, splits, distractors.
- `scripts/label_grounding.py`: NLI grounding labels (GPU).
- `scripts/run_phase3_h4_grounding.py` and `scripts/run_phase3_h4_grounding_analysis.py`.
- Outputs: `runs/phase3_grounding/<model>/<dataset>/`, `runs/phase3_h4g/`, `runs/phase3_h4g_analysis/`.

## 9. Compute Estimate

About 4,500 x 3 + 1,500 = 15,000 answers per model with passages, without semantic entropy. Rough Kaggle T4 estimate: 3-4 hours for Qwen, 8-12 for Phi, about 1-2 for NLI labels: **about 15 GPU hours**, within one week's free quota.

## 10. Decisions Settled at Locking

1. **New held-out dataset: RelationExtraction** (MRQA out-of-domain; 2,948 questions; single-sentence passages, median 22 words in a sample of 100). Its 1,500 test questions are drawn with seed 2033 like the rest.
2. **Abstention allowed.** A new flag in `run_inference_v2.py` adds one line to the context prompt: "If the context does not contain the answer, reply: unanswerable." The Phase 1 context prompt stays the default, so earlier runs are unchanged.
3. **Grounding label: NLI only**, as the theory paper defines it, checked by the audit. A verbatim span match does not count by itself.
4. **Distractor share: one third** of the questions in every split.

## 11. Amendments

**2026-11-22: Implementation details fixed in code, before any model answered a grounding question.** Fixed in `prepare_phase3_grounding.py`, `label_grounding.py`, `run_phase3_h4_grounding.py`, and `run_phase3_h4_grounding_analysis.py` (docstrings); the scoring and analysis scripts were tested on synthetic data only (with a grounding signal, 7 of 8 pairs pass; without one, every pair fails on learnability):
- One question per passage is kept within each MRQA split before sampling. MRQA's own training and development data share 15 NQ and 3 HotpotQA passages, which therefore appear in two of my splits of that dataset. This does not touch any comparison: when a dataset is held out, its own training split is not used.
- Abstentions are detected as answers that contain "unanswerable", match the shared refusal rule, or are empty. They are left out of training, fitting, and evaluation for every arm, so all arms score the same answers, and they are counted per split.
- The NLI premise is truncated to fit 512 tokens; the question and answer are never truncated.
- Inference skips semantic entropy (`--se-splits ""`) and uses the new `--abstain-instruction` flag.

No decision rule, dataset, model, arm, threshold, or metric changes.

## References

- Fisch, A., Talmor, A., Jia, R., Seo, M., Choi, E., & Chen, D. (2019). MRQA 2019 shared task: Evaluating generalization in reading comprehension. In *Proceedings of the 2nd Workshop on Machine Reading for Question Answering*.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. In *ACL*.
