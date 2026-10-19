# TCL Phase 2 Protocol: Does Joint Training Transfer?

Status: locked before any Phase 2 training. Written October 18, 2026.

This protocol fixes how I test H3-T and H3-J from `TCL-Research-Program.md`. I commit it before writing the training code and before any Phase 2 model is trained. Changes after this point are logged as amendments (Section 11) with the reason; the decision rules in Section 8 cannot change once any Phase 2 test result has been computed.

## 1. Questions

- **H3-T (primary):** does a trust head trained jointly with the backbone (TCL-J) rank correct above incorrect answers better than a frozen head (TCL-F) **on a dataset it was not trained on**?
- **H3-J (secondary):** does the joint objective beat trust-only fine-tuning of the backbone on the same transfer test, at equal or better generation quality?

## 2. Model

Qwen2.5-0.5B-Instruct, the same model, prompts, and generation settings as Phase 1 (`run_inference_v2.py`: chat prompt, greedy decoding, stop at end of turn, answer span = text before the first newline).

## 3. Data

| Dataset | Train | Validation | Test | Source |
|---|---:|---:|---:|---|
| TriviaQA (`rc.nocontext`) | 2,500 | 500 | 1,500 | `tcl_experiments/data/phase1_kaggle/triviaqa.csv` |
| SQuAD (article-disjoint splits) | 2,500 | 500 | 1,500 | `tcl_experiments/data/phase1_kaggle/squad.csv` |

These are the Phase 1 question sets and splits, already committed. NQ-Open is not used (Qwen answers 7% correctly).

**Answers.** The base model's answers are taken from the Phase 1 Qwen records (`tcl_experiments/runs/phase1_kaggle/qwen/`), so every arm is trained and tested on exactly the same question-answer pairs.

**Labels.** Rule v3 (`answer_match_v3`, `docs/phase2/TCL-Labeling-Rule-v3.md`), applied to the answer span.

| Dataset | Train correct / total | Validation correct / total | Test correct / incorrect |
|---|---:|---:|---:|
| TriviaQA | 410 / 2,500 | 84 / 500 | 216 / 1,284 |
| SQuAD | 1,741 / 2,500 | 362 / 500 | 1,040 / 460 |

Both test sets meet the size rule (at least 150 of each class). No base-model answer is a refusal.

**Reference answers** (for the LM loss): the first accepted answer in the question set. For TriviaQA that is the canonical answer value (it is listed first by `prepare_phase1_kaggle_data.py`); for SQuAD it is the first annotated answer span.

## 4. Trust Head and Losses

**Input.** The mean of the backbone's hidden states over the answer-span tokens at layer 12 of 24 (50% depth), the best configuration in the Phase 1 layer sweep. The input sequence is the Phase 1 chat prompt followed by the answer.

**Head.** Two-layer MLP: LayerNorm, Linear(896 -> 256), GELU, dropout 0.1, Linear(256 -> 1), sigmoid. The same head is used by every arm.

**Trust loss.** Binary cross-entropy between the head's output and the v3 label, unweighted (class weighting distorts probabilities, as Phase 1 showed).

**LM loss.** Negative log-likelihood of the reference answer tokens, given the chat prompt (teacher forcing). It is computed on reference answers only, never on the model's own answers, so that training does not reinforce wrong answers.

## 5. Arms

All arms see the same training questions, answers, labels, head architecture, and number of epochs.

| Arm | Backbone | Objective | Role |
|---|---|---|---|
| A | Frozen | `L_trust` (head only) | TCL-F, the frozen baseline |
| B | LoRA | `L_trust` | Trust-only fine-tuning |
| C | LoRA | `L_LM + lambda * L_trust` | TCL-J, the proposal |
| D | LoRA, then frozen | `L_LM`, then a head trained on the adapted, frozen backbone | Controls for the effect of LM fine-tuning alone |

**LoRA:** rank 16, alpha 32, dropout 0.05, on all attention and MLP projections (`q, k, v, o, gate, up, down`).

**Optimization:** AdamW, learning rate 2e-4 for LoRA weights and 1e-3 for the head, weight decay 0.01, linear warmup over the first 5% of steps then linear decay, effective batch size 16, 3 epochs, gradient clipping at 1.0, fp16 mixed precision. The checkpoint kept is the epoch with the lowest **validation trust loss** on the training dataset's own validation split.

**Lambda for C:** chosen from {0.1, 0.5, 1.0} by validation AUROC on the training dataset's own validation split, using seed 0. The chosen lambda is then run with seeds 1 and 2. All three lambda values are reported.

**Seeds:** 3 per arm per training dataset (0, 1, 2). Seeds set data order, dropout, LoRA initialization, and head initialization.

**Runs:** 4 arms x 2 training datasets x 3 seeds, plus the two extra lambda values for C with seed 0: 28 trainings.

## 6. Evaluation

Each trained model is evaluated on both test sets: its own dataset (in-distribution) and the other one (**transfer**: TriviaQA -> SQuAD and SQuAD -> TriviaQA).

**Primary evaluation set: fixed base-model answers.** The trust head scores the same 1,500 base-model answers per test set for every arm, with the same labels. This gives a paired comparison: every arm is judged on identical items.

**Secondary evaluation set: regenerated answers.** Each LoRA-trained model (B, C, D) also answers the test questions itself; those answers are labeled with rule v3 and scored by its own head. This is the realistic setting, but the answer sets differ between arms, so it is not used for the H3 decisions.

**Generation quality,** for B, C, and D, relative to the base model:
- answer accuracy (rule v3) on both test sets, from the regenerated answers;
- perplexity on a fixed held-out text sample: the first 200 non-empty paragraphs of the WikiText-2 test split (`wikitext-2-raw-v1`).

**Generation-quality bar:** accuracy drops by at most 1 percentage point on **each** test set, and perplexity rises by at most 3%.

## 7. Metrics and Statistics

- **Primary metric:** AUROC on the fixed base-model answers.
- **Also reported:** AURC; accuracy at 50% and 80% coverage; ECE (15 equal-mass bins) and Brier score after a Platt map fitted on the training dataset's validation split (for transfer, the source validation split, since target labels are unavailable in deployment).
- **Refusals:** reported as a separate count; every trust result is also reported with refusals excluded.
- **Seeds and bootstrap:** for a difference between two arms, each of 1,000 paired bootstrap resamples of the test items (seed 2026) computes the AUROC difference for each seed, then the mean over the 3 seeds. The 95% CI is the 2.5th-97.5th percentile of that mean. Per-seed results are also reported.
- **Reference baselines** on the same fixed answers, from the Phase 1 records: raw confidence (answer span), calibrated raw confidence, P(True), semantic entropy, and a logistic probe on frozen 50%-depth features (the Phase 1 method).

## 8. Decision Rules

Let `d(X, Y)` be the seed-averaged AUROC difference X minus Y on a transfer test set, with its 95% CI.

**H3-T passes** if `d(C, A)` has a CI lower bound above zero for **both** transfer directions, and C meets the generation-quality bar.

**H3-J passes** if `d(C, B)` has a CI lower bound above zero for **both** transfer directions, and C meets the generation-quality bar.

**Reading a tie between C and B** (CI of `d(C, B)` includes zero in at least one direction):
- If B fails the generation-quality bar and C meets it: the joint-training claim narrows to **"generation-preserving trust training"**.
- If both meet the bar: the joint objective adds nothing measurable beyond trust-only fine-tuning at this scale.
- In neither case do I describe the joint objective as producing better trust estimates.

**Interpreting D.** If `d(D, A)` is as large as `d(C, A)` (CI of `d(C, D)` includes zero in both directions), the transfer gain is attributed to LM fine-tuning of the backbone rather than to training for trust, and I say so.

What follows from each outcome is fixed in the program plan, Section 8.

## 9. Label Audit

From labels only, before any metric:
- a fresh random sample of 100 base-model test answers per dataset (seed 2028), excluding the rows audited in Phase 1;
- a random sample of 50 regenerated test answers per dataset from arm C, seed 0.

I report the v3 error rate on these samples. This is the rule's error rate on data it was not developed on.

## 10. Code, Compute, and Outputs

- Training and evaluation: `tcl_experiments/scripts/train_trust_head.py` (to be written in step 4).
- Analysis: `tcl_experiments/scripts/run_phase2_analysis.py` (to be written and committed before any test result exists).
- Notebook: `notebooks/tcl_phase2_kaggle_runs.ipynb`, pinning the code commit.
- Outputs: `tcl_experiments/runs/phase2/` (per-run test scores, regenerated answers, generation metrics, configs) and `tcl_experiments/runs/phase2_analysis/`.
- Compute estimate: about 15-20 Kaggle T4 hours for the 28 trainings, their evaluations, and regeneration.

Before the Kaggle runs, I check the full pipeline end to end on a CPU with a tiny model and a few questions. That check produces no Phase 2 results.

## 11. Amendments

Each amendment was made before any arm A-D was trained on the Phase 2 data.

**2026-10-18: WikiText-2 sample excludes headings.** The perplexity sample is the first 200 non-empty paragraphs of the WikiText-2 test split *that are not section headings* (lines such as `= Robert Boulter =`). Headings are one short line each and would add little but noise. The sample is committed as `tcl_experiments/data/phase2/wikitext2_test_200.json` (27,533 words).

**2026-10-18: Reference baselines seen during a code test (disclosure).** While testing `run_phase2_analysis.py` on synthetic arm outputs, its reference-baseline block ran on the real Phase 1 records and printed transfer AUROCs on the fixed base-model answers with rule v3 labels: raw confidence 0.762 (SQuAD test) and 0.712 (TriviaQA test), P(True) 0.622 and 0.643, semantic entropy 0.775 and 0.773, and a frozen logistic probe on 50%-depth features of 0.575 (trained on TriviaQA, tested on SQuAD) and 0.682 (trained on SQuAD, tested on TriviaQA). These are ungated reference rows. No arm A-D result existed at the time, and no protocol choice was changed after seeing them; the decision rules in Section 8 were already committed.

**2026-10-19: Optimizer step accounting (after the Kaggle code test).** A 20-question Kaggle test of the full pipeline ran all 14 TriviaQA trainings. With so few questions, some LoRA runs made no update at all: there were about 3 optimizer steps per run, and fp16 loss scaling skips steps whose gradients overflow, which is normal at the start of training. To make this visible in the real runs, the training code now records the number of optimizer steps taken and skipped in each epoch, and the last, partial accumulation of an epoch now also gets an optimizer step instead of being dropped. The objectives, hyperparameters, and decision rules are unchanged. The 20-question test produced no Phase 2 results (20 test items per set), and I did not look at its AUROCs to make any choice.
