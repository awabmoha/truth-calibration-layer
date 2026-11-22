# TCL Phase 3 H4 Protocol: Does Predicted Self-Consistency Add to Raw Confidence on New Data?

Status: **locked** on November 22, 2026, before any H4 code was written and before any H4 metric was computed. The only H4 numbers seen before locking are the consistency label counts per test split (Section 2), computed from labels alone to apply the size rule. Changes after this point are logged as amendments (Section 10) with the reason; the decision rule in Section 5 cannot change once any H4 test metric has been computed.

This protocol tests **H4** from my program plan (`TCL-Research-Program.md`, Sections 4 and 7) for the first trust dimension in the plan's order, **self-consistency**. Grounding and evidence density get their own protocols later.

## 1. Question

**H4 (self-consistency):** can a head predict, from **one forward pass**, how consistently the model would answer if asked again, and does that prediction add information about correctness **beyond calibrated raw confidence**, on a dataset held out from training?

Why this comes now:
- H4-T failed (`TCL-Phase3-Checkpoint.md`): no learned **correctness** head beat calibrated raw confidence out of distribution. As fixed in the program plan (Section 8), raw confidence is now TCL's correctness dimension, and every further dimension is judged on whether it **adds** to it.
- Sampling-based consistency is the strongest signal in my data: semantic entropy (10 samples) beat raw confidence on most Phase 3 pairs. But it costs 10 extra generations per answer. TCL's value would be getting most of that signal from a single pass.

**Prior work.** Predicting semantic entropy from the hidden states of a single generation has been done before ("semantic entropy probes"; Kossen et al., 2024), and those probes are reported to generalize out of distribution better than probes that predict accuracy directly. That matches my own Phases 1-3, where accuracy-predicting heads did not transfer, and it is the main reason to expect this dimension to do better. What this protocol adds is a stricter test: whether the one-pass prediction **adds to calibrated raw confidence** (not just to other probes) on held-out datasets, including PopQA, with two models, under a decision rule fixed in advance.

## 2. Data and Label

**Models and datasets:** exactly the Phase 3 setup. Qwen2.5-0.5B-Instruct and Phi-3.5-mini-instruct; TriviaQA, SQuAD, NQ-Open (Phase 1 records and hidden states); PopQA (the Phase 3 sample: 3,000 questions for Qwen, 1,500 for Phi). **No new inference is needed.**

**Consistency label** (theory paper v2, Section 5.2; computed by `run_inference_v2.py` and already stored as `self_consistency`): the model's greedy answer is compared with 10 answers sampled at temperature 1.0, all clustered by bidirectional NLI entailment (DeBERTa-large-MNLI). The label `y_c` is the fraction of the 10 samples in the greedy answer's meaning cluster, in [0, 1].

**Where the label exists:** on every **test** split (1,500 questions per pool dataset, and all of PopQA). Phase 1 sampled only test splits, so training and validation splits have no consistency label. Section 9 records how the head is trained anyway.

**Binary form, for AUROC:** consistent if `y_c >= 0.5` (at least half the samples agree with the answer).

**Size rule and eligible pairs.** A (model, held-out dataset) pair counts only if its test split has at least 150 answers in each **correctness** class (the Phase 3 rule) and at least 150 in each **consistency** class. From labels only (counts, no metric):

| Model | Held-out dataset | Correct / incorrect | Consistent / inconsistent | Eligible |
|---|---|---:|---:|---|
| Qwen | TriviaQA | 216 / 1,284 | 229 / 1,271 | Yes |
| Qwen | SQuAD | 1,040 / 460 | 1,103 / 397 | Yes |
| Qwen | NQ-Open | 108 / 1,392 | 107 / 1,393 | No |
| Qwen | PopQA | 285 / 2,715 | 425 / 2,575 | Yes |
| Phi | TriviaQA | 779 / 721 | 985 / 515 | Yes |
| Phi | SQuAD | 1,405 / 95 | 1,405 / 95 | No |
| Phi | NQ-Open | 322 / 1,178 | 579 / 921 | Yes |
| Phi | PopQA | 278 / 1,222 | 423 / 1,077 | Yes |

So H4 has the same **6 eligible pairs** as H4-T: 4 leave-one-dataset-out pairs and 2 PopQA pairs.

## 3. Heads and Arms

**Consistency head (C-hat):** reads the 50%-depth answer-span hidden state (the Phase 3 feature) and outputs a predicted `y_c`, trained with binary cross-entropy on the soft label. Its design is **not tuned**: it uses the setting Phase 3a selected (MLP: LayerNorm, 256 hidden units, GELU, dropout 0.1; AdamW, learning rate 1e-3, weight decay 1, batch 256, up to 60 epochs, equal weight per training dataset), with early stopping on a held-back 10% of its training data. Seeds 0-4.

**Training data (per pair):** the consistency labels of the **other** pool datasets (for PopQA pairs: all three pool datasets), never the held-out dataset. Where those labels come from is decision 1.

**Arms**, each scoring correctness on the held-out test split:

| Arm | Inputs | Fitted on |
|---|---|---|
| **R** | Raw confidence, Platt-mapped (as in Phase 3) | Training mixture's validation splits |
| **R+C-hat** | Logistic regression on logit(raw confidence) and logit(C-hat) | Training mixture's validation splits (correctness labels, rule v3) |
| *R+C (oracle)* | *The same combination with the sampled label `y_c` instead of C-hat (costs 10 samples)* | *Reference, ungated* |
| *SE* | *Semantic entropy, 10 samples* | *Reference, ungated* |

The combiner is fitted only on validation splits of the training datasets, where C-hat is available from hidden states alone. The held-out dataset is never used to fit anything.

## 4. Metrics

Per eligible pair, on the held-out test split, with C-hat averaged over 5 seeds:

1. **Learnability:** AUROC of C-hat against the binary consistency label.
2. **Non-redundancy:** Spearman correlation between C-hat and calibrated raw confidence.
3. **Added value:** `d_AURC = AURC(R) - AURC(R+C-hat)` for correctness (positive = the combination is better), with a 95% paired bootstrap CI (1,000 resamples of test items, seed 2026).

**Also reported, not gated:** correctness AUROC of R+C-hat and its difference from R; the oracle R+C and SE (how much of the sampling signal one pass recovers); ECE and Brier after the combiner; accuracy at 50% and 80% coverage; per-seed values; every result with refusals excluded.

## 5. Decision Rule

A pair **passes** if all three hold: C-hat AUROC >= 0.70, Spearman correlation with raw confidence < 0.7, and the `d_AURC` CI lower bound above zero. These are the thresholds in my program plan (Section 7, H4), unchanged.

- **H4 (self-consistency) passes** if a **majority of eligible pairs** pass (with 6: at least 4), **including at least one PopQA pair**. This mirrors the H4-T rule.

**What follows:**
- **Passes:** self-consistency becomes TCL's second dimension, predicted from one pass. This is a positive result, and under my publication plan the first point to write up Phases 1-3 together.
- **Fails:** self-consistency, predicted from one pass, does not add to raw confidence out of distribution at this scale. I report it, test the next dimension (grounding) under its own protocol, and carry the question to Phase 4's larger model.

## 6. Label Audit

The consistency label comes from an NLI model, not from a rule I wrote. Before any H4 metric, I review a random sample of 50 answers per model (seed 2032, from the held-out test splits) and judge whether the clustering is right: does each sample in the greedy answer's cluster mean the same thing, and does each sample outside it mean something different? I report the error rate. The correctness labels are the ones already audited in Phases 1-3.

## 7. Order of Work

1. Lock this protocol (done, November 22, 2026). Commit.
2. Write `run_phase3_h4.py` (heads, combiner, scores) and its analysis; test them on synthetic data only. Commit.
3. Run the consistency-label audit. Commit.
4. Train, score, and run the analysis **once**. Write the checkpoint.

## 8. Code and Outputs (planned)

- `tcl_experiments/scripts/run_phase3_h4.py`: trains the consistency heads, fits the combiner, writes test scores.
- `tcl_experiments/scripts/run_phase3_h4_analysis.py`: applies Section 5 once.
- Outputs: `tcl_experiments/runs/phase3_h4/`, `tcl_experiments/runs/phase3_h4_analysis/`.
- Compute: laptop CPU only.

## 9. Decisions Settled at Locking

1. **Training labels for the consistency head: the other pool datasets' test splits.** No new compute. Within a pair, the head is trained on different datasets from the one it is evaluated on, so nothing leaks into the evaluation. The cost, which I state in every write-up: those test splits serve as training data for other pairs. The alternative, new sampling on the training and validation splits (about 15-25 Kaggle GPU hours), was not chosen.
2. **Binary threshold for learnability: `y_c >= 0.5`**, fixed before any metric; not the median of the training labels.
3. **Scope: self-consistency only.** Grounding needs new SQuAD 2.0 inference, and evidence density has a label only on PopQA, so it has no held-out dataset to transfer to. Each gets its own protocol.

## 10. Amendments

**2026-11-22: Implementation details fixed in code, before any run on real data.** The protocol does not specify these; they are fixed in `run_phase3_h4.py` and `run_phase3_h4_analysis.py` (docstrings) and tested on synthetic data only (one dataset where consistency carries signal: every pair passes; one where it is noise: every pair fails):
- The consistency head's early stopping holds back a random 10% of its training rows (seeded per seed).
- The sampled label is smoothed before its logit is taken: `(10 * y_c + 0.5) / 11`.
- The oracle R+C cannot be fitted on validation splits, which have no sampled answers, so its combiner is fitted on the training datasets' test splits. It stays ungated.
- "Averaged over 5 seeds" means each metric is computed per seed and then averaged, as in Phase 3 (and inside every bootstrap resample).
- The audit (Section 6) compares my hand count of samples that mean the same as the greedy answer with the NLI count (`10 * y_c`), drawn by `draw_phase3_h4_audit.py`.

No decision rule, dataset, model, arm, threshold, or metric changes.

## References

- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630, 625-630.
- Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., & Gal, Y. (2024). Semantic entropy probes: Robust and cheap hallucination detection in LLMs. arXiv:2406.15927.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In *ICLR*.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In *ACL*.
