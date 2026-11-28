# TCL Research Program

Status: my master plan. It replaces `docs/archive/tcl-v0/TCL-v0-Roadmap.md` as the document that governs this project. Every phase-level document I write (ablation plans, runbooks) must trace back to a phase and hypothesis here.

Date: October 16, 2026. Last revised: November 24, 2026 (after Phase 3's H4 tests; see the changelog).

## 1. North Star

I believe language models should emit, alongside every claim they generate, a structured and calibrated estimate of how much that claim can be trusted, and that this ability should be **learned inside the model**, not bolted on afterwards.

The Truth Calibration Layer (TCL) is my research program to prove this is possible, useful, and better than post-hoc alternatives.

I am keeping the name. I make claims phase by phase, and at each phase I claim only what that phase's gate has passed.

## 2. What Makes TCL Different

Prior work already shows that frozen hidden states carry correctness signal (Kadavath et al. 2022; Azaria & Mitchell 2023; Burns et al. 2023; Marks & Tegmark 2024; Kossen et al. 2024; Orgad et al. 2025). TCL-v0 replicates that. For me it is the foundation, not the contribution.

Fine-tuning a model to predict its own correctness is also not new: Kadavath et al. (2022) trained P(IK), and Kapoor et al. (2024) report that LoRA fine-tuning on graded examples beats frozen probes. My claims build on that result instead of claiming it.

My distinguishing claims are:

1. **The joint objective specifically helps.** Training the backbone with `L_LM + lambda * L_trust` produces better trust estimates than a frozen head, than trust-only fine-tuning, and than LM-only fine-tuning on the same data, without degrading generation. The controlled comparison that isolates the joint objective is my contribution.
2. **Trust is multi-dimensional.** Separate, operationally defined trust dimensions carry non-redundant information and support better downstream decisions than a single confidence score.
3. **It transfers.** A learned trust signal holds up across datasets, task families, and model scales, where frozen probes are known to degrade.

Every phase below exists to test one of these. Phase 1 showed that frozen probes fail claim 3 (they do not transfer across datasets), so Phase 2 tests claims 1 and 3 together, with transfer as its primary result.

**Status after Phase 2 (November 20, 2026):** claim 1 is not supported. Joint training did not transfer better than a frozen head, in Phase 2 or in its one preregistered rerun (2b), and I have dropped it as a claim. Claims 2 and 3 stand as open questions. Claim 3 is now the central obstacle: every head trained on one dataset's labels transferred below untrained signals (raw token confidence, semantic entropy). From Phase 3 on, every trust result must hold on a held-out dataset.

**Status after Phase 3's H4-T test (November 22, 2026):** claim 3 is not supported for correctness at this scale. A head trained on a mixture of datasets, given raw confidence, beat calibrated raw confidence on 1 of 6 held-out pairs. As fixed in Section 8, TCL's correctness dimension is now calibrated raw confidence (with semantic entropy where affordable), and the H4 dimensions are judged on whether they add to it.

**Status after Phase 3's H4 tests (November 24, 2026):** claim 2 is not supported at this scale. Self-consistency and grounding are both learnable from one pass on held-out datasets and not redundant with raw confidence, but neither adds to it on a majority of pairs (3 of 6 and 2 of 7). As fixed in Section 8, trust is effectively one-dimensional at this scale, and Phase 4 re-tests at 7-8B before I drop multi-dimensionality.

**Status after Phase 4a (November 28, 2026):** claims 2 and 3 are not supported at 7B either. On Qwen2.5-7B no learned correctness head beats calibrated raw confidence (0 of 3 pairs), self-consistency adds to it on 2 of 3 pairs but misses on the PopQA pair by a CI bound of -0.0003, and grounding is not testable (the model abstains on 1,996 of 2,000 irrelevant passages and rarely answers wrongly otherwise). As fixed in the Phase 4a protocol, trust is effectively one-dimensional from 0.5B to 7B on QA, and I drop the multi-dimensional claim for this program's scale. The plan is revised at the Phase 4b gate before any 4b work.

## 3. Theory Issues I Must Resolve First

Status: resolved in `paper/TCL-Theory-Paper-v2.md` (October 2026). Phase 0 gate met: Section 8 of the paper gives every hypothesis a metric, a baseline list, and a pass/fail threshold (recorded November 20, 2026).

My version 1 theory paper had three problems I needed to fix before running more experiments, because they decide what the experiments are.

| Issue | Where | What I will do |
|---|---|---|
| Frozen vs joint contradiction | Sec. 4.1 says the base transformer is frozen; Sec. 4.4 defines `L_total = L_LM + lambda * L_TCL`. With a frozen base, `L_LM` has no gradient and TCL reduces to a probe, which is what TCL-v0 already is. | Commit to the joint version as my central hypothesis. Keep the frozen head as the baseline and deployable variant. |
| Trust vector is not observable | Evidence density, source diversity, and contradiction risk have no label definitions. The paper itself notes they cannot be measured from hidden states. | Define every dimension by a concrete, reproducible labeling procedure (Section 5). A dimension without a label procedure is out of scope. |
| Inconsistent dimension lists | Paper: confidence, evidence density, contradiction risk, source diversity. Roadmap/gap doc: correctness, reasoning validity, grounding, epistemic uncertainty. | Fix one canonical list in theory paper v2 (Section 5). |

The paper is also missing the hidden-state probing and uncertainty-estimation literature (the Section 2 references, plus semantic entropy, P(True), verbalized confidence, and linguistic calibration). In theory paper v2 I will position TCL against them.

## 4. Hypotheses

Each hypothesis is falsifiable and has a test I declare in advance.

| ID | Hypothesis | Tested in |
|---|---|---|
| H1 | Frozen hidden states improve correctness **ranking** (AUROC, AURC) beyond calibrated raw confidence. | Phase 1 |
| H2 | Frozen-probe signal transfers across QA datasets with bounded degradation. | Phase 1 |
| H3-T (primary) | A trust head trained jointly with the backbone (`L_LM + lambda * L_trust`) **transfers across datasets** better than a frozen head trained on identical data. | Phase 2 |
| H3-J (secondary) | The joint objective beats trust-only fine-tuning of the backbone, at equal or better generation quality. | Phase 2 |
| H4-T | A correctness head trained on a mixture of datasets, with calibrated raw confidence among its inputs, beats calibrated raw confidence on a dataset held out from training. | Phase 3 |
| H4 | At least one additional trust dimension is learnable and carries information not captured by correctness confidence, **on a held-out dataset**. | Phase 3 |
| H4-T, H4 at 7B | The Phase 3 tests (H4-T, H4 self-consistency, H4 grounding) re-run unchanged on a 7B model: does a learned head beat, or a second dimension add to, calibrated raw confidence at that scale? | Phase 4a |
| H5 | The trust results of Phases 3-4a hold on a non-QA task family (GSM8K). The joint-versus-frozen comparison is re-run at the larger scale. | Phase 4b |
| H6 | Trust estimates can be produced at claim level inside long-form generation. | Phase 5 |

H3 is the core of TCL. Phase 1 showed that frozen probes do not transfer across datasets, so transfer (H3-T) is where a trained trust head has to prove itself, and it is the primary Phase 2 test. Whether the joint objective specifically matters (H3-J) is the secondary test. If either fails, I follow Section 8.

## 5. Canonical Trust Vector (Adopted In Theory v2)

I only admit dimensions that have a label procedure.

| Dimension | Meaning | Label procedure | Dataset |
|---|---|---|---|
| Correctness | Probability the claim is correct | Answer-match rule v3 (`docs/phase2/TCL-Labeling-Rule-v3.md`), with an audit of a random test sample | TriviaQA, NQ-Open, SQuAD |
| Consistency (contradiction risk) | Probability the model would contradict this claim under resampling | Sample N=10 answers, cluster by bidirectional NLI entailment, label = fraction of samples outside the claim's cluster | Same QA sets |
| Grounding | Whether the claim is supported by provided context | Entailment of answer by context (NLI); distractor passages as negatives (changed from SQuAD 2.0's unanswerable questions on 2026-11-22, so grounding can be tested on held-out datasets) | MRQA: SQuAD, Natural Questions, HotpotQA; RelationExtraction held out |
| Evidence density (proxy) | How well-represented the subject is in training-like data | Entity popularity (Wikipedia page views) as a proxy label | PopQA |

I am dropping source diversity until I have a retrieval or provenance setup; without one it is not observable.

## 6. My Locked Evaluation Protocol

These rules apply to every phase. I change them only through the change-control process in Section 9.

**Primary metrics**
- AUROC (ranking of correct vs incorrect)
- AURC and accuracy at 50% / 80% coverage (selective prediction)

**Secondary metrics**
- ECE (15 equal-mass bins) and Brier score, computed **after** fitting the same post-hoc calibrator on validation for every method
- Generation quality: answer accuracy (EM/F1) and held-out perplexity

**Metrics I no longer use as headlines**
- Wrong answers at confidence >= 0.8 / 0.9 (trivially reduced by lowering all scores)
- `min(raw, probe)` as a headline method (I keep it as one baseline)

**Statistics**
- At least 3 seeds for anything I train
- 1,000-sample bootstrap 95% CIs on every metric I report
- I count a "win" only when the CI of the difference excludes zero
- Calibration metrics use unweighted fits; I use class weighting only for ranking-only probes, and say so when I do

**Baselines I include in every comparison**
1. Raw sequence likelihood
2. Raw likelihood + best of temperature / Platt / isotonic calibration
3. P(True) self-evaluation
4. Semantic entropy (N=10 samples)
5. Frozen hidden-state probe (TCL-v0)

**Data hygiene**
- I fix splits by ID files and commit them before a phase starts
- I evaluate test splits once per phase, at the gate
- I do all model and hyperparameter selection on validation only

**Test-set size (added at the Phase 1 local gate)**
- Every test set has at least 1,000 rows and at least 150 rows of each class (correct and incorrect)
- A model-dataset pairing that cannot reach 150 rows of the smaller class at a reasonable size (about 5,000 questions) is dropped from that phase and recorded as dropped
- Sampling-based baselines (semantic entropy) run on the test split only, to keep GPU cost bounded

## 7. Phases

### Phase 0: Theory Consolidation

- **Goal:** a theory paper whose experiments are unambiguous.
- **Work:** resolve the Section 3 issues; adopt the Section 5 trust vector; write the related work; add the Section 4 hypotheses.
- **Compute:** none.
- **Deliverable:** `TCL-Theory-Paper` v2.
- **Gate:** every hypothesis has a metric, a baseline list, and a pass/fail threshold.
- **Status:** done (October 2026).

### Phase 1: Close TCL-v0

- **Goal:** settle H1 and H2 under my locked protocol, then stop iterating on v0.
- **Work (local CPU):** re-analyze my existing records with unweighted calibrated comparisons, bootstrap CIs, and AURC; run cross-dataset transfer (train SQuAD -> test TriviaQA and the reverse; add NQ-Open).
- **Work (Kaggle GPU):** P(True) and semantic-entropy baselines on the six reviewed runs; one layer sweep (early-middle / middle / final) on one model only.
- **H1 passes if:** probe AUROC beats the best calibrated raw baseline with the CI excluding zero in >= 4 of 6 runs.
- **H2 passes if:** the transferred probe keeps >= 70% of its in-distribution AUROC gain over raw.
- **Deliverable:** a short arXiv note / workshop paper: "Do hidden-state probes add value beyond calibrated likelihood and sampling baselines in small open models?"
- **Status:** closed (October 2026). H1-R passed at its threshold, H2-R failed (`docs/phase1/TCL-Phase1-Kaggle-Checkpoint.md`). The workshop note has not been written; the results are reported in the checkpoint and the README.
- **If H1 fails:** I still proceed to Phase 2 (H3 compares joint vs frozen, not vs raw), but I record that frozen probes add little and that my case for TCL rests entirely on joint training.

### Phase 2: TCL-v1 Joint Training (Core Test)

- **Goal:** test H3-T (transfer, primary) and H3-J (joint objective, secondary) on the correctness dimension.
- **Model:** Qwen2.5-0.5B-Instruct (LoRA, Kaggle T4). The trust head reads answer-mean hidden states at 50% depth, the best layer in the Phase 1 sweep.
- **Data:** the Phase 1 TriviaQA and SQuAD question sets with their committed splits (2,500 train / 500 validation / 1,500 test each; SQuAD split by Wikipedia article), relabeled with rule v3. Every arm is trained twice: once on TriviaQA, once on SQuAD.
  - NQ-Open is not used for training: Qwen answers only 7% of it correctly, too few correct answers for stable training labels or for the size rule.
- **Arms (identical data, identical trust-head architecture):**
  - A. Frozen backbone + MLP trust head (my strong frozen baseline)
  - B. LoRA backbone trained on `L_trust` only (controls for "any fine-tuning helps")
  - C. LoRA backbone trained on `L_LM + lambda * L_trust`, lambda in {0.1, 0.5, 1.0}, chosen on validation (TCL-J, the proposal)
  - D. LoRA backbone trained on `L_LM` only, then a frozen head (controls for LM fine-tuning alone)
- **Evaluation:** each trained model is tested on the *other* dataset's test split (transfer: TriviaQA -> SQuAD and SQuAD -> TriviaQA) and on its own (in-distribution). The base model's answers are labeled once; each trained model also regenerates its own answers, labeled with the same rule, because its answers change after training.
- **Refusals:** reported as a separate outcome. Every trust result is also reported with refusals excluded, so a head cannot score well only by recognizing refusals.
- **H3-T passes if:** C's transfer AUROC exceeds A's with the 95% CI of the difference excluding zero in **both** transfer directions, and C meets the generation-quality bar.
- **H3-J passes if:** C's transfer AUROC exceeds B's with the CI excluding zero in **both** directions, and C meets the generation-quality bar.
- **Generation-quality bar:** answer accuracy drops by at most 1 point and held-out perplexity rises by at most 3%, both relative to the base model.
- **In-distribution results** are reported for every arm but are not gated: Phase 1 showed in-distribution differences are small.
- **Deliverable:** my main TCL paper draft (method + core result).
- **Result (November 20, 2026):** H3-T and H3-J fail in Phase 2 and again in its one preregistered rerun, Phase 2b (end-of-turn token added to the LM target). The joint-transfer claim is dropped; the deliverable becomes a negative-result report. See `docs/phase2/TCL-Phase2-Checkpoint.md` and `docs/phase2/TCL-Phase2b-Checkpoint.md`.

**What a tie between C and B means (fixed before any Phase 2 result):**
- If C ≈ B on trust (CI of the difference includes zero) **and** B fails the generation-quality bar while C meets it, TCL's joint-training claim narrows to **"generation-preserving trust training"**: adapting the backbone for trust helps, and the LM term is what keeps generation intact.
- If C ≈ B on trust **and both** meet the generation-quality bar, the joint objective adds nothing measurable beyond trust-only fine-tuning at this scale. I report that, and TCL's contribution in Phase 2 rests on H3-T alone.
- In neither case do I describe the joint objective as producing better trust estimates.

### Phase 3: Transfer and a Second Trust Dimension

- **Goal:** test H4-T, then H4. Revised November 20, 2026, after Phase 2 (see changelog).
- **H4-T status (November 22, 2026): fails** (`docs/phase3/TCL-Phase3-Checkpoint.md`). H4 follows with calibrated raw confidence as the correctness baseline; the self-consistency protocol is locked (`docs/phase3/TCL-Phase3-H4-Consistency-Protocol.md`).
- **H4 status for self-consistency (November 22, 2026): fails, by one pair** (`docs/phase3/TCL-Phase3-H4-Consistency-Checkpoint.md`). The dimension is learnable from one pass and non-redundant on all 6 pairs, but lowers AURC on only 3 (4 needed). Grounding is the next dimension; its protocol is locked (`docs/phase3/TCL-Phase3-H4-Grounding-Protocol.md`).
- **H4 status for grounding (November 24, 2026): fails** (`docs/phase3/TCL-Phase3-H4-Grounding-Checkpoint.md`). 2 of 7 pairs pass (Qwen / SQuAD, Qwen / NQ), none on RelationExtraction; 4 were needed. The head mostly detects an irrelevant passage. With self-consistency also failed, and evidence density untestable for transfer, the Section 8 row "H4 fails for all dimensions" applies: Phase 4 re-tests at larger scale.
- **Transfer requirement (all of Phase 3):** heads are trained on a mixture of datasets and evaluated on a dataset left out of training (leave-one-dataset-out), and on at least one dataset never used in any earlier phase. In-distribution results are reported but not gated.
- **Method:** frozen-backbone heads (TCL-F), following the Section 8 row for a failed H3-T. Each head receives calibrated raw confidence as an input, so it learns a correction to it rather than starting from nothing.
- **H4-T passes if:** the head's AUROC exceeds calibrated raw confidence's on the held-out dataset with the 95% CI of the difference excluding zero, for a majority of held-out datasets, with at least one being a dataset no earlier phase used. Semantic entropy is reported as a reference.
- **H4-T comes first,** because the extra dimensions are read by the same kind of head; if correctness alone cannot beat raw confidence out of distribution, H4 is judged against that.
- **Order:** consistency first (its labels need no new data), then grounding (SQuAD 2.0), then evidence density (PopQA).
- **H4 passes if:** on the held-out dataset, the dimension is predicted with AUROC >= 0.70, correlates < 0.7 with correctness confidence, and combining it with correctness improves selective-prediction AURC with the CI excluding zero.
- **Deliverable:** multi-dimensional trust-head results.

### Phase 4: Generalization And Scale

Revised November 24, 2026, after Phase 3 (see changelog): Phase 3 left no learned method that beats raw confidence, so there is no "best method" to carry forward. Phase 4 is split in two.

**Phase 4a: the Phase 3 tests at 7B.**
- **Goal:** carry out the Section 8 row "H4 fails for all dimensions": re-test before dropping multi-dimensionality.
- **Work:** Qwen2.5-7B-Instruct (the family of Phase 3's 0.5B model), float16 on Kaggle's two T4s, inference only. The Phase 3 question sets, head design, arms, thresholds, and statistics, unchanged. One fix chosen before any 7B data: the grounding label, selected on the Phase 3 audit set.
- **Compute:** about 20-25 free Kaggle GPU hours.
- **Passes if:** H4-T or H4 (either dimension) passes by the Phase 3 majority rules on the 7B pairs. Protocol: `docs/phase4/TCL-Phase4a-Protocol.md`.
- **Status (November 28, 2026): fails** (`docs/phase4/TCL-Phase4a-Checkpoint.md`). H4-T: 0 of 3 pairs. Self-consistency: 2 of 3 (TriviaQA, NQ-Open), not the PopQA pair. Grounding: not testable (RelationExtraction fails the size rule). Phase 4b's scope is decided at this gate. Ungated: relabeled with the fixed grounding label, Phase 3's grounding test would pass (5 of 7 pairs); its recorded decision stands.

**Phase 4b: task family and joint training at scale.**
- **Goal:** test H5.
- **Work:** one non-QA task family with automatic labels (GSM8K math), and the Phase 2 four-arm comparison (frozen, trust-only, joint, LM-only) re-run on the 7B model via QLoRA, as a new test with its own protocol.
- **Compute:** likely paid GPU for the 7B training (estimate in Section 10).
- **Scope is decided after 4a:** if 4a passes, 4b tests whether the result holds on GSM8K; if it fails, I revise this plan at that gate before any 4b work.
- **Deliverable:** the full TCL paper (conference submission target), under the publication track in Section 11.

### Phase 5: Claim-Level Trust (Long Horizon)

- **Goal:** test H6, the claim-extraction vision from my original paper.
- **Work:** long-form biographies with FActScore-style atomic-claim labels; per-claim trust from token-span hidden states.
- **Deliverable:** TCL as a claim-level reliability layer for long-form generation.

## 8. Decision Gates And Pivots

| Outcome | My decision |
|---|---|
| H3-T and H3-J pass | My core claim stands: joint training transfers better than a frozen head and beats trust-only fine-tuning. I continue to Phase 3 with TCL-J. |
| H3-T passes, H3-J ties | Training the backbone for trust is what transfers; the joint objective's role is decided by the tie rule in Phase 2 ("generation-preserving trust training", or no added value). I continue to Phase 3 with the best arm that meets the generation-quality bar. |
| H3-T fails, A ≈ C | A trained head transfers no better than a frozen one. I continue TCL as a **frozen multi-dimensional trust head** (the TCL-F design in my paper); its value must then come from H4 and H5, and I report the transfer failure as a main result. |
| H3-T fails, C < A | Joint training hurts transfer. I investigate lambda and label staleness once; if it still fails, I drop the joint claim and publish the negative result. |
| H4-T fails | No learned correctness head beats calibrated raw confidence out of distribution at this scale. TCL's correctness dimension becomes calibrated raw confidence (and semantic entropy where sampling is affordable); H4 dimensions are then judged on whether they add to it. I report this as a main result. |
| H4 fails for all dimensions | Trust is effectively one-dimensional at this scale. I re-test at Phase 4 scale before dropping multi-dimensionality. |
| H5 fails | I publish scope-bounded results and make generalization the next goal of the program. |

I publish negative results instead of hiding them. They are part of what makes this program credible.

## 9. Change Control

- I change this plan only at a phase gate, or when a result invalidates one of my assumptions.
- I record every change in the changelog below, with the date, what changed, and the evidence that forced it.
- I start no new experiment unless it maps to a hypothesis in Section 4.
- I cannot change the thresholds in Section 7 after a phase's test split has been evaluated.

## 10. Compute And Schedule

My hardware: my laptop is CPU-only (Ryzen 7 5825U, about 6 GB RAM, integrated AMD graphics, no CUDA). I use it to analyze saved records. Anything that runs a model goes to Kaggle (free T4/P100, about 30 GPU hours a week).

| Phase | Compute | Estimated duration (part-time) |
|---|---|---|
| 0 | Local | 2-3 weeks |
| 1 | Local + about 15 Kaggle hours | 3-4 weeks |
| 2 | About 40-60 Kaggle hours (used: about 24 for the runs that count, plus failed sessions) | 5 weeks (done) |
| 3 | About 30-50 Kaggle hours | 6-8 weeks |
| 4a | About 20-25 Kaggle hours | 3-4 weeks |
| 4b | Kaggle + about $100-300 paid GPU | 6-8 weeks |
| 5 | Paid GPU | open |

Phases 0-2 run on free compute. The Phase 2 result decides whether paid compute is worth it.

## 11. Publication Track

Revised November 22, 2026. I no longer write a separate paper for each phase.

1. **Already public:** the Phase 2 and 2b negative result as a preprint on Zenodo (DOI 10.5281/zenodo.24403374), and every phase's protocol, checkpoint, code, and records in this repository.
2. **One cumulative paper at the first positive result.** If H4 passes, I write up Phases 1-3 together; if not, the next phase that succeeds (4, then 5) is written up with every earlier phase. Earlier negative results are always included, as the evidence that motivates the positive one.
3. **If no phase succeeds,** I publish the whole program as one negative-result paper.

## Changelog

| Date | Change | Reason |
|---|---|---|
| 2026-11-28 | **Phase 4a run and closed** (`docs/phase4/TCL-Phase4a-Checkpoint.md`): H4-T and H4 fail at 7B (self-consistency 2 of 3 pairs without PopQA; grounding not testable). One amendment before any metric: the grounding audit takes the 4 answered distractor rows and 46 gold rows. | Protocol Section 8, row "Both fail (or are not testable)": the multi-dimensional claim is dropped for this program's scale, and Phase 4b is scoped at its gate. |
| 2026-11-24 | **Plan revised after Phase 3** (phase gate) and **Phase 4a protocol locked** (`docs/phase4/TCL-Phase4a-Protocol.md`) before any Phase 4a code or 7B output. Phase 4 split: **4a** re-runs Phase 3's H4-T, H4 self-consistency, and H4 grounding tests unchanged on Qwen2.5-7B-Instruct (float16, two T4s, inference only, about 20-25 free Kaggle hours); **4b** (GSM8K, joint training at 7B) is scoped after 4a. H5 reworded: there is no "best method from Phases 2-3" to carry forward. One fix chosen before any 7B data: the grounding label, selected among three NLI candidates on the Phase 3 audit set. A test passes on a majority of its eligible 7B pairs, including its new-dataset pair; H4 at 7B passes if either dimension passes. | Section 8 row "H4 fails for all dimensions": re-test at larger scale before dropping multi-dimensionality. Phase 3's gains appeared where raw confidence was weak, so a pass at 7B is not expected; the test is run as preregistered either way. |
| 2026-11-24 | **H4 (grounding) analysis run once** (`docs/phase3/TCL-Phase3-H4-Grounding-Checkpoint.md`): **fails**. 2 of 7 eligible pairs pass (Qwen / SQuAD, Qwen / NQ); 4 were needed, including Qwen / RelationExtraction, which fails. Phi / RelationExtraction was dropped by the size rule (125 incorrect). Learnability (6 of 7 at or above 0.70) and non-redundancy (Spearman 0.11-0.61) mostly hold; the AURC gain fails on 5 of 7. By the ungated AUROC the combination beats raw confidence on 5 of 7, but only through distractor rows: among answers from the right passage it adds nothing. Label audit: the NLI grounding label is wrong on 23 of 100 answers (mostly supported short answers rejected), rule v3 on 11 of 100. **H4 has failed for every testable dimension**; per Section 8, trust is effectively one-dimensional at this scale and Phase 4 re-tests at larger scale. | The single preregistered grounding run. Under the publication plan, Phases 1-3 wait for the first successful phase. |
| 2026-11-22 | **H4 protocol for grounding locked** (`docs/phase3/TCL-Phase3-H4-Grounding-Protocol.md`) before any grounding data or metric. **Data source changed** from SQuAD 2.0 to MRQA (SQuAD, Natural Questions, HotpotQA as the pool; RelationExtraction held out), with distractor passages (one third of questions) as negatives instead of SQuAD 2.0's unanswerable questions; the model may answer "unanswerable"; grounding is labeled by NLI entailment. Same head design, arms, thresholds, and majority rule as the self-consistency test (8 eligible pairs at most, including the new dataset for at least one model). About 15 Kaggle GPU hours. | Self-consistency failed by one pair, so the plan's next dimension is tested. SQuAD 2.0 is a single dataset; distractors give the same kind of negative in every dataset, which a held-out test needs. |
| 2026-11-22 | **H4 (self-consistency) analysis run once** (`docs/phase3/TCL-Phase3-H4-Consistency-Checkpoint.md`): **fails by one pair**. 3 of 6 pairs pass (Qwen / TriviaQA, Qwen / PopQA, Phi / NQ-Open); 4 were needed. Learnability (AUROC 0.73-0.95) and non-redundancy (Spearman 0.26-0.62) hold on every pair; the AURC gain over raw confidence fails on 3. By the ungated AUROC, the one-pass combination beats raw confidence on 4 of 6 pairs and matches semantic entropy on Qwen. Consistency-label audit: 92 of 100 binary labels agree with my hand count. As fixed in Section 8, grounding is tested next, and a failed H4 is re-tested at Phase 4 scale. | The preregistered rule, applied once. |
| 2026-11-22 | **H4 protocol for self-consistency locked** (`docs/phase3/TCL-Phase3-H4-Consistency-Protocol.md`) before any H4 code or metric. A one-pass head predicts the sampled self-consistency label; it passes on a pair if its AUROC is at least 0.70, its Spearman correlation with calibrated raw confidence is below 0.7, and combining it with raw confidence lowers correctness AURC (CI above zero); H4 passes on a majority of the 6 eligible pairs, including one PopQA pair. Uses existing Phase 1 and Phase 3 data only; the head's design is fixed to the Phase 3a setting. Only consistency label counts were computed before locking. **Publication plan changed:** no separate papers per phase; Phases 1-3 are written up together at the first positive result (H4, or a later phase), with every earlier result included. | H4-T failed, so raw confidence is the correctness dimension and H4 is the next test in Phase 3; semantic entropy beat raw confidence on most Phase 3 pairs, which makes consistency the most promising next dimension. |
| 2026-11-22 | **Phase 3b analysis run once** (`docs/phase3/TCL-Phase3-Checkpoint.md`): **H4-T fails**. `d(M+R, R)` has its CI above zero on 1 of 6 eligible pairs (Qwen / PopQA, +0.079); on 3 pairs the head is significantly worse. 3a chose an MLP head (weight decay 1, equal weighting) on validation splits only. Qwen's PopQA sample was extended to 3,000 questions by the size rule, decided from labels alone. Label audits: 1 rule error in 100 per model. As fixed in Section 8, TCL's correctness dimension becomes calibrated raw confidence. Phases 2 and 3 will be written up together as one paper. | The preregistered rule, applied once. |
| 2026-11-21 | **Phase 3 protocol locked** (`docs/phase3/TCL-Phase3-Protocol.md`) before any Phase 3 code or metric. Tests H4-T with Qwen2.5-0.5B and Phi-3.5-mini (Gemma dropped: near-chance token confidence, frequent refusals), PopQA as the new held-out dataset, a pilot (3a) that selects the head design on training and validation splits only, and a pass rule of `d(M+R, R)` above zero on a majority of up to 6 eligible pairs, including at least one PopQA pair. Only label counts per split were computed before locking, to apply the size rule. | The four open decisions of the draft, settled; the draft's pilot would have read test splits, so design selection was moved to validation splits to keep all four pool test sets usable in the confirmatory test. |
| 2026-11-20 | **Plan revised after Phase 2** (phase gate). Claim 1 (joint objective) marked not supported. New hypothesis **H4-T**: a correctness head trained on a mixture of datasets, given calibrated raw confidence, beats it on a held-out dataset. H4 is now judged on held-out data. H5 now tests the best method from Phases 2-3, and Phase 4 re-runs the joint-versus-frozen comparison at the larger scale. Phase 3 is renamed "Transfer and a Second Trust Dimension" and uses frozen heads (the Section 8 path for a failed H3-T). New gate row for H4-T failing. Publication track: Phase 2 and 2b become a negative-result paper (draft in `paper/TCL-Phase2-Negative-Result.md`). No Phase 3 threshold was set after seeing Phase 3 data; none exists yet. | Phase 2 and 2b: joint training does not transfer, and every head trained on one dataset's labels transfers below raw token confidence and semantic entropy. Transfer is the obstacle every phase has hit. |
| 2026-11-20 | **Phase 2b analysis run once** (`docs/phase2/TCL-Phase2b-Checkpoint.md`): the end-of-turn fix works (fine-tuned answers back to 1-2 words), but **H3-T fails again** (C - A: +0.034 from TriviaQA to SQuAD, -0.038 from SQuAD to TriviaQA; C fails the generation-quality bar) and **H3-J fails**. LM-only fine-tuning matches or beats C in every comparison. Lambda 1.0 chosen for both datasets. Audit of 100 regenerated answers, reviewed before any metric was read: 5 wrong (all on SQuAD, correct answers marked wrong). As fixed in the Phase 2b protocol, **the claim that joint training improves transfer is dropped**; the negative result (Phase 2 and 2b) will be published, and Phase 3 will be redesigned around transfer before any new data. | The preregistered rules, applied once to the 16 new trainings plus the reused A and B runs. |
| 2026-11-20 | **Phase 2b protocol locked** (`docs/phase2/TCL-Phase2b-Protocol.md`) before any Phase 2b training: the one investigation the "C < A" row allows. Single change: the LM target gains the chat template's end-of-turn token. Arms C and D rerun (lambda reselected; 16 trainings); A and B reused. Same decision rules. Fixed in advance: if H3-T passes, I report it as a second, preregistered attempt after a disclosed fix; if it fails, I drop the joint-transfer claim and publish the negative result. No further rerun. | Phase 2's LM target lacked the end-of-turn token, so C and D learned not to stop; the transfer test of TCL-J as designed was not clean. |
| 2026-11-20 | Theory paper v2 related work extended with four papers, each checked against its source (arXiv, ACL Anthology, ICLR): RAGognizer (Ridder et al., 2026), the closest prior work to TCL-J (joint LoRA + detection head, but no frozen-head control, no seeds, no joint-vs-non-joint transfer comparison); CCPS (Khanmohammadi et al., 2025); LitCab (Liu et al., 2024; LM frozen); and Zhang et al. (2025) on probing reasoning models. References: 39 -> 43. | An outside review named these papers; two of its descriptions were inaccurate (LitCab's LM is frozen; Burns et al. is ICLR 2023), so each was checked before citing. |
| 2026-11-20 | **Phase 2 analysis run once** (`docs/phase2/TCL-Phase2-Checkpoint.md`): **H3-T fails** (C beats A from TriviaQA to SQuAD, +0.022, but is worse from SQuAD to TriviaQA, -0.038; C fails the generation-quality bar) and **H3-J fails**. In-distribution, C is the best arm on both datasets (+0.019 and +0.041 over A), but LM-only fine-tuning (D) gets most of that. Every trained head transfers far below raw token confidence (0.56-0.65 vs 0.71-0.76). Label audit of 300 fresh labels: 4.3% wrong (9% of base SQuAD answers, all correct answers marked wrong). Found: the LM target lacked the end-of-turn token, so fine-tuned models answer in 9-20 words instead of 2. The result matches neither "A ≈ C" nor "C < A" exactly; the closer row ("C < A") allows one investigation before dropping the joint claim; that choice is open. Runs were split into three Kaggle sessions per dataset after the first full sessions were stopped (protocol amendments). | The preregistered decision rules, applied once to all 28 trainings. |
| 2026-10-18 | Phase 2 step 4: training (`train_trust_head.py`), analysis (`run_phase2_analysis.py`), and the Kaggle notebook written and checked end to end on CPU with a tiny model (no Phase 2 results). Two protocol amendments logged: WikiText-2 sample excludes headings; reference baselines were seen during a code test (disclosed, ungated). | The analysis must be committed before any Phase 2 result exists. |
| 2026-10-18 | Phase 2 protocol locked (`docs/phase2/TCL-Phase2-Protocol.md`) before any training: arms A-D on Qwen 0.5B with LoRA, trained on TriviaQA and on SQuAD, primary evaluation on fixed base-model answers (paired), regenerated answers secondary, seed-averaged paired bootstrap, generation bar on both test sets plus WikiText-2 perplexity, fresh label audit. Theory paper: the LM loss uses reference answers only. | Step 3 of the Phase 2 plan. Fixed answers make the arm comparison paired; the paper's "general instruction mix" was replaced by measuring drift with held-out perplexity, which keeps every arm's training data identical. |
| 2026-10-18 | Phase 2 redesigned before any Phase 2 data: H3 split into **H3-T** (transfer across datasets, primary) and **H3-J** (joint objective vs trust-only fine-tuning, secondary), each requiring a CI-excluding-zero win in both transfer directions plus the generation-quality bar; a tie rule for C vs B fixed in advance ("generation-preserving trust training" only if B fails the quality bar and C meets it); training data changed from TriviaQA + NQ-Open to TriviaQA and SQuAD (Phase 1 splits, rule v3 labels), each arm trained on each; trust head reads 50% depth; refusals reported separately. Decision table updated. | Phase 1: frozen probes do not transfer (H2/H2-R failed), in-distribution gains are small, middle layers are best, and Qwen answers only 7% of NQ-Open correctly. Prior work (Kapoor et al. 2024) makes "fine-tuning beats frozen probes" alone not new, so the joint-vs-trust-only comparison needs a predeclared reading of a tie. |
| 2026-10-18 | Phase 2 step 1: labeling rule v3 frozen (`docs/phase2/TCL-Labeling-Rule-v3.md`). Audited error rate 6.7% -> 4.5% (optimistic, developed on the audit set); a fresh audit of Phase 2 test answers will give the reported rate. Phase 1 stays on rule v2. | The Phase 1 audit showed one-directional, form-level label errors (plurals, spelling, units, refusals) concentrated in SQuAD and NQ-Open. |
| 2026-10-18 | **Phase 1 closed.** Label audit complete (`docs/phase1/TCL-Phase1-Label-Audit.md`): automatic labels wrong on 40 of 600 test rows (6.7%; 0-4% TriviaQA, 7-14% SQuAD and NQ-Open). Audit labels were decided by the assistant, not reviewed by hand. Before Phase 2: improve the labeling rule (plurals, spelling, units, abbreviations, refusals never correct, no digit-inside-number matches) and validate it on the 600 audited rows. | The rule's errors are one-directional (correct answers marked wrong) and concentrated in SQuAD and NQ-Open, where two of the four H1-R wins and the Qwen SQuAD loss sit. |
| 2026-10-18 | Phase 1 Kaggle analysis run (`docs/phase1/TCL-Phase1-Kaggle-Checkpoint.md`): **H1-R passes at the threshold** (4 of 6 pairings) and **H2-R fails** (2 of 6 directions), for all seeds. Phase 2 trust-head input set to 50% depth, answer-mean. Phase 2 must evaluate cross-dataset transfer and handle refusals explicitly. Label audit still pending before Phase 1 closes. | With 1,500 test questions, small probe gains are detectable for Qwen TriviaQA and Phi NQ-Open, but Qwen SQuAD is significantly worse and Gemma's gains are partly refusal detection. Frozen probes again fail to transfer. |
| 2026-10-17 | Kaggle Phase 1 amendments (`docs/phase1/TCL-Phase1-Kaggle-Protocol.md`, Section 7): refusals detected on the answer span instead of the full output; Qwen NQ-Open dropped under the size rule. Both decided from labels and output text, before any metric. | Gemma appends disclaimers after real answers, so the full-output flag over-matched; Qwen answers only 6.7% of NQ-Open test questions correctly. |
| 2026-10-16 | Added the test-set size rule to Section 6: at least 1,000 test rows and at least 150 of each class; pairings that cannot reach this are dropped. Applies to the Kaggle part of Phase 1 and to all later phases. It does not change the recorded H1/H2 results. | Phase 1 local tests had 200 test rows (AUROC CIs about ±0.10), too wide to detect the small effects seen for Qwen and Phi. Phi SQuAD had only 7 incorrect test rows. |
| 2026-10-16 | Phase 1 local gate evaluated (`docs/phase1/TCL-Phase1-Local-Checkpoint.md`): **H1 fails** (2 of 6 runs) and **H2 fails** (2 of 6 directions); both hold under all three seeds. Per Section 7, I proceed to Phase 2 and record that the case for TCL now rests on H3 and H4. Added the H2 count rule (>= 4 of 6 directions) in `docs/phase1/TCL-Phase1-Protocol.md` before evaluation. | Frozen-probe gains were significant only for Gemma 2B-it, where raw confidence is near chance. With 200 test rows the comparisons are underpowered (CIs about ±0.1 AUROC). |
| 2026-10-16 | Verified all 39 references in theory paper v2 against OpenAlex, publisher proceedings (PMLR, NeurIPS, ACL Anthology), arXiv, and conference pages. Corrected the Zhao et al. (2021) author initial, added the published venue for Bang et al. (2023), and added pages for Azaria & Mitchell (2023). Tightened claims attributed to Orgad et al. (detectors fail to generalize across *datasets*), Kapoor et al. (wording now follows their abstract), Kadavath et al. (scale and format), and PopQA (monthly page views). | Phase 0 requires that the related work, which positions TCL's novelty, rest on verified sources. |
| 2026-10-16 | Phase 0: wrote theory paper v2 (`paper/TCL-Theory-Paper-v2.md` / `.docx`). Split TCL into TCL-J (joint, the proposal) and TCL-F (frozen, the baseline); adopted the Section 5 trust vector; added related work. Rewrote Section 2, claim 1, to build on prior fine-tuning work. | While writing related work I confirmed that fine-tuned self-knowledge (Kadavath et al. 2022) and LoRA fine-tuning beating frozen probes (Kapoor et al. 2024) are already published, so "joint training helps" alone is not new. My claim is now the controlled isolation of the joint objective. |
| 2026-10-16 | Created this program plan to replace `docs/archive/tcl-v0/TCL-v0-Roadmap.md`. Moved headline metrics from high-confidence error counts to AUROC/AURC; moved calibration comparisons to unweighted post-hoc calibrated scores. | My baseline checkpoint showed raw-only calibration can reduce high-confidence error counts; class-balanced weighting distorts calibration metrics (e.g. Phi SQuAD raw-only ECE 0.307 weighted vs 0.027 unweighted). |
