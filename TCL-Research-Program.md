# TCL Research Program

Status: my master plan. It replaces `docs/archive/tcl-v0/TCL-v0-Roadmap.md` as the document that governs this project. Every phase-level document I write (ablation plans, runbooks) must trace back to a phase and hypothesis here.

Date: October 16, 2026.

## 1. North Star

I believe language models should emit, alongside every claim they generate, a structured and calibrated estimate of how much that claim can be trusted, and that this ability should be **learned inside the model**, not bolted on afterwards.

The Truth Calibration Layer (TCL) is my research program to prove this is possible, useful, and better than post-hoc alternatives.

I am keeping the name. I make claims phase by phase, and at each phase I claim only what that phase's gate has passed.

## 2. What Makes TCL Different

Prior work already shows that frozen hidden states carry correctness signal (Kadavath et al. 2022; Azaria & Mitchell 2023; Burns et al. 2022; Marks & Tegmark 2023; Kossen et al. 2024; Orgad et al. 2024). TCL-v0 replicates that. For me it is the foundation, not the contribution.

Fine-tuning a model to predict its own correctness is also not new: Kadavath et al. (2022) trained P(IK), and Kapoor et al. (2024) report that LoRA fine-tuning on graded examples beats frozen probes. My claims build on that result instead of claiming it.

My distinguishing claims are:

1. **The joint objective specifically helps.** Training the backbone with `L_LM + lambda * L_trust` produces better trust estimates than a frozen head, than trust-only fine-tuning, and than LM-only fine-tuning on the same data, without degrading generation. The controlled comparison that isolates the joint objective is my contribution.
2. **Trust is multi-dimensional.** Separate, operationally defined trust dimensions carry non-redundant information and support better downstream decisions than a single confidence score.
3. **It transfers.** A learned trust signal holds up across datasets, task families, and model scales, where frozen probes are known to degrade.

Every phase below exists to test one of these. Phase 1 showed that frozen probes fail claim 3 (they do not transfer across datasets), so Phase 2 tests claims 1 and 3 together, with transfer as its primary result.

## 3. Theory Issues I Must Resolve First

Status: resolved in `paper/TCL-Theory-Paper-v2.md` (draft, October 16, 2026). Phase 0 gate review pending.

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
| H4 | At least one additional trust dimension is learnable and carries information not captured by correctness confidence. | Phase 3 |
| H5 | The joint-trained trust signal generalizes to a non-QA task family and to a larger model. | Phase 4 |
| H6 | Trust estimates can be produced at claim level inside long-form generation. | Phase 5 |

H3 is the core of TCL. Phase 1 showed that frozen probes do not transfer across datasets, so transfer (H3-T) is where a trained trust head has to prove itself, and it is the primary Phase 2 test. Whether the joint objective specifically matters (H3-J) is the secondary test. If either fails, I follow Section 8.

## 5. Canonical Trust Vector (Adopted In Theory v2)

I only admit dimensions that have a label procedure.

| Dimension | Meaning | Label procedure | Dataset |
|---|---|---|---|
| Correctness | Probability the claim is correct | Answer-match rule v3 (`docs/phase2/TCL-Labeling-Rule-v3.md`), with an audit of a random test sample | TriviaQA, NQ-Open, SQuAD |
| Consistency (contradiction risk) | Probability the model would contradict this claim under resampling | Sample N=10 answers, cluster by bidirectional NLI entailment, label = fraction of samples outside the claim's cluster | Same QA sets |
| Grounding | Whether the claim is supported by provided context | Entailment of answer by context; SQuAD 2.0 unanswerable questions as negatives | SQuAD 2.0 |
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

### Phase 1: Close TCL-v0

- **Goal:** settle H1 and H2 under my locked protocol, then stop iterating on v0.
- **Work (local CPU):** re-analyze my existing records with unweighted calibrated comparisons, bootstrap CIs, and AURC; run cross-dataset transfer (train SQuAD -> test TriviaQA and the reverse; add NQ-Open).
- **Work (Kaggle GPU):** P(True) and semantic-entropy baselines on the six reviewed runs; one layer sweep (early-middle / middle / final) on one model only.
- **H1 passes if:** probe AUROC beats the best calibrated raw baseline with the CI excluding zero in >= 4 of 6 runs.
- **H2 passes if:** the transferred probe keeps >= 70% of its in-distribution AUROC gain over raw.
- **Deliverable:** a short arXiv note / workshop paper: "Do hidden-state probes add value beyond calibrated likelihood and sampling baselines in small open models?"
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

**What a tie between C and B means (fixed before any Phase 2 result):**
- If C ≈ B on trust (CI of the difference includes zero) **and** B fails the generation-quality bar while C meets it, TCL's joint-training claim narrows to **"generation-preserving trust training"**: adapting the backbone for trust helps, and the LM term is what keeps generation intact.
- If C ≈ B on trust **and both** meet the generation-quality bar, the joint objective adds nothing measurable beyond trust-only fine-tuning at this scale. I report that, and TCL's contribution in Phase 2 rests on H3-T alone.
- In neither case do I describe the joint objective as producing better trust estimates.

### Phase 3: Second Trust Dimension

- **Goal:** test H4.
- **Order:** consistency first (its labels need no new data), then grounding (SQuAD 2.0), then evidence density (PopQA).
- **H4 passes if:** the dimension is predicted with AUROC >= 0.70, correlates < 0.7 with correctness confidence, and combining it with correctness improves selective-prediction AURC with the CI excluding zero.
- **Deliverable:** multi-dimensional trust-head results.

### Phase 4: Generalization And Scale

- **Goal:** test H5.
- **Work:** one non-QA task family with automatic labels (GSM8K math); one larger model (Llama-3.1-8B or Qwen2.5-7B via QLoRA).
- **Compute:** likely paid GPU for the 7B/8B runs (estimate in Section 10).
- **H5 passes if:** my Phase 2 and 3 conclusions hold on both, within CIs.
- **Deliverable:** the full TCL paper (conference submission target).

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
| 2 | About 40-60 Kaggle hours | 6-8 weeks |
| 3 | About 30-50 Kaggle hours | 6-8 weeks |
| 4 | Kaggle + about $100-300 paid GPU | 8-10 weeks |
| 5 | Paid GPU | open |

Phases 0-2 run on free compute. The Phase 2 result decides whether paid compute is worth it.

## 11. Publication Track

1. Phase 1: arXiv note / workshop paper (probing beyond calibrated baselines, with transfer).
2. Phases 2-3: main TCL method paper (joint training + multi-dimensional trust).
3. Phases 4-5: full TCL paper with generalization and claim-level results.

## Changelog

| Date | Change | Reason |
|---|---|---|
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
