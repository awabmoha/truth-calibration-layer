# TCL Final Paper: Outline

Written November 28, 2026, after closing the program and the literature review ([reports/TCL novelty literature review.md](reports/TCL%20novelty%20literature%20review.md)). Evidence base: [TCL-Program-Synthesis.md](TCL-Program-Synthesis.md). Target: arXiv (cs.CL, cross-listed cs.LG), then TMLR, using TMLR's own template and anonymity rules.

## Positioning

The literature review settles what this paper is: **a preregistered, held-out-dataset stress test of claims that other papers already make, not a new method.** None of my findings is a new phenomenon. Orgad et al. (2024) already report that probe transfer rarely exceeds logit baselines; CoCoA Light (NeurIPS 2025) already combines a one-pass consistency predictor with sequence probability; Kapoor et al. (2024) report the opposite of my joint-training result. What I add is the strictness of the test (locked protocols, run once, leave-one-dataset-out, every failure reported) and a few specific diagnostics. TMLR judges whether claims are supported, not novelty, which fits this.

**Working title:** *Token Confidence Is Hard to Beat: A Preregistered Study of Learned Trust Heads for LLM Question Answering*

**Wording rules from the review:**
- Say "answer-span token probability" or "answer probability", not "calibrated", when the metric is AUROC or AURC: Platt scaling is monotone, so it cannot change a ranking metric.
- Do not claim "trust is one-dimensional" in general; claim it for short-answer QA, 0.5B-7B, and the heads tested.
- Do not claim "first preregistered" anything.

## Contributions (as scoped by the review)

1. **A fair test of learned correctness heads against answer probability on held-out datasets**, across three model sizes: no head beats it out of distribution, including a head given logit(confidence) as an input, which is slightly worse at 7B (negative transfer, not redundancy). Extends Orgad et al. (2024) with a span-level baseline, a stacked head, and preregistration.
2. **A controlled joint-training null:** LM + trust LoRA, trust-only LoRA, LM-only LoRA, and a frozen head, in a four-arm design run twice. In-distribution gains, no transfer gain; plain LM fine-tuning explains the one positive direction. Must be reconciled with Kapoor et al. (2024).
3. **A preregistered cross-dataset test of one-pass self-consistency** (cf. SEPs, CoCoA Light): it matches or beats 10-sample semantic entropy when combined with answer probability, but passes a preregistered selective-prediction bar on only some pairs, and its gain shrinks from 0.5B to 7B within Qwen.
4. **Grounding diagnostics:** the head's gain comes from irrelevant-passage answers (split by passage type), a case a 7B model removes by abstaining; and an NLI-label hypothesis template flips a preregistered conclusion (2 of 7 to 5 of 7 pairs, exploratory).
5. **The baseline matters:** against the first-token probability the learned head wins on 4 of 9 held-out pairs; against the minimum token probability or the answer span, on 1 of 9 at most. This connects the program to the disagreement in the literature (exploratory; [TCL-Final-Analyses.md](TCL-Final-Analyses.md)).
6. **The weak-confidence pattern**, as a hypothesis: Gemma (near-chance raw confidence, large probe gains) and a self-consistency correlation that survives artifact corrections but not significance.

## Section Plan

1. **Introduction.** The promise of learned trust signals; the overlooked baseline; what a fair test requires (held-out dataset, strong untrained baseline, decision fixed in advance). Summary of results and of what is and is not new.
2. **Related work.** Probes and their transfer (Kadavath, CH-Wang, Orgad, Liu x2, Duan); fine-tuning for uncertainty (Kapoor, Lin, R-Tuning, APRICOT); sampling-based uncertainty and its amortization (semantic entropy, SEPs, CoCoA); grounding detection and irrelevant context (Lookback Lens, ReDeEP, Dubanowska, Wen, Cuconasu, AbstentionBench); UQ benchmarks (LM-Polygraph: maximum sequence probability is a strong baseline for short outputs); preregistration in NLP (van Miltenburg, Søgaard et al., Cacioli).
3. **Setup.** Models, datasets and splits, labeling rule v3 and audits, the answer-span confidence, arms, metrics, statistics, the preregistration process (protocols, amendments, run-once analyses).
4. **Correctness heads do not transfer past answer probability** (Phases 1, 3, 4a).
5. **Joint training buys in-distribution accuracy, not transfer** (Phases 2, 2b).
6. **Self-consistency: real, cheap, small** (Phase 3, 4a).
7. **Grounding: a relevance detector, and a labeling lesson** (Phase 3, 4a, ungated relabel).
8. **Where learned signals help: weak confidence and scale** (Gemma, Qwen PopQA, the correlation, the Qwen 0.5B-7B comparison).
9. **Reconciling with prior positive results.** Kapoor et al. (training size and diversity, model size, baseline), Liu et al., CoCoA Light, SEPs: plausible reasons for different outcomes, without claiming they are wrong.
10. **Limitations.** Short-answer QA only; up to 7B; one probe design; scale confounded with family (Phi); distractors are unrelated rather than related-but-insufficient; NLI also defines the consistency label; power.
11. **Conclusion.**
- **Appendix:** every protocol's decision table, amendments, label audits, per-seed results, compute.

## Analyses Added Before Drafting (Reporting Only, No New Tests)

**Done November 28, 2026:** [TCL-Final-Analyses.md](TCL-Final-Analyses.md) (`scripts/final_analyses.py`). They use saved scores only, are exploratory, and change no recorded decision.

1. **Minimum detectable effect** for each preregistered test, from the bootstrap CIs, so each null says how large a gain it rules out. (Result: Phase 2 is not the weakest; H4-T MDEs are 0.03-0.06 AUROC.)
2. **Weak-confidence correlation, corrected:** a headroom-normalized gain (gain / (1 - raw AUROC)), split-sample estimation (raw AUROC and gain from disjoint halves of the test items) to remove the shared-noise artifact, and a bootstrap clustered by model and dataset.
3. **Paired CIs for the 7B H4-T losses** (already computed in `runs/phase4a_h4t_analysis/details.json`; to be tabulated).
4. **Baseline choice:** first-token and minimum-token probabilities next to the answer-span baseline. (Result: the most useful new diagnostic; see contribution 5.)

## Checks Before Submission (From the Review)

- Whether Liu et al. (2404.15993)'s white-box head beats their logit-only head out of distribution.
- Whether CoCoA Light trains on the dataset it is evaluated on (if not, C3 is almost fully pre-empted).
- Duan (AACL 2026) per-cell numbers; the spurious cue Dubanowska et al. identify.
- Verify every reference marked unverified in the review, including venues for Orgad (ICLR 2025?), Lin et al. 2022 (TMLR?), APRICOT, Kumar et al. 2022.
