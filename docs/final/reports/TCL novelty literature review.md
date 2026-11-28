# Rigor, not discovery, carries the TCL paper

None of my seven claims is a new phenomenon. **Four of them (C1, C2, C3, C6) restate directions that prior work already reports or contests, and C4, C5 and C7 are partly known.** What is new is narrower: the combination of a preregistered, run-once, leave-one-dataset-out protocol with a strong untrained baseline, plus a few specific results that I found nowhere else. These are a stacked head that receives logit(raw confidence) as an input and still fails to beat raw confidence on held-out data (slightly worse at 7B); a controlled four-arm comparison of joint LM+trust LoRA against a frozen head; a within-family scale trend in which learned-signal gains shrink; a grounding head whose gain comes almost entirely from unrelated passages that a 7B model refuses on its own; and a case where a label-template choice flips a preregistered conclusion. The headline direction of C1 is already in Orgad et al. 2024, who write that probe generalization "rarely exceeds what can be achieved by examining logits alone" ([arXiv 2410.02707](https://arxiv.org/html/2410.02707v4)). The core of C3 is already in CoCoA Light (NeurIPS 2025), which trains an MLP on mid-layer greedy hidden states to predict consistency and, combined with sequence probability, beats semantic entropy ([arXiv 2502.04964](https://arxiv.org/html/2502.04964v6)). The main published counterpoint is Kapoor et al. 2024, who argue that "training through the features of a model is necessary" and that the trained estimators generalize ([arXiv 2406.08391](https://arxiv.org/abs/2406.08391)). The honest paper is a preregistered, scoped confirmation and stress test: "calibrated token confidence is hard to beat on short-answer QA from 0.5B to 7B." It should not be framed as a discovery. It also needs to say openly that AUROC cannot tell a calibrated baseline from an uncalibrated one, that the weak-confidence correlation is partly mechanical, and that its nulls need minimum-detectable-effect statements to be informative.

## Per-claim verdicts: one method-level novelty, the rest is protocol and scope

The table below gives a verdict for each claim. "Established" means a peer-reviewed paper already reports the same result. "Partially known" means the direction or main ingredients are published but my exact setup or test is not. "Apparently new" means I found no prior report in this search. That is weak evidence, because each search used roughly 15 tool calls and did not fully cover 2025-2026 OpenReview submissions.

| Claim | Verdict | Closest prior work | What is new, if anything |
|---|---|---|---|
| **C1** Trained hidden-state correctness probes, including a mixture-trained MLP given logit(raw confidence), do not beat Platt-calibrated answer-span probability on a held-out dataset; slightly worse at 7B | **Partially known; the direction is close to established** | Orgad et al. 2024 ([arXiv 2410.02707](https://arxiv.org/html/2410.02707v4)); Duan, AACL 2026, probe advantage concentrated in in-domain and within-family cells ([ACL Anthology](https://aclanthology.org/2026.aacl-long.227/)); LM-Polygraph, MSP "very strong and robust baseline" ([TACL 2025](https://aclanthology.org/2025.tacl-1.11)); Liu et al. 2024 for mixture training ([arXiv 2407.08582](https://arxiv.org/abs/2407.08582)) | A stacked head that has the baseline as an input and still does not beat it on held-out data, and loses 0.004-0.022 AUROC at 7B. Also the preregistered leave-one-dataset-out protocol on four short-answer QA sets. "Calibrated" is **not** a differentiator for AUROC (see risks). |
| **C2** Joint LM+trust LoRA does not transfer better than a frozen head (or trust-only LoRA); all trained heads transfer below raw confidence and semantic entropy | **Partially known; the four-arm design is apparently new; the headline contradicts Kapoor et al.** | Kapoor et al. 2024, LoRA beats a frozen probe and generalizes MC to open-ended ([arXiv 2406.08391](https://arxiv.org/html/2406.08391v3)); Kumar et al. 2022, fine-tuning can distort features and underperform OOD ([arXiv 2202.10054](https://arxiv.org/pdf/2202.10054)); Lin et al. 2022, logit uncertainty generalizes under shift about as well as trained verbal uncertainty ([arXiv 2205.14334](https://arxiv.org/abs/2205.14334)) | A controlled, preregistered null for joint L_LM + λ·L_trust against frozen, trust-only and LM-then-head arms, judged against calibrated confidence and semantic entropy. Scope: 0.5B, single-source training, 2 sets × 3 seeds. |
| **C3** One-pass self-consistency head + raw confidence matches or beats 10-sample semantic entropy on held-out data; only partly passes the AURC bar; gain shrinks 0.5B→7B | **Partially known; the core is established** | Semantic Entropy Probes ([arXiv 2406.15927](https://arxiv.org/html/2406.15927v1)); CoCoA Light, NeurIPS 2025 ([arXiv 2502.04964](https://arxiv.org/html/2502.04964v6)); Phillips et al. 2026, mixed results for probe+SE combinations ([arXiv 2603.21172](https://arxiv.org/html/2603.21172v1)) | A strict train-on-others, test-on-held-out protocol for the head + confidence combiner; an explicit redundancy check (Spearman 0.26-0.64); a preregistered AURC bar passed only partly (3/6, 2/3); the shrinking gain from 0.5B to 7B (two sizes only). |
| **C4** One-pass grounding head mostly detects irrelevant/distractor passages; a 7B model abstains on those itself | **Partially known** | Dubanowska et al. 2025, detector performance "largely driven by a spurious correlation" and near-random OOD ([arXiv 2509.19372](https://arxiv.org/abs/2509.19372)); CH-Wang et al. 2024 ([arXiv 2312.17249](https://arxiv.org/abs/2312.17249)); Wen et al. 2024, irrelevant context can make abstention easier ([arXiv 2404.12452](https://arxiv.org/abs/2404.12452)); Cuconasu et al. 2024 on random vs. distracting passages ([arXiv 2401.14887](https://arxiv.org/abs/2401.14887)) | Splitting the head's gain into relevant-passage and distractor cases, with near-zero gain on relevant passages. Also the methodological point that a detector's measured advantage can be an artifact of a small model failing to abstain, and that it disappears at 7B (1,996/2,000 abstentions). |
| **C5** NLI grounding labels are noisy (76-79% human agreement); a hypothesis-template fix flips a preregistered conclusion (2/7 → 5/7 pairs, ungated) | **Partially known** | QA2D ([arXiv 1809.02922](https://arxiv.org/pdf/1809.02922)); Chen, Choi, Durrett 2021, where the question-to-declarative converter is "critical" ([arXiv 2104.08731](https://arxiv.org/abs/2104.08731)); hypothesis-wording sensitivity ([arXiv 2409.13735](https://arxiv.org/pdf/2409.13735), snippet only) | The agreement figure for DeBERTa-MNLI on MRQA-style short answers. A documented case of a labeling choice flipping a preregistered test. This is a cautionary case study, and the re-analysis is exploratory. |
| **C6** Gains shrink with scale within one family and are larger where raw-confidence AUROC is weaker | **Partially known; the quantitative analysis is apparently new but modest** | Kadavath et al. 2022 on calibration improving with scale ([arXiv 2207.05221](https://arxiv.org/abs/2207.05221)); Liang et al., ACL 2026, gains "particularly strong" on smaller or less-calibrated models ([ACL Anthology](https://aclanthology.org/2026.acl-long.788/)); an HF blog reports the **opposite** scale trend against a first-token baseline ([HF blog](https://huggingface.co/blog/sarimahsan101/llms-know-before-they-hallucinate), snippet only) | A within-Qwen2.5 trend against a calibrated answer-span baseline, and a cross-pair Spearman of about -0.4 to -0.6. Both are partly expected from ceiling effects and shared measurement error. |
| **C7** Fully preregistered multi-phase program reporting every failure | **Partially known as practice; apparently uncommon as a combination** | van Miltenburg et al., NAACL 2021 ([ACL Anthology](https://aclanthology.org/2021.naacl-main.51)); NeurIPS 2020 pre-registration pilot ([PMLR v148](https://proceedings.mlr.press/v148/)); Cacioli 2026, "A Pre-Registered Negative Result" on LLM confidence ([arXiv 2604.24070](https://arxiv.org/pdf/2604.24070)) | A cumulative, multi-phase, run-once program with label audits before metrics and an amendment log. I found no direct equivalent, but I must not claim "first preregistered." |

## Prior positive results the paper must reconcile, not dismiss

The paper's interest to TMLR readers rests on its tension with published positive results. It therefore needs a dedicated "reconciling with prior work" section that explains plausible reasons for the different outcomes, without claiming those papers are wrong.

**Kapoor et al. 2024 is the main counterpoint.** They train on about 20k graded examples from 16 benchmarks, using LLaMA-2 7B/13B and Mistral 7B. They report that a frozen "Probe is insufficient to generalize effectively from MC to OE," while LoRA + Prompt does generalize ([arXiv 2406.08391](https://arxiv.org/html/2406.08391v3)). Their setup differs from mine in ways that could explain the opposite result. Their training data are many-dataset, while my C2 training is single-source. Their models are 7B+, while my C2 model is 0.5B. Their uncertainty estimator is a separate fine-tuned model with no LM loss. Most important, they report no temperature-scaled token-likelihood baseline for open-ended answers, and they call perplexity "generally not predictive." Their companion Calibration-Tuning paper reportedly finds fine-tuning more effective than an auxiliary model on frozen features. I saw that quote only in a search snippet ([ACL Anthology](https://aclanthology.org/2024.uncertainlp-1.1/)). Three other papers report OOD gains for trained confidence: R-Tuning ([arXiv 2311.09677](https://arxiv.org/abs/2311.09677)), RLCR ([arXiv 2507.16806](https://arxiv.org/html/2507.16806v2)) and ConfTuner ([arXiv 2508.18847](https://arxiv.org/abs/2508.18847v2)). All three mostly train on many datasets or at larger scale.

**Liu, Pan, Li and Chen 2024 is the closest methodological precedent for C1's stacked head.** Their supervised estimator mixes logit features and hidden states in black-, grey- and white-box variants, and claims "robust transferability in out-of-distribution settings" ([arXiv 2404.15993](https://arxiv.org/abs/2404.15993)). The most important unread comparison for C1 is whether their white-box head beats their grey-box (logit-only) head out of distribution. That must be checked before submission. Two other papers report favorable probe results. Liu, Chen, Cheng and He (EMNLP 2024) show that training on 40+ datasets improves cross-domain probes, but they do not compare against logits ([arXiv 2407.08582](https://arxiv.org/abs/2407.08582)). Mahaut et al. (ACL 2024) find that "trained hidden-state probes provide the most reliable confidence estimates," and the abstract does not mention held-out-dataset transfer ([ACL Anthology](https://aclanthology.org/2024.acl-long.250/)).

**Orgad et al. 2024 and Duan 2026 support my results and also pre-empt them.** Orgad et al. subtract the strongest logit baseline from probe transfer AUC and conclude that "past evidence for generalization may therefore have been overstated." In-distribution, however, their exact-answer probes do beat logits ([arXiv 2410.02707](https://arxiv.org/html/2410.02707v4)). Duan finds probe advantage over logprob confidence concentrated in in-domain and within-family cells, with probe directions nearly orthogonal across dataset pairs ([ACL Anthology](https://aclanthology.org/2026.aacl-long.227/)). C1 must be presented as a stricter replication of their direction. LM-Polygraph adds benchmark-scale support: MSP "often proves difficult to surpass in tasks with short outputs," and most density-based hidden-state methods "demonstrate poor results across all tasks" ([TACL 2025](https://aclanthology.org/2025.tacl-1.11)). Its main tables, however, do not include supervised correctness probes, and that is the gap my paper fills.

**CoCoA and SEPs pre-empt C3.** SEPs generalize better than accuracy probes OOD, with gains of +7.7 to +10.5 AUROC×100, but "cannot match" sampled semantic entropy, and they are never combined with token probability ([arXiv 2406.15927](https://arxiv.org/html/2406.15927v1)). CoCoA Light uses an MLP on about 50%-depth embeddings, trained without correctness labels and combined with sequence probability. On TriviaQA it reaches PRR 0.674 vs. SE 0.562 on Mistral-7B. On MMLU, sequence probability alone sometimes beats it ([arXiv 2502.04964](https://arxiv.org/html/2502.04964v6)). I could not verify whether CoCoA Light's unlabeled training set comes from the evaluation dataset. If it does, my cross-dataset protocol is a real distinction; if not, the overlap is even larger. Phillips et al. 2026 combine an entropy signal with a correctness probe using two-feature logistic regression. Their results are mixed: semantic entropy alone beats the combination on E-AURC for Llama and Ministral ([arXiv 2603.21172](https://arxiv.org/html/2603.21172v1)). This supports my finding that the AURC bar was passed only partly.

**For grounding, abstention and preregistration, the counterpoints are narrower.** Lookback Lens reports cross-task and cross-model transfer of an attention-ratio detector ([arXiv 2407.07071](https://arxiv.org/abs/2407.07071)). This contrasts with my poor-transfer and distractor-only findings, but it is not a direct contradiction. AbstentionBench finds scale has "almost no effect" on abstention ([arXiv 2506.09038](https://arxiv.org/pdf/2506.09038)), and Sufficient Context finds strong models answer instead of abstaining when context is insufficient ([arXiv 2411.06037](https://arxiv.org/abs/2411.06037v3)). Both mean my 7B abstention result must be scoped to clearly unrelated passages, which are the easy case, and not to related-but-insufficient ones. For C7, Cacioli's two 2026 preprints are the closest preregistered precedents in LLM confidence ([arXiv 2604.24070](https://arxiv.org/pdf/2604.24070); [arXiv 2604.22215](https://arxiv.org/abs/2604.22215)). Stacey et al. 2026 test more than 2,000 probes and report "poor robustness in current methods" ([arXiv 2604.11662](https://arxiv.org/abs/2604.11662)), which is a close neighbor to my overall conclusion.

## Reviewer risks that could sink an honest null

**AUROC cannot see Platt calibration.** Platt scaling is a monotone transform, and both AUROC and AURC depend only on the ranking of scores. So "Platt-calibrated" raw confidence has exactly the same AUROC and AURC as uncalibrated answer-span probability. The calibration cannot be why my baseline is harder to beat than Orgad's. What does differ is the **answer-span** aggregation, as opposed to first-token or mean-logit baselines. That difference matters, and the HF blog that reports the opposite scale trend used a first-token baseline ([HF blog](https://huggingface.co/blog/sarimahsan101/llms-know-before-they-hallucinate)). Calibration matters only for ECE and Brier score, and as a feature scale inside learned combiners. The paper should say this outright and keep "calibrated" out of its novelty claims. LM-Polygraph also notes that isotonic normalization calibrates scores "without hurting ranking" ([TACL 2025](https://aclanthology.org/2025.tacl-1.11)).

**The stacked-head result invites a "why didn't it learn the identity?" question.** A head that receives logit(confidence) and scores below it on held-out data means the hidden-state features fitted source-dataset patterns that then hurt on the target dataset. That is negative transfer, not mere redundancy. Reviewers will ask about regularization, early stopping and capacity. They will also ask whether a simple stacking (logistic regression on confidence plus a held-out-validated probe score) would close the gap. The 0.004-0.022 AUROC losses at 7B are small, so they need paired bootstrap CIs and should not be called "worse" unless the CIs exclude zero.

**The weak-confidence correlation in C6 has a built-in artifact.** Gain is defined as learned AUROC minus raw AUROC. Any noise in raw AUROC therefore appears with opposite signs in the x-axis and the y-axis, which produces a negative correlation even with no real effect. There is also less headroom near AUROC = 1. On top of that, the 27 dataset-model pairs share models and datasets, so they are not independent, and a naive Spearman p-value overstates the evidence. The paper should report a headroom-normalized gain such as gain/(1 − raw AUROC), estimate baseline strength and gain on separate splits, and use a permutation or cluster bootstrap over models and datasets. Even if the pattern survives, it should be described as expected, not as a discovery.

**Scale claims rest on two points.** Phi-3.5-mini is a different family, so "within one family" means Qwen2.5-0.5B vs. 7B only, for both C3 and C6. Kadavath et al. already establish that calibration improves with scale for base models ([arXiv 2207.05221](https://arxiv.org/abs/2207.05221)), and the GPT-4 report shows post-training can reduce it (unverified in the notes). An instruction-tuned two-point trend is suggestive, not a scaling law.

**Nulls need power statements.** TMLR judges whether "claims are supported by accurate, convincing and clear evidence," and it lets authors narrow claims instead of adding experiments ([TMLR acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html)). For a null, that means reporting a minimum detectable effect for each preregistered test, given the held-out test-set sizes and seed variance, so that "does not beat" means "does not beat by more than X." C2 is most at risk, with only 2 training sets × 3 seeds at 0.5B, and so are the partial AURC passes in C3.

**Grounding results have label and design weaknesses.** The distractors are unrelated passages, so reviewers will say the head learned a topic-match detector. Existing work already shows context reliance is linearly decodable (snippet only, [arXiv 2602.22787](https://arxiv.org/html/2602.22787v2)). The 5/7 result after the template fix is ungated and exploratory. It must sit beside the preregistered 2/7, not replace it, or it reads as a forking path. The same NLI model also defines the self-consistency label in C3 ("NLI-equivalent to the greedy answer"). Reviewers may ask whether the 76-79% agreement problem affects C3 as well, so I should check this or state it as a limitation. Chen et al. found most false negatives came from the premise or the NLI model rather than the conversion pipeline ([arXiv 2104.08731](https://ar5iv.labs.arxiv.org/html/2104.08731)). My full-passage premise therefore makes NLI-side errors plausible.

**Scope creep in wording is a risk.** "Trust is one-dimensional" overstates the result. The evidence covers short-answer QA, three models from 0.5B to 7B, and the specific heads I tested. LM-Polygraph reports that MSP's advantage is strongest for short outputs, while sample-diversity methods are preferable for long ones ([TACL 2025](https://aclanthology.org/2025.tacl-1.11)). The conclusion therefore must not extend to long-form generation.

## Recommended framing and titles

The paper should open with a question and a definition of a fair comparison, not with a method. Do learned hidden-state trust signals add transferable information beyond the model's own answer-span probability, when judged on held-out datasets with identical labels and preregistered, run-once analyses? The answer it gives is: mostly no on correctness. Self-consistency adds a modest gain that matches 10-sample semantic entropy in one pass, but it only partly clears its preregistered bar and shrinks with scale. Grounding mostly reduces to distractor detection, which a 7B model already handles by abstaining. The contribution list should be honest. It contains a fair-comparison protocol and its result; a replication of Orgad and CoCoA directions under that protocol; three specific new results (the stacked head, the joint-LoRA ablation, and the stratified grounding gain); and full preregistration artifacts, framed as an application of van Miltenburg et al. ([ACL Anthology](https://aclanthology.org/2021.naacl-main.51)) and Hofman et al. ([arXiv 2311.18807](https://arxiv.org/abs/2311.18807v1)), not as an invention. Following ICBINB's advice, each failure should come with an analysis of why it failed ([ICBINB 2025](https://sites.google.com/view/icbinb-2025)). The defensible wording on preregistration is "one of few preregistered studies of learned LLM confidence signals, and the only multi-phase one I found."

Three candidate titles fit this framing. **"Calibrated Token Confidence Is Hard to Beat: A Preregistered Study of Learned Trust Heads for LLM Question Answering"** is claim-forward and echoes LM-Polygraph's message. **"Do Learned Hidden-State Trust Heads Beat Token Confidence on Held-Out QA? A Preregistered Multi-Phase Study from 0.5B to 7B"** is question-form and states its scope. **"Learned Hidden-State Trust Adds Little Beyond Answer Probability: Preregistered Negative Results for LLM QA"** is the most direct and parallels Cacioli's title style. Given the AUROC-invariance point, I prefer titles that say "token confidence" or "answer probability" over ones that put weight on "calibrated." The first two are the safest.

## Conclusion

The literature changes what this paper is. It is not evidence that probes fail; Orgad, Duan and LM-Polygraph already make that case. It is the most controlled test I could find of the strongest positive claims (Kapoor, Liu, CoCoA Light) against a fair baseline, and its value depends on the fairness controls, CIs and minimum detectable effects more than on any single number. The two most useful new findings are diagnostic: a stacked head can do worse than its own input on new data, and a grounding detector's apparent value can vanish once the base model handles the easy case. Both warn evaluators more than they advance methods, which suits TMLR's "something to be learned" criterion.

Four checks would change verdicts before submission. The first is whether Liu et al.'s white-box head beats its grey-box head OOD; a yes would weaken C1's newness. The second is whether CoCoA Light trains on the evaluation dataset; a no would leave C3 almost fully pre-empted. The third is Duan's per-cell numbers. The fourth is which spurious cue Dubanowska et al. identify; if it is context relevance, C4 overlaps more strongly with their work.

## References

Status key: **Verified** means the arXiv, ACL Anthology, venue page or abstract was read during the research sessions (detailed figures may still come only from the HTML). **Snippet** means the paper was seen only as a search summary, so its existence and content are provisional. **Unverified** means the citation comes from memory and was not re-checked; check IDs and venues before citing.

| Reference | ID / venue | Status |
|---|---|---|
| Orgad et al., "LLMs Know More Than They Show" | arXiv 2410.02707; ICLR 2025 | Verified (arXiv); venue unverified |
| Duan, "Do Correctness Probes Transfer? A Cross-Family Study of LLM Hidden States" | AACL-IJCNLP 2026, 2026.aacl-long.227 | Verified (abstract; per-cell numbers not read) |
| Kadavath et al., "Language Models (Mostly) Know What They Know" | arXiv 2207.05221 | Verified (abstract) |
| CH-Wang, Van Durme, Eisner, Kedzie, "Do Androids Know They're Only Dreaming of Electric Sheep?" | arXiv 2312.17249; Findings of ACL 2024 | Verified |
| Liu, Chen, Cheng, He, "On the Universal Truthfulness Hyperplane Inside LLMs" | arXiv 2407.08582; EMNLP 2024 | Verified |
| Kossen et al., "Semantic Entropy Probes" | arXiv 2406.15927; ICML 2024 workshop poster | Verified |
| Han, Kossen, Razzak, Gal, "Semantic Entropy Neurons" | NeurIPS 2024 workshop poster | Verified |
| Mahaut et al., "Factual Confidence of LLMs" | arXiv 2406.13415; ACL 2024, 2024.acl-long.250 | Verified (abstract) |
| Liu, Pan, Li, Chen, "Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach" | arXiv 2404.15993; no venue found | Verified (abstract only) |
| Kapoor, Gruver et al., "Large Language Models Must Be Taught to Know What They Don't Know" | arXiv 2406.08391; NeurIPS 2024 | Verified |
| Kapoor et al., "Calibration-Tuning: Teaching LLMs to Know What They Don't Know" | UncertaiNLP 2024, 2024.uncertainlp-1.1 | Verified (abstract); frozen-vs-finetune quote snippet only |
| Liang et al., "Your Reasoning Model is Secretly a Reward Model" (Clue) | ACL 2026, 2026.acl-long.788 | Verified |
| Vashurin et al., CoCoA | arXiv 2502.04964; NeurIPS 2025 | Verified (Appendix I not read) |
| Vashurin et al., "Benchmarking UQ Methods for LLMs with LM-Polygraph" | arXiv 2406.15627; TACL 13 (2025) 220-248 | Verified |
| Nikitin, Kossen, Gal, Marttinen, "Kernel Language Entropy" | arXiv 2405.20003; NeurIPS 2024 | Verified |
| Nguyen, Payani, Mirzasoleiman, "Beyond Semantic Entropy" (SNNE) | arXiv 2506.00245; Findings of ACL 2025 | Verified |
| Phillips, Gustafsson, Wu, Thakur, Clifton, "Entropy Alone is Insufficient for Safe Selective Prediction in LLMs" | arXiv 2603.21172; preprint | Verified |
| Phillips, Wu, Gao, Clifton, "Semantic Self-Distillation for Language Model Uncertainty" | arXiv 2602.04577; preprint | Verified |
| Shelmanov et al., "A Head to Predict and a Head to Question" | arXiv 2505.08200; EMNLP 2025 | Verified |
| Cacioli, "Distilling Self-Consistency into Verbal Confidence: A Pre-Registered Negative Result..." | arXiv 2604.24070; preprint | Verified |
| Cacioli, "Verbal Confidence Saturation in 3-9B Open-Weight Instruction-Tuned LLMs" | arXiv 2604.22215; preprint | Verified |
| Lin, Hilton, Evans, "Teaching Models to Express Their Uncertainty in Words" | arXiv 2205.14334; TMLR 2022 | Verified (arXiv); venue unverified |
| Zhang et al., R-Tuning | arXiv 2311.09677; NAACL 2024 | Verified |
| Li, Xiong, Wu, Hooi, ConfTuner | arXiv 2508.18847; NeurIPS 2025 | Verified (venue); numbers from summary |
| Damani et al., RLCR, "Beyond Binary Rewards" | arXiv 2507.16806; ICLR 2026 | Verified |
| Ulmer et al., APRICOT | arXiv 2403.05973; ACL 2024 | Verified (arXiv); venue unverified |
| Ha & Vo, "Toward Better Generalisation in Uncertainty Estimators" | arXiv 2507.03998; preprint | Verified |
| Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform OOD" | arXiv 2202.10054; ICLR 2022 | arXiv seen; venue unverified |
| "Preserving Pre-trained Features Helps Calibrate Fine-tuned Language Models" | arXiv 2305.19249 | Verified (arXiv); venue unknown |
| Stacey, Orgad, Inui, Heinzerling, Moosavi, probe robustness study | arXiv 2604.11662; preprint (title differs between listing and PDF) | Verified |
| Chuang et al., Lookback Lens | arXiv 2407.07071; EMNLP 2024 | Verified |
| Sun et al., ReDeEP | arXiv 2410.11414; ICLR 2025 | Verified |
| Dubanowska et al., "Representation-based Broad Hallucination Detectors Fail to Generalize OOD" | arXiv 2509.19372; Findings of EMNLP 2025 | Verified (abstract) |
| Zhang et al., PRISM | arXiv 2411.04847; ACL 2025 | Verified |
| Yoran, Wolfson, Ram, Berant, "Making Retrieval-Augmented LMs Robust to Irrelevant Context" | arXiv 2310.01558; ICLR 2024 | Verified (venue listings inconsistent) |
| Cuconasu et al., "The Power of Noise" | arXiv 2401.14887; SIGIR 2024 | Verified |
| Joren et al., "Sufficient Context" | arXiv 2411.06037; ICLR 2025 | Verified |
| Wen, Howe, Wang, "Characterizing LLM Abstention Behavior in Science QA with Context Perturbations" | arXiv 2404.12452; Findings of EMNLP 2024 | Verified |
| AbstentionBench | arXiv 2506.09038 | Verified (arXiv) |
| Wen et al., "Know Your Limits: A Survey of Abstention in LLMs" | arXiv 2407.18418; TACL 2025 | Verified |
| Demszky, Guu, Liang, QA2D | arXiv 1809.02922 | Verified |
| Chen, Choi, Durrett, "Can NLI Models Verify QA Systems' Predictions?" | arXiv 2104.08731; Findings of EMNLP 2021 | Verified |
| Adlakha et al., "Evaluating Correctness and Faithfulness of Instruction-Following Models for QA" | arXiv 2307.16877; TACL 2024 | Verified |
| Laurer et al., "Less Annotating, More Classifying" | Political Analysis 32(1), 2024 | Verified |
| van Miltenburg, van der Lee, Krahmer, "Preregistering NLP Research" | arXiv 2103.06944; NAACL 2021 | Verified |
| Søgaard, Hershcovich, de Lhoneux, "A Two-Sided Discussion of Preregistration of NLP Research" | arXiv 2302.10086; EACL 2023 | Verified |
| NeurIPS 2020 pre-registration workshop | PMLR vol. 148 | Verified |
| Hofman et al., "Pre-registration for Predictive Modeling" | arXiv 2311.18807 | Verified |
| Bouvier et al., "Towards Clear Expectations for Uncertainty Estimation" | arXiv 2207.13341 | Verified (arXiv); venue unknown |
| TMLR acceptance criteria; Insights @ NAACL 2025; ICBINB 2025/2026 | Venue pages | Verified |
| "A Heckler in the Hidden State" | arXiv 2609.36783 | Snippet |
| "Operational Proto-Introspection in Looped Language Models" | arXiv 2607.18553 | Snippet |
| "Geometry of Confabulation and Refusal in Small Language Models" | arXiv 2611.01272 | Snippet |
| HF blog, "Hidden-State Hallucination Probes Beat Confidence Baselines — And the Gap Grows with Scale" | Hugging Face community blog; not peer-reviewed | Snippet |
| "The Calls are Coming from Inside the Model" | arXiv 2608.27750 | Snippet |
| GrACE | arXiv 2509.09438 | Snippet |
| "Language Models Can Predict Their Own Behavior" | arXiv 2502.13329 | Snippet (title only) |
| "Too Consistent to Detect" | arXiv 2505.17656; EMNLP 2025 | Snippet |
| "Do small language models know what they don't know?" | arXiv 2609.20824 | Snippet |
| Srey et al., factorised study of probe-based UE | arXiv 2606.27679 | Snippet |
| HalluRAG | arXiv 2412.17056 | Snippet |
| GRADE | arXiv 2604.02830 | Snippet |
| Knowledge-attribution probing | arXiv 2602.22787 | Snippet |
| Probing-RAG | Findings of NAACL 2025 | Snippet |
| CLEAR | arXiv 2510.12460 | Snippet |
| Grounding Probes | arXiv 2610.04642 | Snippet |
| Token-level RAGTruth probe study | arXiv 2608.18115 | Snippet |
| "The Powerless Noise" | arXiv 2607.03615 | Snippet |
| Small frozen RAG models abstention study | arXiv 2608.22228 | Snippet |
| Insufficient-evidence abstention study (135M-32B) | arXiv 2609.37469 | Snippet |
| Feng et al., "Don't Hallucinate, Abstain" | ACL 2024 | Snippet |
| Hypothesis-construction sensitivity in NLI zero-shot | arXiv 2409.13735 | Snippet |
| Hypothesis paraphrase inconsistency in NLI | arXiv 2401.14440 | Snippet |
| Ma et al., entailment-based zero-shot classification issues | ACL 2021 (short) | Snippet |
| "The Labeling Problem in Hallucination Detection Benchmarks" | arXiv 2610.08026 | Snippet |
| Lightweight-detector benchmark (short-answer NLI errors) | arXiv 2606.29809 | Snippet |
| Kuhn, Gal, Farquhar, "Semantic Uncertainty" | arXiv 2302.09664; ICLR 2023 | Unverified |
| Farquhar, Kossen, Kuhn, Gal, "Detecting hallucinations in LLMs using semantic entropy" | Nature 630 (2024), doi:10.1038/s41586-024-07421-0 | Unverified |
| Manakul, Liusie, Gales, SelfCheckGPT | arXiv 2303.08896; EMNLP 2023 | Unverified |
| Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" | arXiv 2203.11171; ICLR 2023 | Unverified |
| OpenAI, GPT-4 Technical Report | arXiv 2303.08774 | Unverified |
| Azaria & Mitchell, SAPLMA | arXiv 2304.13734; Findings of EMNLP 2023 | Unverified |
| Burns et al., CCS | arXiv 2212.03827; ICLR 2023 | Unverified |
| Marks & Tegmark, "Geometry of Truth" | arXiv 2310.06824; COLM 2024 | Unverified |
| Bürger et al., "Truth is Universal" | arXiv 2407.12831; NeurIPS 2024 | Unverified |
| Levinstein & Herrmann, "Still No Lie Detector for Language Models" | arXiv 2307.00175; Philosophical Studies 2024 | Unverified |
| Farquhar et al., "Challenges with Unsupervised LLM Knowledge Discovery" | arXiv 2312.10029 | Unverified |
| Niu et al., RAGTruth | arXiv 2401.00396; ACL 2024 | Unverified |
| Sriramanan et al., LLM-Check | NeurIPS 2024 | Unverified |
| Shi et al., "LLMs Can Be Easily Distracted by Irrelevant Context" | arXiv 2302.00093; ICML 2023 | Unverified |
| Rajpurkar et al., SQuAD 2.0 | ACL 2018 | Unverified |
| Honovich et al., TRUE | arXiv 2204.04991; NAACL 2022 | Unverified |
| Laban et al., SummaC | TACL 2022 | Unverified |
| Zha et al., AlignScore | arXiv 2305.16739; ACL 2023 | Unverified |
| Tang, Laban, Durrett, MiniCheck | arXiv 2404.10774; EMNLP 2024 | Unverified |
| Kryściński et al., FactCC | EMNLP 2020 | Unverified |
| Fadeeva et al., LM-Polygraph (toolkit) | arXiv 2311.07383; EMNLP 2023 demos | Unverified |
| Geng et al., survey of confidence estimation and calibration in LLMs | arXiv 2311.08298; NAACL 2024 | Unverified |
| Huang et al., "Look Before You Leap" | arXiv 2307.10236 | Unverified |
