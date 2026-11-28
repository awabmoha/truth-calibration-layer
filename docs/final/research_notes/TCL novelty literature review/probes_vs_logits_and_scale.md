# Hidden-state correctness probes vs. token-probability confidence: OOD and scale (2022-2026)

Scope: prior work on whether trained hidden-state correctness/truthfulness probes beat the model's own token-probability (logit) confidence, especially on held-out datasets, and how this changes with scale. It is checked against two claims from my preregistered study:
- **C1:** probes, including a mixture-trained MLP that also gets logit(raw confidence) as an input, do not beat Platt-calibrated answer-span token probability on a held-out dataset. Single-dataset probes transfer far worse.
- **C6:** within Qwen2.5 (0.5B to 7B), learned-signal gains over raw confidence shrink as raw confidence improves. Across 27 dataset-model pairs, gains are larger where raw AUROC is weaker (Spearman about -0.4 to -0.6).

Verification legend: **[fetched]** means I read the abstract or page in this session. **[memory, not re-verified]** means the ID, venue, or detail comes from prior knowledge and needs checking before citation.

## Q1. Which papers compare probes with logit/token-probability baselines, do they calibrate the baseline, do they test on held-out datasets, and does anyone report probes losing or tying OOD?

### Takeaway
C1 is **partially known, and its headline direction is close to established**. Orgad et al. 2024 report that cross-dataset probe generalization "rarely exceeds what can be achieved by examining logits alone." Duan (AACL 2026) reports that probe advantage over logprob baselines is concentrated in in-domain and within-family cells. Several 2026 papers report probes tying or losing to token confidence. Three parts of C1 still look new: a **Platt-calibrated** answer-span baseline, a **leave-one-dataset-out** protocol, and a **mixture-trained MLP with logit(confidence) as an input that still fails to beat the baseline, and gets worse at 7B**. I found no prior work combining these. Single-dataset probes transferring poorly is well established (CH-Wang 2023/24, Orgad 2024, Kadavath 2022, Liu 2024).

### Cited Findings
**Direct overlap with C1 (probes vs. logits on held-out data)**
- Orgad et al., "LLMs Know More Than They Show" (arXiv 2410.02707). Error detectors "fail to generalize across datasets." In the generalization analysis they subtract the strongest logit baseline (Logit-min-exact) from probe AUC (Fig. 3b), and write that "most of this performance can be achieved by logit-based truthfulness detection" and that the probe's generalization "rarely exceeds what can be achieved by examining logits alone." They conclude "Past evidence for generalization may therefore have been overstated." Baselines: Logits-mean/min (and -exact variants) and p(True)(-exact). Models: Mistral-7B(-Instruct), Llama3-8B(-Instruct). In-distribution, the exact-answer-token probes beat these baselines (Table 1). The baseline is uncalibrated; AUROC is rank-based, so calibration matters little for AUROC. — [arXiv 2410.02707 abstract](https://arxiv.org/abs/2410.02707), [HTML v4](https://arxiv.org/html/2410.02707v4) [fetched]. Venue: I believe ICLR 2025 [memory, not re-verified; the arXiv page lists none].
- Duan, "Do Correctness Probes Transfer? A Cross-Family Study of LLM Hidden States," AACL-IJCNLP 2026 (long), pp. 5095-5118. Seven datasets, four task families, Qwen3-8B / Llama-3.1-8B / Gemma-2-9B, five seeds. Probes are compared with logprob confidence baselines, and "probe advantage" is concentrated in in-domain and within-family transfer cells. Within-family transfer beats cross-family by 0.075 AUC. Probe directions are nearly orthogonal across dataset pairs. Early-layer probes can look transferable in-domain but fail across sources. Conclusion: transfer is "selective rather than universal." — [ACL Anthology 2026.aacl-long.227](https://aclanthology.org/2026.aacl-long.227/) [fetched; per-cell numbers not read]
- "A Heckler in the Hidden State: Correctness Signals in Diffusion Language Models" (arXiv 2609.36783). Probes roughly tie or lose to token confidence: DiffuCoder-7B probe 0.842 vs. negative token entropy 0.838; Qwen2.5-Coder probe 0.677 vs. log-prob confidence 0.703. — [arXiv](https://arxiv.org/html/2609.36783) [search summary only, not fetched]
- "Operational Proto-Introspection in Looped Language Models" (arXiv 2607.18553). Once the baseline includes answer-level features (length, log-probability, entropy, sample agreement), an answer-free probe adds nothing measurable. — [arXiv](https://arxiv.org/pdf/2607.18553) [search summary only]
- "Geometry of Confabulation and Refusal in Small Language Models" (arXiv 2611.01272). Hidden states add only about 3 points over the strongest prompt baseline for general correctness. — [arXiv](https://arxiv.org/pdf/2611.01272) [search summary only]

**Probes generalize poorly across tasks/datasets (supports the "single-dataset probes transfer worse" part)**
- Kadavath et al. 2022, "Language Models (Mostly) Know What They Know." Trained P(IK) heads "partially generalize across tasks, though they struggle with calibration of P(IK) on new tasks." — [arXiv 2207.05221](https://arxiv.org/abs/2207.05221) [fetched]
- CH-Wang, Van Durme, Eisner, Kedzie, "Do Androids Know They're Only Dreaming of Electric Sheep?", Findings of ACL 2024 (arXiv 2312.17249). Hallucination probes "generalize poorly across tasks" and from synthetic to organic hallucinations. In-domain they beat several contemporary baselines and an expert annotator on response-level F1. — [arXiv 2312.17249](https://arxiv.org/abs/2312.17249) [fetched]
- Liu, Chen, Cheng, He, "On the Universal Truthfulness Hyperplane Inside LLMs," EMNLP 2024 (arXiv 2407.08582). Prior probes "often fail to generalize" OOD. Training on 40+ datasets improves cross-task and cross-domain performance, and diversity matters more than sample count. The abstract mentions no logit baseline. This is the closest precedent for a **mixture-trained** probe, but it argues that mixtures help; it does not ask whether mixtures beat calibrated logits. — [arXiv 2407.08582](https://arxiv.org/abs/2407.08582) [fetched]
- Kossen et al., "Semantic Entropy Probes" (arXiv 2406.15927). "SEPs clearly generalize better to unseen tasks than accuracy probes" (+2.2 to 10.5 AUROC x100). SEPs "approach, but do not match" sampling baselines (semantic entropy, p(True), naive entropy) OOD. I could not read the OOD log-likelihood numbers in Fig. 7. Models include Phi-3-3.8B, Mistral-7B, Llama-2-7B/70B. — [arXiv HTML](https://arxiv.org/html/2406.15927) [fetched]

**Papers reporting probes win (in-distribution, or uncalibrated or weak baselines)**
- Mahaut et al., "Factual Confidence of LLMs," ACL 2024 (long), pp. 4554-4570. "Trained hidden-state probes provide the most reliable confidence estimates" across several LLMs, at the cost of white-box access and supervision. The abstract says nothing on OOD transfer of probes. — [ACL Anthology 2024.acl-long.250](https://aclanthology.org/2024.acl-long.250/) [fetched; full-text details on baselines and generalization not read]
- Liu, Pan, Li, Chen, "Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach" (arXiv 2404.15993). A supervised estimator on hidden activations (black-, grey-, and white-box variants) claims "robust transferability in out-of-distribution settings." It is a close methodological precedent for a supervised head that mixes logit features and hidden states. I did not verify whether hidden states beat grey-box (logit-only) features on held-out data. — [arXiv 2404.15993](https://arxiv.org/abs/2404.15993) [fetched abstract only]
- Kapoor, Gruver et al., "Large Language Models Must Be Taught to Know What They Don't Know," NeurIPS 2024 (arXiv 2406.08391). Prompting is insufficient, "training through the features of a model is necessary," and about 1,000 graded examples beat baselines. The estimates "generalize well." [memory, not re-verified]: frozen-feature probes underperform LoRA fine-tuning, and baselines include sequence likelihood. — [arXiv 2406.08391](https://arxiv.org/abs/2406.08391) [abstract fetched]
- Azaria & Mitchell, SAPLMA, "The Internal State of an LLM Knows When It's Lying," Findings of EMNLP 2023 (arXiv 2304.13734). The probe beats sentence-probability and few-shot baselines with held-out-topic evaluation on true/false statements, not generated QA answers. [memory, not re-verified]
- Ulmer et al., APRICOT, ACL 2024 (arXiv 2403.05973). An auxiliary model on question and answer **text** (not hidden states) predicts confidence and is compared with sequence-likelihood and verbalized baselines. [memory, not re-verified]

**Truth-direction and probe-robustness literature (adjacent, mostly true/false statements)**
- Burns et al., CCS, ICLR 2023 (arXiv 2212.03827). Marks & Tegmark, "Geometry of Truth," COLM 2024 (arXiv 2310.06824). Bürger et al., "Truth is Universal," NeurIPS 2024 (arXiv 2407.12831): a 2D truth subspace generalizes across negations. Levinstein & Herrmann, "Still No Lie Detector for Language Models" (arXiv 2307.00175; Philosophical Studies 2024): probes fail to generalize, e.g. to negations. Farquhar et al. 2023, "Challenges with Unsupervised LLM Knowledge Discovery" (arXiv 2312.10029): CCS-style methods latch onto arbitrary salient features. [all memory, not re-verified this session] These papers mostly evaluate statement truth rather than the correctness of generated QA answers, and rarely use a token-probability baseline.

### Inferences
- The core message of C1, that hidden-state probes' OOD advantage over logits largely disappears, is **already stated** by Orgad et al. 2024 (via logit-subtracted transfer heatmaps) and supported by Duan 2026. I should present C1 as a stricter, preregistered confirmation and extension, not as a discovery.
- What seems new in C1:
  - a Platt-calibrated answer-span baseline;
  - leave-one-dataset-out testing on four standard short-answer QA sets;
  - a **stacked** probe (hidden states plus logit(conf)) that, by construction, has the baseline as an input and still does not beat it on held-out data, and at 7B is 0.004-0.022 AUROC worse;
  - the preregistration itself.
- I found no paper reporting that adding hidden states to a logit feature *hurts* held-out AUROC at 7B. That detail looks new, but it is a small effect, so I should report CIs.
- Orgad's baseline is uncalibrated, and AUROC is invariant to monotone calibration. Platt calibration only matters for calibration metrics (ECE/Brier) or for mixtures trained across datasets. I should not oversell "calibrated" as the reason for the difference.

### Gaps
- I could not read Duan 2026's per-cell numbers, so I cannot say whether probes lost outright or tied in cross-family cells.
- I could not confirm from the full text whether Mahaut et al. 2024 tested probes on held-out datasets.
- I did not verify whether Liu et al. 2404.15993's white-box head beats its grey-box (logit-feature) head out of distribution. This is the most important unread comparison for C1.
- I did not read Kapoor et al.'s OOD (MMLU-subject) results in full text.
- The 2026 arXiv papers (2609.36783, 2607.18553, 2611.01272) are known only from search snippets; their venues and peer-review status are unknown.

## Q2. Evidence on token-probability calibration improving with scale, and on probe advantage vs. scale

### Takeaway
That raw confidence improves with scale is **established** for base models (Kadavath 2022), with the caveat that RLHF and post-training degrade calibration (GPT-4 report). The direction of probe advantage vs. scale is **contested**. One small blog experiment on Qwen2.5 1.5B-7B reports the probe gap *growing* with scale, the opposite of C6. Its baseline was first-token entropy/log-prob, not a calibrated answer-span probability. The ACL 2026 Clue paper reports hidden-state gains are largest on "smaller or less-calibrated" models, which is consistent with C6.

### Cited Findings
- Kadavath et al. 2022: "larger models are well-calibrated on diverse multiple choice and true/false questions" in the right format. P(True) shows "encouraging performance, calibration, and scaling." — [arXiv 2207.05221](https://arxiv.org/abs/2207.05221) [fetched]
- OpenAI GPT-4 Technical Report (arXiv 2303.08774): the pre-trained GPT-4 is well calibrated on MMLU, and post-training (RLHF/PPO) significantly reduces calibration (calibration figure). — [arXiv 2303.08774](https://arxiv.org/abs/2303.08774) [memory, not re-verified]
- Liang et al., "Your Reasoning Model is Secretly a Reward Model: Optimization-Free Verification from Experience," ACL 2026 (long), pp. 17358-17372. A non-parametric hidden-state verifier (Clue). They call token-probability confidence "often miscalibrated"/brittle and say gains are "particularly strong" on smaller or less-calibrated models (e.g., 1.5B on AIME24: 56.7% majority@64 to 70.0%). The method is not a trained probe; the domain is reasoning rather than short-answer QA, and the page gives no OOD test. — [ACL Anthology 2026.acl-long.788](https://aclanthology.org/2026.acl-long.788/) [fetched]
- Hugging Face community blog, "Hidden-State Hallucination Probes Beat Confidence Baselines — And the Gap Grows with Scale." Qwen2.5 1.5B / 3B / 7B; probe minus baseline AUC +0.041 / +0.068 / +0.101; the baseline was entropy and log-prob of the first generated token. The author speculates that instruction-tuned larger models look confident regardless of correctness. Not peer-reviewed, small experiment. — [HF blog](https://huggingface.co/blog/sarimahsan101/llms-know-before-they-hallucinate) [search summary only]
- "The Calls are Coming from Inside the Model" (arXiv 2608.27750, tool calling): probes improve with size (about +0.06 AUROC for 70B-class vs. ~8B), but there is no token-probability comparison. — [arXiv](https://arxiv.org/pdf/2608.27750v1) [search summary only]
- Kossen et al. SEPs: the OOD gain of SEPs over accuracy probes is similar at Llama-2-7B (7.7) and 70B (7.9). There is no clean scale trend, and the gain is not measured against logits. — [arXiv HTML 2406.15927](https://arxiv.org/html/2406.15927) [fetched]

### Inferences
- The C6 premise that raw confidence improves with scale (TriviaQA 0.712 to 0.854) agrees with Kadavath and needs no novelty claim.
- The C6 conclusion that probe and learned-signal gains shrink with scale is **contradicted in direction** by one weak source (the HF blog). That source uses a first-token baseline, which is plausibly a worse baseline at larger instruction-tuned scales. My answer-span, calibrated baseline is a stronger and more appropriate comparator. The discrepancy is worth a sentence in the paper, since the result depends on the baseline.
- I found no peer-reviewed paper that measures probe-minus-logit advantage as a function of scale within one family on held-out QA data. C6's within-family scale trend looks **apparently new**, but my evidence is three points in one family (0.5B, 3.8B Phi is a different family, 7B), so I should frame it cautiously.

### Gaps
- I did not find systematic studies of the AUROC (discrimination) of answer-span probability vs. scale in instruct models for short-answer QA. Kadavath covers base models and MC/TF formats.
- I could not verify the HF blog's setup beyond the search summary.

## Q3. Has anyone reported that "gains are larger where raw confidence is weak"?

### Takeaway
I found the pattern **only qualitatively, and only partially**. The ACL 2026 Clue paper says hidden-state verification helps most on "smaller or less-calibrated models." I found no paper reporting a quantitative across-pairs correlation (e.g., Spearman) between baseline AUROC and probe gain. C6's cross-pair correlation looks **apparently new as a reported analysis**, but it is close to a regression-to-the-mean or ceiling-effect expectation, so its novelty value is modest.

### Cited Findings
- Liang et al. ACL 2026: gains "particularly strong" on smaller or less-calibrated models. — [ACL Anthology 2026.acl-long.788](https://aclanthology.org/2026.acl-long.788/) [fetched]
- Orgad et al. 2024: subtracting the logit baseline from probe transfer AUC shows little residual. This is an implicit "gain over logits" analysis, but it is not correlated with baseline strength. — [arXiv HTML 2410.02707v4](https://arxiv.org/html/2410.02707v4) [fetched]
- Duan 2026: probe advantage over logprob is concentrated in in-domain and within-family cells, which is a cell-level, not baseline-strength, account. — [ACL Anthology 2026.aacl-long.227](https://aclanthology.org/2026.aacl-long.227/) [fetched]

### Inferences
- A negative correlation between baseline AUROC and gain is partly mechanical: there is less headroom near the ceiling, and errors in the baseline estimate feed into the gain with opposite sign. Reviewers will likely raise this. I should (a) state it as expected rather than novel, and (b) where possible, show it holds under a headroom-normalized gain, such as gain / (1 - baseline AUROC), or with split-sample estimation.
- Overall novelty verdicts:
  - **C1:** partially known. Orgad 2024 and Duan 2026 already show probe OOD advantage over logits is mostly absent. New elements are the calibrated answer-span baseline, preregistered leave-one-dataset-out, and the stacked hidden+logit MLP not beating or slightly hurting at 7B.
  - **C6:** partially known or apparently new. The scale-calibration premise is established. A qualitative "helps more where confidence is weak" exists in ACL 2026 (reasoning, non-parametric). The quantitative within-family shrinkage and cross-pair correlation appear unreported, and one weak blog reports the opposite scale trend with a weaker baseline.

### Gaps
- The search was not exhaustive for 2025-2026 OpenReview (ICLR 2026 / NeurIPS 2026) submissions on probe scaling. I made about 14 tool calls.
- I did not verify LM-Polygraph benchmark results (Fadeeva et al. 2023; Vashurin et al., TACL 2025), which compare supervised/density methods with information-based (logit) methods across tasks. They are possibly relevant to C1 and worth a targeted check.
