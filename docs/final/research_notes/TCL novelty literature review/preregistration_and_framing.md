# Preregistration, Negative-Result Reporting, and Framing for C7 (cumulative TCL paper)

Research date: 2026-11-28. Verification key: **[V]** = checked this session against arXiv / ACL Anthology / venue page; **[U]** = from memory or secondary aggregator only, verify before citing.

## Q1. Is preregistration established practice in ML/NLP, and in LLM uncertainty work specifically?

### Takeaway
Preregistration is *advocated* in NLP/ML (position papers 2021-2023, two NeurIPS pilot workshops) but is *not* established practice; I found only a handful of explicitly preregistered LLM-uncertainty studies, all 2026 and all single-author arXiv preprints by one author (Cacioli). Verdict on C7's methodological claim: **(b) partially known** -- preregistration of LLM-confidence experiments and pre-registered negative results already exist, but a multi-phase, cumulative, fully preregistered program (locked protocols per phase, run-once analyses, label audits before metrics, amendment log, all failures reported) on learned trust heads appears uncommon; I found no direct equivalent, which is a weak "apparently new" signal only for the *combination*, not for any single practice.

### Cited Findings
- **van Miltenburg, van der Lee & Krahmer, "Preregistering NLP research", NAACL 2021, pp. 613-623** [V]: argue NLP should adopt preregistration (common in medicine/psychology, "rarely discussed in NLP"), propose preregistration questions per study type, and argue for registered reports as grounds for "slow science in NLP research". Won a Best Thematic Paper award (per Tilburg portal). — [ACL Anthology 2021.naacl-main.51](https://aclanthology.org/2021.naacl-main.51); [arXiv 2103.06944](https://arxiv.org/abs/2103.06944v2); [Tilburg portal](https://research.tilburguniversity.edu/en/publications/preregistering-nlp-research/)
- **Søgaard, Hershcovich & de Lhoneux, "A Two-Sided Discussion of Preregistration of NLP Research", EACL 2023, pp. 83-93** [V]: dialogue-form pros/cons; pros include guarding against HARKing and p-hacking, cons include worries that preregistration could skew the field toward confirmatory work. — [ACL Anthology 2023.eacl-main.6](https://aclanthology.org/2023.eacl-main.6); [arXiv 2302.10086](https://arxiv.org/abs/2302.10086v1)
- **NeurIPS 2020 workshop "The pre-registration experiment: an alternative publication model for machine learning research"** (Bertinetto, Henriques, Albanie, Paganini, Varol; Dec 11 2020) [V]: papers submitted *without results*, reviewed on protocol, experiments run after acceptance; proceedings = PMLR vol. 148 (published 8 July 2021). Repeated at NeurIPS 2021 ("The pre-registration workshop"); the organizers themselves called 2020 a "successful pilot" (self-assessment). — [NeurIPS 2020 workshop](https://neurips.cc/virtual/2020/public/workshop_16158.html); [NeurIPS 2021 workshop](https://neurips.cc/virtual/2021/workshop/21885); [PMLR v148](https://proceedings.mlr.press/v148/)
- **Hofman, Chatzimparmpas, Sharma, Watts & Hullman, "Pre-registration for Predictive Modeling", arXiv 2311.18807 (Nov 2023)** [V]: predictive ML still suffers "researcher degrees of freedom"; they offer a lightweight pre-registration template for predictive studies and a qualitative user study. Coauthor blog caveat: "it's still an honor system". — [arXiv 2311.18807](https://arxiv.org/abs/2311.18807v1); [Statistical Modeling blog](https://statmodeling.stat.columbia.edu/2023/12/06/exploring-pre-registration-for-predictive-modeling/)
- **Cacioli, "Distilling Self-Consistency into Verbal Confidence: A Pre-Registered Negative Result and Post-Hoc Rescue on Gemma 3 4B", arXiv 2604.24070 (Apr 2026)** [V, venue = arXiv preprint only]: preregistered confidence-conditioned SFT with self-consistency targets; confirmatory run *lowered* verbal AUROC2 0.554 -> 0.509, cut accuracy 7.6 pp, preregistered decision tree returned "Stop". Post-hoc (explicitly exploratory) rescue: removing a modal-correct filter gave AUROC2 0.774 vs single-pass logit entropy 0.701 on held-out TriviaQA; one model, one seed. — [arXiv 2604.24070](https://arxiv.org/pdf/2604.24070); [alphaXiv](https://www.alphaxiv.org/abs/2604.24070)
- **Cacioli, "Verbal Confidence Saturation in 3-9B Open-Weight Instruction-Tuned LLMs: A Pre-Registered Psychometric Validity Screen", arXiv 2604.22215 (24 Apr 2026)** [V]: OSF preregistration osf.io/azbvx; 7 models, 524 TriviaQA items, 8,384 trials; all 7 invalid on numeric verbal confidence (mean ceiling rate 91.7%); token logprobs did not predict verbal confidence (CV R^2 < 0.01). — [arXiv 2604.22215](https://arxiv.org/abs/2604.22215)
- A search for "pre-registered LLM calibration/uncertainty OSF" returned large non-preregistered benchmarks (e.g., arXiv 2602.00279, 2505.23854, 2605.23909) but no other preregistered studies. — [search result example: arXiv 2505.23854](https://arxiv.org/html/2505.23854v1)

### Inferences
- Preregistration in NLP has well-cited *advocacy* and *debate* papers (2021, 2023) and ML had a pilot (2020-21), but adoption in LLM-uncertainty work looks rare: the only preregistered examples I found are Cacioli's 2026 preprints. Cacioli 2604.24070 is the **closest precedent** to C7 and must be cited: same topic family (training confidence signals), preregistered, negative, and also compares against a logit/entropy baseline. Key differences to stress honestly: C7 is multi-phase and cumulative (4 phases, 0.5B -> 7B), tests learned hidden-state heads (probes, joint LoRA, mixture-trained heads) rather than verbal confidence, includes extra "trust dimensions" (self-consistency, grounding), and reports *no* post-hoc rescue as a headline.
- Do not claim "first preregistered LLM uncertainty study". A defensible wording: "to my knowledge, one of few preregistered studies of learned LLM confidence signals, and the only multi-phase one I found."
- The C7 practices (label audits before metrics, run-once analyses, amendment log) map onto the registered-report model argued for by van Miltenburg et al. and piloted at NeurIPS 2020; framing them as an application of those proposals (not an invention) is safest.

### Gaps
- I did not check OSF directly for unpublished LLM preregistrations; prevalence estimates (e.g., "% of LLM UQ papers preregistered") do not exist in what I found.
- Did not verify whether Cacioli's papers are under review anywhere; treat as non-peer-reviewed.
- No retrospective on outcomes of the NeurIPS 2020/2021 pre-registration pilot was found.

## Q2. Negative-result venues and precedents (TMLR, ICBINB, Insights), and "X does not beat a simple baseline" papers in uncertainty

### Takeaway
TMLR's criteria explicitly exclude novelty/impact and centre on (1) claims matched by evidence and (2) interest to some audience, and they let authors narrow claims instead of adding experiments -- a good fit for a careful negative study. Dedicated negative-result workshops exist (Insights @ ACL venues, ICBINB @ NeurIPS/ICLR, ICLR 2026 edition LLM-focused). The strongest "simple baseline is hard to beat" precedent in LLM UQ is the LM-Polygraph benchmark (TACL 2025). I did not find a TMLR paper specifically titled as a negative result on LLM trust/uncertainty heads.

### Cited Findings
- **TMLR acceptance criteria** [V]: two questions -- are claims supported by accurate, convincing (and clear) evidence, and would some of TMLR's audience be interested. Claims-vs-evidence is the most important; gaps can be closed by more experiments *or* by authors reducing their claims. TMLR avoids "significant", "impactful", "novel"; interest is satisfied if "there is something to be learned by some researchers"; papers should be accepted even when the contribution is modest, but rejected for bold claims without rigorous evidence or unclear writing. — [TMLR acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html); [TMLR reviewer guide](https://jmlr.org/tmlr/reviewer-guide.html); [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html)
- **Insights from Negative Results in NLP** [V]: 6th edition at NAACL 2025 (Albuquerque, May 2025), 16 papers; aims to give negative results a venue because NLP's "near-universal focus on improvements in benchmarks" makes them hard to publish; also invites methodological critiques. — [ACL Anthology 2025.insights-1](https://aclanthology.org/2025.insights-1); [event page](https://aclanthology.org/events/insights-2025/)
- **ICBINB ("I Can't Believe It's Not Better")** [V]: ICLR 2025 edition "Challenges in Applied Deep Learning" (Singapore, 28 Apr 2025); promotes "slow science" and discussion of surprising/negative results; asks papers to describe the negative outcome and investigate why it failed; proceedings PMLR vol. 296 (Aug 2025). ICLR 2026 edition is themed on LLMs ("Where LLMs Need to Improve"). — [ICBINB 2025 site](https://sites.google.com/view/icbinb-2025); [PMLR v296](https://proceedings.mlr.press/v296/); [ICBINB 2026 site](https://sites.google.com/view/icbinb-2026); [ICLR 2026 CFP](https://groups.google.com/g/ML-news/c/ZOPDi5FT39M)
- **Simple-baseline-is-strong precedent outside LLMs**: Bouvier et al., "Towards Clear Expectations for Uncertainty Estimation" (arXiv 2207.13341, 2022): on 7 classification datasets, no statistical superiority of SOTA intrinsic UQ methods over simple baselines. Venue not confirmed [U for venue]. — [arXiv 2207.13341](https://arxiv.org/abs/2207.13341v1)
- **Probe robustness failures (closest technical neighbours of C7's hidden-state heads)**:
  - Stacey, Orgad, Inui, Heinzerling & Moosavi, arXiv 2604.11662 (Apr 2026) [V; title differs between listing "Hidden Failures in Robustness: Why Supervised Uncertainty Quantification Needs Better Evaluation" and PDF "A Robust Evaluation of Probe Robustness..."]: >2,000 probes; "poor robustness in current methods", especially long-form; middle layers and token aggregation generalise better; robustness driven by inputs more than architecture. — [arXiv 2604.11662](https://arxiv.org/abs/2604.11662)
  - arXiv 2507.03998 "Toward Better Generalisation in Uncertainty Estimators": probes trained on one dataset "often struggle to generalise" to other tasks/domains. — [arXiv 2507.03998](https://arxiv.org/html/2507.03998v1)
  - Srey et al., arXiv 2606.27679 (factorised study of probe-based UE): raw hidden states hard to beat in-domain; structured/compressed features more robust under shift. — [arXiv 2606.27679](https://arxiv.org/abs/2606.27679)
- **Counter-evidence (positive results for trained signals) that C7 must engage with**:
  - Kapoor et al., "Large Language Models Must Be Taught to Know What They Don't Know", NeurIPS 2024 [V]: ~1,000 graded examples suffice to beat baselines; training through features (LoRA) needed; claims generalisation to new formats/subjects; says perplexity-style black-box methods have limited power. — [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract.html); [arXiv 2406.08391](https://arxiv.org/abs/2406.08391)
  - Liu et al., "Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach", arXiv 2404.15993: supervised hidden-state UE with claimed robust OOD transfer. — [arXiv 2404.15993](https://arxiv.org/html/2404.15993v3)
  - Kossen et al., "Semantic Entropy Probes", arXiv 2406.15927: probes claimed to generalise OOD better than accuracy-predicting probes. — [arXiv 2406.15927](https://arxiv.org/html/2406.15927v1)

### Inferences
- C7's substantive conclusion ("learned hidden-state trust adds little beyond calibrated token confidence") is **(b) partially known**: LM-Polygraph shows MSP is hard to beat on short-output QA, and 2025-26 probe-robustness papers show probe gains shrink OOD. It directly *contradicts* the positive claims of Kapoor et al. (2024) and Liu et al. (2024); the paper's interest to TMLR readers lies largely in that tension, so the fair-comparison details (calibrated baseline, held-out datasets, same labels) are the load-bearing evidence and should be foregrounded.
- A likely reviewer objection: differences from Kapoor et al. (scale, training data, label grading, baseline calibration). The paper should include an explicit "why my result differs from Kapoor et al. / Liu et al." section rather than claim they are wrong.
- The TMLR "reduce claims" route matters: scope the claim to QA, Qwen-family 0.5B-7B (as tested), and the specific heads tested.
- Fallback venues if TMLR rejects: Insights @ ACL-venue or ICBINB (LLM-themed in 2026) -- these are workshops (non-archival or lighter-weight); TMLR remains the journal option consistent with my publication strategy.

### Gaps
- No TMLR paper found that is explicitly a multi-phase preregistered negative result; I could not search OpenReview TMLR full-text effectively.
- Orgad et al., "LLMs Know More Than They Show" (ICLR 2025, arXiv 2410.02707) reports probes for error detection generalise poorly across tasks [U -- not verified this session, but highly relevant; verify].
- Desk-rejection reasons from TMLR 2026-11-22 (per my memory notes) were not part of this research; framing below assumes the cumulative paper is a fresh submission.

## Q3. What do UQ benchmarks report about simple token-probability baselines vs trained/supervised methods?

### Takeaway
LM-Polygraph (Vashurin et al., TACL 2025) concludes that Maximum Sequence Probability is "a very strong and robust baseline across all tasks", that information-based methods are on par with costlier ones for short answers (and substantially superior on multiple-choice QA), and that most density-based (hidden-state, training-data-fitted) methods perform poorly -- direct support for C7's conclusion, though LM-Polygraph's main benchmark does not include fully supervised correctness probes like C7's.

### Cited Findings
- **Vashurin et al., "Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph", TACL vol. 13 (2025), pp. 220-248, DOI 10.1162/tacl_a_00737; arXiv 2406.15627 (v1 Jun 2024, v4 Jun 2025)** [V]; eleven tasks. — [ACL Anthology 2025.tacl-1.11](https://aclanthology.org/2025.tacl-1.11); [arXiv 2406.15627](https://arxiv.org/abs/2406.15627). Statements from the paper text (TACL PDF):
  - Evaluation protocols "overlook simple yet effective baselines"; "MSP often proves difficult to surpass in tasks with short outputs" (Sec. on evaluation issues).
  - "Despite being a simplistic baseline, MSP demonstrates strong performance": best on MMLU for StableLM, second-best for Mistral; second-best on CoQA for Mistral; lags on GSM8k.
  - "The majority of density-based methods demonstrate poor results across all tasks", except hybrid HUQ-MD (information + density).
  - Conclusion: for shorter answers use information-based methods, which "often perform on par with diversity-based techniques" at lower cost; for multiple-choice QA MSP/Perplexity/CCP "would be substantially superior"; sample-diversity methods (Semantic Entropy, DegMat, SAR) are preferable for long outputs; reflexive/verbalised methods generally weak except for large models; "MSP appears to be a very strong and robust baseline across all tasks and should not be discarded".
  - Isotonic PCC normalisation calibrates scores without hurting ranking (relevant to C7's "calibrated raw confidence" baseline).
- Information-based methods described as "simple to implement and cost-effective" while "often on par with more computationally demanding techniques" (Sec. 2). — [ACL Anthology 2025.tacl-1.11](https://aclanthology.org/2025.tacl-1.11)
- Fadeeva et al. 2023, "LM-Polygraph: Uncertainty Estimation for Language Models", EMNLP 2023 System Demonstrations, arXiv 2311.07383 [U -- not re-verified this session].
- Geng et al., "A Survey of Confidence Estimation and Calibration in Large Language Models", NAACL 2024, arXiv 2311.08298 [U -- not re-verified this session].
- Huang et al., "Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models", arXiv 2307.10236 [U -- not re-verified; I recall it found simple measures competitive in some settings but have not confirmed the exact finding].

### Inferences
- C7 can position itself as the *trained-head* counterpart of LM-Polygraph's MSP finding: LM-Polygraph shows unsupervised/density hidden-state methods rarely beat MSP; C7 tests *supervised* hidden-state heads under a preregistered, fair comparison and finds the same pattern at 0.5B-7B on QA. That is a coherent, modest, interest-satisfying contribution.
- Because MSP's advantage is strongest for short-answer QA (LM-Polygraph), C7 must state that its tasks are short-answer QA and should not extrapolate to long-form generation, where probes and sample-diversity methods may matter more.

### Gaps
- LM-Polygraph's main tables do not include supervised correctness probes (e.g., Kapoor-style LoRA heads); I found no benchmark that pits supervised probes against calibrated MSP head-to-head across held-out datasets under preregistration -- this is the gap C7 fills, but I cannot rule out a 2026 paper I did not find.

## Q4. Recommended titles and framing for a TMLR paper

### Takeaway
Frame the paper as a careful, preregistered empirical study whose main finding is a well-scoped null ("learned hidden-state trust signals add little beyond calibrated token confidence on QA, 0.5B-7B"), with the preregistration process as supporting methodology, not the headline novelty. Lead with the question and the fair comparison; report every phase including the two Phase 2 failures; position against LM-Polygraph (MSP strong), Kapoor et al. and Liu et al. (trained signals help), probe-robustness work (Stacey et al.), and Cacioli (preregistered negative result).

### Cited Findings
- TMLR accepts modest contributions if claims match evidence and some audience learns something; reduced claims are an accepted remedy. — [TMLR acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html); [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html)
- Precedent title phrasing "A Pre-Registered Negative Result" exists in this exact sub-area (Cacioli 2026). — [arXiv 2604.24070](https://arxiv.org/pdf/2604.24070)
- ICBINB asks negative-result papers to investigate *why* the method failed — a useful structuring device even for TMLR. — [ICBINB 2025](https://sites.google.com/view/icbinb-2025)

### Inferences (recommendations, my judgement)
Candidate titles:
1. **"Do Learned Trust Heads Beat Calibrated Token Confidence? A Preregistered Multi-Phase Study from 0.5B to 7B"** (question-form, states scope; recommended).
2. **"Calibrated Token Confidence Is Hard to Beat: Four Preregistered Phases of Learned Trust Signals for LLM Question Answering"** (echoes LM-Polygraph's "MSP is a strong baseline" message; claim-forward).
3. **"Learned Hidden-State Trust Adds Little Beyond Calibrated Confidence: A Preregistered Negative Result for LLM QA"** (most direct; parallels Cacioli's title style).

Framing checklist:
- Abstract: question -> fair comparison definition (calibrated raw confidence, held-out datasets, same labels) -> preregistration/run-once -> result per phase -> scoped conclusion (QA, Qwen2.5 0.5B-7B, heads tested).
- Contribution list: (i) a fair-comparison protocol and result; (ii) evidence across 4 phases and model scales; (iii) negative results for self-consistency and grounding as extra trust dimensions; (iv) full preregistration artifacts and amendment log (methodological transparency, citing van Miltenburg et al. 2021, NeurIPS 2020 pilot, Hofman et al. 2023).
- Avoid: "first preregistered", "trust is one-dimensional" as a general claim (scope it to the signals and tasks tested), and any implication that Kapoor et al./Liu et al. are wrong; instead give a "reconciling with prior positive results" section (different baselines, calibration of baseline, OOD evaluation, scale).
- Include an "evidence of power" element (CIs, minimum detectable effect) so the null is informative rather than underpowered -- the main way TMLR reviewers will test "claims supported by evidence" for a null.

### Gaps
- I did not find TMLR-specific statistics on acceptance rates for negative-result papers.
- Title-length/style norms for TMLR were not checked.
