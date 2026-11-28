# One-pass prediction of sampling-based uncertainty (self-consistency / semantic entropy) and its combination with token-probability confidence: prior work check for claim C3

Claim under review (C3): a one-pass hidden-state MLP head (50%-depth layer, answer-span mean) trained to predict the sampled self-consistency label (fraction of 10 samples NLI-equivalent to the greedy answer), trained on other datasets and tested on a held-out QA dataset, is learnable (AUROC 0.73-0.95) and not redundant with raw confidence (Spearman 0.26-0.64). Combined with calibrated raw confidence via logistic regression, it matches or beats 10-sample semantic entropy (SE) on several held-out pairs (Qwen2.5-0.5B and Qwen2.5-7B). The AURC gain over raw confidence passed the preregistered bar on only some pairs, and the gain shrinks from 0.5B to 7B.

Verification legend: [verified] = checked on arXiv, ACL Anthology, NeurIPS, or MLAnthology pages this session. [memory, unverified] = from prior knowledge, not re-fetched this session.

## Q1. Semantic Entropy Probes (Kossen et al. 2024): setup, OOD results, combination with token probability, comparison to sampled SE and accuracy probes

### Takeaway
SEPs are linear probes on a single hidden state that predict a binarized SE label. Under leave-one-dataset-out evaluation they generalize better than accuracy probes, but they do not match sampled SE, and they are not combined with token probability. They establish the core idea behind C3: a one-pass probe of sampling-based uncertainty that transfers across datasets better than a correctness probe. They do not cover the "combine with raw confidence and match sampled SE" part.

### Cited Findings
- Title and venue: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs", by Kossen, Han, Razzak, Schut, Malik and Gal, arXiv:2406.15927, v1 dated 22 Jun 2024 [verified]. It was presented as an ICML 2024 workshop poster [verified via search listing], and I found no main-track peer-reviewed venue. — [arXiv](https://arxiv.org/abs/2406.15927); [ICML virtual](https://icml.cc/virtual/2024/39442)
- Abstract: SEPs "approximate SE directly from the hidden states of a single generation". They "retain high performance for hallucination detection and generalize better to out-of-distribution data than previous probing methods that directly predict model accuracy." — [arXiv](https://arxiv.org/abs/2406.15927)
- Models: Llama-2-7B/70B, Mistral-7B and Phi-3 Mini (3.8B) for short-form; Llama-2-70B and Llama-3-70B for long-form. Datasets: TriviaQA, SQuAD, BioASQ, NQ-Open. Short answers are clustered with DeBERTa-Large entailment, and correctness is scored by SQuAD F1. — [arXiv HTML](https://arxiv.org/html/2406.15927v1)
- Probe input is a single token position, not an answer-span mean. The two positions are TBG (token before generation) and SLT (second-last token of the answer). The best layers are mid-to-late for short-form, and hidden states beat residual-stream and MLP outputs. The probe is a linear logistic regression, not an MLP. — [arXiv HTML](https://arxiv.org/html/2406.15927v1)
- The target is binarized SE, split at a "best split" threshold. Soft labels "did not improve performance". By contrast, C3 regresses a continuous agreement fraction. — [arXiv HTML](https://arxiv.org/html/2406.15927v1)
- OOD design is leave-one-dataset-out: train on one dataset and test on the others, the reverse of C3's train-on-others, test-on-held-out. AUROC gains of SEP over the accuracy probe (×100):

  | Model | In-distribution | OOD |
  |---|---|---|
  | Mistral-7B | +2.8 | +10.5 |
  | Phi-3 | +2.1 | +9.9 |
  | Llama-2-7B | −0.5 | +7.7 |
  | Llama-2-70B short-form | +1.3 | +7.9 |
  | Llama-3-70B long-form | −2.0 | +6.2 |

  — [arXiv HTML](https://arxiv.org/html/2406.15927v1)
- Versus sampled SE, the paper states that "SEPs cannot match the performance of these methods" (sampled SE, p(True), naive entropy). — [arXiv HTML](https://arxiv.org/html/2406.15927v1)
- No combination of SEP with token log-probability or any other confidence is reported, and there is no systematic scale study. — [arXiv HTML](https://arxiv.org/html/2406.15927v1)
- Follow-up: "Semantic Entropy Neurons: Encoding Semantic Uncertainty in the Latent Space of LLMs" by Han, Kossen, Razzak and Gal, a NeurIPS 2024 workshop poster (Foundation Model Interventions). A small L1-selected set of neurons predicts SE about as well as full probes, and intervening on these neurons shifts semantic uncertainty. — [NeurIPS virtual](https://neurips.cc/virtual/2024/104099)

### Inferences
- The idea "a one-pass hidden-state probe can predict sampling-based semantic uncertainty and transfers across QA datasets better than a correctness probe" is established by SEP (2024). C3's learnability result (AUROC 0.73-0.95 on held-out datasets) is a replication and extension of it, not a new finding.
- C3 differs from SEP in its target (a continuous greedy-answer agreement fraction instead of binarized SE), its features (answer-span mean at 50% depth instead of a single token), and its probe (an MLP instead of a linear probe). These are design variations. CoCoA Light (Q3) already uses a similar target: average similarity of the greedy answer to samples.

### Gaps
- The SEP paper's exact raw AUROCs for SEP vs. sampled SE are given only in figures. I could not extract numbers, so the size of the gap is not quantified here.

## Q2. Foundational sampling-based methods (Kuhn 2023, Farquhar 2024, SelfCheckGPT, self-consistency, KLE, Beyond SE)

### Takeaway
These define the sampled baselines that C3 approximates. None of them is a one-pass method. KLE and SNNE ("Beyond Semantic Entropy") are refinements of SE, and SNNE explicitly allows adding token probabilities in a white-box variant.

### Cited Findings
- Kernel Language Entropy, by Nikitin, Kossen, Gal and Marttinen, NeurIPS 2024, arXiv:2405.20003 [verified]. It computes von Neumann entropy of semantic kernels over sampled answers and is proven more expressive than SE. — [arXiv](https://arxiv.org/abs/2405.20003); [NeurIPS](https://neurips.cc/virtual/2024/poster/93979)
- "Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity", by Nguyen, Payani and Mirzasoleiman, Findings of ACL 2025, arXiv:2506.00245 [verified]. SNNE is a black-box nearest-neighbour entropy that the authors say "can be extended to white-box settings by incorporating token probabilities." — [ACL Anthology](https://aclanthology.org/2025.findings-acl.234)
- Kuhn, Gal and Farquhar, "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation", ICLR 2023, arXiv:2302.09664. [memory, unverified this session] — [arXiv](https://arxiv.org/abs/2302.09664)
- Farquhar, Kossen, Kuhn and Gal, "Detecting hallucinations in large language models using semantic entropy", Nature 630:625-630 (2024), doi:10.1038/s41586-024-07421-0. [memory, unverified this session] — [Nature](https://www.nature.com/articles/s41586-024-07421-0)
- Manakul, Liusie and Gales, "SelfCheckGPT", EMNLP 2023, arXiv:2303.08896. [memory, unverified this session] — [arXiv](https://arxiv.org/abs/2303.08896)
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023, arXiv:2203.11171. [memory, unverified this session] — [arXiv](https://arxiv.org/abs/2203.11171)
- Kadavath et al., "Language Models (Mostly) Know What They Know", arXiv:2207.05221 (2022). [memory, unverified this session] As I recall, it trains a P(IK) head on the model to predict the fraction of sampled answers that are correct. It reports partial cross-task generalization and better calibration and self-knowledge with scale. This is an early one-pass predictor of a sampling-derived target, though the target is correctness rather than agreement. Verify before citing. — [arXiv](https://arxiv.org/abs/2207.05221)

### Inferences
- C3's label, the fraction of samples NLI-equivalent to the greedy answer, is close to SelfCheckGPT-NLI and CoCoA's consistency term, and to the "agreement with greedy" variants of SE. It is a standard sampled-consistency quantity, not a new target.

### Gaps
- The four foundational citations marked [memory] should be checked against their DOIs and arXiv IDs before submission.

## Q3. Hybrid scores (logit confidence × consistency) and amortized one-pass consistency: has anyone combined predicted consistency with raw confidence and matched SE?

### Takeaway
Yes, substantially. CoCoA (Vashurin et al., NeurIPS 2025) shows that combining token-probability confidence with sampled consistency beats both sequence probability and SE. Its "CoCoA Light" variant trains an MLP on mid-layer hidden states of the greedy output to predict the consistency score, using unlabeled data. Combined with sequence probability, Light stays above SE and close to full CoCoA. This is the closest prior work to C3, and it covers the core of the "one-pass predicted consistency + raw confidence ≥ SE" claim. What CoCoA Light does not clearly report (I could not verify it) is a strict held-out-dataset transfer protocol, a scale trend, or a selective-prediction AURC test against raw confidence alone.

### Cited Findings
- CoCoA citation: Vashurin, Goloburda, Ilina, Rubashevskii, Nakov, Shelmanov and Panov. arXiv:2502.04964, first titled "Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency". It was published at NeurIPS 2025 as "CoCoA: A Minimum Bayes Risk Framework Bridging Confidence and Consistency for Uncertainty Quantification in LLMs". The v6 PDF header and MLAnthology both list NeurIPS 2025 [verified]. — [arXiv](https://arxiv.org/abs/2502.04964); [MLAnthology](https://mlanthology.org/neurips/2025/vashurin2025neurips-cocoa)
- The combination is multiplicative, U = u(y*|x) × (1/M) Σ(1 − s(y*, y_i)), where u is confidence (1−p or −log p) and s is similarity to the greedy output. The paper's ablation found the additive form worse. C3 uses logistic regression instead. — [arXiv HTML v6](https://arxiv.org/html/2502.04964v6)
- QA mean PRR, greedy setting:

  | Model | CoCoA-SP | SP | SE |
  |---|---|---|---|
  | Llama 3.1 8B | 0.451 | 0.409 | 0.356 |
  | Mistral 7B | 0.509 | 0.475 | 0.423 |
  | Falcon 3 7B | 0.511 | 0.475 | 0.439 |

  — [arXiv HTML v6](https://arxiv.org/html/2502.04964v6)
- On why simple combinations are not enough, the authors note prior hybrids "sometimes fail to outperform much simpler baseline methods". — [arXiv](https://arxiv.org/abs/2502.04964)
- CoCoA Light: "We use a multilayer perceptron network as an auxiliary model." "The features used are embeddings from the middle layers of the base LLM". These are layer 16 for Llama 3.1 8B and Mistral 7B, and layer 14 for Falcon 3 7B, roughly 50% depth, as in C3. Embeddings come from greedy generations, and the target is the MBR consistency score. Training uses "a held-out set of input sequences" that "is not labeled". At inference Light requires "only greedy output generation... eliminating the need for additional sampling." — [arXiv HTML v6](https://arxiv.org/html/2502.04964v6)
- CoCoA_SP Light vs. full CoCoA_SP, sequence probability (SP) and SE, PRR on TriviaQA (Table 11, greedy):

  | Model | CoCoA_SP Light | CoCoA_SP | SP | SE |
  |---|---|---|---|---|
  | Mistral-7B | 0.674 | 0.670 | 0.628 | 0.562 |
  | Llama-8B | 0.598 | 0.597 | 0.526 | 0.549 |
  | Falcon-7B | 0.691 | 0.684 | 0.662 | 0.581 |

  Light beats SE on TriviaQA, CoQA, MMLU, WMT and XSUM for all three models, with GSM8k mixed. On MMLU, SP alone sometimes beats Light (Falcon: SP 0.539 vs. Light 0.522; Llama: 0.508 vs. 0.504). — [arXiv HTML v6](https://arxiv.org/html/2502.04964v6)
- Phillips, Gustafsson, Wu, Thakur and Clifton, "Entropy Alone is Insufficient for Safe Selective Prediction in LLMs", arXiv:2603.21172 (March 2026) [verified, preprint]. It combines an entropy signal (sequence NLL, 10-sample SE, or SEP) with a linear correctness probe through two-feature logistic regression, and evaluates risk-coverage (E-AURC). Models are 3B-8B (Ministral-8B, Llama-3.2-3B, Qwen3-4B, Gemma-3-4B). Datasets are TriviaQA, BioASQ and MedicalQA, all within-dataset splits with no cross-dataset transfer. Results are mixed. Qwen PC+SE has AUROC 0.910 vs. SE 0.844, but Llama PC+SE has 0.863 vs. SE 0.884. On E-AURC, SE beats PC+SE for Llama (0.045 vs. 0.055) and Ministral (0.066 vs. 0.073). — [arXiv HTML](https://arxiv.org/html/2603.21172v1)
- Shelmanov et al., "A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs", EMNLP 2025, arXiv:2505.08200 [verified]. Supervised transformer heads on attention maps and token probabilities give claim-level hallucination detection that is robust in- and out-of-domain. The target is supervised correctness, not consistency. — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1809)
- Phillips, Wu, Gao and Clifton, "Semantic Self-Distillation for Language Model Uncertainty", arXiv:2602.04577 (Feb 2026) [verified, preprint]. A mixture density network on the last-prompt-token hidden state predicts the distribution of 32 sampled answer embeddings. Mean AUROC on TriviaQA across 7 models (3-8B) is SE 0.804, SSD 0.708, and the scalar probes PCP 0.771 and SEP 0.759. All results are TriviaQA-only, with no token-probability combination and no cross-dataset test. — [arXiv HTML](https://arxiv.org/html/2602.04577v1)
- Cacioli, "Distilling Self-Consistency into Verbal Confidence: A Pre-Registered Negative Result and Post-Hoc Rescue on Gemma 3 4B", arXiv:2604.24070 (27 Apr 2026) [verified, preprint, single author, single model]. Fine-tuning on self-consistency targets reaches held-out TriviaQA AUROC2 0.774, above logit entropy (0.701) and below 10-sample self-consistency (0.999 as reported). The preregistered phase failed (0.554 → 0.509). It distills into verbal output, not a probe head. — [arXiv](https://arxiv.org/abs/2604.24070)
- "Language Models Can Predict Their Own Behavior", arXiv:2502.13329. Probes on input-token hidden states predict the model's eventual CoT answer, with conformal abstention. It is adjacent work on one-pass prediction of sampled behaviour. [verified title only via search snippet; details not checked] — [arXiv](https://arxiv.org/pdf/2502.13329)
- "Too Consistent to Detect", EMNLP 2025, arXiv:2505.17656. Errors where all samples agree with the greedy answer are hard for both consistency methods and hidden-state probes. This is a relevant limitation for any consistency-based head. [search snippet only] — [arXiv](https://arxiv.org/pdf/2505.17656v1)

### Inferences
- These parts of C3 are already established by CoCoA Light (NeurIPS 2025):
  - an MLP on about 50%-depth hidden states of the greedy output predicts the sampled greedy-vs-samples consistency;
  - it is trained without correctness labels;
  - combined with token-probability confidence, it beats 10-sample SE and nearly matches the full sampled hybrid.
- C3's AUROC numbers matching or beating SE (for example 0.785 vs. 0.773 and 0.869 vs. 0.864) are consistent with CoCoA Light's results. They should be framed as an independent preregistered replication under a stricter protocol, not as a first demonstration.
- These parts of C3 appear new or at least not shown in what I could verify:
  - a strict train-on-other-datasets, test-on-held-out-QA-dataset protocol for the consistency head plus confidence combiner;
  - an explicit redundancy check against raw confidence (Spearman 0.26-0.64);
  - a preregistered AURC-gain bar over raw confidence that passes only partially (3/6 and 2/3);
  - a within-family scale comparison (Qwen2.5 0.5B vs. 7B) showing the gain shrinking.
- The partial AURC pass and the near-tie with raw confidence fit what other work reports. SP alone sometimes beats CoCoA Light (MMLU), and Phillips et al. 2026 find combined scores sometimes lose to SE alone. Hybrid gains over a strong confidence baseline look modest and setting-dependent.

### Gaps
- I could not read CoCoA Appendix I ("CoCoA Light Training Details"). The pooling (answer-span mean or otherwise), the number of Monte Carlo samples, and whether the unlabeled held-out set comes from the same dataset or task as evaluation are therefore unverified. If it is the same dataset, C3's cross-dataset protocol is a real distinction. If not, the overlap is even larger.
- I could not confirm whether CoCoA reports AURC or selective-prediction curves for Light, as opposed to PRR. PRR is itself a rejection-curve metric, so it is close to AURC in spirit.

## Q4. Does the advantage of predicted or sampled consistency over raw confidence shrink with model scale?

### Takeaway
I found no study that directly tests whether the gain of a one-pass consistency head, or of sampled SE, over token-probability confidence shrinks from small to larger models within one family. This part of C3 appears new, with weak indirect support.

### Cited Findings
- SEP reports similar OOD gains at Llama-2-7B and 70B (+7.7 vs. +7.9), but the gain is measured over accuracy probes, not raw confidence, and there is no systematic scale analysis. — [arXiv HTML](https://arxiv.org/html/2406.15927v1)
- SSD's results across 3-8B models are not monotonic in size, and the authors attribute differences to distillation fidelity. — [arXiv HTML](https://arxiv.org/html/2602.04577v1)
- Phillips et al. 2026 do not analyse scale, and they note larger models "may exhibit different failure profiles." — [arXiv HTML](https://arxiv.org/html/2603.21172v1)
- A search for "semantic entropy advantage over token probability shrinks with scale" returned no direct test. One 2026 paper argues token-level entropy fails specifically in small models, because answers are short with near-unity token probabilities. That would predict larger consistency gains at small scale, consistent in direction with C3's shrinking gain. — ["Do small language models know what they don't know?" arXiv:2609.20824, search snippet only, unverified](https://arxiv.org/pdf/2609.20824)

### Inferences
- The scale finding is plausibly new but rests on only two model sizes (0.5B, 7B) in one family, and should be stated cautiously.
- Kadavath et al. 2022 ([memory, unverified]) report that self-evaluation and calibration improve with scale. That is compatible with raw confidence becoming harder to beat at larger sizes, but it is not the same test.

### Gaps
- There is no direct prior evidence for or against a shrinking hybrid gain with scale. The search was limited, and large UQ benchmarks (for example LM-Polygraph-style evaluations across sizes) were not checked in detail.

## Overall verdict for C3
- **Partially known (b), close to established for the core.**
  - The one-pass consistency/SE probe is established (SEP 2024; SEP's leave-one-dataset-out OOD advantage over accuracy probes).
  - The amortized consistency MLP on mid-layer greedy embeddings combined with token probability, beating SE, is established in CoCoA Light (NeurIPS 2025).
  - Combining confidence-type and consistency signals with logistic regression for selective prediction appears in Phillips et al. 2026, which is within-dataset and gives mixed results.
- **Apparently new (c) components:**
  - the strict leave-dataset-out evaluation of the consistency head plus confidence combiner against 10-sample SE;
  - the explicit non-redundancy check against raw confidence;
  - the preregistered and only partially passed AURC-gain bar;
  - the within-family 0.5B-to-7B shrinkage of the gain.

  These are contributions of rigour and calibration of expectations, not of method.
- **Recommended framing:** "A preregistered, cross-dataset replication and stress test of amortized consistency + confidence (cf. CoCoA Light; SEPs), finding parity with SE on correctness AUROC but only partial and scale-shrinking selective-prediction gains over raw confidence." CoCoA Light must be cited as the closest prior work.
