# Zenodo Upload: Phase 2 Paper

What to enter on zenodo.org (New upload) to publish the paper with a DOI. After publishing, add the DOI to the README, `CITATION.cff`, and the paper's first page.

## File

- `tcl-phase2-negative-result.pdf` (this folder)

## Metadata

| Field | Value |
|---|---|
| Resource type | Publication / Preprint |
| Title | Joint Training Does Not Make Trust Heads Transfer: A Preregistered Negative Result |
| Creator | Mohamed, Awab (Affiliation: Independent researcher) |
| Publication date | The day of upload |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Language | English |
| Keywords | trust estimation; uncertainty estimation; calibration; hallucination detection; probing; LoRA; negative result; preregistration; language models |
| Related works | "Is supplemented by" https://github.com/awabmoha/truth-calibration-layer (Resource type: Software) |
| Version | v1 |

## Description (paste as the abstract)

A trust head is a small model that reads a language model's hidden states and predicts whether its answer is correct. A natural hope is that training this head jointly with the language model, under L_LM + lambda * L_trust, will shape the backbone's representations so that trust estimates generalize better than a head on a frozen model. I test this with a preregistered comparison of four arms on identical data: a frozen-backbone head (A), LoRA fine-tuning for trust only (B), joint LoRA fine-tuning (C), and LoRA fine-tuning for language modeling only followed by a head (D). The model is Qwen2.5-0.5B-Instruct, trained on TriviaQA or SQuAD and tested on the other dataset, with 3 seeds per arm and paired bootstrap confidence intervals. The pass rule, fixed before training, required C to beat A in both transfer directions while keeping generation quality. It fails. Joint training beats the frozen head from TriviaQA to SQuAD (+0.022 AUROC) but is significantly worse from SQuAD to TriviaQA (-0.038), and its gain in the first direction is matched by language-model fine-tuning alone (+0.031). After the analysis I found an implementation flaw: the language-modeling target lacked the end-of-turn token, so fine-tuned models answered in a median of 5-19 words instead of 2. I fixed it and ran the one rerun my protocol allowed. Answers returned to 1-2 words, but the transfer result did not change (+0.034 and -0.038; LM-only fine-tuning +0.041 and +0.002). In-distribution, joint training improves on trust-only fine-tuning by about 0.01 AUROC but never beats LM-only fine-tuning. Across every arm and both runs, a head trained on one dataset's correctness labels transfers far below the model's raw token confidence (0.56-0.65 vs 0.71-0.76) and semantic entropy (0.77), which are trained on neither. An earlier study of frozen probes on three models (Qwen2.5-0.5B, Phi-3.5-mini, Gemma 2B) shows the same failure for the two models whose token confidence is informative. At this scale, joint training does not solve the transfer problem of learned trust estimators.

## Later Versions

To update the paper, open the record on Zenodo and click "New version". The new version gets its own DOI; the concept DOI always points to the latest one. Later phases (3-5) go in new records, as separate papers that cite this one.
