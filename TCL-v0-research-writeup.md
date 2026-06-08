# TCL-v0: Frozen Hidden States as Confidence Signals for Answer Correctness

**Author:** Awab Mohamed  
**Project:** Truth Calibration Layer (TCL)  
**Status:** Reviewed TCL-v0 empirical note; not full TCL validation
**Checkpoint:** v0.1.1 raw-baseline checkpoint

## Abstract

Truth Calibration Layer (TCL) is a theory-stage framework for confidence-aware language models. This note reports TCL-v0, a narrow empirical diagnostic that asks whether frozen language-model hidden states contain useful signal for estimating answer correctness. TCL-v0 leaves the base model unchanged, extracts hidden states from generated answers, trains a lightweight probe, and compares probe-based confidence with raw generation confidence and raw-only post-hoc calibration baselines.

The current evidence supports a cautious claim: frozen hidden states appear to contain answer-correctness calibration signal under tested QA conditions, especially in ranking/AUC and several Brier-score comparisons. However, raw-only calibration explains a meaningful part of the earlier high-confidence wrong-answer reductions. TCL-v0 therefore motivates the broader TCL hypothesis, but it does not validate the full TCL architecture.

## 1. Research Question

Large language models often produce fluent answers even when those answers are wrong. Raw token probabilities measure how likely the generated text is under the model, but they do not reliably measure whether the answer is correct.

TCL-v0 asks:

```text
Can frozen LLM hidden states support better answer-correctness confidence than raw generation confidence and raw-only post-hoc calibration?
```

This is narrower than full TCL. TCL-v0 does not implement a joint token-and-trust training objective, an integrated trust head, or a four-dimensional trust vector. It tests one supporting assumption for the broader theory:

```text
frozen hidden states contain answer-correctness calibration signal
```

## 2. Contributions

This work contributes:

- a reproducible diagnostic pipeline for answer records, hidden states, labels, confidence scores, calibration metrics, and reviewed artifacts
- reviewed 1,000-example runs across Qwen, Phi-3.5 Mini, and Gemma 2B-it on SQuAD and TriviaQA
- a conservative TCL-v0 score:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

- raw-only baseline comparisons using temperature scaling, Platt/logistic calibration, isotonic regression, and raw-only conservative fusion
- an explicit mapping from TCL-v0 evidence to the still-unvalidated full TCL proposal

## 3. Method

For each QA example, TCL-v0 records:

- question
- context, when available
- accepted answers
- model answer
- correctness label
- raw generation confidence
- hidden-state vector
- train/validation/test split
- run metadata

The current implementation uses:

- frozen base model
- final-layer hidden states
- `answer_mean` hidden-state pooling
- logistic regression probe with balanced class weighting
- geometric mean generated-token probability as raw confidence
- conservative fusion using `min(raw, probe)`

The conservative rule was introduced because a plain hidden-state probe can become overconfident on fluent wrong answers. It lets the probe lower confidence but prevents it from raising confidence above raw generation confidence.

## 4. Why The Baselines Matter

The first TCL-v0 results compared mostly against raw generation confidence. That is not enough. Raw confidence is often badly calibrated, especially in small models. A strong result must show that hidden-state probing adds value beyond simple post-hoc calibration of raw confidence.

The v0.1.1 checkpoint therefore adds:

- temperature scaling on raw confidence
- Platt/logistic calibration on raw confidence
- isotonic regression on raw confidence
- `min(raw, raw_only_calibrated)`
- `min(raw, hidden_state_probe)`

This comparison is important because high-confidence wrong-answer reduction can happen simply by pushing confidence downward. Therefore, high-confidence-error reduction alone is no longer sufficient evidence for hidden-state-specific calibration.

## 5. Experimental Setup

### Reviewed Free-GPU Runs

The reviewed extended-validation checkpoint covers:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- Kaggle-hosted Gemma 2B-it
- SQuAD validation with context
- TriviaQA `rc.nocontext`
- six 1,000-example runs
- 200 held-out test examples per run
- targeted manual review completed for all six runs

### Earlier Local Runs

Earlier local CPU diagnostics also used:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- TriviaQA-500
- SQuAD-500
- NQ-Open protocol checks

Those runs established the first empirical foothold, but the reviewed six-run checkpoint and raw-baseline checkpoint are the current evidence standard.

### Metrics

The evaluation reports:

- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score
- reliability bins
- accuracy at threshold 0.5
- AUC when both classes are present
- wrong answers with confidence >= 0.8 and >= 0.9

## 6. Reviewed Extended-Validation Results

The formal Qwen/Phi gate produced:

```text
decision: mixed_continue_cautiously
stopping-rule choice: continue TCL-v0 research
```

The later Gemma exploratory extension produced:

```text
decision: supports_continuing_tcl_v0
```

The strongest repeated pre-baseline pattern was:

```text
conservative TCL-v0 reduces high-confidence wrong answers
```

For example, conservative TCL-v0 reduced wrong answers with confidence >= 0.8 in every reviewed run:

| Run | Raw Wrong >= 0.8 | Conservative TCL-v0 Wrong >= 0.8 |
|---|---:|---:|
| Qwen SQuAD-1000 | 30 | 10 |
| Qwen TriviaQA-1000 | 9 | 1 |
| Phi SQuAD-1000 | 5 | 4 |
| Phi TriviaQA-1000 | 42 | 9 |
| Gemma SQuAD-1000 | 62 | 10 |
| Gemma TriviaQA-1000 | 76 | 10 |

This pattern is practically interesting, but the raw-baseline checkpoint changes how it should be interpreted.

## 7. Raw-Baseline Checkpoint

The raw-baseline checkpoint asks whether hidden-state TCL-v0 still looks useful after adding stronger raw-only calibration baselines.

Weighted raw-only calibrators use balanced class weighting. Under that comparison:

| Run | Best Raw-Only Brier | Hidden Conservative Brier | Best Raw-Only ECE | Hidden Conservative ECE | Best Raw-Only AUC | Hidden AUC |
|---|---:|---:|---:|---:|---:|---:|
| Qwen SQuAD | 0.195974 | 0.195644 | 0.128586 | 0.152570 | 0.678534 | 0.783394 |
| Qwen TriviaQA | 0.190504 | 0.131989 | 0.228478 | 0.099740 | 0.747093 | 0.804905 |
| Phi SQuAD | 0.163009 | 0.046439 | 0.342367 | 0.040488 | 0.705774 | 0.806810 |
| Phi TriviaQA | 0.174747 | 0.198426 | 0.058779 | 0.152008 | 0.837561 | 0.828226 |
| Gemma SQuAD | 0.243452 | 0.134160 | 0.140702 | 0.121125 | 0.552640 | 0.883021 |
| Gemma TriviaQA | 0.219489 | 0.197273 | 0.214738 | 0.176408 | 0.711928 | 0.760785 |

Weighted interpretation:

- hidden-state conservative TCL-v0 has better Brier score in 5 of 6 runs
- hidden-state conservative TCL-v0 has better ECE in 4 of 6 runs
- hidden-state conservative TCL-v0 has better AUC in 5 of 6 runs
- raw-only calibration often wins high-confidence wrong-answer count by suppressing confidence
- Phi TriviaQA is a clear negative case where raw-only calibration is stronger

Unweighted raw-only calibration is an even stronger competitor for ECE and high-confidence wrong-answer counts. Under the unweighted sensitivity check, hidden-state TCL-v0 still shows stronger AUC in 5 of 6 runs, but raw-only baselines often win ECE.

The updated interpretation is:

```text
Hidden states may improve ranking/AUC and some calibration metrics beyond raw-only baselines, but high-confidence-error reduction is partly explained by ordinary post-hoc calibration.
```

## 8. Distance From Full TCL

TCL-v0 is not a direct validation of full TCL.

Full TCL proposes:

- joint training for next-token prediction and trust-score prediction
- an integrated trust head or architectural module
- a multi-dimensional trust vector
- trust dimensions beyond answer correctness, including reasoning validity, grounding/provenance, and epistemic uncertainty

TCL-v0 uses:

- frozen models
- one trust dimension: answer correctness
- short-answer QA benchmarks
- an external post-hoc probe
- no model training changes

Therefore, TCL-v0 can justify this:

```text
Hidden states in frozen LLMs contain calibration-relevant signal for answer correctness. This motivates the TCL hypothesis that joint token-and-trust training may be worth testing.
```

It cannot justify this:

```text
Full TCL is validated.
```

## 9. Limitations

Current limitations:

- TCL-v0 is still mostly short-answer QA correctness calibration
- raw-only baselines explain part of the high-confidence-error improvement
- no jointly trained trust head has been implemented
- only one trust dimension has been tested
- correctness labels can be noisy even after targeted review
- the hidden-state probe is high-dimensional relative to training size
- broader task families such as code, math reasoning, long-form factuality, and grounding are not yet tested
- the current codebase is still an experiment pipeline rather than a polished library

## 10. Claim Boundaries

Allowed claim:

```text
TCL-v0 provides reviewed preliminary evidence that frozen hidden states contain answer-correctness calibration signal under tested QA settings. After adding raw-only baselines, the strongest remaining evidence is competitive calibration and improved ranking/AUC in several runs, not high-confidence-error reduction alone.
```

Not allowed:

- TCL is validated
- TCL makes LLMs truthful
- TCL solves hallucination
- TCL-v0 is a general truth detector
- TCL-v0 validates the full TCL architecture
- TCL-v0 generalizes to all tasks, models, or domains
- the four-dimensional TCL trust vector has been implemented
- the system is ready for deployment or user-facing reliability decisions

## 11. Future Work

The next experiment should be a baseline-and-ablation checkpoint, not a scale-only run.

Minimum next design:

- keep raw, temperature scaling, Platt/logistic raw calibration, and isotonic raw calibration
- compare `min(raw, hidden_probe)` with `min(raw, raw_only_calibrated)`
- test hidden layer choice
- test pooling method
- compare raw-only, hidden-only, and raw-plus-hidden features
- preserve targeted manual-review rules
- include at least one cross-dataset or prompt-shift test

Only after that should larger 3,000-example runs be treated as useful evidence.

The path toward full TCL requires a later TCL-v1:

- fine-tune a model with a trust-score prediction head
- train with a joint token-and-trust objective
- add at least one second trust dimension, such as grounding/provenance
- compare the jointly trained model against raw-only calibration and frozen-probe TCL-v0

## 12. Related Work

Guo et al. studied calibration in modern neural networks and showed that temperature scaling is a strong post-hoc calibration method. TCL-v0 uses calibration metrics from this tradition but evaluates hidden-state probes in addition to output-probability calibration.

Kadavath et al. studied whether language models know what they know. TCL-v0 shares the interest in correctness awareness, but uses frozen hidden-state vectors and a small external probe.

Lin, Hilton, and Evans introduced TruthfulQA to evaluate whether models mimic human falsehoods. TCL-v0 does not directly evaluate truthfulness in that sense, but addresses the related confidence problem: estimating whether an answer should be trusted.

Kuhn, Gal, and Farquhar proposed semantic uncertainty for natural language generation. TCL-v0 differs by avoiding multiple sampled generations and instead probing the hidden state of a single generated answer.

## 13. References

- Guo, Pleiss, Sun, and Weinberger. "On Calibration of Modern Neural Networks." ICML 2017. https://arxiv.org/abs/1706.04599
- Kadavath et al. "Language Models (Mostly) Know What They Know." 2022. https://arxiv.org/abs/2207.05221
- Lin, Hilton, and Evans. "TruthfulQA: Measuring How Models Mimic Human Falsehoods." ACL 2022. https://arxiv.org/abs/2109.07958
- Kuhn, Gal, and Farquhar. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." 2023. https://arxiv.org/abs/2302.09664

## 14. Conclusion

TCL-v0 has moved from theory to a stricter empirical position. The project now has reviewed QA evidence, raw-only calibration baselines, and explicit scope boundaries.

The clean conclusion is:

```text
Frozen hidden states show evidence of containing answer-correctness calibration signal under tested QA settings, but TCL-v0 is empirical motivation for full TCL, not validation of full TCL.
```
