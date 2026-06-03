# TCL-v0 Experiment Plan

Status: living protocol, updated after first diagnostics

This document defines the first practical experiment for the Truth Calibration Layer (TCL) idea. TCL remains theory-stage as a full framework. TCL-v0 is an early confidence-only diagnostic implementation used to test whether frozen hidden states can support correctness calibration. Results should be reported only within the exact tested conditions.

## 1. Purpose

TCL-v0 tests one narrow question:

Can frozen LLM hidden states predict answer correctness better than raw generation confidence?

The goal is not to build the full four-dimensional TCL trust vector yet. TCL-v0 only studies the first dimension:

```text
frozen LLM hidden state -> small probe -> correctness/confidence score
```

Current best diagnostic variant after the first benchmark run:

```text
conservative TCL-v0 confidence = min(raw_generation_confidence, probe_confidence)
```

This conservative variant prevents the hidden-state probe from increasing confidence above the model's raw generation confidence.

## 2. Research Hypothesis

Primary hypothesis:

Frozen hidden states from a language model contain a learnable signal about factual answer correctness that can produce better-calibrated confidence estimates than raw token-probability confidence.

Operational version:

Given factual QA examples, a lightweight probe trained on frozen hidden-state vectors will achieve lower Expected Calibration Error (ECE) and/or lower Brier score than raw generation confidence on held-out examples.

## 3. What TCL-v0 Is

TCL-v0 is:

- A confidence-only probe.
- Trained on frozen hidden states.
- Evaluated against raw generation confidence.
- A first empirical test of the TCL research hypothesis.
- Currently strongest as a conservative confidence rule that combines raw confidence and probe confidence.

TCL-v0 is not:

- A full TCL implementation.
- A four-dimensional trust-vector system.
- A truth detector.
- Evidence that the model has become truthful.
- A replacement for retrieval, citation checking, or human review.

## 4. Base Model Scope

The base LLM must stay frozen. TCL-v0 should not update base-model weights.

Recommended stages:

1. Smoke-test model: a very small model only to validate scripts and output format.
2. First meaningful model: a stronger instruction-tuned model that can answer factual QA with nontrivial accuracy.
3. Later comparison: repeat with a second model only after the first model pipeline is stable.

The smoke test must not be treated as evidence for or against TCL.

## 5. Dataset Scope

Starter data:

- The current `tcl_experiments/data/seed_questions.csv` file is only for pipeline validation.
- It has too few examples for research claims.

First meaningful dataset:

- Use a factual QA dataset with accepted answers, such as TriviaQA or Natural Questions.
- Start with a manageable subset before scaling.
- Keep train, validation, and test splits separate.

Recommended first evidence run:

- At least several hundred examples for a preliminary result.
- Prefer 1,000+ examples before treating calibration metrics as meaningful.
- Preserve a held-out test split that is not used for probe selection.

## 6. Record Schema

Each example should produce one record with these fields:

```text
id
dataset
question
accepted_answers
model_name
prompt_template
model_answer
correctness_label
correctness_method
raw_generation_confidence
raw_confidence_method
hidden_state_layer
hidden_state_method
hidden_state_vector
tcl_v0_confidence
tcl_v0_conservative_confidence
split
run_id
timestamp
```

Required notes:

- `correctness_label` should be binary for TCL-v0: 1 correct, 0 incorrect.
- `correctness_method` should record whether the label came from exact match, substring match, normalized match, dataset evaluator, or manual review.
- `raw_confidence_method` should be explicit, for example geometric mean generated-token probability.
- `hidden_state_method` should be explicit, because hidden-state pooling choices are part of the experiment.
- `tcl_v0_conservative_confidence` should be recorded when the conservative rule is evaluated.

## 7. Correctness Labeling

Initial automatic labeling:

- Normalize case.
- Strip punctuation.
- Collapse whitespace.
- Accept exact match or accepted-answer substring match.

Known weakness:

Automatic string matching can mislabel paraphrases, partial answers, or answers with extra explanation.

Plan:

- Use automatic labels for the first run.
- Add a manual review pass for borderline or surprising examples.
- Keep the labeling method recorded in every example.

## 8. Raw Confidence Baseline

The first baseline is raw generation confidence.

Recommended v0 definition:

```text
raw_generation_confidence = geometric mean probability of generated answer tokens
```

Reason:

Using the geometric mean avoids unfairly penalizing longer answers as strongly as multiplying token probabilities directly.

Caution:

This is a generation-confidence proxy, not a true probability that the answer is factually correct.

## 9. Hidden-State Extraction

TCL-v0 should compare multiple hidden-state extraction methods before locking one in.

Recommended variants:

1. Final generated-token hidden state.
2. Mean-pooled generated-answer hidden states.
3. Mean-pooled prompt-plus-answer hidden states.

Preferred starting point:

Mean-pooled generated-answer hidden states, because TCL-v0 is trying to predict correctness of the produced answer rather than the prompt alone.

Important:

The chosen layer must be recorded. Start with the final transformer layer, then optionally test earlier layers later.

## 10. Probe Model

Start simple:

- Standardize hidden-state vectors.
- Train logistic regression.
- Output probability of correctness.

Why logistic regression first:

- It is easy to interpret.
- It reduces overfitting risk.
- It keeps the experiment focused on whether the hidden state already contains signal.

Later options:

- Small MLP probe.
- Layer-wise probe comparison.
- Probe calibration with temperature scaling or isotonic regression.

These should come only after the logistic-regression baseline is stable.

## 11. Splitting Strategy

Use fixed splits:

- Train: fit the TCL-v0 probe.
- Validation: tune probe settings or choose hidden-state method.
- Test: final held-out evaluation.

Avoid:

- Choosing the best method based on test metrics.
- Reporting test-set improvements after repeated test probing.

For small early runs, use a simple train/test split only as a debugging step.

## 12. Metrics

Primary metrics:

- Expected Calibration Error (ECE).
- Brier score.

Secondary metrics:

- Maximum Calibration Error (MCE).
- Reliability bins.
- Accuracy at threshold 0.5.
- AUC, if both classes are present in the test split.

Report separately:

- Raw generation confidence metrics.
- TCL-v0 probe confidence metrics.
- Conservative TCL-v0 confidence metrics.
- Calibrated TCL-v0 confidence metrics, when calibration is used.

Do not report a single headline number without the reliability bins.
Always report high-confidence error counts, especially wrong examples with confidence greater than or equal to 0.8 and 0.9.

## 13. Success Criteria

TCL-v0 shows preliminary support for the hypothesis if:

- TCL-v0 has lower ECE than raw generation confidence on held-out examples.
- TCL-v0 has lower Brier score than raw generation confidence on held-out examples.
- Reliability bins show better alignment between confidence and accuracy.
- High-confidence false positives do not increase relative to raw confidence, or are explicitly bounded by a conservative confidence rule.
- The improvement is not explained by a tiny or highly imbalanced test set.

TCL-v0 does not support the hypothesis if:

- Raw generation confidence is equally or better calibrated.
- TCL-v0 overfits the training set and fails on held-out data.
- TCL-v0 improves average ECE but creates severe high-confidence false positives.
- The result depends on a labeling artifact or dataset shortcut.

## 14. Threats to Validity

Main risks:

- The probe may learn dataset-specific answer patterns instead of general reliability.
- The correctness labels may be noisy.
- Raw confidence may be measured in a way that is too weak or unfair.
- The hidden-state vector may include answer wording artifacts rather than correctness signal.
- Small datasets can produce unstable calibration metrics.

Mitigations:

- Use held-out test data.
- Keep raw confidence definitions explicit.
- Compare multiple hidden-state extraction methods.
- Inspect reliability bins, not only scalar metrics.
- Manually review a sample of labels.
- Repeat on another dataset only after the first protocol is stable.

## 15. Claim Boundaries

Allowed future claim if results support it:

TCL-v0 provides preliminary evidence that frozen hidden states can improve confidence calibration for factual QA under the tested conditions.

Current cautious diagnostic claim from the first 200-example TriviaQA run:

Conservative TCL-v0 improved ECE, Brier score, and threshold accuracy on the tested TriviaQA subset while eliminating test errors above 0.8 confidence, but this does not yet establish generalization.

Not allowed:

- TCL makes LLMs truthful.
- TCL detects truth in general.
- TCL solves hallucination.
- TCL has been validated before the experiment is actually run.
- TCL-v0 proves the full four-dimensional TCL framework.

## 16. Recommended Next Implementation Changes

Completed implementation changes:

1. Raw confidence metadata is explicit.
2. Multiple hidden-state extraction methods are supported.
3. Dataset, split, run ID, timestamp, hidden-state layer, and correctness method are recorded.
4. Config metadata is saved for each run.
5. Smoke-test, diagnostic, and benchmark runs are separated.
6. Conservative TCL-v0 confidence is evaluated.
7. High-confidence error counts are included.

Next implementation/research change:

Run the conservative variant on a larger TriviaQA subset only after manual review of the current 200-example benchmark labels and high-confidence failures.
