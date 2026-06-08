# TCL-v0 Roadmap

Status: next-step roadmap after the June 2026 reviewed extended-validation checkpoint.

This roadmap is intentionally conservative. TCL-v0 has promising QA calibration signals, especially for reducing high-confidence wrong answers, but it is not a general truth detector and is not deployment-ready.

## Current Position

TCL-v0 currently means:

```text
frozen LLM hidden states -> small probe -> answer-correctness confidence score
```

The best supported claim is:

```text
Frozen hidden states appear to contain useful answer-correctness calibration signal under tested QA settings, especially for reducing high-confidence wrong answers.
```

The current evidence supports continuing TCL-v0 research. It does not support claiming that full TCL is validated, that LLMs become truthful, or that hallucination is solved.

## Milestone 1: Strengthen Baselines

Goal: show whether hidden-state probing adds value beyond standard calibration.

Add baselines:

- raw generation confidence
- temperature scaling on raw confidence or logits
- Platt scaling / logistic calibration on raw confidence
- isotonic regression
- learned fusion of raw confidence and probe confidence
- conservative fusion: `min(raw_generation_confidence, tcl_v0_probe_confidence)`

Critical first comparison:

```text
min(raw_generation_confidence, hidden_state_probe_confidence)
vs
min(raw_generation_confidence, raw_only_calibrated_confidence)
```

If the raw-only conservative baseline matches hidden-state conservative TCL-v0, the hidden state is not yet carrying the claimed extra load. In that case the result should be framed as ordinary post-hoc calibration, not hidden-state truth calibration.

Decision rule:

```text
TCL-v0 is stronger only if hidden-state methods improve high-confidence error behavior or calibration metrics beyond these simpler baselines.
```

## Milestone 2: Run Ablations

Goal: understand what part of TCL-v0 is doing the work.

Planned ablations:

- hidden layer choice: final layer vs earlier layers vs several layers
- hidden-state pooling: answer mean vs last token vs question mean vs context mean
- probe type: logistic regression vs small MLP
- feature set: raw confidence only vs hidden states only vs raw confidence plus hidden states
- fusion rule: conservative min vs learned fusion vs calibrated probe only

Primary reporting metrics:

- ECE
- MCE
- Brier score
- AUC
- reliability bins
- wrong answers above confidence thresholds 0.8 and 0.9

## Milestone 3: Test Generalization

Goal: check whether the signal survives outside the easiest same-distribution setting.

Priority tests:

- train on SQuAD, test on TriviaQA-style data
- train on TriviaQA, test on another short-answer QA dataset
- train on one model, evaluate transfer behavior on another model where feasible
- prompt distribution shift: same questions with different prompt formats
- context shift: with context vs no context

Expected result:

```text
Some degradation is expected. The goal is to measure where TCL-v0 breaks, not to hide failures.
```

## Milestone 4: Add Non-QA Tasks

Goal: stop overfitting the research story to short-answer QA.

Candidate task families:

- math or symbolic reasoning with automatically checkable answers
- code generation with unit-test labels
- short factual generation with external answer matching
- multi-sentence answers with human or assisted review

Do not claim broad truth calibration until at least one non-QA task family has been tested.

## Milestone 5: Clean The API

Goal: move from research-log scripts toward a reusable prototype.

Target API shape:

```python
calibrator = TCLCalibrator(
    model_name="...",
    hidden_state_method="answer_mean",
    fusion="conservative_min",
)

calibrator.fit(train_records)
scores = calibrator.score(eval_records)
```

Expected outputs:

- raw confidence
- probe confidence
- conservative confidence
- calibrated confidence
- correctness prediction
- reliability summary
- high-confidence error report

This does not need to become a full package immediately, but the code should make the core method easy to rerun without reading every experiment script.

## Milestone 6: Document Negative Results

Goal: make the research more trustworthy.

Every major report should include:

- where raw confidence beats TCL-v0
- where TCL-v0 hurts Brier score or ECE
- where the probe becomes overconfident
- where correctness labels are noisy
- where dataset/task choice makes the result too easy or too sparse

Negative results are part of the contribution because they define the boundary of the method.

## Naming And Framing

Keep the project name for now, but frame it carefully:

```text
TCL-v0 is a confidence calibration prototype for answer correctness in bounded benchmark settings.
```

Do not frame TCL-v0 as:

- a general truth detector
- a hallucination solution
- a factuality verifier for open-ended generation
- a safety layer for deployment
- proof that full TCL works

If the name becomes a repeated source of confusion, consider reserving "Truth Calibration Layer" for the broader theory and describing the current prototype as:

```text
Confidence Calibration Layer for QA Correctness
```

## Recommended Next Experiment

Before scaling to larger data, run a baseline-and-ablation checkpoint on the strongest current setting:

```text
model: Gemma 2B-it
benchmark: TriviaQA rc.nocontext
size: 1,000 examples first, then 3,000 only if the baseline checkpoint is useful
compare: raw, temperature scaling, Platt/logistic raw calibration, isotonic, hidden-state probe, conservative min
manual review: preserve targeted manual-review rules
```

The next useful result is not simply a bigger run. The next useful result is knowing whether hidden-state TCL-v0 still matters after stronger baselines are included.
