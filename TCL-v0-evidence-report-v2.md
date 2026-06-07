# TCL-v0 Evidence Report v2

Status: reviewed extended-validation evidence report, not full TCL validation

Generated from the June 7, 2026 checkpoint.

## Research Question

TCL-v0 asks a narrow empirical question:

```text
Can frozen LLM hidden states support better answer-correctness confidence than raw generation confidence?
```

This report does not validate the full Truth Calibration Layer. Full TCL remains a theory-stage framework. TCL-v0 only tests one practical path:

```text
frozen LLM hidden states -> small probe -> correctness/confidence score
```

## Current Position

The project has moved beyond pure theory. TCL-v0 now has reviewed preliminary evidence across multiple models and benchmarks.

Best current claim:

```text
Frozen hidden states appear to contain useful answer-correctness calibration signal under tested QA settings, especially for reducing high-confidence wrong answers.
```

The evidence supports continuing TCL-v0 research. It does not justify claiming full TCL validation, model truthfulness, hallucination solving, or deployment readiness.

## Method Summary

Each run records:

- question
- context when available
- accepted answers
- model answer
- automatic correctness label
- raw generation confidence
- final-layer hidden-state vector
- TCL-v0 probe confidence

Default configuration:

- hidden-state method: `answer_mean`
- probe: logistic regression
- baseline: raw generation confidence
- main TCL-v0 variant: conservative confidence

Conservative rule:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

This rule was chosen because the plain probe can become overconfident. The conservative score allows the probe to lower confidence, but prevents it from raising confidence above raw generation confidence.

## Formal Four-Run Gate

The predeclared extended-validation gate was completed with:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- SQuAD validation with context
- TriviaQA `rc.nocontext`
- 1,000 examples per run
- 200 held-out test examples per run
- targeted manual review completed for all four runs

Decision:

```text
mixed_continue_cautiously
```

Stopping-rule choice:

```text
continue TCL-v0 research
```

Interpretation:

Conservative TCL-v0 showed useful signal, but the formal gate was not strong enough to justify moving directly to TCL-v1.

## Formal Gate Results

### Qwen SQuAD-1000

Reviewed result: mixed.

```text
raw brier: 0.189956
conservative brier: 0.195644
raw ece: 0.119559
conservative ece: 0.152570
raw wrong >= 0.8: 30
conservative wrong >= 0.8: 10
raw wrong >= 0.9: 11
conservative wrong >= 0.9: 2
```

Conservative TCL-v0 reduced high-confidence wrong answers, but raw confidence had better Brier score and ECE.

### Qwen TriviaQA-1000

Reviewed result: strong support.

```text
raw brier: 0.255366
conservative brier: 0.131989
raw ece: 0.344028
conservative ece: 0.099740
raw wrong >= 0.8: 9
conservative wrong >= 0.8: 1
raw wrong >= 0.9: 3
conservative wrong >= 0.9: 0
```

### Phi SQuAD-1000

Reviewed result: mixed.

```text
raw brier: 0.029361
conservative brier: 0.046439
raw ece: 0.009211
conservative ece: 0.040488
raw wrong >= 0.8: 5
conservative wrong >= 0.8: 4
raw wrong >= 0.9: 3
conservative wrong >= 0.9: 2
```

Conservative TCL-v0 reduced high-confidence wrong answers, but raw confidence had better Brier score and ECE.

### Phi TriviaQA-1000

Reviewed result: strong support.

```text
raw brier: 0.278213
conservative brier: 0.198426
raw ece: 0.296857
conservative ece: 0.152008
raw wrong >= 0.8: 42
conservative wrong >= 0.8: 9
raw wrong >= 0.9: 17
conservative wrong >= 0.9: 3
```

## Gemma Exploratory Extension

After the formal gate, Gemma 2B-it was added as an exploratory third model using Kaggle's local model path:

```text
/kaggle/input/models/google/gemma/transformers/2b-it/3
```

Gemma required a notebook-local prompt patch because its chat template rejected system-role messages. The patch retried with a user-only chat message.

The Gemma extension is not a replacement for the formal gate. It is additional exploratory evidence.

Exploratory six-run decision:

```text
supports_continuing_tcl_v0
```

### Gemma SQuAD-1000

Reviewed result: strong support.

```text
raw accuracy@0.5: 0.66
conservative accuracy@0.5: 0.835
raw brier: 0.281062
conservative brier: 0.134160
raw ece: 0.238979
conservative ece: 0.121125
raw wrong >= 0.8: 62
conservative wrong >= 0.8: 10
raw wrong >= 0.9: 33
conservative wrong >= 0.9: 5
raw auc: 0.534425
conservative auc: 0.883021
```

### Gemma TriviaQA-1000

Reviewed result: strong support.

```text
raw accuracy@0.5: 0.26
conservative accuracy@0.5: 0.755
raw brier: 0.473393
conservative brier: 0.197273
raw ece: 0.546427
conservative ece: 0.176408
raw wrong >= 0.8: 76
conservative wrong >= 0.8: 10
raw wrong >= 0.9: 19
conservative wrong >= 0.9: 2
raw auc: 0.711928
conservative auc: 0.760785
```

## Main Pattern

The strongest repeated pattern is:

```text
conservative TCL-v0 reduces high-confidence wrong answers.
```

This is the most important result for the TCL idea because dangerous overconfidence is closer to the central problem than ordinary accuracy alone.

The pattern is strongest on:

- TriviaQA
- Gemma 2B-it
- high-confidence wrong-answer counts

The pattern is mixed on:

- SQuAD for Qwen and Phi
- Brier/ECE when raw confidence is already strong

## Interpretation

The results support the idea that hidden states carry correctness-relevant signal. They also show why TCL-v0 must be constrained:

- plain probe confidence can help but may become overconfident
- calibrated confidence can reduce high-confidence wrong answers but can harm other metrics
- conservative confidence is the most stable variant so far

This supports continuing TCL-v0 research, not jumping to TCL-v1.

## Claim Boundaries

Allowed:

- TCL is a theory-stage framework.
- TCL-v0 is an early empirical diagnostic.
- Reviewed TCL-v0 evidence suggests hidden states can improve confidence calibration under tested QA conditions.
- Conservative TCL-v0 is currently the strongest practical variant.
- The clearest benefit is reducing high-confidence wrong answers.

Not allowed:

- TCL is validated.
- TCL makes models truthful.
- TCL solves hallucination.
- TCL-v0 generalizes to all tasks or models.
- The full four-dimensional TCL trust vector has been implemented.
- The system is deployment-ready.

## Recommended Next Step

Stop running new experiments until this checkpoint is frozen and summarized.

Immediate work:

- update README and result summaries
- keep this report as the current empirical checkpoint
- package the project state
- optionally commit the checkpoint

Next experiment, only after documentation is clean:

```text
Gemma 2B-it on TriviaQA-3000
same split rule
same answer_mean hidden-state method
same conservative TCL-v0 comparison
same targeted manual-review rules
```

Reason:

TriviaQA is where TCL-v0 repeatedly shows the clearest practical value, and Gemma produced the strongest reviewed exploratory signal.

## Bottom Line

TCL-v0 is now a real empirical foothold for the broader TCL theory. The evidence is preliminary, but it is no longer just a theoretical sketch.

Best final wording:

```text
TCL remains a theory-stage framework, but TCL-v0 now has reviewed preliminary evidence across multiple models and QA benchmarks that frozen hidden-state probes can reduce dangerous overconfidence in answer correctness estimates.
```
