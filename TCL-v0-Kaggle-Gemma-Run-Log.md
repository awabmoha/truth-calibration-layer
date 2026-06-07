# TCL-v0 Kaggle Gemma Run Log

Status: active third-model exploratory validation path

This note documents the Kaggle-hosted Gemma path started on 2026-06-07. Gemma is not part of the original two-model stopping gate, which is already complete, but it is useful as an additional model-family stress test.

## Access Path

Hugging Face access to `google/gemma-2-2b-it` was still awaiting repository approval, even after a Hugging Face login token was added to Kaggle. Kaggle model consent was available, so the run used the Kaggle-hosted local model instead:

```text
/kaggle/input/models/google/gemma/transformers/2b-it/3
```

## Notebook Patch

Gemma loaded successfully, but the first run failed with:

```text
jinja2.exceptions.TemplateError: System role not supported
```

Cause: Gemma's chat template rejected the `system` role used by `scripts/run_inference.py`.

Temporary Kaggle-notebook patch: wrap `tokenizer.apply_chat_template(...)` in a try/except and retry with a user-only message when the error contains `System role not supported`.

This patch was applied only in the Kaggle notebook copy before the Gemma runs.

## SQuAD-50 Smoke Test

Command:

```bash
!bash scripts/run_free_gpu_squad_dry_run.sh 50 /kaggle/input/models/google/gemma/transformers/2b-it/3
```

Outcome: completed. This confirmed that the local Kaggle Gemma path and user-only chat fallback worked.

## SQuAD-1000

Command:

```bash
!bash scripts/run_free_gpu_squad_dry_run.sh 1000 /kaggle/input/models/google/gemma/transformers/2b-it/3
```

Outcome: completed and passed local artifact verification.

Run id:

```text
freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean
```

Local import:

```text
tcl_experiments/runs/freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean
```

Observed pre-review metrics on 200 held-out examples:

```text
raw_generation_confidence:
  accuracy_at_0_5: 0.65
  ece: 0.248978631084034
  brier: 0.2909880753764364
  wrong_conf_ge_0_8: 64
  wrong_conf_ge_0_9: 35
  auc: 0.5125274725274725

tcl_v0_conservative_confidence:
  accuracy_at_0_5: 0.835
  ece: 0.12112506752930503
  brier: 0.13385631967675657
  wrong_conf_ge_0_8: 11
  wrong_conf_ge_0_9: 6
  auc: 0.8757142857142857

tcl_v0_calibrated_confidence:
  accuracy_at_0_5: 0.8
  ece: 0.11713384851549678
  brier: 0.12345447759685932
  wrong_conf_ge_0_8: 5
  wrong_conf_ge_0_9: 1
  auc: 0.9018681318681319
```

Pre-review interpretation: Gemma SQuAD-1000 is a strong exploratory result for TCL-v0. Conservative TCL-v0 beat raw confidence on six of seven reported metrics, with raw confidence only better on MCE. However, targeted manual review has not yet been completed, so this is not a final reviewed-label result.

Post-review update:

- targeted review candidates: 86
- reviewed candidates: 86
- label changes: 2
- aggregate interpretation changes: 2

Reviewed-label result: conservative TCL-v0 remained strong on SQuAD. It beat raw confidence on accuracy, AUC, Brier score, ECE, and high-confidence wrong-answer counts; raw confidence remained better on MCE.

```text
raw_generation_confidence:
  accuracy_at_0_5: 0.66
  ece: 0.23897863108403392
  brier: 0.2810621643240196
  wrong_conf_ge_0_8: 62
  wrong_conf_ge_0_9: 33
  auc: 0.5344251336898396

tcl_v0_conservative_confidence:
  accuracy_at_0_5: 0.835
  ece: 0.12112506752930503
  brier: 0.13415980894089724
  wrong_conf_ge_0_8: 10
  wrong_conf_ge_0_9: 5
  auc: 0.8830213903743316
```

## TriviaQA-1000

Command:

```bash
!bash scripts/run_free_gpu_triviaqa_dry_run.sh 1000 /kaggle/input/models/google/gemma/transformers/2b-it/3
```

Outcome: completed and passed local artifact verification.

Run id:

```text
freegpu-triviaqa1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean
```

Local import:

```text
tcl_experiments/runs/freegpu-triviaqa1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean
```

Observed pre-review metrics on 200 held-out examples:

```text
raw_generation_confidence:
  accuracy_at_0_5: 0.23
  ece: 0.5764265596376102
  brier: 0.4922919812216266
  wrong_conf_ge_0_8: 80
  wrong_conf_ge_0_9: 19
  auc: 0.7291078486730661

tcl_v0_conservative_confidence:
  accuracy_at_0_5: 0.755
  ece: 0.17501330168157467
  brier: 0.18839819782165576
  wrong_conf_ge_0_8: 10
  wrong_conf_ge_0_9: 2
  auc: 0.7708921513269339

tcl_v0_calibrated_confidence:
  accuracy_at_0_5: 0.735
  ece: 0.21638931860110158
  brier: 0.1997225038056823
  wrong_conf_ge_0_8: 6
  wrong_conf_ge_0_9: 0
  auc: 0.7636928289102203
```

Pre-review interpretation: Gemma TriviaQA-1000 is also a strong exploratory result for TCL-v0. Conservative TCL-v0 beat raw confidence on six of seven reported metrics, with raw confidence only better on MCE. Calibrated TCL-v0 beat raw confidence on all seven reported metrics. Targeted manual review has not yet been completed, so this remains pre-review evidence.

Post-review update:

- targeted review candidates: 175
- reviewed candidates: 175
- label changes: 6
- aggregate interpretation changes: 6

Reviewed-label result: conservative TCL-v0 remained strong on TriviaQA. It beat raw confidence on accuracy, AUC, Brier score, ECE, and high-confidence wrong-answer counts; raw confidence remained better on MCE. Calibrated TCL-v0 kept the strongest high-confidence wrong-answer reduction.

```text
raw_generation_confidence:
  accuracy_at_0_5: 0.26
  ece: 0.5464265596376101
  brier: 0.4733927770684538
  wrong_conf_ge_0_8: 76
  wrong_conf_ge_0_9: 19
  auc: 0.7119282744282744

tcl_v0_conservative_confidence:
  accuracy_at_0_5: 0.755
  ece: 0.17640799198834387
  brier: 0.1972734148114247
  wrong_conf_ge_0_8: 10
  wrong_conf_ge_0_9: 2
  auc: 0.7607848232848233
```

## Next Step

The two Gemma runs should be treated as an additional exploratory third-model validation path, not as a replacement for the already completed two-model stopping gate. They strengthen the case for continuing TCL-v0 research, but they still do not validate full TCL or justify deployment claims.

## Claim Boundary

This log concerns TCL-v0 only: frozen LLM hidden states, a small probe, and confidence calibration for answer correctness. It does not validate the full Truth Calibration Layer theory, does not prove model truthfulness, and does not establish deployment readiness.
