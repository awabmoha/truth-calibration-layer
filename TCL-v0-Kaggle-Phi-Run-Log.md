# TCL-v0 Kaggle Phi Run Log

Status: active second-family model validation path

This note documents the manual Kaggle work started on 2026-06-07 for the next TCL-v0 extended-validation step. It records what happened before any stronger claim is made.

## Purpose

The Qwen 0.5B SQuAD-1000 and TriviaQA-1000 runs are complete and manually reviewed, but the extended-validation matrix is still incomplete because it needs another model beyond Qwen. The current path tests a stronger non-Qwen model before deciding whether TCL-v0 support generalizes beyond one model family.

## Kaggle Runtime

Observed runtime:

```text
python: 3.12.13
torch: 2.10.0+cu128
transformers: 5.0.0
cuda_available: true
cuda_device_count: 2
cuda_device_name: Tesla T4
cuda_total_memory_gb: 14.56
```

Kaggle showed dependency warnings around CUDA-related packages during install, but CUDA was available and model inference ran.

## Gated Model Attempts

Preferred stronger model:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Result: blocked by Hugging Face gated repository access. The access request was submitted and was awaiting review.

Update: the Hugging Face gated repository status later showed `REJECTED` for Meta's Llama 3.2 language models and evals gating group. Llama 3.2 is therefore not available for this validation path unless access changes later.

Second attempted gated option:

```text
google/gemma-2-2b-it
```

Result: blocked by Hugging Face gated repository access.

## Active Second-Family Model

Current practical second-family model:

```text
microsoft/Phi-3.5-mini-instruct
```

Rationale: stronger than the tiny fallback options, non-Qwen family, accessible without waiting for Llama/Gemma approval, and able to load on Kaggle T4.

## Completed Smoke Tests

### SQuAD-50

Command:

```bash
!bash scripts/run_free_gpu_squad_dry_run.sh 50 microsoft/Phi-3.5-mini-instruct
```

Outcome:

- model loaded successfully
- 50 records were generated
- probe training failed because the tiny test split contained only one label class

Interpretation: load/inference smoke test only, not usable evidence.

### SQuAD-200

Command:

```bash
!bash scripts/run_free_gpu_squad_dry_run.sh 200 microsoft/Phi-3.5-mini-instruct
```

Outcome:

- model loaded successfully
- 200 records were generated
- probe metrics were produced
- targeted manual review candidates were produced
- artifact verification failed because calibrated confidence outputs were unavailable

Observed smoke metrics on 40 held-out test examples:

```text
raw_generation_confidence:
  accuracy_at_0_5: 0.925
  ece: 0.05401671373019516
  brier: 0.06217850236699719
  wrong_conf_ge_0_8: 2
  wrong_conf_ge_0_9: 2
  auc: 0.8018018018018018

tcl_v0_conservative_confidence:
  accuracy_at_0_5: 0.875
  ece: 0.10174607023538548
  brier: 0.11037850939889857
  wrong_conf_ge_0_8: 2
  wrong_conf_ge_0_9: 2
  auc: 0.7837837837837839
```

Interpretation: this is a smoke test, not a decision run. On the small SQuAD-200 split, raw confidence beat conservative TCL-v0 on most available metrics. The result should be recorded, not hidden, but it should not be overinterpreted because the test split is only 40 examples and calibrated confidence was unavailable.

## Next Planned Commands

### SQuAD-1000

```bash
!bash scripts/run_free_gpu_squad_dry_run.sh 1000 microsoft/Phi-3.5-mini-instruct
```

Outcome: completed and passed artifact verification.

Run id:

```text
freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean
```

Artifact:

```text
runs/freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean_artifact.zip
```

Local import:

```text
tcl_experiments/runs/freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean
```

Local verification: passed with 1000 records, 200 test predictions, manual review CSV, targeted review CSV, calibrated confidence outputs, reliability bins, and metrics present.

Observed pre-review test metrics on 200 held-out examples:

```text
raw_generation_confidence:
  accuracy_at_0_5: 0.93
  ece: 0.03832357490119614
  brier: 0.0539940631298646
  wrong_conf_ge_0_8: 11
  wrong_conf_ge_0_9: 4
  auc: 0.8298771121351766

tcl_v0_conservative_confidence:
  accuracy_at_0_5: 0.935
  ece: 0.03409268829848641
  brier: 0.05153021695747673
  wrong_conf_ge_0_8: 7
  wrong_conf_ge_0_9: 3
  auc: 0.8767281105990783
```

Interpretation: this is a successful second-family SQuAD-1000 run before targeted manual review. Conservative TCL-v0 beat raw generation confidence on all seven reported comparison metrics in artifact verification. This should still be treated as pre-review evidence until the targeted manual review candidates are checked and reviewed-label metrics are recomputed.

Post-review update:

- targeted review candidates: 17
- reviewed candidates: 17
- label changes: 7
- aggregate interpretation changes: 7

Reviewed-label result: conservative TCL-v0 became mixed on SQuAD. It reduced high-confidence wrong-answer counts, but raw confidence had better Brier score and ECE after review.

```text
raw_generation_confidence:
  ece: 0.009210628035142825
  brier: 0.029360529394794034
  wrong_conf_ge_0_8: 5
  wrong_conf_ge_0_9: 3
  auc: 0.7054034048852702

tcl_v0_conservative_confidence:
  ece: 0.040488235736664976
  brier: 0.04643921703906897
  wrong_conf_ge_0_8: 4
  wrong_conf_ge_0_9: 2
  auc: 0.8068097705403405
```

### TriviaQA-1000

Only after the SQuAD-1000 artifact is saved, run TriviaQA-1000:

```bash
!bash scripts/run_free_gpu_triviaqa_dry_run.sh 1000 microsoft/Phi-3.5-mini-instruct
```

Outcome: completed and passed artifact verification.

Run id:

```text
freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean
```

Artifact:

```text
runs/freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean_artifact.zip
```

Local import:

```text
tcl_experiments/runs/freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean
```

Local verification: passed with 1000 records, 200 test predictions, manual review CSV, targeted review CSV, calibrated confidence outputs, reliability bins, and metrics present.

Observed pre-review test metrics on 200 held-out examples:

```text
raw_generation_confidence:
  accuracy_at_0_5: 0.565
  ece: 0.3168565231878469
  brier: 0.2891792605167781
  wrong_conf_ge_0_8: 44
  wrong_conf_ge_0_9: 17
  auc: 0.8514492753623188

tcl_v0_conservative_confidence:
  accuracy_at_0_5: 0.73
  ece: 0.15765352055025275
  brier: 0.20116561737108754
  wrong_conf_ge_0_8: 11
  wrong_conf_ge_0_9: 3
  auc: 0.8306159420289855

tcl_v0_calibrated_confidence:
  accuracy_at_0_5: 0.74
  ece: 0.04466933187255227
  brier: 0.17910571633745523
  wrong_conf_ge_0_8: 4
  wrong_conf_ge_0_9: 1
  auc: 0.8065619967793881
```

Interpretation: this is a successful second-family TriviaQA-1000 run before targeted manual review. Conservative TCL-v0 substantially reduced Brier score, ECE, and high-confidence wrong counts versus raw confidence, while raw confidence still had better AUC and MCE. Calibrated TCL-v0 was strongest on ECE, Brier, MCE, and high-confidence wrong counts, but it also reduced AUC versus raw. These results remain pre-review until targeted manual review is completed and reviewed-label metrics are recomputed.

Post-review update:

- targeted review candidates: 116
- reviewed candidates: 116
- label changes: 6
- aggregate interpretation changes: 6

Reviewed-label result: conservative TCL-v0 remained strongly useful on TriviaQA for Brier score, ECE, and high-confidence wrong-answer counts, while raw confidence still had better AUC and MCE.

```text
raw_generation_confidence:
  ece: 0.29685652318784694
  brier: 0.27821288223641305
  wrong_conf_ge_0_8: 42
  wrong_conf_ge_0_9: 17
  auc: 0.8375608766233766

tcl_v0_conservative_confidence:
  ece: 0.152007635696681
  brier: 0.19842573260897103
  wrong_conf_ge_0_8: 9
  wrong_conf_ge_0_9: 3
  auc: 0.828226461038961
```

## Extended Validation Gate Result

After adding Phi-3.5 Mini as the second model and completing targeted manual review for all four 1000-example runs, the extended-validation matrix is complete:

```text
models: Qwen/Qwen2.5-0.5B-Instruct, microsoft/Phi-3.5-mini-instruct
benchmarks: SQuAD, TriviaQA
runs: 4
manual-review completed runs: 4
```

Final gate decision:

```text
mixed_continue_cautiously
```

Required stopping-rule choice:

```text
continue TCL-v0 research
```

Interpretation: TCL-v0 shows useful signal, especially on open-domain TriviaQA and high-confidence wrong-answer reduction, but the mixed SQuAD results are not strong enough to justify moving directly to TCL-v1.

## Claim Boundary

This log concerns TCL-v0 only: frozen LLM hidden states, a small probe, and confidence calibration for answer correctness. It does not validate the full Truth Calibration Layer theory, does not prove model truthfulness, and does not establish deployment readiness.
