#!/usr/bin/env bash
set -euo pipefail

# Kaggle/Colab smoke runner for TCL-v0 ablations.
#
# Required environment variables:
#   MODEL       Model id or local model path.
#   QUESTIONS   CSV with benchmark questions.
#   SPLITS      CSV with id,split columns.
#   DATASET     Dataset label to write into records.
#
# Optional:
#   RUN_PREFIX  Output prefix under runs/.
#   LIMIT       Number of examples for smoke test, default 100.
#   DEVICE      auto/cuda/cpu, default auto.
#   DTYPE       auto/float16/bfloat16/float32, default auto.

MODEL="${MODEL:?Set MODEL to a Hugging Face model id or Kaggle local model path.}"
QUESTIONS="${QUESTIONS:?Set QUESTIONS to the benchmark CSV path.}"
SPLITS="${SPLITS:?Set SPLITS to the split CSV path.}"
DATASET="${DATASET:?Set DATASET to the dataset label.}"
RUN_PREFIX="${RUN_PREFIX:-ablation-smoke}"
LIMIT="${LIMIT:-100}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

METHODS=("answer_mean" "answer_last" "prompt_answer_mean")
LAYERS=("early_middle" "middle" "final")
RECORDS=()

for method in "${METHODS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    run_id="${RUN_PREFIX}-${method}-${layer}"
    run_dir="runs/${run_id}"
    records="${run_dir}/records_${method}.jsonl"
    echo "=== Running ${run_id} ==="
    python scripts/run_inference.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --questions "${QUESTIONS}" \
      --splits "${SPLITS}" \
      --out "${records}" \
      --run-id "${run_id}" \
      --limit "${LIMIT}" \
      --max-new-tokens 24 \
      --hidden-state-method "${method}" \
      --hidden-state-layer-position "${layer}" \
      --chat-template-mode auto \
      --device "${DEVICE}" \
      --torch-dtype "${DTYPE}"

    python scripts/train_probe.py \
      --records "${records}" \
      --out-dir "${run_dir}/analysis" \
      --split-mode record \
      --hidden-state-method "${method}" \
      --calibrate

    python scripts/run_raw_calibration_baselines.py \
      --run-dir "${run_dir}" \
      --method "${method}" \
      --records "${records}" \
      --predictions "${run_dir}/analysis/${method}/test_predictions.csv" \
      --out-dir "${run_dir}/baseline_calibration"

    RECORDS+=("${records}")
  done
done

python scripts/run_ablation_analysis.py \
  --records "${RECORDS[@]}" \
  --out-dir "runs/${RUN_PREFIX}-ablation_summary" \
  --split-mode record

echo "Completed ablation smoke run."
echo "Save runs/${RUN_PREFIX}-ablation_summary/ and all runs/${RUN_PREFIX}-*/ folders before closing the notebook."
