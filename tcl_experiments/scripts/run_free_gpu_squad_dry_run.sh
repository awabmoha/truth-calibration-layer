#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-200}"
MODEL="${2:-Qwen/Qwen2.5-0.5B-Instruct}"
SAFE_MODEL_NAME="$(echo "${MODEL}" | tr '/:' '__')"
RUN_ID="freegpu-squad${LIMIT}-${SAFE_MODEL_NAME}-answermean"

cd "$(dirname "$0")/.."

python scripts/check_runtime.py

python scripts/prepare_squad_subset.py --limit "${LIMIT}"

python scripts/run_inference.py \
  --model "${MODEL}" \
  --dataset "squad_validation_${LIMIT}" \
  --questions "data/benchmarks/squad/squad_validation_${LIMIT}.csv" \
  --splits "data/benchmarks/squad/squad_validation_${LIMIT}_splits.csv" \
  --run-id "${RUN_ID}" \
  --hidden-state-method answer_mean \
  --device cuda \
  --torch-dtype float16 \
  --max-new-tokens 12

python scripts/train_probe.py \
  --records "runs/${RUN_ID}/records_answer_mean.jsonl" \
  --out-dir "runs/${RUN_ID}/analysis" \
  --split-mode record \
  --calibrate

python scripts/make_review_csv.py \
  --records "runs/${RUN_ID}/records_answer_mean.jsonl" \
  --out "runs/${RUN_ID}/manual_review_all.csv"

echo "Completed ${RUN_ID}"
echo "Save runs/${RUN_ID}/ before closing the notebook session."
