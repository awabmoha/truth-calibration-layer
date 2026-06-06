#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-200}"
MODEL="${2:-Qwen/Qwen2.5-0.5B-Instruct}"
SAFE_MODEL_NAME="$(echo "${MODEL}" | tr '/:' '__')"
RUN_ID="freegpu-triviaqa${LIMIT}-${SAFE_MODEL_NAME}-answermean"
STEM="triviaqa_rc_nocontext_validation_${LIMIT}"

cd "$(dirname "$0")/.."

python scripts/check_runtime.py

python scripts/prepare_triviaqa_subset.py --limit "${LIMIT}"

python scripts/run_inference.py \
  --model "${MODEL}" \
  --dataset "${STEM}" \
  --questions "data/benchmarks/triviaqa/${STEM}.csv" \
  --splits "data/benchmarks/triviaqa/${STEM}_splits.csv" \
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

python scripts/make_targeted_review_csv.py \
  --records "runs/${RUN_ID}/records_answer_mean.jsonl" \
  --predictions "runs/${RUN_ID}/analysis/answer_mean/test_predictions.csv" \
  --out "runs/${RUN_ID}/targeted_manual_review_candidates.csv"

python scripts/verify_run_artifact.py \
  --run-dir "runs/${RUN_ID}" \
  --method answer_mean \
  --min-records "${LIMIT}" \
  --require-manual-review \
  --require-targeted-review \
  --require-calibrated \
  --out-json "runs/${RUN_ID}/artifact_verification.json" \
  --out-md "runs/${RUN_ID}/ARTIFACT_VERIFICATION.md"

python scripts/package_run_artifact.py \
  --run-dir "runs/${RUN_ID}" \
  --benchmark-glob "data/benchmarks/triviaqa/${STEM}*" \
  --out "runs/${RUN_ID}_artifact.zip"

echo "Completed ${RUN_ID}"
echo "Save runs/${RUN_ID}_artifact.zip and runs/${RUN_ID}/ before closing the notebook session."
