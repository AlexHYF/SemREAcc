#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

dataset="${DATASET:-core}"
concurrency="${CONCURRENCY:-32}"
run_prefix="${RUN_PREFIX:-ensemble}"
ensemble_run="${ENSEMBLE_RUN_NAME:-ensemble-qwen-gemma-llama}"

if [[ "$dataset" != "core" && "$dataset" != "full" ]]; then
  echo "DATASET must be core or full" >&2
  exit 2
fi
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONCURRENCY must be a positive integer" >&2
  exit 2
fi
if [[ ! "$run_prefix" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_PREFIX contains unsupported characters" >&2
  exit 2
fi
if [[ ! "$ensemble_run" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ENSEMBLE_RUN_NAME contains unsupported characters" >&2
  exit 2
fi

models=(qwen35-4b gemma3-4b llama32-3b)
runs=()
for model_key in "${models[@]}"; do
  run_name="${run_prefix}-${model_key}"
  runs+=("$run_name")
  INFERENCE_BACKEND=transformers \
    SERVER_LOG="$repo_root/results/$run_name/server.log" \
    scripts/run_model_experiment.sh "$model_key" \
    --dataset "$dataset" \
    --concurrency "$concurrency" \
    --run-name "$run_name"
done

.venv/bin/python scripts/evaluate_ensemble.py \
  --runs "${runs[@]}" \
  --dataset "$dataset" \
  --output-run "$ensemble_run"

echo "Ensemble summary: results/$ensemble_run/summary_${dataset}.csv"
