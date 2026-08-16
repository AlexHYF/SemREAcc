#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
batch_size="${BATCH_SIZE:-8}"
output_root="${OUTPUT_ROOT:-results/atomic-roc}"
device="${DEVICE:-cuda}"
dtype="${DTYPE:-bfloat16}"
limit_per_class="${LIMIT_PER_CLASS:-}"
model_keys_text="${MODEL_KEYS:-qwen35-4b gemma3-4b llama32-3b}"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing $python_bin. Run scripts/bootstrap_transformers.sh first." >&2
  exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ "$dtype" != "bfloat16" && "$dtype" != "float16" && "$dtype" != "float32" ]]; then
  echo "DTYPE must be bfloat16, float16, or float32" >&2
  exit 2
fi
limit_args=()
if [[ -n "$limit_per_class" ]]; then
  if [[ ! "$limit_per_class" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT_PER_CLASS must be a positive integer" >&2
    exit 2
  fi
  limit_args=(--limit-per-class "$limit_per_class")
fi

read -r -a models <<< "$model_keys_text"
if (( ${#models[@]} == 0 )); then
  echo "MODEL_KEYS must contain at least one model key" >&2
  exit 2
fi
declare -A seen_models=()
for model_key in "${models[@]}"; do
  if [[ ! "$model_key" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "MODEL_KEYS contains an unsupported model key: $model_key" >&2
    exit 2
  fi
  if [[ -n "${seen_models[$model_key]:-}" ]]; then
    echo "MODEL_KEYS contains a duplicate: $model_key" >&2
    exit 2
  fi
  seen_models[$model_key]=1
done

if ! "$python_bin" -c \
  'import torch, transformers; print(f"torch {torch.__version__}; transformers {transformers.__version__}")'; then
  echo "PyTorch and Transformers are required. Run scripts/bootstrap_transformers.sh first." >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
for model_key in "${models[@]}"; do
  echo "Running atomic ROC experiment for $model_key"
  "$python_bin" scripts/run_atomic_roc.py \
    --model-key "$model_key" \
    --run-name "$model_key" \
    --batch-size "$batch_size" \
    --output-root "$output_root" \
    --device "$device" \
    --dtype "$dtype" \
    "${limit_args[@]}"
done

"$python_bin" scripts/summarize_atomic_roc.py \
  --runs "${models[@]}" \
  --output-root "$output_root"

echo "Combined ROC summary: $output_root/summary.csv"
echo "Combined ROC plot: $output_root/roc_curves_all_models.svg"
