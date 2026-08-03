#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

"$python_bin" -m pip install --upgrade uv
uv venv .venv --python "$python_bin"

# Qwen3.5 and GLM-4.7-Flash currently require vLLM main/nightly. The
# --torch-backend option selects wheels matching the machine's accelerator.
uv pip install --python .venv/bin/python \
  vllm \
  --torch-backend=auto \
  --index-url https://pypi.org/simple \
  --extra-index-url https://wheels.vllm.ai/nightly

# GLM-4.7-Flash currently asks for Transformers main as well.
uv pip install --python .venv/bin/python \
  'git+https://github.com/huggingface/transformers.git'

echo "Environment ready. Run: scripts/run_model_experiment.sh qwen35-0.8b"
