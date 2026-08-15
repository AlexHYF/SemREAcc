#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  active_venv="$(cd "$VIRTUAL_ENV" 2>/dev/null && pwd -P || true)"
  target_venv="$repo_root/.venv"
  if [[ "$active_venv" == "$target_venv" ]]; then
    echo "Deactivate the existing .venv before rebuilding it: deactivate" >&2
    exit 2
  fi
fi

python_bin="${PYTHON_BIN:-python3}"
transformers_revision="${TRANSFORMERS_REVISION:-a61d5f9e4fc184cff66938ff6c521cc358b5e024}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

"$python_bin" -m pip install --upgrade uv

# This environment intentionally contains no vLLM. PyTorch's CUDA 12.4 wheels
# run on the RTX 4090 hosts used for this experiment, while the backend selector
# leaves PyPI available for NVIDIA runtime dependencies such as cuDNN. Pin the
# source revision and Mistral tokenizer so token-level scores are reproducible.
uv venv --clear .venv --python "$python_bin"
uv pip install --python .venv/bin/python \
  torch==2.6.0 \
  torchvision==0.21.0 \
  --torch-backend=cu124
uv pip install --python .venv/bin/python \
  "transformers[serving] @ git+https://github.com/huggingface/transformers.git@${transformers_revision}" \
  'mistral-common==1.11.7' \
  'requests>=2.32'
printf '%s\n' transformers > .venv/.semre_backend

.venv/bin/python - <<'PY'
import torch
import transformers
import mistral_common
print(f"PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}; GPU available {torch.cuda.is_available()}")
print(f"Transformers {transformers.__version__}")
print(f"mistral-common {mistral_common.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU")
PY
.venv/bin/transformers serve --help >/dev/null

echo "Transformers environment ready. Run: CONCURRENCY=16 scripts/run_small_model_ensemble.sh"
