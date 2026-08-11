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
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

"$python_bin" -m pip install --upgrade uv
# The serving environment is disposable. Clear it so switching between CUDA
# variants cannot leave a CUDA 13 torch/vLLM wheel in a CUDA 12 environment (or
# vice versa). Hugging Face model downloads live outside .venv and are retained.
uv venv --clear .venv --python "$python_bin"

cuda_variant="${SEMRE_CUDA_VARIANT:-auto}"
if [[ "$cuda_variant" == "auto" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  detected_cuda="$({ nvidia-smi 2>/dev/null || true; } | sed -n \
    's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
  case "$detected_cuda" in
    13.*) cuda_variant="cu130" ;;
    12.*) cuda_variant="cu129" ;;
  esac
fi

case "$cuda_variant" in
  cu124)
    # Current vLLM wheels no longer target CUDA 12.4, while the older cu124
    # vLLM releases predate Qwen3.5. Use Transformers' OpenAI-compatible
    # evaluation server only when this compatibility fallback is requested
    # explicitly. CUDA 12.x hosts normally use the faster cu129 vLLM path.
    uv pip install --python .venv/bin/python \
      torch==2.6.0 \
      torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python .venv/bin/python \
      'transformers[serving] @ git+https://github.com/huggingface/transformers.git@main' \
      'requests>=2.32'
    printf '%s\n' transformers > .venv/.semre_backend
    ;;
  cu129)
    # Stable vLLM wheels now use CUDA 12.9 by default. Avoid the nightly
    # index here: it can temporarily contain a wheel for only one CPU
    # architecture while a nightly release is being published.
    uv pip install --python .venv/bin/python \
      vllm \
      --torch-backend=cu129
    uv pip install --python .venv/bin/python \
      'git+https://github.com/huggingface/transformers.git@main'
    printf '%s\n' vllm > .venv/.semre_backend
    ;;
  cu130)
    # CUDA 13.0 remains an alternate vLLM build, so use its CUDA-specific
    # nightly index and keep the PyTorch and vLLM CUDA ABIs aligned.
    uv pip install --python .venv/bin/python \
      vllm \
      --torch-backend=cu130 \
      --extra-index-url=https://wheels.vllm.ai/nightly/cu130
    uv pip install --python .venv/bin/python \
      'git+https://github.com/huggingface/transformers.git@main'
    printf '%s\n' vllm > .venv/.semre_backend
    ;;
  auto)
    # Fall back to uv's accelerator detection when nvidia-smi is unavailable.
    uv pip install --python .venv/bin/python \
      vllm \
      --torch-backend=auto
    uv pip install --python .venv/bin/python \
      'git+https://github.com/huggingface/transformers.git@main'
    printf '%s\n' vllm > .venv/.semre_backend
    ;;
  *)
    echo "Unsupported SEMRE_CUDA_VARIANT=$cuda_variant (use auto, cu124, cu129, or cu130)" >&2
    exit 2
    ;;
esac

if [[ "$cuda_variant" != "cu124" ]]; then
  # FlashInfer may JIT-compile GPU kernels during vLLM startup.
  uv pip install --python .venv/bin/python 'ninja>=1.11'
fi

if [[ "$cuda_variant" == "cu124" ]]; then
  .venv/bin/python - <<'PY'
import torch
import transformers
print(f"PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}; GPU available {torch.cuda.is_available()}")
print(f"Transformers {transformers.__version__}")
PY
  .venv/bin/transformers serve --help >/dev/null
else
  .venv/bin/python - <<'PY'
import torch
import vllm
print(f"PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}; GPU available {torch.cuda.is_available()}")
print(f"vLLM {vllm.__version__}")
PY
fi

echo "Environment ready for $cuda_variant. Run: scripts/run_model_experiment.sh qwen35-0.8b"
