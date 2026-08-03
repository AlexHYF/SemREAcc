#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_KEY [run_xor_experiment.py options]" >&2
  exit 2
fi

model_key="$1"
shift
if [[ ! "$model_key" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid model key: $model_key" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
port="${PORT:-8000}"
server_timeout="${SERVER_START_TIMEOUT:-7200}"
server_log="$repo_root/results/$model_key/server.log"
base_url="${OPENAI_BASE_URL:-http://127.0.0.1:$port/v1}"
mkdir -p "$(dirname "$server_log")"

if [[ "$python_bin" == */* ]]; then
  python_available=0
  [[ -x "$python_bin" ]] && python_available=1
else
  python_available=0
  command -v "$python_bin" >/dev/null 2>&1 && python_available=1
fi
if [[ "$python_available" != "1" ]]; then
  echo "Python executable not found: $python_bin" >&2
  echo "Run scripts/bootstrap_vllm.sh, or set PYTHON_BIN." >&2
  exit 1
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && [[ "$server_pid" =~ ^[0-9]+$ ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${USE_EXISTING_SERVER:-0}" != "1" ]]; then
  "$python_bin" scripts/serve_model.py "$model_key" --port "$port" >"$server_log" 2>&1 &
  server_pid="$!"
  echo "Starting $model_key (PID $server_pid); log: $server_log"

  deadline=$((SECONDS + server_timeout))
  while ! curl --fail --silent --show-error "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Model server exited before becoming ready. Last log lines:" >&2
      tail -n 80 "$server_log" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Model server did not become ready within ${server_timeout}s." >&2
      echo "Last log lines:" >&2
      tail -n 80 "$server_log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "Model server is ready."
else
  echo "Using existing OpenAI-compatible server at $base_url"
fi

"$python_bin" scripts/run_xor_experiment.py \
  --model-key "$model_key" \
  --base-url "$base_url" \
  "$@"
