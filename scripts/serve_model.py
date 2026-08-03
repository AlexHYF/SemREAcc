#!/usr/bin/env python3
"""Launch one configured model with an OpenAI-compatible local server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "models.json"


def load_models(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        models = json.load(handle)
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No models found in {path}")
    return models


def build_command(
    config: dict,
    host: str,
    port: int,
    tensor_parallel_size: int,
    extra_args: list[str],
) -> list[str]:
    return [
        "vllm",
        "serve",
        config["model_id"],
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--served-model-name",
        config["model_id"],
        "--gpu-memory-utilization",
        "0.92",
        *config.get("vllm_args", []),
        *extra_args,
    ]


def build_transformers_command(
    config: dict,
    host: str,
    port: int,
    extra_args: list[str],
) -> list[str]:
    return [
        "transformers",
        "serve",
        config["model_id"],
        "--host",
        host,
        "--port",
        str(port),
        "--continuous-batching",
        "--dtype",
        "bfloat16",
        *config.get("transformers_args", []),
        *extra_args,
    ]


def find_executable(name: str) -> str | None:
    """Find an executable globally or beside the active virtualenv's Python."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    virtualenv_candidate = Path(sys.executable).parent / name
    if virtualenv_candidate.is_file() and os.access(virtualenv_candidate, os.X_OK):
        return str(virtualenv_candidate)
    return None


def find_vllm_executable() -> str | None:
    """Backward-compatible helper used by tests and callers."""
    return find_executable("vllm")


def select_backend() -> str:
    requested = os.getenv("INFERENCE_BACKEND")
    if requested:
        backend = requested.strip().lower()
    else:
        marker = Path(sys.executable).parent.parent / ".semre_backend"
        backend = (
            marker.read_text(encoding="utf-8").strip().lower()
            if marker.is_file()
            else "auto"
        )
    if backend == "auto":
        if find_executable("vllm"):
            return "vllm"
        if find_executable("transformers"):
            return "transformers"
        raise SystemExit(
            "Neither vllm nor transformers serve is installed. "
            "Run scripts/bootstrap_vllm.sh first."
        )
    if backend not in {"vllm", "transformers"}:
        raise SystemExit("INFERENCE_BACKEND must be vllm or transformers")
    return backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a model from configs/models.json."
    )
    parser.add_argument("model_key", nargs="?", help="Short model key from the manifest")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor-parallel GPU count (default: manifest, or TP_SIZE environment variable)",
    )
    parser.add_argument("--list", action="store_true", help="List configured model keys")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    args.server_extra_args = extra_args
    return args


def main() -> int:
    args = parse_args()
    models = load_models(args.config)
    if args.list:
        for key, config in models.items():
            print(f"{key:24s} {config['model_id']}")
        return 0
    if not args.model_key:
        raise SystemExit("model_key is required unless --list is used")
    if args.model_key not in models:
        choices = ", ".join(models)
        raise SystemExit(f"Unknown model key {args.model_key!r}. Choose one of: {choices}")

    config = models[args.model_key]
    env_tp = os.getenv("TP_SIZE")
    tp_size = args.tp_size
    if tp_size is None and env_tp:
        try:
            tp_size = int(env_tp)
        except ValueError as exc:
            raise SystemExit(f"TP_SIZE must be an integer, got {env_tp!r}") from exc
    if tp_size is None:
        tp_size = int(config["recommended_tensor_parallel_size"])
    if tp_size < 1:
        raise SystemExit("Tensor-parallel size must be at least 1")

    backend = select_backend()
    if backend == "vllm":
        command = build_command(config, args.host, args.port, tp_size, args.server_extra_args)
    else:
        if tp_size != 1:
            raise SystemExit("The Transformers fallback supports only TP_SIZE=1")
        command = build_transformers_command(
            config, args.host, args.port, args.server_extra_args
        )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    executable = find_executable(command[0])
    if executable is None:
        raise SystemExit(
            f"{command[0]} is not installed or is not on PATH. "
            "Run scripts/bootstrap_vllm.sh first."
        )
    command[0] = executable
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
