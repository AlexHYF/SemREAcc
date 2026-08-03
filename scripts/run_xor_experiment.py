#!/usr/bin/env python3
"""Run the crossed Chinese/Japanese name experiment against an OpenAI API.

The client uses only the Python standard library. It works with vLLM, SGLang,
or another server implementing POST /v1/chat/completions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "models.json"
DATA_DIR = ROOT / "data" / "xor_names"
PROMPT_VERSION = "xor-names-v1"
SYSTEM_PROMPT = (
    "You are a careful binary classifier. Follow the user's definitions exactly. "
    "Your entire response must be exactly YES or NO."
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_prompt(row: dict[str, str]) -> str:
    role = "given name" if row["role"] == "given" else "surname"
    origin = "Chinese" if row["target_origin"] == "C" else "Japanese"
    return (
        f'Treat "{row["name"]}" as a romanized {role}, not as evidence about '
        f"a person's nationality. Is it commonly used as a {origin} {role}?\n"
        "Answer only YES or NO."
    )


def direct_prompt(row: dict[str, str]) -> str:
    return (
        f'The text "{row["full_name"]}" contains two romanized tokens in '
        "GIVEN-NAME SURNAME order. Does it satisfy either of these conditions?\n"
        "1. The first token is commonly used as a Chinese given name and the "
        "second as a Japanese surname.\n"
        "2. The first token is commonly used as a Japanese given name and the "
        "second as a Chinese surname.\n"
        "Answer only YES or NO."
    )


def parse_yes_no(text: str | None) -> int | None:
    if not text:
        return None
    lowered = text.lower()
    if "<think>" in lowered and "</think>" not in lowered:
        # A truncated reasoning trace is not a classifier answer, even if it
        # happens to contain the words "yes" or "no".
        return None
    cleaned = THINK_BLOCK_RE.sub(" ", text).strip()
    if cleaned.startswith("{"):
        try:
            answer = json.loads(cleaned).get("answer")
            if isinstance(answer, bool):
                return int(answer)
            if isinstance(answer, str) and answer.strip().upper() in {"YES", "NO"}:
                return int(answer.strip().upper() == "YES")
        except (json.JSONDecodeError, AttributeError):
            pass
    final_line = cleaned.splitlines()[-1].strip() if cleaned else ""
    final_match = re.fullmatch(
        r"(?:final\s+answer|answer)?\s*[:=-]?\s*(YES|NO)[.!]?",
        final_line,
        re.IGNORECASE,
    )
    if final_match:
        return int(final_match.group(1).upper() == "YES")
    matches = ANSWER_RE.findall(cleaned)
    if not matches:
        return None
    normalized = {match.upper() for match in matches}
    if len(normalized) != 1:
        return None
    return int(matches[0].upper() == "YES")


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


class ChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        timeout: float,
        retries: int,
        extra_body: dict[str, Any],
    ) -> None:
        self.url = normalize_base_url(base_url) + "/chat/completions"
        self.api_key = api_key
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout
        self.retries = retries
        self.extra_body = extra_body

    def complete(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        payload.update(self.extra_body)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        last_error = ""
        for attempt in range(1, self.retries + 2):
            request = Request(self.url, data=body, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                choice = parsed["choices"][0]
                message = choice["message"]
                raw_response = message.get("content") or ""
                reasoning_response = message.get("reasoning_content") or None
                usage = parsed.get("usage") or {}
                return {
                    "raw_response": raw_response,
                    "reasoning_response": reasoning_response,
                    "prediction": parse_yes_no(raw_response),
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "attempt_count": attempt,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": choice.get("finish_reason"),
                    "error": None,
                }
            except HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:2000]
                except Exception:  # pragma: no cover - defensive around error streams
                    detail = ""
                last_error = f"HTTP {exc.code}: {detail or exc.reason}"
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= self.retries:
                time.sleep(min(2 ** (attempt - 1), 20))
        return {
            "raw_response": None,
            "reasoning_response": None,
            "prediction": None,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "attempt_count": min(attempt, self.retries + 1),
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "error": last_error or "Unknown request error",
        }


def latest_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[record["id"]] = record
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return latest


def make_job(row: dict[str, str], task: str) -> dict[str, Any]:
    prompt = atomic_prompt(row) if task == "atomic" else direct_prompt(row)
    return {
        "id": row["id"],
        "task": task,
        "prompt": prompt,
        "gold_label": int(row["label"]),
        "metadata": row,
    }


def execute_job(client: ChatClient, job: dict[str, Any]) -> dict[str, Any]:
    result = client.complete(job["prompt"])
    return {
        "id": job["id"],
        "task": job["task"],
        "model_id": client.model_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
        "prompt": job["prompt"],
        "gold_label": job["gold_label"],
        **job["metadata"],
        **result,
        "timestamp_utc": utc_now(),
    }


def run_jobs(
    *,
    jobs: list[dict[str, Any]],
    output_path: Path,
    client: ChatClient,
    concurrency: int,
    rerun_failures: bool,
) -> dict[str, dict[str, Any]]:
    existing = latest_jsonl(output_path)
    pending = []
    for job in jobs:
        old = existing.get(job["id"])
        if old is None or (rerun_failures and old.get("prediction") is None):
            pending.append(job)
    print(
        f"{output_path.name}: {len(existing)} cached, {len(pending)} pending "
        f"of {len(jobs)} selected",
        flush=True,
    )
    if not pending:
        return existing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    completed = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures: dict[Future[dict[str, Any]], str] = {
                executor.submit(execute_job, client, job): job["id"] for job in pending
            }
            try:
                for future in as_completed(futures):
                    record = future.result()
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    existing[record["id"]] = record
                    completed += 1
                    failures += int(record["prediction"] is None)
                    if completed % 50 == 0 or completed == len(pending):
                        print(
                            f"  completed {completed}/{len(pending)}; "
                            f"unparsed/errors={failures}",
                            flush=True,
                        )
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                print("Interrupted; completed records are safely checkpointed.", file=sys.stderr)
                raise
    return existing


def kleene_and(left: int | None, right: int | None) -> int | None:
    if left == 0 or right == 0:
        return 0
    if left == 1 and right == 1:
        return 1
    return None


def kleene_or(left: int | None, right: int | None) -> int | None:
    if left == 1 or right == 1:
        return 1
    if left == 0 and right == 0:
        return 0
    return None


def component_prediction(
    row: dict[str, str], atomic_by_key: dict[tuple[str, str, str], int | None]
) -> int | None:
    def atom(name: str, role: str, origin: str) -> int | None:
        return atomic_by_key.get((name.casefold(), role, origin))

    c_given = atom(row["first_name"], "given", "C")
    j_given = atom(row["first_name"], "given", "J")
    c_surname = atom(row["last_name"], "surname", "C")
    j_surname = atom(row["last_name"], "surname", "J")
    return kleene_or(kleene_and(c_given, j_surname), kleene_and(j_given, c_surname))


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def classification_metrics(items: Iterable[tuple[int, int | None]]) -> dict[str, Any]:
    pairs = list(items)
    tp = tn = fp = fn = invalid = 0
    for gold, prediction in pairs:
        if prediction is None:
            invalid += 1
        elif gold == 1 and prediction == 1:
            tp += 1
        elif gold == 0 and prediction == 0:
            tn += 1
        elif gold == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1
    total = len(pairs)
    valid = total - invalid
    correct = tp + tn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "n": total,
        "valid": valid,
        "invalid": invalid,
        "coverage": safe_div(valid, total),
        "accuracy": safe_div(correct, total),
        "accuracy_ci95_wilson": wilson_interval(correct, total),
        "valid_accuracy": safe_div(correct, valid),
        "precision": precision,
        "recall": recall,
        "specificity": safe_div(tn, tn + fp),
        "f1": safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def exact_mcnemar_p(direct_only: int, component_only: int) -> float | None:
    discordant = direct_only + component_only
    if discordant == 0:
        return None
    smaller = min(direct_only, component_only)
    # Sum the exact Binomial(n, 0.5) tail in log space. This remains fast and
    # numerically stable for the 40,000-row exhaustive dataset.
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2)
        for k in range(smaller + 1)
    ]
    largest = max(log_probabilities)
    log_tail = largest + math.log(
        sum(math.exp(log_probability - largest) for log_probability in log_probabilities)
    )
    return min(1.0, 2 * math.exp(log_tail))


def latency_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    latencies = sorted(
        float(row["latency_seconds"])
        for row in rows
        if row.get("latency_seconds") is not None
    )
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    if not latencies:
        return {
            "requests": len(rows),
            "latency_seconds_sum": None,
            "latency_seconds_mean": None,
            "latency_seconds_median": None,
            "latency_seconds_p95": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "requests": len(rows),
        "latency_seconds_sum": sum(latencies),
        "latency_seconds_mean": statistics.fmean(latencies),
        "latency_seconds_median": statistics.median(latencies),
        "latency_seconds_p95": latencies[p95_index],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def by_group(
    scored: list[dict[str, Any]], prediction_field: str, group_field: str
) -> dict[str, dict[str, Any]]:
    values = sorted({row[group_field] for row in scored})
    return {
        value: classification_metrics(
            (int(row["gold_label"]), row[prediction_field])
            for row in scored
            if row[group_field] == value
        )
        for value in values
    }


def write_scores_and_metrics(
    *,
    rows: list[dict[str, str]],
    atomic_records: dict[str, dict[str, Any]],
    direct_records: dict[str, dict[str, Any]],
    output_dir: Path,
    dataset_name: str,
    model_key: str,
    model_id: str,
) -> dict[str, Any]:
    atomic_by_key: dict[tuple[str, str, str], int | None] = {}
    for record in atomic_records.values():
        key = (record["name"].casefold(), record["role"], record["target_origin"])
        atomic_by_key[key] = record.get("prediction")

    scored: list[dict[str, Any]] = []
    for row in rows:
        direct = direct_records.get(row["id"], {}).get("prediction")
        component = component_prediction(row, atomic_by_key)
        gold = int(row["label"])
        scored.append(
            {
                **row,
                "gold_label": gold,
                "direct_prediction": direct,
                "component_prediction": component,
                "direct_correct": None if direct is None else int(direct == gold),
                "component_correct": None if component is None else int(component == gold),
            }
        )

    predictions_path = output_dir / f"predictions_{dataset_name}.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(scored[0]) if scored else [
        "id", "full_name", "gold_label", "direct_prediction", "component_prediction"
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)

    atomic_selected_ids: set[str] = set()
    needed_keys = required_atomic_keys(rows)
    atomic_scored: list[dict[str, Any]] = []
    for record in atomic_records.values():
        key = (record["name"].casefold(), record["role"], record["target_origin"])
        if key in needed_keys:
            atomic_selected_ids.add(record["id"])
            atomic_scored.append(record)

    paired = {
        "n": len(scored),
        "both_valid": 0,
        "both_correct": 0,
        "direct_only_correct": 0,
        "component_only_correct": 0,
        "both_wrong": 0,
        "either_invalid": 0,
    }
    for row in scored:
        direct_correct = row["direct_correct"]
        component_correct = row["component_correct"]
        if direct_correct is None or component_correct is None:
            paired["either_invalid"] += 1
        else:
            paired["both_valid"] += 1
            if direct_correct and component_correct:
                paired["both_correct"] += 1
            elif direct_correct:
                paired["direct_only_correct"] += 1
            elif component_correct:
                paired["component_only_correct"] += 1
            else:
                paired["both_wrong"] += 1
    paired["mcnemar_exact_p_two_sided"] = exact_mcnemar_p(
        paired["direct_only_correct"], paired["component_only_correct"]
    )

    direct_metric = classification_metrics(
        (row["gold_label"], row["direct_prediction"]) for row in scored
    )
    component_metric = classification_metrics(
        (row["gold_label"], row["component_prediction"]) for row in scored
    )
    metrics = {
        "model_key": model_key,
        "model_id": model_id,
        "dataset": dataset_name,
        "evaluated_rows": len(rows),
        "prompt_version": PROMPT_VERSION,
        "generated_at_utc": utc_now(),
        "atomic": {
            "overall": classification_metrics(
                (int(row["gold_label"]), row.get("prediction")) for row in atomic_scored
            ),
            "by_role": {
                role: classification_metrics(
                    (int(row["gold_label"]), row.get("prediction"))
                    for row in atomic_scored
                    if row["role"] == role
                )
                for role in ("given", "surname")
            },
            "by_target_origin": {
                origin: classification_metrics(
                    (int(row["gold_label"]), row.get("prediction"))
                    for row in atomic_scored
                    if row["target_origin"] == origin
                )
                for origin in ("C", "J")
            },
        },
        "direct": {
            "overall": direct_metric,
            "by_pair_type": by_group(scored, "direct_prediction", "pair_type"),
        },
        "component_semre": {
            "overall": component_metric,
            "by_pair_type": by_group(scored, "component_prediction", "pair_type"),
        },
        "paired_comparison": paired,
        "request_statistics": {
            "atomic": latency_summary(
                atomic_records[record_id] for record_id in atomic_selected_ids
            ),
            "direct": latency_summary(
                direct_records[row["id"]] for row in rows if row["id"] in direct_records
            ),
        },
    }
    metrics_path = output_dir / f"metrics_{dataset_name}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metrics


def required_atomic_keys(rows: Iterable[dict[str, str]]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        for origin in ("C", "J"):
            keys.add((row["first_name"].casefold(), "given", origin))
            keys.add((row["last_name"].casefold(), "surname", origin))
    return keys


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct and component-wise classification on the XOR name benchmark."
    )
    parser.add_argument("--model-key", required=True, help="Key in configs/models.json")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--model-id",
        help="API model name override (defaults to the model_id in the manifest)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--dataset", choices=("core", "full"), default="core")
    parser.add_argument("--mode", choices=("all", "atomic", "direct", "score"), default="all")
    parser.add_argument("--run-name", help="Results subdirectory (default: model key)")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test only: evaluate the first N pair rows and just their needed atoms",
    )
    parser.add_argument(
        "--rerun-failures",
        action="store_true",
        help="Append fresh attempts for cached records with no parseable prediction",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.retries < 0:
        raise SystemExit("concurrency/max-tokens must be positive and retries cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    models = load_json(args.model_config)
    if args.model_key not in models:
        raise SystemExit(
            f"Unknown model key {args.model_key!r}; choose one of: {', '.join(models)}"
        )
    model_config = models[args.model_key]
    model_id = args.model_id or model_config["model_id"]
    request_extra_body = model_config.get("request_extra_body", {})

    dataset_path = DATA_DIR / f"xor_names_{args.dataset}.csv"
    rows = read_csv(dataset_path)
    dataset_row_count = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    all_atomic_rows = read_csv(DATA_DIR / "individual_queries.csv")
    needed_keys = required_atomic_keys(rows)
    atomic_rows = [
        row
        for row in all_atomic_rows
        if (row["name"].casefold(), row["role"], row["target_origin"]) in needed_keys
    ]

    run_name = args.run_name or args.model_key
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise SystemExit("--run-name may contain only letters, digits, dot, underscore, and hyphen")
    output_dir = args.output_root.resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "model_key": args.model_key,
        "model_id": model_id,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "request_extra_body": request_extra_body,
        "atomic_dataset_sha256": file_sha256(DATA_DIR / "individual_queries.csv"),
    }
    fingerprint = protocol_fingerprint(protocol)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        old_config = load_json(config_path)
        if old_config.get("protocol_fingerprint") != fingerprint:
            raise SystemExit(
                f"Protocol differs from existing {config_path}. Use a new --run-name to avoid "
                "mixing incompatible cached answers."
            )
    else:
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **protocol,
                    "protocol_fingerprint": fingerprint,
                    "base_url": normalize_base_url(args.base_url),
                    "created_at_utc": utc_now(),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

    dataset_config = {
        "dataset": args.dataset,
        "path": str(dataset_path),
        "rows": dataset_row_count,
        "sha256": file_sha256(dataset_path),
    }
    dataset_config_path = output_dir / f"dataset_config_{args.dataset}.json"
    if dataset_config_path.exists():
        if load_json(dataset_config_path) != dataset_config:
            raise SystemExit(
                f"Dataset differs from existing {dataset_config_path}. Use a new --run-name "
                "to avoid reusing predictions for changed rows."
            )
    else:
        with dataset_config_path.open("w", encoding="utf-8") as handle:
            json.dump(dataset_config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    atomic_path = output_dir / "atomic.jsonl"
    direct_path = output_dir / f"direct_{args.dataset}.jsonl"
    client = ChatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model_id=model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        timeout=args.timeout,
        retries=args.retries,
        extra_body=request_extra_body,
    )

    if args.mode in {"all", "atomic"}:
        run_jobs(
            jobs=[make_job(row, "atomic") for row in atomic_rows],
            output_path=atomic_path,
            client=client,
            concurrency=args.concurrency,
            rerun_failures=args.rerun_failures,
        )
    if args.mode in {"all", "direct"}:
        run_jobs(
            jobs=[make_job(row, "direct") for row in rows],
            output_path=direct_path,
            client=client,
            concurrency=args.concurrency,
            rerun_failures=args.rerun_failures,
        )

    atomic_records = latest_jsonl(atomic_path)
    direct_records = latest_jsonl(direct_path)
    metrics = write_scores_and_metrics(
        rows=rows,
        atomic_records=atomic_records,
        direct_records=direct_records,
        output_dir=output_dir,
        dataset_name=args.dataset,
        model_key=args.model_key,
        model_id=model_id,
    )
    direct_accuracy = metrics["direct"]["overall"]["accuracy"]
    component_accuracy = metrics["component_semre"]["overall"]["accuracy"]
    print(f"Results: {output_dir}")
    print(f"Direct accuracy:    {direct_accuracy if direct_accuracy is not None else 'N/A'}")
    print(f"Component accuracy: {component_accuracy if component_accuracy is not None else 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
