#!/usr/bin/env python3
"""Combine per-model atomic ROC outputs into tables and one comparison plot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from run_atomic_roc import COLORS, QUERY_ORDER, QUERY_TITLES, write_roc_svg


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "atomic-roc"
DEFAULT_RUNS = ("qwen35-4b", "gemma3-4b", "llama32-3b")
COMPATIBILITY_FIELDS = (
    "dataset_sha256",
    "score_version",
    "prompt_version",
    "system_prompt",
    "answer_texts",
    "score_definition",
    "dtype",
    "transformers_version",
    "transformers_vcs_commit",
    "mistral_common_version",
    "torch_version",
)
POINT_NUMBER_FIELDS = (
    "point_index",
    "threshold_log_odds",
    "threshold_yes_probability",
    "fpr",
    "tpr",
    "tp",
    "fp",
    "tn",
    "fn",
)
SUMMARY_COLUMNS = (
    "model_key",
    "model_id",
    "query",
    "query_title",
    "n",
    "positives",
    "negatives",
    "single_token_examples",
    "multi_token_examples",
    "auc",
    "threshold_log_odds",
    "threshold_yes_probability",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "tp",
    "fp",
    "tn",
    "fn",
)
DEFAULT_METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "tp",
    "fp",
    "tn",
    "fn",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def resolve_run(output_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()
    return (output_root / candidate).resolve()


def required(mapping: dict[str, Any], key: str, source: Path) -> Any:
    if key not in mapping:
        raise SystemExit(f"Missing {key!r} in {source}")
    return mapping[key]


def compatibility_protocol(config: dict[str, Any], source: Path) -> dict[str, Any]:
    return {field: required(config, field, source) for field in COMPATIBILITY_FIELDS}


def validate_summary(
    config: dict[str, Any], summary: dict[str, Any], config_path: Path, summary_path: Path
) -> None:
    for field in ("model_key", "model_id", "dataset_sha256", "score_version"):
        config_value = required(config, field, config_path)
        summary_value = required(summary, field, summary_path)
        if summary_value != config_value:
            raise SystemExit(
                f"{field} disagrees between {config_path} and {summary_path}: "
                f"{config_value!r} != {summary_value!r}"
            )
    queries = required(summary, "queries", summary_path)
    if not isinstance(queries, dict) or set(queries) != set(QUERY_ORDER):
        raise SystemExit(
            f"{summary_path} must contain exactly these queries: "
            f"{', '.join(QUERY_ORDER)}"
        )


def number(value: str, *, integer: bool, source: Path, field: str) -> int | float:
    try:
        parsed: int | float = int(value) if integer else float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid {field} value {value!r} in {source}") from exc
    if not integer and not math.isfinite(parsed):
        if field != "threshold_log_odds" or parsed != math.inf:
            raise SystemExit(f"Non-finite {field} value {value!r} in {source}")
    return parsed


def load_points(path: Path, model_key: str) -> dict[str, list[dict[str, Any]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit(f"Missing CSV header in {path}")
            missing = {
                "model_key",
                "query",
                *POINT_NUMBER_FIELDS,
            } - set(reader.fieldnames)
            if missing:
                raise SystemExit(f"Missing columns in {path}: {', '.join(sorted(missing))}")
            rows = list(reader)
    except OSError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    grouped: dict[str, list[dict[str, Any]]] = {query: [] for query in QUERY_ORDER}
    integer_fields = {"point_index", "tp", "fp", "tn", "fn"}
    for row in rows:
        if row["model_key"] != model_key:
            raise SystemExit(
                f"Unexpected model_key {row['model_key']!r} in {path}; "
                f"expected {model_key!r}"
            )
        query = row["query"]
        if query not in grouped:
            raise SystemExit(f"Unexpected query {query!r} in {path}")
        converted: dict[str, Any] = dict(row)
        for field in POINT_NUMBER_FIELDS:
            converted[field] = number(
                row[field], integer=field in integer_fields, source=path, field=field
            )
        grouped[query].append(converted)
    for query, points in grouped.items():
        if not points:
            raise SystemExit(f"No {query} ROC points in {path}")
        points.sort(key=lambda point: point["point_index"])
        expected_indexes = list(range(len(points)))
        if [point["point_index"] for point in points] != expected_indexes:
            raise SystemExit(f"Non-contiguous point_index values for {query} in {path}")
        if points[0]["fpr"] != 0.0 or points[0]["tpr"] != 0.0:
            raise SystemExit(f"The {query} ROC curve in {path} does not start at (0, 0)")
        if points[0]["threshold_log_odds"] != math.inf or any(
            not math.isfinite(point["threshold_log_odds"])
            for point in points[1:]
        ):
            raise SystemExit(f"Invalid threshold sequence for {query} in {path}")
        if points[-1]["fpr"] != 1.0 or points[-1]["tpr"] != 1.0:
            raise SystemExit(f"The {query} ROC curve in {path} does not end at (1, 1)")
    return grouped


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="Run directory names under --output-root, or explicit run paths",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if len(set(args.runs)) != len(args.runs):
        raise SystemExit("--runs contains a duplicate")

    reference_protocol: dict[str, Any] | None = None
    reference_counts: dict[str, tuple[int, int, int]] | None = None
    models: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_point_rows: list[dict[str, Any]] = []
    series_by_query: dict[str, list[dict[str, Any]]] = {
        query: [] for query in QUERY_ORDER
    }

    for model_index, run_value in enumerate(args.runs):
        run_dir = resolve_run(output_root, run_value)
        config_path = run_dir / "run_config.json"
        summary_path = run_dir / "roc_summary.json"
        points_path = run_dir / "roc_points.csv"
        config = load_json(config_path)
        summary = load_json(summary_path)
        validate_summary(config, summary, config_path, summary_path)
        protocol = compatibility_protocol(config, config_path)
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            differences = [
                field
                for field in COMPATIBILITY_FIELDS
                if protocol[field] != reference_protocol[field]
            ]
            raise SystemExit(
                f"Incompatible experiment protocol in {config_path}; differing "
                f"fields: {', '.join(differences)}"
            )

        model_key = str(summary["model_key"])
        model_id = str(summary["model_id"])
        points = load_points(points_path, model_key)
        query_counts: dict[str, tuple[int, int, int]] = {}
        for query in QUERY_ORDER:
            query_summary = summary["queries"][query]
            if not isinstance(query_summary, dict):
                raise SystemExit(f"Expected an object for {query} in {summary_path}")
            n = int(required(query_summary, "n", summary_path))
            positives = int(required(query_summary, "positives", summary_path))
            negatives = int(required(query_summary, "negatives", summary_path))
            single_token_examples = int(
                required(query_summary, "single_token_examples", summary_path)
            )
            multi_token_examples = int(
                required(query_summary, "multi_token_examples", summary_path)
            )
            if n != positives + negatives or positives < 1 or negatives < 1:
                raise SystemExit(f"Invalid class counts for {query} in {summary_path}")
            if n != single_token_examples + multi_token_examples:
                raise SystemExit(
                    f"Invalid tokenization counts for {query} in {summary_path}"
                )
            query_counts[query] = (n, positives, negatives)
            auc = float(required(query_summary, "auc", summary_path))
            if not 0.0 <= auc <= 1.0:
                raise SystemExit(f"Invalid AUC for {query} in {summary_path}: {auc}")
            default = required(query_summary, "default_threshold", summary_path)
            if not isinstance(default, dict):
                raise SystemExit(
                    f"Expected default_threshold object for {query} in {summary_path}"
                )
            all_summary_rows.append(
                {
                    "model_key": model_key,
                    "model_id": model_id,
                    "query": query,
                    "query_title": query_summary.get("title", QUERY_TITLES[query]),
                    "n": n,
                    "positives": positives,
                    "negatives": negatives,
                    "single_token_examples": single_token_examples,
                    "multi_token_examples": multi_token_examples,
                    "auc": auc,
                    "threshold_log_odds": default.get("threshold_log_odds"),
                    "threshold_yes_probability": default.get(
                        "threshold_yes_probability"
                    ),
                    **{
                        field: default.get(field)
                        for field in DEFAULT_METRIC_COLUMNS
                    },
                }
            )
            series_by_query[query].append(
                {
                    "label": model_key,
                    "points": points[query],
                    "auc": auc,
                    "operating_point": default,
                    "color": COLORS[model_index % len(COLORS)],
                }
            )
            all_point_rows.extend(points[query])

        if reference_counts is None:
            reference_counts = query_counts
        elif query_counts != reference_counts:
            raise SystemExit(
                f"{summary_path} evaluated different per-query class counts than "
                "the first run"
            )
        models.append(
            {
                "model_key": model_key,
                "model_id": model_id,
                "run_dir": str(run_dir),
                "protocol_fingerprint": config.get("protocol_fingerprint"),
            }
        )

    assert reference_protocol is not None
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "summary.csv", all_summary_rows, list(SUMMARY_COLUMNS))
    point_fields = list(all_point_rows[0])
    write_csv(output_root / "roc_points_all_models.csv", all_point_rows, point_fields)
    write_roc_svg(
        output_root / "roc_curves_all_models.svg",
        series_by_query,
        title="Atomic YES/NO ROC curves — model comparison",
    )
    aggregate = {
        "generated_at_utc": utc_now(),
        "compatibility_protocol": reference_protocol,
        "query_counts": {
            query: {
                "n": counts[0],
                "positives": counts[1],
                "negatives": counts[2],
            }
            for query, counts in (reference_counts or {}).items()
        },
        "models": models,
        "outputs": {
            "summary": "summary.csv",
            "roc_points": "roc_points_all_models.csv",
            "roc_plot": "roc_curves_all_models.svg",
        },
    }
    with (output_root / "aggregate_config.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Combined {len(models)} models in {output_root}")
    print(f"Summary: {output_root / 'summary.csv'}")
    print(f"ROC plot: {output_root / 'roc_curves_all_models.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
