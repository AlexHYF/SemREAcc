#!/usr/bin/env python3
"""Collect per-model XOR metrics into a model-scaling CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("core", "full"), default="core")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "models.json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.model_config.open(encoding="utf-8") as handle:
        models = json.load(handle)
    rows: list[dict[str, Any]] = []
    pattern = f"*/metrics_{args.dataset}.json"
    for path in args.results_root.glob(pattern):
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        key = metrics["model_key"]
        config = models.get(key, {})
        direct = metrics["direct"]["overall"]
        component = metrics["component_semre"]["overall"]
        rows.append(
            {
                "run_name": path.parent.name,
                "model_key": key,
                "model_id": metrics["model_id"],
                "architecture": config.get("architecture"),
                "total_parameters_b": config.get("total_parameters_b"),
                "active_parameters_b": config.get("active_parameters_b"),
                "precision": config.get("precision"),
                "evaluated_rows": metrics["evaluated_rows"],
                "atomic_accuracy": nested(metrics, "atomic", "overall", "accuracy"),
                "direct_accuracy": direct["accuracy"],
                "component_accuracy": component["accuracy"],
                "component_minus_direct_accuracy": (
                    component["accuracy"] - direct["accuracy"]
                    if component["accuracy"] is not None and direct["accuracy"] is not None
                    else None
                ),
                "direct_false_positive_rate": direct["false_positive_rate"],
                "component_false_positive_rate": component["false_positive_rate"],
                "direct_precision": direct["precision"],
                "component_precision": component["precision"],
                "direct_coverage": direct["coverage"],
                "component_coverage": component["coverage"],
                "mcnemar_exact_p_two_sided": nested(
                    metrics, "paired_comparison", "mcnemar_exact_p_two_sided"
                ),
                "atomic_requests": nested(metrics, "request_statistics", "atomic", "requests"),
                "direct_requests": nested(metrics, "request_statistics", "direct", "requests"),
            }
        )
    rows.sort(
        key=lambda row: (
            float("inf") if row["total_parameters_b"] is None else row["total_parameters_b"],
            row["run_name"],
        )
    )
    if not rows:
        raise SystemExit(f"No {pattern} files found under {args.results_root}")

    output = args.output or args.results_root / f"summary_{args.dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs to {output}")
    print("model_key\ttotal_B\tdirect_acc\tcomponent_acc\tdelta")
    for row in rows:
        print(
            f"{row['model_key']}\t{row['total_parameters_b']}\t"
            f"{row['direct_accuracy']}\t{row['component_accuracy']}\t"
            f"{row['component_minus_direct_accuracy']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
