#!/usr/bin/env python3
"""Evaluate majority-vote ensembles over completed XOR-name model runs."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any, Iterable

from run_xor_experiment import (
    classification_metrics,
    component_prediction,
    exact_mcnemar_p,
    latest_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
PROTOCOL_KEYS = (
    "prompt_version",
    "system_prompt",
    "temperature",
    "max_tokens",
    "seed",
    "atomic_dataset_sha256",
)
DATASET_KEYS = ("dataset", "rows", "sha256")


def parse_prediction(value: Any, *, context: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        prediction = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction {value!r} in {context}") from exc
    if prediction not in {0, 1}:
        raise ValueError(f"Invalid prediction {prediction!r} in {context}")
    return prediction


def majority_vote(values: Iterable[int | None], voter_count: int) -> int | None:
    """Return a strict majority of all configured voters, or None."""
    votes = list(values)
    if len(votes) != voter_count:
        raise ValueError(f"Expected {voter_count} votes, received {len(votes)}")
    positives = sum(value == 1 for value in votes)
    negatives = sum(value == 0 for value in votes)
    threshold = voter_count // 2 + 1
    if positives >= threshold:
        return 1
    if negatives >= threshold:
        return 0
    return None


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing predictions file: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            row_id = row.get("id")
            if not row_id:
                raise ValueError(f"Missing id at {path}:{line_number}")
            if row_id in rows:
                raise ValueError(f"Duplicate id {row_id!r} in {path}")
            row["gold_label"] = int(row["gold_label"])
            row["direct_prediction"] = parse_prediction(
                row.get("direct_prediction"), context=f"{path}:{line_number} direct"
            )
            row["component_prediction"] = parse_prediction(
                row.get("component_prediction"), context=f"{path}:{line_number} component"
            )
            rows[row_id] = row
    return rows


def atomic_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["name"]).casefold(),
        str(record["role"]),
        str(record["target_origin"]),
    )


def read_atoms(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    records = latest_jsonl(path)
    if not records:
        raise ValueError(f"No atomic records found in {path}")
    atoms: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records.values():
        key = atomic_key(record)
        if key in atoms:
            raise ValueError(f"Duplicate atomic key {key!r} in {path}")
        record = dict(record)
        record["gold_label"] = int(record["gold_label"])
        record["prediction"] = parse_prediction(
            record.get("prediction"), context=f"{path}:{record.get('id')}"
        )
        atoms[key] = record
    return atoms


def load_run(results_root: Path, run_name: str, dataset: str) -> dict[str, Any]:
    run_dir = results_root / run_name
    required = {
        "run_config": run_dir / "run_config.json",
        "dataset_config": run_dir / f"dataset_config_{dataset}.json",
        "metrics": run_dir / f"metrics_{dataset}.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise ValueError(f"Run {run_name!r} is missing {label}: {path}")
    metrics = read_json(required["metrics"])
    dataset_config = read_json(required["dataset_config"])
    predictions = read_predictions(run_dir / f"predictions_{dataset}.csv")
    atoms = read_atoms(run_dir / "atomic.jsonl")
    expected_rows = int(dataset_config["rows"])
    evaluated_rows = int(metrics["evaluated_rows"])
    if len(predictions) != expected_rows or evaluated_rows != expected_rows:
        raise ValueError(
            f"Run {run_name!r} is incomplete: dataset has {expected_rows} rows, "
            f"metrics report {evaluated_rows}, and predictions contain "
            f"{len(predictions)}. Finish the run without --limit before voting."
        )
    reported_atoms = int(metrics["atomic"]["overall"]["n"])
    if len(atoms) != reported_atoms:
        raise ValueError(
            f"Run {run_name!r} has {len(atoms)} atomic records but its metrics "
            f"report {reported_atoms}"
        )
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "run_config": read_json(required["run_config"]),
        "dataset_config": dataset_config,
        "metrics": metrics,
        "predictions": predictions,
        "atoms": atoms,
        "model_key": metrics["model_key"],
        "model_id": metrics["model_id"],
    }


def validate_runs(runs: list[dict[str, Any]]) -> None:
    reference = runs[0]
    reference_ids = set(reference["predictions"])
    reference_atom_keys = set(reference["atoms"])
    reference_dataset = {
        key: reference["dataset_config"].get(key) for key in DATASET_KEYS
    }
    reference_protocol = {
        key: reference["run_config"].get(key) for key in PROTOCOL_KEYS
    }
    metadata_fields = (
        "full_name",
        "first_name",
        "last_name",
        "first_origin",
        "last_origin",
        "pair_type",
        "gold_label",
    )
    atom_fields = ("name", "role", "target_origin", "true_origin", "gold_label")

    for run in runs[1:]:
        dataset = {key: run["dataset_config"].get(key) for key in DATASET_KEYS}
        if dataset != reference_dataset:
            raise ValueError(
                f"Dataset configuration differs between {reference['run_name']!r} "
                f"and {run['run_name']!r}"
            )
        protocol = {key: run["run_config"].get(key) for key in PROTOCOL_KEYS}
        if protocol != reference_protocol:
            raise ValueError(
                f"Prompt or sampling protocol differs between {reference['run_name']!r} "
                f"and {run['run_name']!r}"
            )
        if set(run["predictions"]) != reference_ids:
            raise ValueError(f"Prediction ids differ for run {run['run_name']!r}")
        if set(run["atoms"]) != reference_atom_keys:
            raise ValueError(f"Atomic query keys differ for run {run['run_name']!r}")

        for row_id in reference_ids:
            left = reference["predictions"][row_id]
            right = run["predictions"][row_id]
            if any(left[field] != right[field] for field in metadata_fields):
                raise ValueError(
                    f"Row metadata differs for {row_id!r} in run {run['run_name']!r}"
                )
        for key in reference_atom_keys:
            left = reference["atoms"][key]
            right = run["atoms"][key]
            if any(left[field] != right[field] for field in atom_fields):
                raise ValueError(
                    f"Atomic metadata differs for {key!r} in run {run['run_name']!r}"
                )


def metrics_by_pair_type(
    rows: list[dict[str, Any]], prediction_field: str
) -> dict[str, dict[str, Any]]:
    pair_types = sorted({str(row["pair_type"]) for row in rows})
    return {
        pair_type: classification_metrics(
            (int(row["gold_label"]), row[prediction_field])
            for row in rows
            if row["pair_type"] == pair_type
        )
        for pair_type in pair_types
    }


def scored_metrics(
    rows: list[dict[str, Any]], prediction_field: str
) -> dict[str, Any]:
    return {
        "overall": classification_metrics(
            (int(row["gold_label"]), row[prediction_field]) for row in rows
        ),
        "by_pair_type": metrics_by_pair_type(rows, prediction_field),
    }


def agreement_summary(votes_by_item: Iterable[list[int | None]]) -> dict[str, int]:
    summary = {"unanimous": 0, "split_vote": 0, "no_majority": 0}
    for votes in votes_by_item:
        valid = [vote for vote in votes if vote is not None]
        if len(valid) == len(votes) and len(set(valid)) == 1:
            summary["unanimous"] += 1
        elif majority_vote(votes, len(votes)) is None:
            summary["no_majority"] += 1
        else:
            summary["split_vote"] += 1
    return summary


def pairwise_disagreement(
    runs: list[dict[str, Any]], item_kind: str, prediction_field: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for left, right in combinations(runs, 2):
        left_items = left[item_kind]
        right_items = right[item_kind]
        valid = disagree = 0
        for key in left_items:
            left_prediction = left_items[key][prediction_field]
            right_prediction = right_items[key][prediction_field]
            if left_prediction is None or right_prediction is None:
                continue
            valid += 1
            disagree += int(left_prediction != right_prediction)
        name = f"{left['run_name']}__{right['run_name']}"
        output[name] = {
            "valid_comparisons": valid,
            "disagreements": disagree,
            "disagreement_rate": disagree / valid if valid else None,
        }
    return output


def paired_comparison(
    rows: list[dict[str, Any]], left_field: str, right_field: str
) -> dict[str, Any]:
    left_only = right_only = both_correct = both_wrong = either_invalid = 0
    for row in rows:
        gold = int(row["gold_label"])
        left = row[left_field]
        right = row[right_field]
        if left is None or right is None:
            either_invalid += 1
            continue
        left_correct = left == gold
        right_correct = right == gold
        if left_correct and right_correct:
            both_correct += 1
        elif left_correct:
            left_only += 1
        elif right_correct:
            right_only += 1
        else:
            both_wrong += 1
    return {
        "both_correct": both_correct,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_wrong": both_wrong,
        "either_invalid": either_invalid,
        "mcnemar_exact_p_two_sided": exact_mcnemar_p(left_only, right_only),
    }


def write_predictions(
    path: Path, rows: list[dict[str, Any]], run_names: list[str]
) -> None:
    metadata = [
        "id",
        "full_name",
        "first_name",
        "last_name",
        "first_origin",
        "last_origin",
        "pair_type",
        "gold_label",
    ]
    individual_fields = [
        f"{run_name}_{method}_prediction"
        for run_name in run_names
        for method in ("direct", "component")
    ]
    ensemble_fields = [
        "direct_majority_prediction",
        "component_majority_prediction",
        "oracle_majority_component_prediction",
    ]
    fields = metadata + individual_fields + ensemble_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def summary_rows(
    individual_metrics: dict[str, Any], ensemble_metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, run_metrics in individual_metrics.items():
        for method in ("direct", "component"):
            rows.append(
                {
                    "method": f"{run_name}:{method}",
                    "kind": "individual",
                    "run_name": run_name,
                    "model_id": run_metrics["model_id"],
                    **run_metrics[method]["overall"],
                }
            )
    for method in (
        "direct_majority",
        "component_majority",
        "oracle_majority_component",
    ):
        individual_method = "direct" if method == "direct_majority" else "component"
        candidates = [
            (
                run_name,
                run_metrics[individual_method]["overall"]["accuracy"],
            )
            for run_name, run_metrics in individual_metrics.items()
            if run_metrics[individual_method]["overall"]["accuracy"] is not None
        ]
        best_run, best_accuracy = max(candidates, key=lambda item: item[1])
        overall = ensemble_metrics[method]["overall"]
        rows.append(
            {
                "method": f"ensemble:{method}",
                "kind": "ensemble",
                "run_name": "",
                "model_id": "",
                "comparison_target": f"{best_run}:{individual_method}",
                "accuracy_delta_vs_best_individual": (
                    overall["accuracy"] - best_accuracy
                    if overall["accuracy"] is not None
                    else None
                ),
                **overall,
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred_fields = [
        "method",
        "kind",
        "run_name",
        "model_id",
        "comparison_target",
        "accuracy_delta_vs_best_individual",
        "n",
        "valid",
        "invalid",
        "coverage",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in preferred_fields} for row in rows
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Odd number of completed result-directory names",
    )
    parser.add_argument("--dataset", choices=("core", "full"), default="core")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-run", default="ensemble-qwen-gemma-llama")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.runs) < 3 or len(args.runs) % 2 == 0:
        raise SystemExit("--runs must contain an odd number of runs (at least three)")
    if len(set(args.runs)) != len(args.runs):
        raise SystemExit("--runs contains duplicate run names")
    for run_name in [*args.runs, args.output_run]:
        if not RUN_NAME_RE.fullmatch(run_name):
            raise SystemExit(
                "Run names may contain only letters, digits, dots, underscores, and hyphens"
            )
    if args.output_run in args.runs:
        raise SystemExit("--output-run must differ from every input run")

    results_root = args.results_root.resolve()
    try:
        runs = [load_run(results_root, run_name, args.dataset) for run_name in args.runs]
        if len({run["model_id"] for run in runs}) != len(runs):
            raise ValueError("Every input run must use a distinct model")
        validate_runs(runs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    voter_count = len(runs)
    reference_rows = runs[0]["predictions"]
    row_ids = list(reference_rows)
    atom_keys = list(runs[0]["atoms"])

    voted_atoms: dict[tuple[str, str, str], int | None] = {}
    for key in atom_keys:
        voted_atoms[key] = majority_vote(
            [run["atoms"][key]["prediction"] for run in runs], voter_count
        )

    scored_rows: list[dict[str, Any]] = []
    for row_id in row_ids:
        base = reference_rows[row_id]
        scored: dict[str, Any] = {
            field: base[field]
            for field in (
                "id",
                "full_name",
                "first_name",
                "last_name",
                "first_origin",
                "last_origin",
                "pair_type",
                "gold_label",
            )
        }
        direct_votes: list[int | None] = []
        component_votes: list[int | None] = []
        for run in runs:
            prediction = run["predictions"][row_id]
            direct = prediction["direct_prediction"]
            component = prediction["component_prediction"]
            scored[f"{run['run_name']}_direct_prediction"] = direct
            scored[f"{run['run_name']}_component_prediction"] = component
            direct_votes.append(direct)
            component_votes.append(component)
        scored["direct_majority_prediction"] = majority_vote(
            direct_votes, voter_count
        )
        scored["component_majority_prediction"] = majority_vote(
            component_votes, voter_count
        )
        scored["oracle_majority_component_prediction"] = component_prediction(
            base, voted_atoms
        )
        scored_rows.append(scored)

    individual_metrics: dict[str, Any] = {}
    for run in runs:
        run_name = run["run_name"]
        individual_metrics[run_name] = {
            "model_key": run["model_key"],
            "model_id": run["model_id"],
            "atomic": classification_metrics(
                (
                    int(run["atoms"][key]["gold_label"]),
                    run["atoms"][key]["prediction"],
                )
                for key in atom_keys
            ),
            "direct": scored_metrics(scored_rows, f"{run_name}_direct_prediction"),
            "component": scored_metrics(
                scored_rows, f"{run_name}_component_prediction"
            ),
        }

    ensemble_metrics = {
        "direct_majority": scored_metrics(scored_rows, "direct_majority_prediction"),
        "component_majority": scored_metrics(
            scored_rows, "component_majority_prediction"
        ),
        "oracle_majority_component": scored_metrics(
            scored_rows, "oracle_majority_component_prediction"
        ),
        "atomic_majority": classification_metrics(
            (
                int(runs[0]["atoms"][key]["gold_label"]),
                voted_atoms[key],
            )
            for key in atom_keys
        ),
    }

    direct_votes_by_row = [
        [run["predictions"][row_id]["direct_prediction"] for run in runs]
        for row_id in row_ids
    ]
    component_votes_by_row = [
        [run["predictions"][row_id]["component_prediction"] for run in runs]
        for row_id in row_ids
    ]
    atomic_votes_by_key = [
        [run["atoms"][key]["prediction"] for run in runs] for key in atom_keys
    ]

    comparisons: dict[str, Any] = {}
    for ensemble_name, ensemble_field, individual_method in (
        ("direct_majority", "direct_majority_prediction", "direct"),
        ("component_majority", "component_majority_prediction", "component"),
        (
            "oracle_majority_component",
            "oracle_majority_component_prediction",
            "component",
        ),
    ):
        comparisons[ensemble_name] = {
            run["run_name"]: paired_comparison(
                scored_rows,
                ensemble_field,
                f"{run['run_name']}_{individual_method}_prediction",
            )
            for run in runs
        }

    output_dir = results_root / args.output_run
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / f"metrics_{args.dataset}.json",
        output_dir / f"predictions_{args.dataset}.csv",
        output_dir / "ensemble_config.json",
        output_dir / f"summary_{args.dataset}.csv",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing ensemble outputs: "
            + ", ".join(str(path) for path in existing)
        )

    metrics = {
        "dataset": args.dataset,
        "evaluated_rows": len(scored_rows),
        "voter_count": voter_count,
        "input_runs": [
            {
                "run_name": run["run_name"],
                "model_key": run["model_key"],
                "model_id": run["model_id"],
            }
            for run in runs
        ],
        "individuals": individual_metrics,
        "ensembles": ensemble_metrics,
        "agreement": {
            "direct": agreement_summary(direct_votes_by_row),
            "component": agreement_summary(component_votes_by_row),
            "atomic": agreement_summary(atomic_votes_by_key),
        },
        "pairwise_disagreement": {
            "direct": pairwise_disagreement(
                runs, "predictions", "direct_prediction"
            ),
            "component": pairwise_disagreement(
                runs, "predictions", "component_prediction"
            ),
            "atomic": pairwise_disagreement(runs, "atoms", "prediction"),
        },
        "paired_comparisons": comparisons,
    }

    with output_paths[0].open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_predictions(output_paths[1], scored_rows, args.runs)
    with output_paths[2].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_config": runs[0]["dataset_config"],
                "protocol": {
                    key: runs[0]["run_config"].get(key) for key in PROTOCOL_KEYS
                },
                "input_runs": args.runs,
                "voting_rule": "strict majority of all configured models",
                "invalid_vote_policy": (
                    "A prediction is invalid unless more than half of all models vote "
                    "for the same Boolean label."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    comparison_rows = summary_rows(individual_metrics, ensemble_metrics)
    write_summary(output_paths[3], comparison_rows)

    print(
        "method\taccuracy\tprecision\trecall\tfalse_positive_rate\t"
        "accuracy_delta_vs_best_individual"
    )
    for row in comparison_rows:
        print(
            f"{row['method']}\t{row['accuracy']}\t"
            f"{row['precision']}\t{row['recall']}\t"
            f"{row['false_positive_rate']}\t"
            f"{row.get('accuracy_delta_vs_best_individual', '')}"
        )
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
