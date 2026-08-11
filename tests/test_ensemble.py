from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_ensemble as ensemble  # noqa: E402
import run_xor_experiment as experiment  # noqa: E402


class MajorityVoteTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        self.assertEqual(ensemble.majority_vote([1, 1, 0], 3), 1)
        self.assertEqual(ensemble.majority_vote([0, 0, 1], 3), 0)
        self.assertEqual(ensemble.majority_vote([1, 1, None], 3), 1)
        self.assertIsNone(ensemble.majority_vote([1, 0, None], 3))

    def test_vote_then_compose_can_differ_from_compose_then_vote(self) -> None:
        row = {"first_name": "Wei", "last_name": "Sato"}
        keys = [
            ("wei", "given", "C"),
            ("wei", "given", "J"),
            ("sato", "surname", "C"),
            ("sato", "surname", "J"),
        ]
        atom_maps = [
            dict(zip(keys, [1, 0, 0, 1], strict=True)),
            dict(zip(keys, [0, 1, 1, 0], strict=True)),
            dict(zip(keys, [0, 0, 0, 0], strict=True)),
        ]
        final_votes = [
            experiment.component_prediction(row, atoms) for atoms in atom_maps
        ]
        voted_atoms = {
            key: ensemble.majority_vote([atoms[key] for atoms in atom_maps], 3)
            for key in keys
        }

        self.assertEqual(final_votes, [1, 1, 0])
        self.assertEqual(ensemble.majority_vote(final_votes, 3), 1)
        self.assertEqual(experiment.component_prediction(row, voted_atoms), 0)

    def test_invalid_prediction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ensemble.parse_prediction("YES", context="test")
        with self.assertRaises(ValueError):
            ensemble.parse_prediction(2, context="test")

    def test_end_to_end_evaluator(self) -> None:
        rows = [
            {
                "id": "row-1",
                "full_name": "Wei Sato",
                "first_name": "Wei",
                "last_name": "Sato",
                "first_origin": "C",
                "last_origin": "J",
                "pair_type": "CJ",
                "gold_label": 1,
            },
            {
                "id": "row-2",
                "full_name": "Wei Li",
                "first_name": "Wei",
                "last_name": "Li",
                "first_origin": "C",
                "last_origin": "C",
                "pair_type": "CC",
                "gold_label": 0,
            },
        ]
        atom_specs = [
            ("Wei", "given", "C", "C", 1),
            ("Wei", "given", "J", "C", 0),
            ("Sato", "surname", "C", "J", 0),
            ("Sato", "surname", "J", "J", 1),
            ("Li", "surname", "C", "C", 1),
            ("Li", "surname", "J", "C", 0),
        ]
        direct_votes = ([1, 1], [1, 0], [0, 0])

        with tempfile.TemporaryDirectory() as temporary:
            results_root = Path(temporary)
            for index, run_name in enumerate(("small-a", "small-b", "small-c")):
                run_dir = results_root / run_name
                run_dir.mkdir()
                (run_dir / "run_config.json").write_text(
                    json.dumps(
                        {
                            "prompt_version": "test-v1",
                            "system_prompt": "test",
                            "temperature": 0,
                            "max_tokens": 1,
                            "seed": 0,
                            "atomic_dataset_sha256": "atoms",
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "dataset_config_core.json").write_text(
                    json.dumps(
                        {
                            "dataset": "core",
                            "path": f"/different/machine/{index}",
                            "rows": 2,
                            "sha256": "rows",
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "metrics_core.json").write_text(
                    json.dumps(
                        {
                            "model_key": f"model-{index}",
                            "model_id": f"Model {index}",
                            "evaluated_rows": 2,
                            "atomic": {"overall": {"n": 6}},
                        }
                    ),
                    encoding="utf-8",
                )
                with (run_dir / "predictions_core.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    fieldnames = [
                        *rows[0],
                        "direct_prediction",
                        "component_prediction",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row, direct in zip(rows, direct_votes[index], strict=True):
                        writer.writerow(
                            {
                                **row,
                                "direct_prediction": direct,
                                "component_prediction": row["gold_label"],
                            }
                        )
                with (run_dir / "atomic.jsonl").open("w", encoding="utf-8") as handle:
                    for atom_index, atom in enumerate(atom_specs):
                        name, role, target, true_origin, gold = atom
                        handle.write(
                            json.dumps(
                                {
                                    "id": f"atom-{atom_index}",
                                    "name": name,
                                    "role": role,
                                    "target_origin": target,
                                    "true_origin": true_origin,
                                    "gold_label": gold,
                                    "prediction": gold,
                                }
                            )
                            + "\n"
                        )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_ensemble.py"),
                    "--runs",
                    "small-a",
                    "small-b",
                    "small-c",
                    "--results-root",
                    str(results_root),
                    "--output-run",
                    "voted",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("ensemble:direct_majority\t1.0", completed.stdout)
            with (results_root / "voted" / "summary_core.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                summary = {row["method"]: row for row in csv.DictReader(handle)}
            self.assertEqual(summary["ensemble:direct_majority"]["accuracy"], "1.0")
            self.assertEqual(
                summary["ensemble:oracle_majority_component"]["accuracy"], "1.0"
            )


if __name__ == "__main__":
    unittest.main()
