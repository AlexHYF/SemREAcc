from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_xor_experiment as experiment  # noqa: E402
import serve_model  # noqa: E402


class AnswerParsingTests(unittest.TestCase):
    def test_plain_answers(self) -> None:
        self.assertEqual(experiment.parse_yes_no("YES"), 1)
        self.assertEqual(experiment.parse_yes_no("No."), 0)
        self.assertIsNone(experiment.parse_yes_no("possibly"))

    def test_thinking_is_ignored(self) -> None:
        response = "<think>I must choose yes or no.</think>\nYES"
        self.assertEqual(experiment.parse_yes_no(response), 1)
        self.assertIsNone(experiment.parse_yes_no("<think>I must choose yes or no"))

    def test_conflicting_unstructured_answer_is_invalid(self) -> None:
        self.assertIsNone(experiment.parse_yes_no("It could be YES or NO."))
        self.assertEqual(experiment.parse_yes_no("It could be yes or no.\nAnswer: NO"), 0)

    def test_json_answer(self) -> None:
        self.assertEqual(experiment.parse_yes_no('{"answer": "NO"}'), 0)


class LogicTests(unittest.TestCase):
    def test_three_valued_logic(self) -> None:
        self.assertEqual(experiment.kleene_and(1, 1), 1)
        self.assertEqual(experiment.kleene_and(None, 0), 0)
        self.assertIsNone(experiment.kleene_and(None, 1))
        self.assertEqual(experiment.kleene_or(None, 1), 1)
        self.assertEqual(experiment.kleene_or(0, 0), 0)
        self.assertIsNone(experiment.kleene_or(None, 0))

    def test_crossed_name_recombination(self) -> None:
        atoms = {
            ("wei", "given", "C"): 1,
            ("wei", "given", "J"): 0,
            ("sato", "surname", "C"): 0,
            ("sato", "surname", "J"): 1,
        }
        row = {"first_name": "Wei", "last_name": "Sato"}
        self.assertEqual(experiment.component_prediction(row, atoms), 1)


class MetricTests(unittest.TestCase):
    def test_invalid_is_wrong_in_primary_accuracy(self) -> None:
        metrics = experiment.classification_metrics([(1, 1), (0, 1), (0, 0), (1, None)])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["valid_accuracy"], 2 / 3)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["invalid"], 1)

    def test_mcnemar(self) -> None:
        self.assertIsNone(experiment.exact_mcnemar_p(0, 0))
        self.assertEqual(experiment.exact_mcnemar_p(1, 0), 1.0)


class ManifestTests(unittest.TestCase):
    def test_all_requested_models_build_vllm_commands(self) -> None:
        models = json.loads((ROOT / "configs" / "models.json").read_text())
        self.assertEqual(len(models), 12)
        for config in models.values():
            command = serve_model.build_command(
                config,
                host="0.0.0.0",
                port=8000,
                tensor_parallel_size=config["recommended_tensor_parallel_size"],
                extra_args=[],
            )
            self.assertEqual(command[:2], ["vllm", "serve"])
            self.assertIn(config["model_id"], command)

    def test_ensemble_models_have_similar_sizes_and_distinct_families(self) -> None:
        models = json.loads((ROOT / "configs" / "models.json").read_text())
        keys = ("qwen35-4b", "phi4-mini-3.8b", "ministral3-3b")
        sizes = [models[key]["total_parameters_b"] for key in keys]
        publishers = {models[key]["model_id"].split("/", 1)[0] for key in keys}
        self.assertLessEqual(max(sizes) - min(sizes), 1.0)
        self.assertEqual(len(publishers), 3)

    def test_finds_vllm_beside_virtualenv_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary) / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            vllm = bin_dir / "vllm"
            python.touch()
            vllm.touch(mode=0o755)
            with patch.object(serve_model.shutil, "which", return_value=None), patch.object(
                serve_model.sys, "executable", str(python)
            ):
                self.assertEqual(serve_model.find_vllm_executable(), str(vllm))

    def test_builds_transformers_serve_command(self) -> None:
        config = {"model_id": "Qwen/Qwen3.5-4B"}
        command = serve_model.build_transformers_command(
            config, host="0.0.0.0", port=8000, extra_args=[]
        )
        self.assertEqual(command[:3], ["transformers", "serve", config["model_id"]])
        self.assertIn("--continuous-batching", command)


class EndToEndTests(unittest.TestCase):
    def test_checkpoint_and_score_against_openai_compatible_endpoint(self) -> None:
        calls: list[str] = []

        def complete_no(client: experiment.ChatClient, prompt: str) -> dict[str, object]:
            calls.append(prompt)
            return {
                "raw_response": "NO",
                "prediction": 0,
                "latency_seconds": 0.01,
                "attempt_count": 1,
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "finish_reason": "stop",
                "error": None,
            }

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(experiment.ChatClient, "complete", complete_no):
                argv = [
                    "run_xor_experiment.py",
                    "--model-key",
                    "qwen35-0.8b",
                    "--output-root",
                    temporary,
                    "--run-name",
                    "e2e",
                    "--limit",
                    "6",
                    "--concurrency",
                    "4",
                ]
                with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(experiment.main(), 0)
                first_count = len(calls)
                self.assertGreater(first_count, 6)
                run_dir = Path(temporary) / "e2e"
                metrics = json.loads((run_dir / "metrics_core.json").read_text())
                self.assertEqual(metrics["evaluated_rows"], 6)
                self.assertEqual(metrics["direct"]["overall"]["accuracy"], 1.0)
                self.assertTrue((run_dir / "predictions_core.csv").exists())

                # The second invocation should use both JSONL checkpoints.
                with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(experiment.main(), 0)
                self.assertEqual(len(calls), first_count)


if __name__ == "__main__":
    unittest.main()
