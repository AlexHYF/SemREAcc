from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import csv
import json
import math
from types import ModuleType, SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_atomic_roc as atomic_roc  # noqa: E402
import summarize_atomic_roc as summarize_roc  # noqa: E402


class RocCurveTests(unittest.TestCase):
    def test_perfect_ranking_has_unit_auc(self) -> None:
        points = atomic_roc.roc_curve(
            labels=[0, 0, 1, 1],
            scores=[-2.0, -1.0, 1.0, 2.0],
        )

        self.assertAlmostEqual(atomic_roc.roc_auc(points), 1.0)
        self.assertEqual((points[0]["fpr"], points[0]["tpr"]), (0.0, 0.0))
        self.assertEqual((points[-1]["fpr"], points[-1]["tpr"]), (1.0, 1.0))

    def test_reversed_ranking_has_zero_auc(self) -> None:
        points = atomic_roc.roc_curve(
            labels=[0, 0, 1, 1],
            scores=[2.0, 1.0, -1.0, -2.0],
        )

        self.assertAlmostEqual(atomic_roc.roc_auc(points), 0.0)

    def test_tied_scores_are_processed_as_one_threshold_group(self) -> None:
        # Each score group contains one positive and one negative. Treating tied
        # examples one-by-one would make AUC depend on their input ordering.
        points = atomic_roc.roc_curve(
            labels=[1, 0, 1, 0],
            scores=[0.8, 0.8, 0.1, 0.1],
        )

        self.assertEqual(len(points), 3)
        self.assertEqual((points[1]["fpr"], points[1]["tpr"]), (0.5, 0.5))
        self.assertAlmostEqual(atomic_roc.roc_auc(points), 0.5)

    def test_all_tied_scores_have_chance_auc(self) -> None:
        points = atomic_roc.roc_curve(
            labels=[0, 1, 0, 1],
            scores=[0.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(atomic_roc.roc_auc(points), 0.5)

    def test_rejects_invalid_binary_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "label|binary"):
            atomic_roc.roc_curve([0, 1, 2], [0.1, 0.2, 0.3])

    def test_rejects_nonfinite_scores(self) -> None:
        for nonfinite in (math.nan, math.inf, -math.inf):
            with self.subTest(score=nonfinite):
                with self.assertRaisesRegex(ValueError, "finite|score"):
                    atomic_roc.roc_curve([0, 1], [0.0, nonfinite])

    def test_rejects_empty_or_one_class_input(self) -> None:
        for labels, scores in (([], []), ([0, 0], [0.0, 1.0]), ([1, 1], [0.0, 1.0])):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    atomic_roc.roc_curve(labels, scores)

    def test_stable_sigmoid_handles_extreme_log_odds(self) -> None:
        self.assertEqual(atomic_roc.stable_sigmoid(1000.0), 1.0)
        self.assertEqual(atomic_roc.stable_sigmoid(-1000.0), 0.0)
        self.assertEqual(atomic_roc.stable_sigmoid(0.0), 0.5)


class CandidateTokenizationTests(unittest.TestCase):
    class BoundaryMergingTokenizer:
        """Tiny tokenizer fixture whose prompt token changes at the boundary."""

        encodings = {
            "prompt": [99],
            "promptYES": [10, 21, 22],
            "promptNO": [10, 31],
            # These deliberately differ from the continuation suffixes. A
            # standalone-label fallback would therefore produce a wrong test.
            "YES": [121],
            "NO": [131],
        }

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if add_special_tokens:
                raise AssertionError("candidate encoding must not add special tokens")
            return list(self.encodings[text])

    def test_candidate_split_uses_common_prefix_of_complete_encodings(self) -> None:
        tokenizer = self.BoundaryMergingTokenizer()

        prompt_ids, suffixes = atomic_roc.split_candidate_tokens(
            tokenizer,
            "prompt",
            {"yes": "YES", "no": "NO"},
        )

        self.assertEqual(prompt_ids, [10])
        self.assertEqual(suffixes, {"yes": [21, 22], "no": [31]})

    def test_answer_positions_predict_each_continuation_token(self) -> None:
        # For a continuation beginning at input position 7, causal-LM logits at
        # positions 6, 7, and 8 predict its three tokens.
        self.assertEqual(list(atomic_roc.answer_logit_positions(7, 3)), [6, 7, 8])

    def test_multimodal_adapter_wraps_text_content_for_processor(self) -> None:
        class Processor:
            tokenizer = self.BoundaryMergingTokenizer()
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "rendered"

        processor = Processor()
        adapter = atomic_roc.ChatTokenizerAdapter(
            processor, tokenizer=processor.tokenizer, multimodal=True
        )
        rendered = adapter.apply_chat_template(
            [{"role": "user", "content": "hello"}], tokenize=False
        )

        self.assertEqual(rendered, "rendered")
        self.assertEqual(
            processor.messages,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        )
        self.assertEqual(processor.kwargs, {"tokenize": False})


class BatchScoringTests(unittest.TestCase):
    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = np.asarray(values)
            self.device = "cpu"

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, key):
            return BatchScoringTests.FakeTensor(self.values[key])

        def __setitem__(self, key, value) -> None:
            values = value.values if isinstance(value, self.__class__) else value
            self.values[key] = values

        def float(self):
            return self.__class__(self.values.astype(float))

        def index_select(self, dimension: int, indexes):
            return self.__class__(
                np.take(self.values, indexes.values.astype(int), axis=dimension)
            )

        def gather(self, dimension: int, indexes):
            return self.__class__(
                np.take_along_axis(
                    self.values, indexes.values.astype(int), axis=dimension
                )
            )

        def unsqueeze(self, dimension: int):
            return self.__class__(np.expand_dims(self.values, axis=dimension))

        def squeeze(self, dimension: int):
            return self.__class__(np.squeeze(self.values, axis=dimension))

        def sum(self):
            return self.__class__(self.values.sum())

        def item(self):
            return self.values.item()

        def tolist(self):
            return self.values.tolist()

    class FakeTorch:
        long = np.int64

        @staticmethod
        def full(shape, value, *, dtype, device):
            return BatchScoringTests.FakeTensor(
                np.full(shape, value, dtype=dtype)
            )

        @staticmethod
        def zeros(shape, *, dtype, device):
            return BatchScoringTests.FakeTensor(np.zeros(shape, dtype=dtype))

        @staticmethod
        def tensor(values, *, dtype, device):
            return BatchScoringTests.FakeTensor(np.asarray(values, dtype=dtype))

        @staticmethod
        def inference_mode():
            return nullcontext()

        @staticmethod
        def log_softmax(tensor, dim: int):
            values = tensor.values.astype(float)
            maximum = np.max(values, axis=dim, keepdims=True)
            shifted = values - maximum
            normalized = shifted - np.log(
                np.exp(shifted).sum(axis=dim, keepdims=True)
            )
            return BatchScoringTests.FakeTensor(normalized)

    class FakeModel:
        config = SimpleNamespace(pad_token_id=0, eos_token_id=0)

        def __call__(self, *, input_ids, attention_mask, use_cache):
            self.input_ids = input_ids.values.copy()
            self.attention_mask = attention_mask.values.copy()
            batch, length = self.input_ids.shape
            logits = np.zeros((batch, length, 10), dtype=float)
            # Row 0 is the paired single-token candidate. Rows 1 and 2 are
            # the YES and NO teacher-forced multi-token candidates.
            logits[0, 1, 3] = 2.0
            logits[0, 1, 4] = 0.5
            logits[1, 0, 6] = logits[2, 0, 6] = 1.0
            logits[1, 0, 8] = logits[2, 0, 8] = 0.25
            logits[1, 1, 7] = 1.5
            logits[2, 1, 9] = -0.5
            return SimpleNamespace(logits=BatchScoringTests.FakeTensor(logits))

    def test_scores_single_token_logits_and_multi_token_logprobs(self) -> None:
        prepared = [
            {
                "prompt_ids": [1, 2],
                "candidates": {
                    "yes": {"answer_ids": [3]},
                    "no": {"answer_ids": [4]},
                },
            },
            {
                "prompt_ids": [5],
                "candidates": {
                    "yes": {"answer_ids": [6, 7]},
                    "no": {"answer_ids": [8, 9]},
                },
            },
        ]
        model = self.FakeModel()

        scores = atomic_roc.score_prepared_batch(
            model, self.FakeTorch, prepared, "cpu"
        )

        self.assertEqual(model.input_ids.shape, (3, 3))
        np.testing.assert_array_equal(model.attention_mask[0], [1, 1, 0])
        self.assertEqual(scores[0]["yes"]["token_logits"], [2.0])
        self.assertEqual(scores[0]["no"]["token_logits"], [0.5])

        def log_probability(logits, target):
            logits = np.asarray(logits, dtype=float)
            return logits[target] - np.log(np.exp(logits).sum())

        first = np.zeros(10)
        first[6], first[8] = 1.0, 0.25
        yes_second = np.zeros(10)
        yes_second[7] = 1.5
        no_second = np.zeros(10)
        no_second[9] = -0.5
        expected_yes = log_probability(first, 6) + log_probability(
            yes_second, 7
        )
        expected_no = log_probability(first, 8) + log_probability(no_second, 9)
        self.assertAlmostEqual(
            scores[1]["yes"]["sequence_logprob"], expected_yes
        )
        self.assertAlmostEqual(
            scores[1]["no"]["sequence_logprob"], expected_no
        )
        self.assertEqual(scores[1]["yes"]["token_logits"], [1.0, 1.5])
        self.assertEqual(scores[1]["no"]["token_logits"], [0.25, -0.5])


class ModelLoaderTests(unittest.TestCase):
    def test_dispatches_configured_processor_and_model_classes(self) -> None:
        torch_module = ModuleType("torch")
        torch_module.bfloat16 = "bf16"
        torch_module.float16 = "fp16"
        torch_module.float32 = "fp32"
        torch_module.__version__ = "test-torch"
        torch_module.cuda = SimpleNamespace(is_available=lambda: False)

        processor = SimpleNamespace(tokenizer=SimpleNamespace())

        class ProcessorLoader:
            calls = []

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.calls.append((model_id, kwargs))
                return processor

        model = SimpleNamespace(
            to=lambda device: setattr(model, "device", device),
            eval=lambda: setattr(model, "evaluated", True),
        )

        class ModelLoader:
            calls = []

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.calls.append((model_id, kwargs))
                return model

        transformers_module = ModuleType("transformers")
        transformers_module.__version__ = "test-transformers"
        transformers_module.TestProcessor = ProcessorLoader
        transformers_module.TestModel = ModelLoader

        config = {
            "model_id": "org/model",
            "transformers_processor_class": "TestProcessor",
            "transformers_model_class": "TestModel",
            "transformers_args": ["--trust-remote-code"],
        }
        with patch.dict(
            sys.modules,
            {"torch": torch_module, "transformers": transformers_module},
        ):
            loaded = atomic_roc.load_model_stack(config, "cpu", "bfloat16")

        self.assertIs(loaded[0], torch_module)
        self.assertIs(loaded[1], transformers_module)
        self.assertIs(loaded[3], model)
        self.assertEqual(loaded[4], "TestModel")
        self.assertEqual(
            ProcessorLoader.calls,
            [("org/model", {"trust_remote_code": True})],
        )
        self.assertEqual(
            ModelLoader.calls,
            [
                (
                    "org/model",
                    {"dtype": "bf16", "trust_remote_code": True},
                )
            ],
        )
        self.assertEqual(model.device, "cpu")
        self.assertTrue(model.evaluated)


class DatasetTests(unittest.TestCase):
    def test_atomic_dataset_has_four_balanced_roc_groups(self) -> None:
        dataset = ROOT / "data" / "xor_names" / "individual_queries.csv"
        with dataset.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 800)
        self.assertEqual(len({row["id"] for row in rows}), 800)
        counts = Counter(
            (f"{row['role']}_{row['target_origin']}", int(row["label"]))
            for row in rows
        )
        self.assertEqual(set(query for query, _ in counts), set(atomic_roc.QUERY_ORDER))
        for query in atomic_roc.QUERY_ORDER:
            self.assertEqual(counts[(query, 0)], 100)
            self.assertEqual(counts[(query, 1)], 100)


class SvgTests(unittest.TestCase):
    def test_writes_all_four_roc_panels(self) -> None:
        points = atomic_roc.roc_curve([0, 1], [-1.0, 1.0])
        series_by_query = {
            query: [{"label": "test-model", "points": points, "auc": 1.0}]
            for query in atomic_roc.QUERY_ORDER
        }

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "roc.svg"
            atomic_roc.write_roc_svg(
                output,
                series_by_query,
                title="Atomic ROC <smoke test>",
            )
            svg = output.read_text(encoding="utf-8")

        self.assertIn("<svg", svg)
        self.assertIn("Atomic ROC &lt;smoke test&gt;", svg)
        self.assertEqual(svg.count("test-model (AUC 1.000)"), 4)
        for title in atomic_roc.QUERY_TITLES.values():
            self.assertIn(title, svg)


class AggregateTests(unittest.TestCase):
    def test_combines_model_summaries_without_losing_identity_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for model_index, model_key in enumerate(("model-a", "model-b")):
                run_dir = output_root / model_key
                records = []
                for query_index, query in enumerate(atomic_roc.QUERY_ORDER):
                    role, target_origin = query.split("_")
                    for label, score in ((0, -1.0), (1, 1.0)):
                        records.append(
                            {
                                "id": f"{model_key}-{query}-{label}",
                                "name": f"name-{query_index}-{label}",
                                "role": role,
                                "target_origin": target_origin,
                                "true_origin": target_origin if label else "J",
                                "source_rank": label + 1,
                                "gold_label": label,
                                "yes_token_count": 1,
                                "no_token_count": 1,
                                "yes_first_token_logit": score,
                                "no_first_token_logit": 0.0,
                                "single_token_logit_difference": score,
                                "yes_logprob": score,
                                "no_logprob": 0.0,
                                "sequence_logprob_difference": score,
                                "score_mode": "single-token-logit",
                                "log_odds_yes_vs_no": score,
                                "yes_probability_normalized": (
                                    atomic_roc.stable_sigmoid(score)
                                ),
                                "prediction_at_zero": int(score >= 0.0),
                            }
                        )
                atomic_roc.analyze_records(
                    records=records,
                    model_key=model_key,
                    model_id=f"org/{model_key}",
                    dataset_sha256="dataset-hash",
                    output_dir=run_dir,
                )
                config = {
                    "model_key": model_key,
                    "model_id": f"org/{model_key}",
                    "dataset_sha256": "dataset-hash",
                    "score_version": atomic_roc.SCORE_VERSION,
                    "prompt_version": "prompt-v1",
                    "system_prompt": "system",
                    "answer_texts": atomic_roc.ANSWER_TEXTS,
                    "score_definition": "test score",
                    "dtype": "bfloat16",
                    "transformers_version": "test",
                    "transformers_vcs_commit": "test-commit",
                    "mistral_common_version": "test",
                    "torch_version": "test",
                    "protocol_fingerprint": f"fingerprint-{model_index}",
                }
                (run_dir / "run_config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )

            with patch.object(
                sys,
                "argv",
                [
                    "summarize_atomic_roc.py",
                    "--runs",
                    "model-a",
                    "model-b",
                    "--output-root",
                    str(output_root),
                ],
            ):
                self.assertEqual(summarize_roc.main(), 0)

            with (output_root / "summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                {row["model_key"] for row in rows}, {"model-a", "model-b"}
            )
            self.assertEqual({row["n"] for row in rows}, {"2"})
            self.assertEqual({row["auc"] for row in rows}, {"1.0"})


if __name__ == "__main__":
    unittest.main()
