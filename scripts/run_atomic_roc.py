#!/usr/bin/env python3
"""Score atomic YES/NO answers with Transformers and produce ROC curves."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
from importlib import metadata
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable

from run_xor_experiment import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    atomic_prompt,
    classification_metrics,
    file_sha256,
    latest_jsonl,
    read_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "models.json"
DEFAULT_DATASET = ROOT / "data" / "xor_names" / "individual_queries.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "atomic-roc"
RUN_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
SCORE_VERSION = "atomic-yes-no-sequence-logprob-v1"
ANSWER_TEXTS = {"yes": "YES", "no": "NO"}
QUERY_ORDER = ("given_C", "given_J", "surname_C", "surname_J")
QUERY_TITLES = {
    "given_C": "Chinese given name",
    "given_J": "Japanese given name",
    "surname_C": "Chinese surname",
    "surname_J": "Japanese surname",
}
COLORS = ("#1769aa", "#d1495b", "#2a9d8f", "#7b2cbf", "#e76f51")


class ChatTokenizerAdapter:
    """Expose one text-tokenizer interface for text and multimodal checkpoints."""

    def __init__(
        self,
        processor: Any,
        *,
        tokenizer: Any | None = None,
        multimodal: bool,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor
        self.multimodal = multimodal

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(
            self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Any:
        if self.multimodal:
            messages_for_template = [
                {
                    "role": message["role"],
                    "content": [
                        {"type": "text", "text": message["content"]}
                    ],
                }
                for message in messages
            ]
        else:
            messages_for_template = messages
        return self.processor.apply_chat_template(messages_for_template, **kwargs)

    @property
    def chat_template(self) -> Any:
        return getattr(self.processor, "chat_template", None) or getattr(
            self.tokenizer, "chat_template", None
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def content_sha256(value: Any) -> str:
    if isinstance(value, str):
        content = value
    else:
        content = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    return hashlib.sha256(content.encode()).hexdigest()


def checkpoint_commit_hash(model: Any, tokenizer: ChatTokenizerAdapter) -> str | None:
    candidates = (
        getattr(getattr(model, "config", None), "_commit_hash", None),
        getattr(tokenizer.processor, "_commit_hash", None),
        getattr(tokenizer.tokenizer, "_commit_hash", None),
        (getattr(tokenizer.processor, "init_kwargs", None) or {}).get(
            "_commit_hash"
        ),
        (getattr(tokenizer.tokenizer, "init_kwargs", None) or {}).get(
            "_commit_hash"
        ),
    )
    return next((value for value in candidates if value), None)


def distribution_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def distribution_vcs_commit(package: str) -> str | None:
    try:
        direct_url = metadata.distribution(package).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        parsed = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    return parsed.get("vcs_info", {}).get("commit_id")


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def validated_label_score_pairs(
    labels: Iterable[int], scores: Iterable[float]
) -> list[tuple[int, float]]:
    pairs = [
        (int(label), float(score))
        for label, score in zip(labels, scores, strict=True)
    ]
    for label, score in pairs:
        if label not in (0, 1):
            raise ValueError(f"ROC labels must be 0 or 1, got {label!r}")
        if not math.isfinite(score):
            raise ValueError(f"ROC scores must be finite, got {score!r}")
    return pairs


def query_key(row: dict[str, Any]) -> str:
    return f"{row['role']}_{row['target_origin']}"


def select_limit_per_class(
    rows: list[dict[str, str]], limit_per_class: int | None
) -> list[dict[str, str]]:
    if limit_per_class is None:
        return rows
    selected: list[dict[str, str]] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        group = (query_key(row), row["label"])
        if counts[group] < limit_per_class:
            selected.append(row)
            counts[group] += 1
    expected = {
        (query, label) for query in QUERY_ORDER for label in ("0", "1")
    }
    missing = sorted(expected - set(counts))
    if missing:
        raise ValueError(f"Dataset is missing query/label groups: {missing}")
    return selected


def roc_curve(labels: Iterable[int], scores: Iterable[float]) -> list[dict[str, Any]]:
    pairs = validated_label_score_pairs(labels, scores)
    positives = sum(label == 1 for label, _ in pairs)
    negatives = sum(label == 0 for label, _ in pairs)
    if not pairs or positives == 0 or negatives == 0:
        raise ValueError("ROC requires at least one positive and one negative example")
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    points: list[dict[str, Any]] = [
        {
            "threshold_log_odds": math.inf,
            "threshold_yes_probability": 1.0,
            "fpr": 0.0,
            "tpr": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": negatives,
            "fn": positives,
        }
    ]
    tp = fp = 0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][1]
        while index < len(pairs) and pairs[index][1] == threshold:
            label, _ = pairs[index]
            tp += int(label == 1)
            fp += int(label == 0)
            index += 1
        points.append(
            {
                "threshold_log_odds": threshold,
                "threshold_yes_probability": stable_sigmoid(threshold),
                "fpr": fp / negatives,
                "tpr": tp / positives,
                "tp": tp,
                "fp": fp,
                "tn": negatives - fp,
                "fn": positives - tp,
            }
        )
    return points


def roc_auc(points: list[dict[str, Any]]) -> float:
    area = 0.0
    for left, right in zip(points, points[1:]):
        width = float(right["fpr"]) - float(left["fpr"])
        area += width * (float(left["tpr"]) + float(right["tpr"])) / 2.0
    return area


def operating_point(labels: list[int], scores: list[float], threshold: float = 0.0) -> dict[str, Any]:
    pairs = validated_label_score_pairs(labels, scores)
    if not math.isfinite(threshold):
        raise ValueError("Operating-point threshold must be finite")
    metrics = classification_metrics(
        (label, int(score >= threshold)) for label, score in pairs
    )
    return {
        "threshold_log_odds": threshold,
        "threshold_yes_probability": stable_sigmoid(threshold),
        **metrics,
    }


def _svg_text(x: float, y: float, value: str, **attributes: Any) -> str:
    attrs = " ".join(f'{key.replace("_", "-")}="{escape(str(item))}"' for key, item in attributes.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{escape(value)}</text>'


def write_roc_svg(
    path: Path,
    series_by_query: dict[str, list[dict[str, Any]]],
    *,
    title: str,
) -> None:
    width, height = 1000, 820
    panel_width, panel_height = 430, 315
    origins = ((75, 100), (555, 100), (75, 465), (555, 465))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 38, title, text_anchor="middle", font_size="24", font_family="sans-serif", font_weight="600"),
    ]
    for query, (origin_x, origin_y) in zip(QUERY_ORDER, origins, strict=True):
        plot_x = origin_x + 48
        plot_y = origin_y + 36
        plot_width = panel_width - 82
        plot_height = panel_height - 80
        elements.append(
            _svg_text(
                origin_x + panel_width / 2,
                origin_y + 18,
                QUERY_TITLES[query],
                text_anchor="middle",
                font_size="17",
                font_family="sans-serif",
                font_weight="600",
            )
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = plot_x + tick * plot_width
            y = plot_y + (1.0 - tick) * plot_height
            elements.append(
                f'<line x1="{x:.1f}" y1="{plot_y:.1f}" x2="{x:.1f}" y2="{plot_y + plot_height:.1f}" stroke="#e6e6e6" stroke-width="1"/>'
            )
            elements.append(
                f'<line x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>'
            )
            elements.append(_svg_text(x, plot_y + plot_height + 18, f"{tick:g}", text_anchor="middle", font_size="11", font_family="sans-serif"))
            elements.append(_svg_text(plot_x - 10, y + 4, f"{tick:g}", text_anchor="end", font_size="11", font_family="sans-serif"))
        elements.extend(
            [
                f'<rect x="{plot_x:.1f}" y="{plot_y:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}" fill="none" stroke="#333" stroke-width="1.2"/>',
                f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y:.1f}" stroke="#999" stroke-width="1.2" stroke-dasharray="5 4"/>',
                _svg_text(plot_x + plot_width / 2, plot_y + plot_height + 39, "False positive rate", text_anchor="middle", font_size="13", font_family="sans-serif"),
                f'<text x="{plot_x - 38:.1f}" y="{plot_y + plot_height / 2:.1f}" text-anchor="middle" font-size="13" font-family="sans-serif" transform="rotate(-90 {plot_x - 38:.1f} {plot_y + plot_height / 2:.1f})">True positive rate</text>',
            ]
        )
        legend_y = plot_y + 15
        for series_index, series in enumerate(series_by_query.get(query, [])):
            color = series.get("color") or COLORS[series_index % len(COLORS)]
            points = series["points"]
            polyline = " ".join(
                f"{plot_x + float(point['fpr']) * plot_width:.2f},{plot_y + (1.0 - float(point['tpr'])) * plot_height:.2f}"
                for point in points
            )
            elements.append(
                f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"/>'
            )
            operating = series.get("operating_point")
            if operating:
                circle_x = plot_x + float(operating["false_positive_rate"]) * plot_width
                circle_y = plot_y + (1.0 - float(operating["recall"])) * plot_height
                elements.append(
                    f'<circle cx="{circle_x:.2f}" cy="{circle_y:.2f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>'
                )
            legend_text = f"{series['label']} (AUC {float(series['auc']):.3f})"
            legend_width = max(126, 7.0 * len(legend_text))
            legend_x = plot_x + plot_width - legend_width - 6
            row_y = legend_y + series_index * 19
            if series_index == 0:
                elements.append(
                    f'<rect x="{legend_x - 6:.1f}" y="{legend_y - 14:.1f}" width="{legend_width + 12:.1f}" height="{19 * len(series_by_query.get(query, [])) + 4:.1f}" rx="3" fill="white" fill-opacity="0.9" stroke="#ddd"/>'
                )
            elements.append(
                f'<line x1="{legend_x:.1f}" y1="{row_y - 4:.1f}" x2="{legend_x + 20:.1f}" y2="{row_y - 4:.1f}" stroke="{color}" stroke-width="2.4"/>'
            )
            elements.append(_svg_text(legend_x + 26, row_y, legend_text, font_size="11", font_family="sans-serif"))
    elements.append(_svg_text(width / 2, height - 14, "Open circles mark the default log-odds threshold of 0 (P(YES)=0.5).", text_anchor="middle", font_size="12", font_family="sans-serif", fill="#555"))
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def render_chat_prompt(tokenizer: Any, prompt: str, template_kwargs: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Chat template did not return a non-empty string")
    return rendered


def split_candidate_tokens(
    tokenizer: Any,
    rendered_prompt: str,
    answers: dict[str, str] = ANSWER_TEXTS,
) -> tuple[list[int], dict[str, list[int]]]:
    """Tokenize candidates jointly and split at their longest common prefix.

    Tokenizers can merge the end of the rendered prompt with the first answer
    characters. Tokenizing the prompt and labels separately would then score a
    token sequence that is not the model's actual YES/NO continuation. The
    longest common prefix of the complete candidate sequences is the exact
    shared conditioning context, including this boundary behavior.
    """

    full_ids = {
        name: list(
            tokenizer.encode(rendered_prompt + answer, add_special_tokens=False)
        )
        for name, answer in answers.items()
    }
    if len(full_ids) < 2:
        raise ValueError("At least two answer candidates are required")
    sequences = list(full_ids.values())
    common_length = 0
    for token_group in zip(*sequences):
        if len(set(token_group)) != 1:
            break
        common_length += 1
    prompt_ids = sequences[0][:common_length]
    answer_ids = {
        name: sequence[common_length:] for name, sequence in full_ids.items()
    }
    if not prompt_ids:
        raise ValueError("Candidate tokenizations have no shared prompt token")
    empty = [name for name, ids in answer_ids.items() if not ids]
    if empty:
        raise ValueError(
            "Candidate tokenizations are not distinguishable after their shared "
            f"prefix: {', '.join(empty)}"
        )
    return prompt_ids, answer_ids


def prepare_atom(tokenizer: Any, row: dict[str, str], template_kwargs: dict[str, Any]) -> dict[str, Any]:
    prompt = atomic_prompt(row)
    rendered = render_chat_prompt(tokenizer, prompt, template_kwargs)
    prompt_ids, candidate_ids = split_candidate_tokens(tokenizer, rendered)
    candidates = {
        answer_name: {
            "answer": answer_text,
            "answer_ids": candidate_ids[answer_name],
        }
        for answer_name, answer_text in ANSWER_TEXTS.items()
    }
    return {
        "row": row,
        "prompt": prompt,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "shared_prefix_token_count": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "candidates": candidates,
    }


def answer_logit_positions(answer_start: int, answer_length: int) -> list[int]:
    if answer_start < 1:
        raise ValueError("An answer needs at least one conditioning token")
    if answer_length < 1:
        raise ValueError("An answer must contain at least one token")
    return list(range(answer_start - 1, answer_start - 1 + answer_length))


def score_prepared_batch(model: Any, torch: Any, prepared: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for atom_index, atom in enumerate(prepared):
        prompt_ids = atom["prompt_ids"]
        candidate_ids = {
            answer_name: atom["candidates"][answer_name]["answer_ids"]
            for answer_name in ("yes", "no")
        }
        if all(len(answer_ids) == 1 for answer_ids in candidate_ids.values()):
            sequences.append(
                {
                    "atom_index": atom_index,
                    "mode": "paired-single-token",
                    "input_ids": prompt_ids,
                    "answer_start": len(prompt_ids),
                    "candidate_ids": candidate_ids,
                }
            )
            continue
        for answer_name, answer_ids in candidate_ids.items():
            sequences.append(
                {
                    "atom_index": atom_index,
                    "mode": "sequence",
                    "answer_name": answer_name,
                    "input_ids": [*prompt_ids, *answer_ids],
                    "answer_start": len(prompt_ids),
                    "answer_ids": answer_ids,
                }
            )
    max_length = max(len(sequence["input_ids"]) for sequence in sequences)
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(model.config, "eos_token_id", None)
    if isinstance(pad_id, list):
        pad_id = pad_id[0]
    if pad_id is None:
        pad_id = 0
    input_ids = torch.full(
        (len(sequences), max_length), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length), dtype=torch.long, device=device
    )
    for index, sequence in enumerate(sequences):
        ids = torch.tensor(sequence["input_ids"], dtype=torch.long, device=device)
        input_ids[index, : len(ids)] = ids
        attention_mask[index, : len(ids)] = 1

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits
    scored: list[dict[str, Any]] = [dict() for _ in prepared]
    for sequence_index, sequence in enumerate(sequences):
        start = int(sequence["answer_start"])
        if sequence["mode"] == "paired-single-token":
            next_token_logits = logits[sequence_index, start - 1].float()
            candidate_names = ("yes", "no")
            targets = torch.tensor(
                [
                    sequence["candidate_ids"][answer_name][0]
                    for answer_name in candidate_names
                ],
                dtype=torch.long,
                device=logits.device,
            )
            selected_logits = next_token_logits.index_select(0, targets)
            selected_logprobs = torch.log_softmax(
                next_token_logits, dim=-1
            ).index_select(0, targets)
            for answer_index, answer_name in enumerate(candidate_names):
                scored[sequence["atom_index"]][answer_name] = {
                    "sequence_logprob": float(
                        selected_logprobs[answer_index].item()
                    ),
                    "token_logits": [
                        float(selected_logits[answer_index].item())
                    ],
                    "token_logprobs": [
                        float(selected_logprobs[answer_index].item())
                    ],
                }
            continue
        answer_ids = sequence["answer_ids"]
        positions = torch.tensor(
            answer_logit_positions(start, len(answer_ids)),
            dtype=torch.long,
            device=logits.device,
        )
        token_logits = logits[sequence_index].index_select(0, positions).float()
        targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
        selected_logits = token_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        token_log_probs = torch.log_softmax(token_logits, dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        scored[sequence["atom_index"]][sequence["answer_name"]] = {
            "sequence_logprob": float(token_log_probs.sum().item()),
            "token_logits": [float(value) for value in selected_logits.tolist()],
            "token_logprobs": [
                float(value) for value in token_log_probs.tolist()
            ],
        }
    del output, logits, input_ids, attention_mask
    return scored


def load_model_stack(
    model_config: dict[str, Any], device: str, dtype_name: str
) -> tuple[Any, Any, ChatTokenizerAdapter, Any, str]:
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised on deployment
        raise SystemExit(
            "PyTorch and Transformers are required. Run "
            "scripts/bootstrap_transformers.sh first."
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_by_name[dtype_name]
    model_id = model_config["model_id"]
    trust_remote_code = "--trust-remote-code" in model_config.get(
        "transformers_args", []
    )
    processor_name = model_config.get(
        "transformers_processor_class", "AutoTokenizer"
    )
    processor_loader = getattr(transformers, processor_name, None)
    if processor_loader is None:
        raise SystemExit(
            f"Installed Transformers {transformers.__version__} has no "
            f"{processor_name}"
        )
    remote_code_kwargs = {"trust_remote_code": True} if trust_remote_code else {}
    processor = processor_loader.from_pretrained(model_id, **remote_code_kwargs)
    text_tokenizer = (
        getattr(processor, "tokenizer", processor)
        if processor_name == "AutoProcessor"
        else processor
    )
    tokenizer = ChatTokenizerAdapter(
        processor,
        tokenizer=text_tokenizer,
        multimodal=processor_name == "AutoProcessor",
    )
    expected_tokenizer_class = model_config.get(
        "expected_transformers_tokenizer_class"
    )
    actual_tokenizer_class = tokenizer.tokenizer.__class__.__name__
    if (
        expected_tokenizer_class is not None
        and actual_tokenizer_class != expected_tokenizer_class
    ):
        raise SystemExit(
            f"Expected tokenizer class {expected_tokenizer_class}, got "
            f"{actual_tokenizer_class}"
        )
    loader_name = model_config.get(
        "transformers_model_class", "AutoModelForCausalLM"
    )
    loader = getattr(transformers, loader_name, None)
    if loader is None:
        raise SystemExit(
            f"Installed Transformers {transformers.__version__} has no {loader_name}"
        )
    print(f"Loading {model_id} with {loader_name} on {device} ({dtype_name})", flush=True)
    model = loader.from_pretrained(
        model_id,
        dtype=dtype,
        **remote_code_kwargs,
    )
    model.to(device)
    model.eval()
    return torch, transformers, tokenizer, model, loader_name


def score_rows(
    *,
    model: Any,
    torch: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    existing: dict[str, dict[str, Any]],
    output_path: Path,
    template_kwargs: dict[str, Any],
    batch_size: int,
    device: str,
    model_key: str,
    model_id: str,
    protocol_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    pending = [row for row in rows if row["id"] not in existing]
    print(
        f"{output_path.name}: {len(existing)} cached, {len(pending)} pending "
        f"of {len(rows)} selected",
        flush=True,
    )
    if not pending:
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for offset in range(0, len(pending), batch_size):
            batch_rows = pending[offset : offset + batch_size]
            started = time.perf_counter()
            prepared = [
                prepare_atom(tokenizer, row, template_kwargs) for row in batch_rows
            ]
            batch_scores = score_prepared_batch(model, torch, prepared, device)
            elapsed = time.perf_counter() - started
            for atom, candidate_scores in zip(prepared, batch_scores, strict=True):
                row = atom["row"]
                yes_score = candidate_scores["yes"]
                no_score = candidate_scores["no"]
                yes_logprob = yes_score["sequence_logprob"]
                no_logprob = no_score["sequence_logprob"]
                sequence_logprob_difference = yes_logprob - no_logprob
                single_token_logit_difference = None
                if (
                    len(yes_score["token_logits"]) == 1
                    and len(no_score["token_logits"]) == 1
                ):
                    single_token_logit_difference = (
                        yes_score["token_logits"][0]
                        - no_score["token_logits"][0]
                    )
                log_odds = (
                    single_token_logit_difference
                    if single_token_logit_difference is not None
                    else sequence_logprob_difference
                )
                score_mode = (
                    "single-token-logit"
                    if single_token_logit_difference is not None
                    else "sequence-logprob"
                )
                record = {
                    "id": row["id"],
                    "model_key": model_key,
                    "model_id": model_id,
                    "score_version": SCORE_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "name": row["name"],
                    "role": row["role"],
                    "target_origin": row["target_origin"],
                    "true_origin": row["true_origin"],
                    "source_rank": int(row["source_rank"]),
                    "gold_label": int(row["label"]),
                    "prompt": atom["prompt"],
                    "rendered_prompt_sha256": atom["rendered_prompt_sha256"],
                    "yes_text": ANSWER_TEXTS["yes"],
                    "no_text": ANSWER_TEXTS["no"],
                    "yes_token_ids": atom["candidates"]["yes"]["answer_ids"],
                    "no_token_ids": atom["candidates"]["no"]["answer_ids"],
                    "yes_token_count": len(yes_score["token_logits"]),
                    "no_token_count": len(no_score["token_logits"]),
                    "yes_first_token_logit": yes_score["token_logits"][0],
                    "no_first_token_logit": no_score["token_logits"][0],
                    "yes_token_logits": yes_score["token_logits"],
                    "no_token_logits": no_score["token_logits"],
                    "yes_token_logprobs": yes_score["token_logprobs"],
                    "no_token_logprobs": no_score["token_logprobs"],
                    "single_token_logit_difference": (
                        single_token_logit_difference
                    ),
                    "shared_prefix_token_count": atom[
                        "shared_prefix_token_count"
                    ],
                    "yes_logprob": yes_logprob,
                    "no_logprob": no_logprob,
                    "sequence_logprob_difference": (
                        sequence_logprob_difference
                    ),
                    "score_mode": score_mode,
                    "log_odds_yes_vs_no": log_odds,
                    "yes_probability_normalized": stable_sigmoid(log_odds),
                    "prediction_at_zero": int(log_odds >= 0.0),
                    "batch_latency_seconds": elapsed,
                    "timestamp_utc": utc_now(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing[row["id"]] = record
                completed += 1
            print(
                f"  completed {completed}/{len(pending)} pending atoms",
                flush=True,
            )
    return existing


def validate_cached_records(
    existing: dict[str, dict[str, Any]],
    rows: list[dict[str, str]],
    *,
    model_key: str,
    model_id: str,
    protocol_fingerprint: str,
) -> None:
    for row in rows:
        record = existing.get(row["id"])
        if record is None:
            continue
        expected = {
            "model_key": model_key,
            "model_id": model_id,
            "score_version": SCORE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "gold_label": int(row["label"]),
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ValueError(
                    f"Cached record {row['id']} has unexpected {field}: "
                    f"{record.get(field)!r} != {value!r}"
                )
        for field in (
            "yes_logprob",
            "no_logprob",
            "sequence_logprob_difference",
            "log_odds_yes_vs_no",
        ):
            try:
                value = float(record[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cached record {row['id']} has no valid {field}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Cached record {row['id']} has non-finite {field}"
                )
        for field in ("yes_token_count", "no_token_count"):
            if int(record.get(field, 0)) < 1:
                raise ValueError(
                    f"Cached record {row['id']} has no valid {field}"
                )


def analyze_records(
    *,
    records: list[dict[str, Any]],
    model_key: str,
    model_id: str,
    dataset_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[query_key(record)].append(record)
    missing = [query for query in QUERY_ORDER if query not in grouped]
    if missing:
        raise ValueError(f"No scored records for queries: {', '.join(missing)}")

    summary: dict[str, Any] = {
        "model_key": model_key,
        "model_id": model_id,
        "score_version": SCORE_VERSION,
        "dataset_sha256": dataset_sha256,
        "evaluated_atoms": len(records),
        "generated_at_utc": utc_now(),
        "queries": {},
    }
    point_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    score_rows_output: list[dict[str, Any]] = []
    plot_series: dict[str, list[dict[str, Any]]] = {}
    for query in QUERY_ORDER:
        query_records = sorted(grouped[query], key=lambda record: record["id"])
        labels = [int(record["gold_label"]) for record in query_records]
        scores = [float(record["log_odds_yes_vs_no"]) for record in query_records]
        points = roc_curve(labels, scores)
        auc = roc_auc(points)
        default = operating_point(labels, scores)
        positives = sum(labels)
        negatives = len(labels) - positives
        single_token_examples = sum(
            int(record["yes_token_count"]) == 1
            and int(record["no_token_count"]) == 1
            for record in query_records
        )
        summary["queries"][query] = {
            "title": QUERY_TITLES[query],
            "n": len(labels),
            "positives": positives,
            "negatives": negatives,
            "single_token_examples": single_token_examples,
            "multi_token_examples": len(labels) - single_token_examples,
            "auc": auc,
            "default_threshold": default,
        }
        summary_rows.append(
            {
                "model_key": model_key,
                "model_id": model_id,
                "query": query,
                "query_title": QUERY_TITLES[query],
                "n": len(labels),
                "positives": positives,
                "negatives": negatives,
                "single_token_examples": single_token_examples,
                "multi_token_examples": len(labels) - single_token_examples,
                "auc": auc,
                "default_accuracy": default["accuracy"],
                "default_tpr": default["recall"],
                "default_fpr": default["false_positive_rate"],
                "default_tp": default["tp"],
                "default_tn": default["tn"],
                "default_fp": default["fp"],
                "default_fn": default["fn"],
            }
        )
        plot_series[query] = [
            {
                "label": model_key,
                "points": points,
                "auc": auc,
                "operating_point": default,
                "color": COLORS[0],
            }
        ]
        for point_index, point in enumerate(points):
            point_rows.append(
                {
                    "model_key": model_key,
                    "model_id": model_id,
                    "query": query,
                    "query_title": QUERY_TITLES[query],
                    "point_index": point_index,
                    **point,
                }
            )
        for record in query_records:
            score_rows_output.append(
                {
                    key: record[key]
                    for key in (
                        "id",
                        "name",
                        "role",
                        "target_origin",
                        "true_origin",
                        "source_rank",
                        "gold_label",
                        "yes_token_count",
                        "no_token_count",
                        "yes_first_token_logit",
                        "no_first_token_logit",
                        "single_token_logit_difference",
                        "yes_logprob",
                        "no_logprob",
                        "sequence_logprob_difference",
                        "score_mode",
                        "log_odds_yes_vs_no",
                        "yes_probability_normalized",
                        "prediction_at_zero",
                    )
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "roc_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "roc_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    with (output_dir / "roc_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows_output[0]))
        writer.writeheader()
        writer.writerows(score_rows_output)
    write_roc_svg(
        output_dir / "roc_curves.svg",
        plot_series,
        title=f"Atomic YES/NO ROC curves — {model_key}",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="Smoke-test N positive and N negative atoms for each query",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.limit_per_class is not None and args.limit_per_class < 1:
        raise SystemExit("--limit-per-class must be positive")
    models = load_json(args.model_config)
    if args.model_key not in models:
        raise SystemExit(
            f"Unknown model key {args.model_key!r}; choose one of: {', '.join(models)}"
        )
    model_config = models[args.model_key]
    model_id = model_config["model_id"]
    run_name = args.run_name or args.model_key
    if not RUN_NAME_RE.fullmatch(run_name):
        raise SystemExit("--run-name contains unsupported characters")
    all_rows = read_csv(args.dataset)
    if not all_rows:
        raise SystemExit(f"Dataset is empty: {args.dataset}")
    rows = select_limit_per_class(all_rows, args.limit_per_class)
    dataset_sha256 = file_sha256(args.dataset)
    template_kwargs = model_config.get("request_extra_body", {}).get(
        "chat_template_kwargs", {}
    )

    torch, transformers, tokenizer, model, loader_name = load_model_stack(
        model_config, args.device, args.dtype
    )
    example_atom = prepare_atom(tokenizer, rows[0], template_kwargs)
    protocol = {
        "score_version": SCORE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "answer_texts": ANSWER_TEXTS,
        "score_definition": (
            "raw_logit(YES) - raw_logit(NO) when both candidates are one "
            "token; otherwise sum_logprob(YES token sequence) - "
            "sum_logprob(NO token sequence)"
        ),
        "model_key": args.model_key,
        "model_id": model_id,
        "model_loader": loader_name,
        "processor_loader": model_config.get(
            "transformers_processor_class", "AutoTokenizer"
        ),
        "processor_class": tokenizer.processor.__class__.__name__,
        "tokenizer_class": tokenizer.tokenizer.__class__.__name__,
        "checkpoint_commit_hash": checkpoint_commit_hash(model, tokenizer),
        "dtype": args.dtype,
        "template_kwargs": template_kwargs,
        "chat_template_sha256": content_sha256(tokenizer.chat_template),
        "candidate_tokenization_example": {
            "row_id": rows[0]["id"],
            "shared_prefix_token_count": example_atom[
                "shared_prefix_token_count"
            ],
            "yes_token_ids": example_atom["candidates"]["yes"][
                "answer_ids"
            ],
            "no_token_ids": example_atom["candidates"]["no"]["answer_ids"],
        },
        "dataset_sha256": dataset_sha256,
        "transformers_version": transformers.__version__,
        "transformers_vcs_commit": distribution_vcs_commit("transformers"),
        "mistral_common_version": distribution_version("mistral-common"),
        "torch_version": torch.__version__,
    }
    protocol["protocol_fingerprint"] = protocol_fingerprint(protocol)
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        existing_config = load_json(config_path)
        if existing_config.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise SystemExit(
                f"Protocol differs from existing {config_path}; use a new --run-name"
            )
    else:
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    scores_path = output_dir / "scores.jsonl"
    existing = latest_jsonl(scores_path)
    validate_cached_records(
        existing,
        rows,
        model_key=args.model_key,
        model_id=model_id,
        protocol_fingerprint=protocol["protocol_fingerprint"],
    )
    records = score_rows(
        model=model,
        torch=torch,
        tokenizer=tokenizer,
        rows=rows,
        existing=existing,
        output_path=scores_path,
        template_kwargs=template_kwargs,
        batch_size=args.batch_size,
        device=args.device,
        model_key=args.model_key,
        model_id=model_id,
        protocol_fingerprint=protocol["protocol_fingerprint"],
    )
    selected_records = [records[row["id"]] for row in rows if row["id"] in records]
    summary = analyze_records(
        records=selected_records,
        model_key=args.model_key,
        model_id=model_id,
        dataset_sha256=dataset_sha256,
        output_dir=output_dir,
    )
    print("query\tn\tauc\tdefault_accuracy\tdefault_tpr\tdefault_fpr")
    for query in QUERY_ORDER:
        metrics = summary["queries"][query]
        default = metrics["default_threshold"]
        print(
            f"{query}\t{metrics['n']}\t{metrics['auc']}\t"
            f"{default['accuracy']}\t{default['recall']}\t"
            f"{default['false_positive_rate']}"
        )
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
