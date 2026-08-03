#!/usr/bin/env python3
"""Build the crossed Chinese/Japanese name benchmark deterministically."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "xor_names"
VOCAB_DIR = DATA_DIR / "vocab"


@dataclass(frozen=True)
class Name:
    text: str
    rank: int


def read_vocab(path: Path) -> list[Name]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [Name(row["name"], int(row["source_rank"])) for row in rows]
    if len(names) != 100:
        raise ValueError(f"{path} contains {len(names)} names, expected 100")
    folded = [name.text.casefold() for name in names]
    if len(set(folded)) != len(folded):
        raise ValueError(f"{path} contains duplicate case-insensitive names")
    return names


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_vocabularies() -> dict[tuple[str, str], list[Name]]:
    vocabularies = {
        ("C", "given"): read_vocab(VOCAB_DIR / "chinese_given.csv"),
        ("C", "surname"): read_vocab(VOCAB_DIR / "chinese_surnames.csv"),
        ("J", "given"): read_vocab(VOCAB_DIR / "japanese_given.csv"),
        ("J", "surname"): read_vocab(VOCAB_DIR / "japanese_surnames.csv"),
    }
    for role in ("given", "surname"):
        chinese = {name.text.casefold() for name in vocabularies[("C", role)]}
        japanese = {name.text.casefold() for name in vocabularies[("J", role)]}
        overlap = sorted(chinese & japanese)
        if overlap:
            raise ValueError(f"Cross-origin overlap for {role}: {overlap}")
    return vocabularies


def make_row(
    dataset: str,
    row_id: int,
    first_origin: str,
    last_origin: str,
    first: Name,
    last: Name,
    rotation: int | None,
) -> dict[str, object]:
    return {
        "id": f"{dataset}-{row_id:05d}",
        "full_name": f"{first.text} {last.text}",
        "first_name": first.text,
        "last_name": last.text,
        "first_origin": first_origin,
        "last_origin": last_origin,
        "first_source_rank": first.rank,
        "last_source_rank": last.rank,
        "pair_type": f"{first_origin}_{last_origin}",
        "label": int(first_origin != last_origin),
        "rotation": "" if rotation is None else rotation,
    }


def build_full(vocab: dict[tuple[str, str], list[Name]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_id = 1
    for first_origin, last_origin in (("C", "C"), ("C", "J"), ("J", "C"), ("J", "J")):
        for first in vocab[(first_origin, "given")]:
            for last in vocab[(last_origin, "surname")]:
                rows.append(
                    make_row("full", row_id, first_origin, last_origin, first, last, None)
                )
                row_id += 1
    return rows


def build_core(
    vocab: dict[tuple[str, str], list[Name]], rotations: int = 10
) -> list[dict[str, object]]:
    """Build a balanced 4,000-row subset with uniform vocabulary coverage."""
    rows: list[dict[str, object]] = []
    row_id = 1
    for first_origin, last_origin in (("C", "C"), ("C", "J"), ("J", "C"), ("J", "J")):
        first_names = vocab[(first_origin, "given")]
        last_names = vocab[(last_origin, "surname")]
        for rotation in range(rotations):
            # The odd stride avoids repeatedly pairing nearby source ranks.
            for index, first in enumerate(first_names):
                last = last_names[(index * 37 + rotation * 11) % len(last_names)]
                rows.append(
                    make_row(
                        "core",
                        row_id,
                        first_origin,
                        last_origin,
                        first,
                        last,
                        rotation,
                    )
                )
                row_id += 1
    return rows


def build_individual_queries(
    vocab: dict[tuple[str, str], list[Name]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_id = 1
    for role in ("given", "surname"):
        for true_origin in ("C", "J"):
            for name in vocab[(true_origin, role)]:
                for target_origin in ("C", "J"):
                    rows.append(
                        {
                            "id": f"atom-{row_id:04d}",
                            "name": name.text,
                            "role": role,
                            "target_origin": target_origin,
                            "true_origin": true_origin,
                            "source_rank": name.rank,
                            "label": int(target_origin == true_origin),
                        }
                    )
                    row_id += 1
    return rows


def main() -> None:
    vocab = load_vocabularies()
    fields = [
        "id",
        "full_name",
        "first_name",
        "last_name",
        "first_origin",
        "last_origin",
        "first_source_rank",
        "last_source_rank",
        "pair_type",
        "label",
        "rotation",
    ]
    full_rows = build_full(vocab)
    core_rows = build_core(vocab)
    atom_rows = build_individual_queries(vocab)
    write_csv(DATA_DIR / "xor_names_full.csv", fields, full_rows)
    write_csv(DATA_DIR / "xor_names_core.csv", fields, core_rows)
    write_csv(
        DATA_DIR / "individual_queries.csv",
        ["id", "name", "role", "target_origin", "true_origin", "source_rank", "label"],
        atom_rows,
    )
    print(
        f"Wrote {len(full_rows):,} exhaustive pairs, "
        f"{len(core_rows):,} core pairs, and {len(atom_rows):,} atomic queries."
    )


if __name__ == "__main__":
    main()
