"""Split each source first, then merge and deterministically shuffle each split."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import Any

from network_sft import config
from network_sft.io import jsonl_rows, setup_logging, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")


def _hash(value: str) -> str:
    return hashlib.sha256(f"{config.SPLIT_SEED}:{value}".encode()).hexdigest()


def _allocate_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted((str(row["id"]) for row in rows), key=_hash)
    total = len(ordered)
    validation = round(total * config.VALIDATION_RATIO)
    test = round(total * config.TEST_RATIO)
    if total >= 20:
        validation = max(1, validation)
        test = max(1, test)
    train_end = total - validation - test
    validation_end = train_end + validation
    assignments = {identifier: "train" for identifier in ordered[:train_end]}
    assignments.update(
        {identifier: "validation" for identifier in ordered[train_end:validation_end]}
    )
    assignments.update({identifier: "test" for identifier in ordered[validation_end:]})
    return assignments


def _general_assignments(rows: list[dict[str, Any]]) -> dict[str, str]:
    return _allocate_ids(rows)


def _functiongemma_assignments(rows: list[dict[str, Any]]) -> dict[str, str]:
    assignments = {}
    for row in rows:
        split = str(row.get("metadata", {}).get("upstream_split") or "train")
        assignments[str(row["id"])] = split if split in SPLITS else "train"
    return assignments


def _nika_assignments(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Hold out complete failure types, making every NIKA test type unseen in train."""
    failure_types = sorted(
        {str(row["metadata"]["failure_type"]) for row in rows},
        key=_hash,
    )
    total = len(failure_types)
    if total < 3:
        raise ValueError("NIKA needs at least three successful failure types for group split")
    validation = max(1, round(total * config.VALIDATION_RATIO))
    test = max(1, round(total * config.TEST_RATIO))
    train_end = total - validation - test
    type_split = {name: "train" for name in failure_types[:train_end]}
    type_split.update({name: "validation" for name in failure_types[train_end:-test]})
    type_split.update({name: "test" for name in failure_types[-test:]})
    return {
        str(row["id"]): type_split[str(row["metadata"]["failure_type"])]
        for row in rows
    }


def run_split_merge() -> dict[str, Any]:
    sources = {
        "general_sft": list(jsonl_rows(config.intermediate_path("general_sft"))),
        "functiongemma_network": list(
            jsonl_rows(config.intermediate_path("functiongemma_network"))
        ),
        "nika": list(jsonl_rows(config.intermediate_path("nika"))),
    }
    assignments = {
        "general_sft": _general_assignments(sources["general_sft"]),
        "functiongemma_network": _functiongemma_assignments(sources["functiongemma_network"]),
        "nika": _nika_assignments(sources["nika"]),
    }
    merged: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    counts: Counter[tuple[str, str]] = Counter()

    for source, rows in sources.items():
        for row in rows:
            split = assignments[source][str(row["id"])]
            row["metadata"]["split"] = split
            merged[split].append(row)
            counts[(source, split)] += 1

    for split, rows in merged.items():
        rows.sort(key=lambda row: _hash(f"{split}:{row['id']}"))
        write_jsonl(config.CANONICAL_DIR / f"{split}.jsonl", rows)

    by_source = {
        source: {split: counts[(source, split)] for split in SPLITS}
        for source in sources
    }
    nika_test_types = sorted(
        {
            row["metadata"]["failure_type"]
            for row in merged["test"]
            if row["source"] == "nika"
        }
    )
    manifest = {
        "split_order": "each source first, merge second",
        "seed": config.SPLIT_SEED,
        "rows": {split: len(rows) for split, rows in merged.items()},
        "rows_by_source": by_source,
        "nika_test_unseen_failure_types": nika_test_types,
    }
    write_json(config.CANONICAL_DIR / "dataset_manifest.json", manifest)
    LOGGER.info("split + merge 完成：%s", manifest["rows"])
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_split_merge()
