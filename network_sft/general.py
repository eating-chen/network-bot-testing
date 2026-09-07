"""Convert a small, deterministic UltraChat subset to canonical SFT rows."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
from pathlib import Path
from typing import Any

from network_sft import config
from network_sft.io import (
    setup_logging,
    stable_id,
    upstream_parquet_rows,
    write_json,
    write_jsonl,
)
from network_sft.schema import canonical_row

LOGGER = logging.getLogger(__name__)


def _clean_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages = [
        {"role": str(item.get("role") or ""), "content": str(item.get("content") or "").strip()}
        for item in value
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
    ]
    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None
    expected = "user"
    for message in messages:
        if message["role"] != expected or not message["content"]:
            return None
        expected = "assistant" if expected == "user" else "user"
    return messages


def _excluded_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def run_general() -> dict[str, Any]:
    root = config.RAW_DIR / "general_sft"
    files = sorted(
        path
        for path in root.rglob("*.parquet")
        if config.GENERAL_SOURCE_SPLIT in path.name
    )
    if not files:
        raise FileNotFoundError(f"No train_sft Parquet files under {root}; run download.py first")

    excluded = _excluded_ids(config.GENERAL_EXCLUDE_IDS)
    seen: set[str] = set()
    # A max-heap represented by negative hash keeps only the deterministic best K rows.
    selected: list[tuple[int, str, dict[str, Any]]] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for path in files:
        for source_row in upstream_parquet_rows(path):
            upstream_id = str(source_row.get("prompt_id") or "")
            if upstream_id in excluded:
                reject("reserved_for_cpt")
                continue
            messages = _clean_messages(source_row.get("messages"))
            if messages is None:
                reject("empty_or_bad_role_sequence")
                continue
            serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            if len(serialized) > config.GENERAL_MAX_CHARS:
                reject("overly_long")
                continue
            fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
            if fingerprint in seen:
                reject("exact_duplicate")
                continue
            seen.add(fingerprint)
            identifier = upstream_id or stable_id(fingerprint)
            row = canonical_row(
                f"general_{stable_id(identifier)}",
                "general_sft",
                "general_instruction",
                "",
                [],
                messages,
                {"original_source": config.GENERAL_DATASET_ID, "original_id": identifier},
            )
            digest = hashlib.sha256(f"{config.SPLIT_SEED}:{identifier}".encode()).hexdigest()
            score = int(digest, 16)
            item = (-score, identifier, row)
            if len(selected) < config.GENERAL_SAMPLE_SIZE:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)

    rows = [item[2] for item in sorted(selected, reverse=True)]
    write_jsonl(config.intermediate_path("general_sft"), rows)
    manifest = {
        "input_rows_seen": len(seen) + sum(rejected.values()),
        "output_rows": len(rows),
        "rejected": rejected,
    }
    write_json(config.REPORTS_DIR / "general_stats.json", manifest)
    LOGGER.info("General SFT 完成：%d rows", len(rows))
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_general()
