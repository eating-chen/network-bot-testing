"""Measure chat-template token lengths without storing tokenized examples."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from network_sft import config
from network_sft.adapters.llama import render_for_llama
from network_sft.io import jsonl_rows, setup_logging, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def _percentile(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def _summary(values: list[int]) -> dict[str, int]:
    return {
        "rows": len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def run_stats() -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install token statistics support with: uv sync --extra sft") from error

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
    lengths: defaultdict[str, list[int]] = defaultdict(list)
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows = list(jsonl_rows(config.CANONICAL_DIR / f"{split}.jsonl"))
        split_counts[split] = len(rows)
        for row in rows:
            messages, tools = render_for_llama(row)
            kwargs = {
                "conversation": messages,
                "tokenize": True,
                "add_generation_prompt": False,
            }
            if tools:
                kwargs["tools"] = tools
            input_ids = tokenizer.apply_chat_template(**kwargs)
            lengths[str(row["source"])].append(len(input_ids))

    report = {
        "created_at": utc_now(),
        "tokenizer": config.TOKENIZER_ID,
        "note": "apply_chat_template only; tokenized samples were not saved",
        "splits": split_counts,
        "datasets": {source: _summary(values) for source, values in sorted(lengths.items())},
    }
    write_json(config.REPORTS_DIR / "dataset_stats.json", report)
    LOGGER.info("token statistics 完成")
    return report


if __name__ == "__main__":
    setup_logging()
    run_stats()
