"""Build one shared-ID corpus for a controlled model comparison."""

import json
import logging
from collections import Counter, defaultdict
from typing import Any

from network_sft import config
from network_sft.io import jsonl_rows, setup_logging, utc_now, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def run_controlled() -> dict[str, Any]:
    stats = json.loads((config.REPORTS_DIR / "token_stats.json").read_text())
    compared = set(config.CONTROLLED_TOKENIZER_IDS)
    for tokenizer_id in compared:
        result = stats["tokenizers"].get(tokenizer_id, {})
        if not result or "load_error" in result:
            raise RuntimeError(f"missing compatibility result: {tokenizer_id}")

    reasons: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for failure in jsonl_rows(config.REPORTS_DIR / "template_incompatible.jsonl"):
        if failure["tokenizer_id"] in compared:
            reasons[failure["id"]].append(
                {"tokenizer_id": failure["tokenizer_id"], "error": failure["error"]}
            )

    kept_by_split, excluded = {}, []
    for split in config.SPLIT_RATIOS:
        rows = list(jsonl_rows(config.FINAL_DIR / f"{split}.jsonl"))
        kept = []
        for row in rows:
            if row["id"] in reasons:
                excluded.append(
                    {
                        "id": row["id"],
                        "source": row["source"],
                        "split": split,
                        "category": row["metadata"]["category"],
                        "incompatible_with": reasons[row["id"]],
                    }
                )
            else:
                kept.append(row)
        write_jsonl(config.CONTROLLED_DIR / f"{split}.jsonl", kept)
        kept_by_split[split] = len(kept)

    excluded.sort(key=lambda row: (row["split"], row["source"], row["id"]))
    write_jsonl(config.CONTROLLED_DIR / "excluded.jsonl", excluded)
    manifest = {
        "created_at": utc_now(),
        "method": "intersection_of_chat_template_compatible_row_ids",
        "tokenizer_ids": list(config.CONTROLLED_TOKENIZER_IDS),
        "source_dir": str(config.FINAL_DIR),
        "rows": kept_by_split,
        "total_rows": sum(kept_by_split.values()),
        "excluded_rows": len(excluded),
        "excluded_by_source": dict(Counter(row["source"] for row in excluded)),
        "excluded_by_category": dict(Counter(row["category"] for row in excluded)),
    }
    write_json(config.CONTROLLED_DIR / "manifest.json", manifest)
    LOGGER.info("controlled rows: %s; excluded: %d", kept_by_split, len(excluded))
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_controlled()
