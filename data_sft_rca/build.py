"""Build canonical data, grounded views, group-aware splits, and Qwen messages."""

import json
import re
from collections import Counter

from data_sft_rca import config
from data_sft_rca.build_views import build_views
from data_sft_rca.cloudops_converter import build_cloudops
from data_sft_rca.faultbench_converter import build_faultbench
from data_sft_rca.io_utils import write_json, write_jsonl
from data_sft_rca.legacy_sources import build_legacy_sources
from data_sft_rca.rcaeval_converter import build_rcaeval
from data_sft_rca.render_qwen_sft import to_sft
from data_sft_rca.schema import validate_row
from data_sft_rca.split import split_rows
from data_sft_rca.symptom_graph import build_graph
from data_sft_rca.validate_views import group_leaks, validate_rows


def _frozen_ids() -> set[str]:
    if not config.FROZEN_EVAL_IDS.exists():
        return set()
    return {
        line.strip()
        for line in config.FROZEN_EVAL_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _deduplicate(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        key = json.dumps(
            {
                "problem": re.sub(r"\s+", " ", row["problem"].lower()),
                "evidence": [
                    re.sub(r"\s+", " ", item["output"].lower()) for item in row["evidence"]
                ],
                "task": row["task_type"],
                "causes": [item["cause_id"] for item in row["diagnoses"]],
            },
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _unseen_slice(splits: dict[str, list[dict]]) -> list[dict]:
    train_causes = {
        diagnosis["cause_id"] for row in splits["train"] for diagnosis in row["diagnoses"]
    }
    return [
        row
        for row in splits["test"]
        if any(diagnosis["cause_id"] not in train_causes for diagnosis in row["diagnoses"])
    ]


def build() -> dict:
    sources = {
        "legacy": build_legacy_sources(),
        "cloudopsbench": build_cloudops(),
        "faultbench": build_faultbench(),
        "rcaeval": build_rcaeval(),
    }
    rows = [row for source_rows in sources.values() for row in source_rows]
    frozen = _frozen_ids()
    rows = [
        row
        for row in rows
        if not {row["id"], row["root_case_id"], row["group_id"]}.intersection(frozen)
    ]
    canonical_errors = [
        {"id": row["id"], "reasons": errors} for row in rows if (errors := validate_row(row))
    ]
    if canonical_errors:
        write_jsonl(config.OUTPUT_DIR / "rejected.jsonl", canonical_errors)
        raise ValueError(f"canonical validation failed for {len(canonical_errors)} rows")

    rows.sort(key=lambda row: row["id"])
    for source in sorted({row["source"] for row in rows}):
        write_jsonl(
            config.CANONICAL_DIR / f"{source}.jsonl",
            (row for row in rows if row["source"] == source),
        )
    write_jsonl(config.CANONICAL_DIR / "all.jsonl", rows)

    graph = build_graph(rows)
    write_json(config.OUTPUT_DIR / "symptom_cause_graph.json", graph)
    views = _deduplicate(build_views(rows, graph))
    errors = validate_rows(views, graph)
    write_jsonl(config.OUTPUT_DIR / "rejected.jsonl", errors)
    if errors:
        raise ValueError(f"view validation failed for {len(errors)} rows")
    write_jsonl(config.VIEWS_DIR / "all.jsonl", views)

    splits, split_manifest = split_rows(views)
    leaks = group_leaks(splits)
    if leaks:
        raise ValueError(f"group leakage detected: {leaks[:5]}")
    for name, split_rows_ in splits.items():
        write_jsonl(config.SFT_DIR / f"{name}.jsonl", (to_sft(row) for row in split_rows_))
    unseen = _unseen_slice(splits)
    write_jsonl(config.SFT_DIR / "unseen_fault_family_test.jsonl", (to_sft(row) for row in unseen))

    by_source = Counter(row["source"] for row in views)
    by_task = Counter(row["task_type"] for row in views)
    manifest = {
        "dataset": "Network-RCA-SFT-v2",
        "sft_rows": len(views),
        "unique_root_cases": len({row["root_case_id"] for row in views}),
        "unique_groups": len({row["group_id"] for row in views}),
        "rows_by_source": dict(sorted(by_source.items())),
        "rows_by_task": dict(sorted(by_task.items())),
        "excluded_frozen_eval_ids": len(frozen),
        "unseen_fault_family_test_rows": len(unseen),
        "missing_sources": [name for name, source_rows in sources.items() if not source_rows],
        "splits": split_manifest,
        "llm_generated": False,
    }
    write_json(config.OUTPUT_DIR / "dataset_manifest.json", manifest)
    (config.OUTPUT_DIR / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
