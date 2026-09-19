"""Quality gates for canonical rows and derived evidence views."""

import json
import re
from collections import defaultdict

from data_sft_rca.schema import validate_row


def _normalized(row: dict) -> str:
    payload = {
        "problem": row["problem"],
        "evidence": [item["output"] for item in row["evidence"]],
        "task": row["task_type"],
        "causes": [item["cause_id"] for item in row["diagnoses"]],
    }
    return re.sub(r"\s+", " ", json.dumps(payload, sort_keys=True).lower()).strip()


def validate_rows(rows: list[dict], graph: dict) -> list[dict]:
    rejected = []
    seen_ids = set()
    seen_content = set()
    graph_causes = {
        family: {item["cause_id"] for item in candidates} for family, candidates in graph.items()
    }
    for row in rows:
        reasons = validate_row(row)
        if row["id"] in seen_ids:
            reasons.append("duplicate id")
        seen_ids.add(row["id"])
        normalized = _normalized(row)
        if normalized in seen_content:
            reasons.append("normalized duplicate")
        seen_content.add(normalized)
        visible_ids = {item["id"] for item in row["evidence"]}
        if visible_ids.intersection(row.get("masked_evidence_ids", [])):
            reasons.append("masked evidence remains visible")
        if (
            row["task_type"] in {"probable_cause_analysis", "next_diagnostic_step"}
            and row.get("evidence_mode") == "partial"
        ):
            allowed = graph_causes.get(row["fault_family"], set())
            unsupported = {item["cause_id"] for item in row["diagnoses"]} - allowed
            if unsupported:
                reasons.append(f"unsupported candidate causes: {sorted(unsupported)}")
        if reasons:
            rejected.append({"id": row.get("id"), "reasons": reasons})
    return rejected


def group_leaks(splits: dict[str, list[dict]]) -> list[str]:
    locations: dict[str, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            locations[row["group_id"]].add(split)
    return sorted(group for group, values in locations.items() if len(values) > 1)
