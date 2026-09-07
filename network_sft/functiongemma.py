"""Convert English FunctionGemma rows into the model-agnostic schema."""

from __future__ import annotations

import json
import logging
from typing import Any

from network_sft import config
from network_sft.io import (
    jsonl_rows,
    setup_logging,
    stable_id,
    upstream_parquet_rows,
    write_json,
    write_jsonl,
)
from network_sft.schema import canonical_row, validate_row

LOGGER = logging.getLogger(__name__)


def _argument_object(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("FunctionGemma tool arguments must be an object")
    return parsed


def _canonical_tools(source_tools: Any) -> list[dict[str, Any]]:
    tools = []
    for source_tool in source_tools or []:
        function = source_tool.get("function") or source_tool
        tools.append(
            {
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters") or {},
            }
        )
    return tools


def convert_row(source_row: dict[str, Any], index: int) -> dict[str, Any]:
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    call_order: list[str] = []
    call_names: dict[str, str] = {}
    tool_result_index = 0

    for message_index, message in enumerate(source_row.get("messages") or []):
        role = message.get("role")
        if role in {"developer", "system"}:
            content = str(message.get("content") or "").strip()
            if content:
                system_parts.append(content)
            continue
        if role == "assistant":
            turn: dict[str, Any] = {
                "role": "assistant",
                "content": str(message.get("content") or ""),
            }
            calls = []
            for call_index, source_call in enumerate(message.get("tool_calls") or []):
                function = source_call.get("function") or source_call
                call_id = str(
                    source_call.get("id") or f"call_{message_index}_{call_index}"
                )
                name = str(function.get("name") or "")
                calls.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "arguments": _argument_object(function.get("arguments") or {}),
                    }
                )
                call_order.append(call_id)
                call_names[call_id] = name
            if calls:
                turn["tool_calls"] = calls
            turns.append(turn)
        elif role == "tool":
            fallback = call_order[tool_result_index] if tool_result_index < len(call_order) else ""
            call_id = str(message.get("tool_call_id") or fallback)
            turns.append(
                {
                    "role": "tool",
                    "call_id": call_id,
                    "name": str(message.get("name") or call_names.get(call_id) or ""),
                    "content": str(message.get("content") or ""),
                }
            )
            tool_result_index += 1
        elif role == "user":
            turns.append({"role": "user", "content": str(message.get("content") or "")})

    identifier = stable_id(index, json.dumps(turns, ensure_ascii=False, sort_keys=True))
    return canonical_row(
        f"functiongemma_{identifier}",
        "functiongemma_network",
        "simple_network_tool",
        "\n\n".join(system_parts),
        _canonical_tools(source_row.get("tools")),
        turns,
        {
            "original_source": config.FUNCTIONGEMMA_DATASET_ID,
            "module_id": str(source_row.get("module_id") or ""),
            "upstream_split": str(source_row.get("split") or "train"),
        },
    )


def run_functiongemma() -> dict[str, int]:
    root = config.RAW_DIR / "functiongemma_network" / "data" / config.FUNCTIONGEMMA_CONFIG
    files = sorted([*root.rglob("*.jsonl"), *root.rglob("*.parquet")])
    if not files:
        raise FileNotFoundError(f"No English data files under {root}; run download.py first")
    rows = []
    rejected = 0
    index = 0
    for path in files:
        reader = jsonl_rows(path) if path.suffix == ".jsonl" else upstream_parquet_rows(path)
        for source_row in reader:
            source_row.setdefault("split", path.stem)
            row = convert_row(source_row, index)
            index += 1
            if validate_row(row):
                rejected += 1
                continue
            rows.append(row)
    write_jsonl(config.intermediate_path("functiongemma_network"), rows)
    manifest = {"input_rows": index, "output_rows": len(rows), "rejected_incomplete": rejected}
    write_json(config.REPORTS_DIR / "functiongemma_stats.json", manifest)
    LOGGER.info("FunctionGemma 完成：%d rows", len(rows))
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_functiongemma()
