"""Parse successful NIKA event traces into canonical agentic SFT conversations."""

from __future__ import annotations

import ast
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from network_sft import config
from network_sft.io import setup_logging, write_json, write_jsonl
from network_sft.schema import canonical_row

LOGGER = logging.getLogger(__name__)
ERROR_EVENTS = {"chain_error", "llm_error", "tool_error", "fatal_error"}
OUTPUT_NAME = re.compile(r"\bname=['\"]([^'\"]+)['\"]")
OUTPUT_CONTENT = re.compile(r"^content=(?P<value>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")", re.DOTALL)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _as_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)} if value not in (None, "") else set()


def submission_is_correct(ground_truth: dict[str, Any], submission: dict[str, Any]) -> bool:
    return (
        ground_truth.get("is_anomaly") == submission.get("is_anomaly")
        and _as_set(ground_truth.get("faulty_devices"))
        == _as_set(submission.get("faulty_devices"))
        and _as_set(ground_truth.get("root_cause_name"))
        == _as_set(submission.get("root_cause_name"))
    )


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("tool input is neither object nor string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool input does not encode an object")
    return parsed


def _tool_output(value: Any) -> str:
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    match = OUTPUT_CONTENT.match(value)
    if match:
        try:
            return str(ast.literal_eval(match.group("value")))
        except (SyntaxError, ValueError):
            pass
    return value


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _tool_schemas(observed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for name in sorted(observed):
        item = observed[name]
        calls: list[dict[str, Any]] = item["calls"]
        keys = set().union(*(call.keys() for call in calls)) if calls else set()
        required = set.intersection(*(set(call) for call in calls)) if calls else set()
        properties = {}
        for key in sorted(keys):
            sample = next((call[key] for call in calls if key in call), "")
            properties[key] = {"type": _json_type(sample)}
        tools.append(
            {
                "name": name,
                "description": item["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(required),
                    "additionalProperties": False,
                },
            }
        )
    return tools


def _events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} invalid JSON") from error
        if not isinstance(event, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        events.append(event)
    return events


def parse_incident(incident_dir: Path) -> dict[str, Any]:
    ground_truth = _load_json(incident_dir / "ground_truth.json")
    submission = _load_json(incident_dir / "submission.json")
    if not submission_is_correct(ground_truth, submission):
        raise ValueError("incorrect_submission")
    session = _load_json(incident_dir / "session_meta.json")
    task = str(session.get("task_description") or "").strip()
    if not task:
        raise ValueError("missing_task_description")

    turns: list[dict[str, Any]] = [{"role": "user", "content": task}]
    pending: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    pending_content: str | None = None
    observed: dict[str, dict[str, Any]] = {}
    call_number = 0

    for event in _events(incident_dir / "conversation_diagnosis_agent.log"):
        kind = event.get("event")
        if kind in ERROR_EVENTS or event.get("invalid_tool_calls"):
            raise ValueError("fatal_or_invalid_tool_event")
        if kind == "llm_end":
            text = str(event.get("text") or "").strip()
            finish_reason = (event.get("generation_info") or {}).get("finish_reason")
            if finish_reason == "tool_calls":
                pending_content = text or None
            elif text:
                if pending or active:
                    raise ValueError("assistant_before_tool_group_complete")
                turns.append({"role": "assistant", "content": text})
        elif kind == "tool_start":
            tool = event.get("tool") or {}
            name = str(tool.get("name") or "").strip()
            if not name:
                raise ValueError("missing_tool_name")
            arguments = _arguments(event.get("input") or {})
            call_id = f"call_{call_number}"
            call_number += 1
            call = {
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }
            pending.append(call)
            item = observed.setdefault(
                name,
                {"description": str(tool.get("description") or ""), "calls": []},
            )
            item["calls"].append(arguments)
        elif kind == "tool_end":
            if pending:
                turns.append(
                    {
                        "role": "assistant",
                        "content": pending_content or "",
                        "tool_calls": pending.copy(),
                    }
                )
                active.extend(pending)
                pending.clear()
                pending_content = None
            if not active:
                raise ValueError("tool_result_without_call")
            raw_output = event.get("output")
            output_name = (
                OUTPUT_NAME.search(raw_output or "") if isinstance(raw_output, str) else None
            )
            match_name = output_name.group(1) if output_name else None
            position = next(
                (i for i, call in enumerate(active) if call["name"] == match_name),
                0,
            )
            call = active.pop(position)
            content = _tool_output(raw_output)
            if not content.strip():
                raise ValueError("empty_tool_output")
            turns.append(
                {
                    "role": "tool",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "content": content,
                }
            )

    if pending or active or turns[-1]["role"] != "assistant":
        raise ValueError("incomplete_trajectory")
    parts = incident_dir.parts
    issue_category, failure_type, incident_id = parts[-3:]
    metadata = {
        "failure_type": failure_type,
        "scenario": str(session.get("scenario_name") or ""),
        "issue_category": issue_category,
        "faulty_devices": sorted(_as_set(ground_truth.get("faulty_devices"))),
        "root_cause_name": sorted(_as_set(ground_truth.get("root_cause_name"))),
        "is_anomaly": bool(ground_truth.get("is_anomaly")),
        "success": True,
        "incident_id": incident_id,
        "backend_model": str(session.get("backend_model") or ""),
        "scenario_topo_size": str(session.get("scenario_topo_size") or ""),
        "original_source": "NIKA Traces v1 / Zenodo 17971675",
    }
    return canonical_row(
        f"nika_{incident_id}",
        "nika",
        "network_agent",
        config.NIKA_SYSTEM_PROMPT,
        _tool_schemas(observed),
        turns,
        metadata,
    )


def run_nika(root: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    root = root or config.RAW_DIR / "nika"
    output = output or config.intermediate_path("nika")
    incident_dirs = sorted(path.parent for path in root.rglob("ground_truth.json"))
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for incident_dir in incident_dirs:
        if not (incident_dir / "submission.json").exists():
            rejected["missing_submission"] += 1
            continue
        try:
            rows.append(parse_incident(incident_dir))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            rejected[str(error)] += 1
    write_jsonl(output, rows)
    manifest = {
        "incidents": len(incident_dirs),
        "output_rows": len(rows),
        "rejected": dict(rejected),
    }
    write_json(config.REPORTS_DIR / "nika_stats.json", manifest)
    LOGGER.info("NIKA 完成：accepted=%d rejected=%d", len(rows), sum(rejected.values()))
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_nika()
