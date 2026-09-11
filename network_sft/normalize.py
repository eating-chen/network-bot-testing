"""Normalize all six sources into one transparent messages/tools schema."""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from network_sft import config
from network_sft.io import (
    data_rows,
    setup_logging,
    source_files,
    stable_id,
    write_json,
    write_jsonl,
)
from network_sft.nika import audit_incident, parse_incident
from network_sft.schema import json_object, make_row, normalize_tool, text, validate_row

LOGGER = logging.getLogger(__name__)
TOOLCALL = re.compile(r"^\s*<TOOLCALL>\s*(.*?)\s*</TOOLCALL>\s*$", re.DOTALL | re.I)
CLARIFY = re.compile(
    r"\b(could you|please (?:provide|specify|clarify)|need (?:to know|the)|which|what .*\?)", re.I
)
CANNOT = re.compile(
    r"\b(unable|cannot|can't|do not have|don't have|no (?:access|tool)|not able)\b", re.I
)


def _files(name: str) -> list[Path]:
    files = source_files(config.RAW_DIR / name)
    if not files:
        raise FileNotFoundError(f"no raw data for {name}; run download.py")
    return files


def _diagnostic(source: str, index: int, raw: dict[str, Any]) -> dict[str, Any]:
    if source == "5g_faults":
        question, answer = text(raw.get("input")), text(raw.get("output"))
        instruction = text(raw.get("instruction"))
        messages = ([{"role": "system", "content": instruction}] if instruction else []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        original = text(raw.get("id")) or stable_id(question, answer)
        metadata = {"original_id": original, "category": "5g_fault"}
    elif source == "telelogs":
        question, answer = text(raw.get("q")), text(raw.get("CoT"))
        original = text(raw.get("id_MD5")) or stable_id(question)
        category = text(raw.get("RCA")) or text(raw.get("c")) or "unknown"
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        metadata = {
            "original_id": original,
            "category": category,
            "answer_class": text(raw.get("c")),
            "root_cause": text(raw.get("RCA")),
        }
    else:
        question, answer = text(raw.get("question")), text(raw.get("answer"))
        original = text(raw.get("id")) or stable_id(question, answer)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        metadata = {"original_id": original, "category": "unclassified"}
    return make_row(
        f"{source}_{stable_id(original, index)}", source, "diagnostic", messages, [], metadata
    )


def _tools(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    return [normalize_tool(item) for item in (value or [])]


def _openai_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    output, waiting = [], []
    call_number = 0
    for raw in value or []:
        role = raw.get("role") or raw.get("from")
        role = {"human": "user", "gpt": "assistant", "function": "tool"}.get(role, role)
        if role == "assistant":
            message: dict[str, Any] = {"role": role, "content": text(raw.get("content"))}
            calls = []
            for source_call in raw.get("tool_calls") or []:
                function = source_call.get("function") or source_call
                call_id = text(source_call.get("id")) or f"call_{call_number}"
                call_number += 1
                call = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": text(function.get("name")),
                        "arguments": json_object(function.get("arguments")),
                    },
                }
                calls.append(call)
                waiting.append(call)
            if calls:
                message["tool_calls"] = calls
            output.append(message)
        elif role == "tool":
            name = text(raw.get("name"))
            given_id = text(raw.get("tool_call_id"))
            position = next(
                (
                    i
                    for i, call in enumerate(waiting)
                    if call["id"] == given_id or (name and call["function"]["name"] == name)
                ),
                0,
            )
            if not waiting:
                raise ValueError("tool result has no preceding call")
            call = waiting.pop(position)
            output.append(
                {
                    "role": "tool",
                    "name": name or call["function"]["name"],
                    "tool_call_id": call["id"],
                    "content": text(raw.get("content")),
                }
            )
        elif role in {"system", "user"}:
            output.append({"role": role, "content": text(raw.get("content"))})
    return output


def _complexity(messages: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    calls_by_turn = [len(message.get("tool_calls") or []) for message in messages]
    metrics = {
        "num_tool_calls": sum(calls_by_turn),
        "num_tool_results": sum(message["role"] == "tool" for message in messages),
        "num_user_turns": sum(message["role"] == "user" for message in messages),
        "max_calls_per_turn": max(calls_by_turn, default=0),
    }
    if metrics["max_calls_per_turn"] > 1:
        category = "parallel"
    elif metrics["num_user_turns"] > 1:
        category = "multi_turn"
    elif metrics["num_tool_calls"] > 1:
        category = "sequential"
    elif metrics["num_tool_calls"] == 1:
        category = "single_tool"
    else:
        category = "no_tool"
    return category, metrics


def normalize_toolace(index: int, raw: dict[str, Any]) -> dict[str, Any]:
    messages, tools = _openai_messages(raw.get("messages")), _tools(raw.get("tools"))
    category, metrics = _complexity(messages)
    original = text(raw.get("id")) or str(index)
    metadata = {"original_id": original, "category": category, **metrics}
    metadata["original_dataset"] = "Team-ACE/ToolACE via minpeter/toolace-parsed"
    return make_row(
        f"toolace_{stable_id(original, index)}",
        "toolace",
        "tool_calling",
        messages,
        tools,
        metadata,
    )


def _decision_answer(content: str) -> tuple[str, dict[str, Any]]:
    match = TOOLCALL.fullmatch(content)
    if match:
        calls = json.loads(match.group(1))
        if isinstance(calls, dict):
            calls = [calls]
        message = {"role": "assistant", "content": "", "tool_calls": []}
        for index, call in enumerate(calls):
            message["tool_calls"].append(
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": text(call.get("name")),
                        "arguments": json_object(call.get("arguments")),
                    },
                }
            )
        return "tool_call", message
    if CLARIFY.search(content) and "?" in content:
        category = "ask_clarification"
    elif CANNOT.search(content):
        category = "cannot_solve"
    else:
        category = "direct_answer"
    return category, {"role": "assistant", "content": content}


def normalize_when2call(index: int, raw: dict[str, Any]) -> dict[str, Any]:
    source_messages = raw.get("messages") or []
    if len(source_messages) != 2 or source_messages[0].get("role") != "user":
        raise ValueError("expected one user and one assistant message")
    category, assistant = _decision_answer(text(source_messages[1].get("content")))
    messages = [{"role": "user", "content": text(source_messages[0].get("content"))}, assistant]
    original = text(raw.get("uuid")) or stable_id(messages[0]["content"], index)
    return make_row(
        f"when2call_{stable_id(original)}",
        "when2call",
        "tool_decision",
        messages,
        _tools(raw.get("tools")),
        {"original_id": original, "category": category, "label_method": "V1 transparent rules"},
    )


def run_normalize() -> dict[str, Any]:
    report, rejected_rows = {}, []
    for source in ("5g_faults", "telelogs", "ccna", "toolace", "when2call"):
        rows, rejected = [], Counter()
        index = 0
        for path in _files(source):
            for raw in data_rows(path):
                try:
                    row = (
                        normalize_toolace(index, raw)
                        if source == "toolace"
                        else normalize_when2call(index, raw)
                        if source == "when2call"
                        else _diagnostic(source, index, raw)
                    )
                    rows.append(row)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    rejected[str(error)] += 1
                    rejected_rows.append({"source": source, "row": index, "reason": str(error)})
                index += 1
        write_jsonl(config.normalized_path(source), rows)
        report[source] = {"input": index, "normalized": len(rows), "rejected": dict(rejected)}

    nika_rows, nika_partial_rows, nika_audit, nika_rejected = [], [], [], Counter()
    incidents = sorted(path.parent for path in (config.RAW_DIR / "nika").rglob("ground_truth.json"))
    for path in incidents:
        if not (path / "submission.json").exists():
            nika_rejected["missing_submission"] += 1
            issue, failure, incident = path.parts[-3:]
            nika_audit.append(
                {
                    "incident_path": str(path),
                    "incident_id": incident,
                    "issue_category": issue,
                    "failure_type": failure,
                    "tier": "missing_submission",
                    "trajectory_status": "not_checked",
                }
            )
            rejected_rows.append(
                {"source": "nika", "row": str(path), "reason": "missing_submission"}
            )
            continue
        try:
            audit = audit_incident(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reason = f"invalid_submission_data: {error}"
            nika_rejected[reason] += 1
            nika_audit.append(
                {
                    "incident_path": str(path),
                    "incident_id": path.name,
                    "tier": "unreadable_submission",
                    "trajectory_status": "not_checked",
                    "trajectory_error": str(error),
                }
            )
            rejected_rows.append({"source": "nika", "row": str(path), "reason": reason})
            continue
        tier = audit["tier"]
        if tier == "wrong":
            audit["trajectory_status"] = "not_checked"
            nika_audit.append(audit)
            nika_rejected["wrong_submission"] += 1
            rejected_rows.append(
                {"source": "nika", "row": str(path), "reason": "wrong_submission"}
            )
            continue
        try:
            row = parse_incident(path, accepted_tier=tier)
            quality_errors = validate_row(row)
            audit["trajectory_status"] = "parsed"
            audit["quality_errors"] = quality_errors
            if tier == "exact_correct":
                nika_rows.append(row)
            elif not quality_errors:
                nika_partial_rows.append(row)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            audit["trajectory_status"] = "parse_rejected"
            audit["trajectory_error"] = str(error)
            if tier == "exact_correct":
                nika_rejected[str(error)] += 1
                rejected_rows.append(
                    {"source": "nika", "row": str(path), "reason": str(error)}
                )
        nika_audit.append(audit)
        if tier == "partial_correct":
            nika_rejected["partial_correct_submission"] += 1
            rejected_rows.append(
                {"source": "nika", "row": str(path), "reason": "partial_correct_submission"}
            )
    write_jsonl(config.normalized_path("nika"), nika_rows)
    write_jsonl(config.NORMALIZED_DIR / "nika_partial.jsonl", nika_partial_rows)
    write_jsonl(config.REPORTS_DIR / "nika_submission_audit.jsonl", nika_audit)
    trajectory_by_tier = {}
    for tier in sorted({row["tier"] for row in nika_audit}):
        tier_rows = [row for row in nika_audit if row["tier"] == tier]
        status = Counter(row["trajectory_status"] for row in tier_rows)
        status["quality_clean"] = sum(
            row["trajectory_status"] == "parsed" and not row.get("quality_errors")
            for row in tier_rows
        )
        status["quality_rejected"] = sum(bool(row.get("quality_errors")) for row in tier_rows)
        trajectory_by_tier[tier] = dict(status)
    report["nika"] = {
        "input": len(incidents),
        "normalized": len(nika_rows),
        "partial_trajectories_saved": len(nika_partial_rows),
        "submission_tiers": dict(Counter(row["tier"] for row in nika_audit)),
        "trajectory_by_tier": trajectory_by_tier,
        "rejected": dict(nika_rejected),
    }
    write_jsonl(config.REPORTS_DIR / "normalize_rejected.jsonl", rejected_rows)
    write_json(config.REPORTS_DIR / "normalize.json", report)
    LOGGER.info("normalize: %s", {source: value["normalized"] for source, value in report.items()})
    return report


if __name__ == "__main__":
    setup_logging()
    run_normalize()
