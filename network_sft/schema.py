"""Model-agnostic canonical SFT schema and trajectory validation."""

from __future__ import annotations

from typing import Any

TURN_ROLES = {"user", "assistant", "tool"}
REQUIRED_COLUMNS = {"id", "source", "task_type", "system", "tools", "turns", "metadata"}
FORBIDDEN_MODEL_TOKENS = (
    "<|start_header_id|>",
    "<|eot_id|>",
    "<tool_call>",
    "<start_of_turn>",
)


def canonical_row(
    identifier: str,
    source: str,
    task_type: str,
    system: str,
    tools: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct semantic data with no API wrappers or model tokens."""
    return {
        "id": identifier,
        "source": source,
        "task_type": task_type,
        "system": system,
        "tools": tools,
        "turns": turns,
        "metadata": metadata or {},
    }


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - row.keys()
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    if not row["id"] or not row["source"] or not row["task_type"]:
        errors.append("id/source/task_type must be non-empty")
    if not isinstance(row["system"], str):
        errors.append("system must be a string")
    elif any(token in row["system"] for token in FORBIDDEN_MODEL_TOKENS):
        errors.append("system contains a model-specific token")
    if not isinstance(row["metadata"], dict):
        errors.append("metadata must be an object")

    tools = row.get("tools")
    if not isinstance(tools, list):
        return errors + ["tools must be a list"]
    tool_names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            errors.append(f"tools[{index}] is not an object")
            continue
        name = str(tool.get("name") or "")
        if not name or name in tool_names:
            errors.append(f"tools[{index}] has missing/duplicate name")
        tool_names.add(name)
        if not isinstance(tool.get("description", ""), str):
            errors.append(f"tools[{index}].description must be a string")
        if not isinstance(tool.get("parameters"), dict):
            errors.append(f"tools[{index}].parameters must be an object")

    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        return errors + ["turns must be a non-empty list"]
    if turns[0].get("role") != "user":
        errors.append("first turn must be user")

    open_calls: dict[str, str] = {}
    saw_assistant_content = False
    previous_role: str | None = None
    for index, turn in enumerate(turns):
        unresolved_before_turn = bool(open_calls)
        if not isinstance(turn, dict):
            errors.append(f"turns[{index}] is not an object")
            continue
        role = turn.get("role")
        if role not in TURN_ROLES:
            errors.append(f"turns[{index}] has invalid role {role!r}")
            continue
        content = turn.get("content")
        calls = turn.get("tool_calls") or []
        if isinstance(content, str) and any(
            token in content for token in FORBIDDEN_MODEL_TOKENS
        ):
            errors.append(f"turns[{index}] contains a model-specific token")

        if role == "user" and not str(content or "").strip():
            errors.append(f"turns[{index}] has empty user content")
        if role == "user" and open_calls:
            errors.append(f"turns[{index}] user appears before tool results")
        if role == "assistant":
            if open_calls:
                errors.append(f"turns[{index}] assistant appears before tool results")
            if str(content or "").strip():
                saw_assistant_content = True
            if not str(content or "").strip() and not calls:
                errors.append(f"turns[{index}] assistant is empty")
            for call_index, call in enumerate(calls):
                if not isinstance(call, dict):
                    errors.append(f"turns[{index}].tool_calls[{call_index}] is not an object")
                    continue
                call_id = str(call.get("call_id") or "")
                name = str(call.get("name") or "")
                if not call_id or call_id in open_calls:
                    errors.append(f"turns[{index}] has missing/duplicate call_id")
                elif not name or name not in tool_names:
                    errors.append(f"turns[{index}] references unknown tool {name!r}")
                elif not isinstance(call.get("arguments"), dict):
                    errors.append(f"turns[{index}] tool arguments must be an object")
                else:
                    open_calls[call_id] = name
        elif calls:
            errors.append(f"turns[{index}] only assistant may contain tool_calls")

        if role == "tool":
            call_id = str(turn.get("call_id") or "")
            name = str(turn.get("name") or "")
            if call_id not in open_calls:
                errors.append(f"turns[{index}] has unmatched call_id {call_id!r}")
            else:
                if name != open_calls[call_id]:
                    errors.append(f"turns[{index}] tool name does not match its call")
                del open_calls[call_id]
            if content is None or not str(content).strip():
                errors.append(f"turns[{index}] has empty tool output")
        elif previous_role == "tool" and unresolved_before_turn:
            errors.append(f"turns[{index}] interrupts a tool-result group")
        elif previous_role == "tool" and role != "assistant":
            errors.append(f"turns[{index}] must be assistant after tool results")
        previous_role = role

    if open_calls:
        errors.append(f"missing tool results for {sorted(open_calls)}")
    if not saw_assistant_content:
        errors.append("at least one non-empty assistant response is required")
    if turns[-1].get("role") != "assistant":
        errors.append("conversation must end with assistant")
    return errors
