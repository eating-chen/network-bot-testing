"""Canonical messages/tools schema and model-independent validation."""

import json
from typing import Any

MODEL_TOKENS = ("<|start_header_id|>", "<|eot_id|>", "<TOOLCALL>", "<tool_call>")
JSON_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
TYPE_ALIASES = {
    "dict": "object",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments are not a JSON object")
    return parsed


def _fix_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_fix_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    fixed = {key: _fix_schema(item) for key, item in value.items()}
    raw_type = fixed.get("type")
    if isinstance(raw_type, str):
        simple = raw_type.split(",", 1)[0].strip().removesuffix(" optional").lower()
        if simple.startswith("list["):
            simple = "array"
        fixed["type"] = TYPE_ALIASES.get(simple, simple if simple in JSON_TYPES else "string")
    return fixed


def normalize_tool(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    function = raw.get("function") or raw
    parameters = _fix_schema(function.get("parameters") or {})
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    return {
        "type": "function",
        "function": {
            "name": text(function.get("name")),
            "description": text(function.get("description")),
            "parameters": parameters,
        },
    }


def make_row(
    identifier: str,
    source: str,
    task_type: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": identifier,
        "source": source,
        "task_type": task_type,
        "messages": messages,
        "tools": tools,
        "metadata": metadata,
    }


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"id", "source", "task_type", "messages", "tools", "metadata"}
    if missing := required - row.keys():
        return [f"missing fields: {sorted(missing)}"]
    if not all(text(row.get(key)) for key in ("id", "source", "task_type")):
        errors.append("id/source/task_type must be non-empty")
    if not isinstance(row["metadata"], dict):
        errors.append("metadata must be an object")

    tools = row["tools"]
    if not isinstance(tools, list):
        return errors + ["tools must be a list"]
    names = set()
    for index, tool in enumerate(tools):
        try:
            function = tool["function"]
            name = function["name"]
            parameters = function["parameters"]
        except (KeyError, TypeError):
            errors.append(f"tools[{index}] is not an OpenAI JSON schema")
            continue
        if tool.get("type") != "function" or not name or name in names:
            errors.append(f"tools[{index}] has bad type/name")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            errors.append(f"tools[{index}] parameters must be an object schema")
        names.add(name)

    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        return errors + ["messages must be a non-empty list"]
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if not roles or roles[0] not in {"system", "user"}:
        errors.append("conversation must start with system or user")
    if roles.count("system") > 1 or ("system" in roles and roles[0] != "system"):
        errors.append("system message may only appear once at the start")

    open_calls: dict[str, str] = {}
    assistant_answer = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] is not an object")
            continue
        role = message.get("role")
        content = text(message.get("content"))
        if role not in {"system", "user", "assistant", "tool"}:
            errors.append(f"messages[{index}] has bad role {role!r}")
            continue
        if role in {"system", "user"} and not content:
            errors.append(f"messages[{index}] has empty {role} content")
        if any(token in content for token in MODEL_TOKENS):
            errors.append(f"messages[{index}] contains model/template tokens")
        calls = message.get("tool_calls") or []
        if role == "assistant":
            assistant_answer |= bool(content)
            if not content and not calls:
                errors.append(f"messages[{index}] empty assistant")
            for call in calls:
                function = call.get("function") or {}
                call_id, name = text(call.get("id")), text(function.get("name"))
                if not call_id or call_id in open_calls or name not in names:
                    errors.append(f"messages[{index}] has invalid tool call")
                else:
                    try:
                        json_object(function.get("arguments"))
                        open_calls[call_id] = name
                    except (TypeError, ValueError, json.JSONDecodeError):
                        errors.append(f"messages[{index}] has invalid arguments")
        elif calls:
            errors.append(f"messages[{index}] only assistant may have tool_calls")
        if role == "tool":
            call_id = text(message.get("tool_call_id"))
            if call_id not in open_calls:
                errors.append(f"messages[{index}] has unmatched tool_call_id")
            else:
                if text(message.get("name")) != open_calls[call_id]:
                    errors.append(f"messages[{index}] tool name does not match its call")
                del open_calls[call_id]
            if not content:
                errors.append(f"messages[{index}] has empty tool result")

    last = messages[-1] if isinstance(messages[-1], dict) else {}
    final_call_ids = {call.get("id") for call in last.get("tool_calls") or []}
    terminal_call = (
        row["task_type"] in {"tool_decision", "tool_calling"}
        and final_call_ids
        and set(open_calls) == final_call_ids
    )
    if open_calls and not terminal_call:
        errors.append("trajectory has calls without results")
    if not assistant_answer and not terminal_call:
        errors.append("conversation needs a non-empty assistant answer")
    if last.get("role") != "assistant":
        errors.append("conversation must end with assistant")
    return errors
