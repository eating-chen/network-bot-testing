"""Translate canonical semantics to Hugging Face chat-template input."""

from __future__ import annotations

from typing import Any


def to_hf_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
        },
    }


def render_for_hf(
    example: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    if example["system"]:
        messages.append({"role": "system", "content": example["system"]})
    for turn in example["turns"]:
        role = turn["role"]
        if role == "assistant" and turn.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.get("content", ""),
                    "tool_calls": [
                        {
                            "id": call["call_id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in turn["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "name": turn["name"],
                    "tool_call_id": turn["call_id"],
                    "content": turn["content"],
                }
            )
        else:
            messages.append({"role": role, "content": turn.get("content", "")})
    return messages, [to_hf_tool(tool) for tool in example["tools"]]
