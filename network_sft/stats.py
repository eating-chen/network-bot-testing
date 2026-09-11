"""Token/assistant-loss statistics and real chat-template compatibility checks."""

import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from network_sft import config
from network_sft.io import jsonl_rows, setup_logging, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def _words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _summary(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)

    def percentile(value: int) -> int:
        return ordered[max(0, math.ceil(value / 100 * len(ordered)) - 1)]

    return {
        "rows": len(values),
        "total": sum(values),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": max(values),
    }


def apply_template(tokenizer: Any, row: dict[str, Any], *, assistant_mask: bool = False) -> Any:
    kwargs = {
        "conversation": row["messages"],
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if row["tools"]:
        kwargs["tools"] = row["tools"]
    if assistant_mask:
        kwargs.update(return_dict=True, return_assistant_tokens_mask=True)
    result = tokenizer.apply_chat_template(**kwargs)
    if not assistant_mask and isinstance(result, Mapping):
        return result["input_ids"]
    return result


def _check_render(tokenizer: Any, row: dict[str, Any]) -> None:
    kwargs = {
        "conversation": row["messages"],
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if row["tools"]:
        kwargs["tools"] = row["tools"]
    rendered = tokenizer.apply_chat_template(**kwargs)
    rendered_words = _words(rendered)
    for message in row["messages"]:
        content = message.get("content") or ""
        if content and content not in rendered:
            raise ValueError(f"chat_template dropped {message['role']} content")
        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            if name not in rendered:
                raise ValueError(f"chat_template dropped tool call {name!r}")
    for tool in row["tools"]:
        function = tool["function"]
        name = function["name"]
        if name not in rendered:
            raise ValueError(f"chat_template dropped tool definition {name!r}")
        description = function.get("description") or ""
        description_probe = " ".join(_words(description).split()[:6])
        if description_probe and description_probe not in rendered_words:
            raise ValueError(f"chat_template dropped schema for {name!r}")


def _assistant_tokens(tokenizer: Any, row: dict[str, Any], rendered: list[int]) -> tuple[int, str]:
    try:
        result = apply_template(tokenizer, row, assistant_mask=True)
        mask = result.get("assistant_masks") or result.get("assistant_tokens_mask")
        if mask and sum(mask):
            return sum(mask), "chat_template_generation_mask"
    except (TypeError, ValueError):
        pass
    content = []
    for message in row["messages"]:
        if message["role"] == "assistant":
            content.append(message.get("content", ""))
            if message.get("tool_calls"):
                content.append(
                    json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
                )
    estimate = len(tokenizer("\n".join(content), add_special_tokens=False)["input_ids"])
    return min(estimate, len(rendered)), "assistant_content_estimate"


def _one_tokenizer(tokenizer_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat_template")
    lengths: defaultdict[str, list[int]] = defaultdict(list)
    assistant: defaultdict[str, list[int]] = defaultdict(list)
    source_lengths: defaultdict[str, list[int]] = defaultdict(list)
    source_assistant: defaultdict[str, list[int]] = defaultdict(list)
    methods, failures = defaultdict(int), []
    over_model_limit = 0
    for row in rows:
        try:
            _check_render(tokenizer, row)
            input_ids = apply_template(tokenizer, row)
            assistant_count, method = _assistant_tokens(tokenizer, row, input_ids)
            key = row["task_type"]
            lengths[key].append(len(input_ids))
            assistant[key].append(assistant_count)
            source_lengths[row["source"]].append(len(input_ids))
            source_assistant[row["source"]].append(assistant_count)
            methods[method] += 1
            if len(input_ids) > tokenizer.model_max_length:
                over_model_limit += 1
        except (KeyError, TypeError, ValueError) as error:
            failures.append({"id": row["id"], "source": row["source"], "error": str(error)})
    diagnostic_tokens = sum(lengths["diagnostic"])
    agentic_tokens = sum(sum(values) for key, values in lengths.items() if key != "diagnostic")
    diagnostic_assistant = sum(assistant["diagnostic"])
    agentic_assistant = sum(sum(values) for key, values in assistant.items() if key != "diagnostic")

    def shares(diagnostic: int, agentic: int) -> dict[str, Any]:
        total = diagnostic + agentic
        return {
            "diagnostic": diagnostic,
            "agentic": agentic,
            "diagnostic_percent": round(diagnostic / total * 100, 2) if total else 0,
            "agentic_percent": round(agentic / total * 100, 2) if total else 0,
        }

    return {
        "compatible_rows": sum(map(len, lengths.values())),
        "incompatible_rows": len(failures),
        "failure_examples": failures[:20],
        "assistant_token_methods": dict(methods),
        "model_max_length": tokenizer.model_max_length,
        "rows_over_model_max_length": over_model_limit,
        "tokens_by_task": {key: _summary(value) for key, value in sorted(lengths.items())},
        "tokens_by_source": {key: _summary(value) for key, value in sorted(source_lengths.items())},
        "assistant_tokens_by_task": {
            key: _summary(value) for key, value in sorted(assistant.items())
        },
        "assistant_tokens_by_source": {
            key: _summary(value) for key, value in sorted(source_assistant.items())
        },
        "token_share_by_capability": shares(diagnostic_tokens, agentic_tokens),
        "assistant_token_share_by_capability": shares(diagnostic_assistant, agentic_assistant),
    }


def run_stats() -> dict[str, Any]:
    try:
        import transformers  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install with: uv sync --extra sft") from error
    rows = []
    for split in config.SPLIT_RATIOS:
        rows.extend(jsonl_rows(config.FINAL_DIR / f"{split}.jsonl"))
    report = {"created_at": utc_now(), "rows": len(rows), "tokenizers": {}}
    for tokenizer_id in config.TOKENIZER_IDS:
        try:
            report["tokenizers"][tokenizer_id] = _one_tokenizer(tokenizer_id, rows)
        except (OSError, TypeError, ValueError) as error:
            report["tokenizers"][tokenizer_id] = {"load_error": str(error)}
        write_json(config.REPORTS_DIR / "token_stats.json", report)
        LOGGER.info("template check: %s", tokenizer_id)
    return report


if __name__ == "__main__":
    setup_logging()
    run_stats()
