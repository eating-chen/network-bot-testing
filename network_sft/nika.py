"""Turn one successful NIKA incident log into one canonical trajectory."""

import ast
import json
import re
from pathlib import Path
from typing import Any

from network_sft import config
from network_sft.io import stable_id
from network_sft.schema import make_row, normalize_tool, text

ERROR_EVENTS = {"chain_error", "llm_error", "tool_error", "fatal_error"}
OUTPUT_NAME = re.compile(r"\bname=['\"]([^'\"]+)['\"]")
OUTPUT_CONTENT = re.compile(r"^content=(?P<v>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")", re.DOTALL)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)} if value not in (None, "") else set()


def evaluate_submission(truth: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    """Assign a traceable exact/partial/wrong tier without changing V1's strict gate."""
    truth_devices = _set(truth.get("faulty_devices"))
    submitted_devices = _set(submission.get("faulty_devices"))
    truth_causes = _set(truth.get("root_cause_name"))
    submitted_causes = _set(submission.get("root_cause_name"))
    matches = {
        "is_anomaly": truth.get("is_anomaly") == submission.get("is_anomaly"),
        "faulty_devices_exact": truth_devices == submitted_devices,
        "root_causes_exact": truth_causes == submitted_causes,
    }
    device_overlap = truth_devices & submitted_devices
    cause_overlap = truth_causes & submitted_causes
    if all(matches.values()):
        tier = "exact_correct"
    elif matches["is_anomaly"] and (
        (matches["faulty_devices_exact"] and bool(cause_overlap))
        or (matches["root_causes_exact"] and bool(device_overlap))
    ):
        tier = "partial_correct"
    else:
        tier = "wrong"
    return {
        "tier": tier,
        "matches": matches,
        "overlap": {
            "faulty_devices": sorted(device_overlap),
            "root_cause_name": sorted(cause_overlap),
        },
    }


def audit_incident(path: Path) -> dict[str, Any]:
    """Return the benchmark comparison needed to trace why an incident was kept."""
    truth = _object(path / "ground_truth.json")
    submission = _object(path / "submission.json")
    issue, failure, incident = path.parts[-3:]
    return {
        "incident_path": str(path),
        "incident_id": incident,
        "issue_category": issue,
        "failure_type": failure,
        **evaluate_submission(truth, submission),
        "ground_truth": {
            "is_anomaly": truth.get("is_anomaly"),
            "faulty_devices": sorted(_set(truth.get("faulty_devices"))),
            "root_cause_name": sorted(_set(truth.get("root_cause_name"))),
        },
        "submission": {
            "is_anomaly": submission.get("is_anomaly"),
            "faulty_devices": sorted(_set(submission.get("faulty_devices"))),
            "root_cause_name": sorted(_set(submission.get("root_cause_name"))),
        },
    }


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool input is not an object")
    return parsed


def _output(value: Any) -> str:
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if match := OUTPUT_CONTENT.match(value):
        try:
            return str(ast.literal_eval(match.group("v")))
        except (SyntaxError, ValueError):
            pass
    return value.strip()


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _tools(observed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name, item in sorted(observed.items()):
        calls = item["calls"]
        keys = set().union(*(call.keys() for call in calls)) if calls else set()
        required = set.intersection(*(set(call) for call in calls)) if calls else set()
        properties = {
            key: {"type": _json_type(next(call[key] for call in calls if key in call))}
            for key in sorted(keys)
        }
        output.append(
            normalize_tool(
                {
                    "name": name,
                    "description": item["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": sorted(required),
                    },
                }
            )
        )
    return output


def _events(path: Path):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid event JSON at line {number}") from error


def parse_incident(path: Path, *, accepted_tier: str = "exact_correct") -> dict[str, Any]:
    if accepted_tier not in {"exact_correct", "partial_correct"}:
        raise ValueError(f"unsupported positive tier: {accepted_tier}")
    truth, submission = _object(path / "ground_truth.json"), _object(path / "submission.json")
    evaluation = evaluate_submission(truth, submission)
    if evaluation["tier"] != accepted_tier:
        raise ValueError(f"{evaluation['tier']}_submission")
    session = _object(path / "session_meta.json")
    task = text(session.get("task_description"))
    if not task:
        raise ValueError("missing_task_description")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": config.NIKA_SYSTEM},
        {"role": "user", "content": task},
    ]
    pending, active, observed = [], [], {}
    pending_content, call_number = "", 0
    for event in _events(path / "conversation_diagnosis_agent.log"):
        kind = event.get("event")
        if kind in ERROR_EVENTS or event.get("invalid_tool_calls"):
            raise ValueError("fatal_or_invalid_tool_event")
        if kind == "llm_end":
            answer = text(event.get("text"))
            if (event.get("generation_info") or {}).get("finish_reason") == "tool_calls":
                pending_content = answer
            elif answer:
                if pending or active:
                    raise ValueError("assistant_before_tools_complete")
                messages.append({"role": "assistant", "content": answer})
        elif kind == "tool_start":
            tool = event.get("tool") or {}
            name = text(tool.get("name"))
            if not name:
                raise ValueError("missing_tool_name")
            arguments = _arguments(event.get("input") or {})
            call = {
                "id": f"call_{call_number}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            call_number += 1
            pending.append(call)
            observed.setdefault(name, {"description": text(tool.get("description")), "calls": []})[
                "calls"
            ].append(arguments)
        elif kind == "tool_end":
            if pending:
                messages.append(
                    {"role": "assistant", "content": pending_content, "tool_calls": pending.copy()}
                )
                active.extend(pending)
                pending, pending_content = [], ""
            if not active:
                raise ValueError("tool_result_without_call")
            raw = event.get("output")
            match = OUTPUT_NAME.search(raw or "") if isinstance(raw, str) else None
            name = match.group(1) if match else ""
            matches = [
                i for i, call in enumerate(active) if call["function"]["name"] == name
            ]
            if len(matches) != 1:
                raise ValueError("ambiguous_tool_result")
            position = matches[0]
            call = active.pop(position)
            messages.append(
                {
                    "role": "tool",
                    "name": call["function"]["name"],
                    "tool_call_id": call["id"],
                    "content": _output(raw),
                }
            )

    if pending or active or messages[-1]["role"] != "assistant":
        raise ValueError("incomplete_trajectory")
    issue, failure, incident = path.parts[-3:]
    metadata = {
        "original_id": incident,
        "scenario": text(session.get("scenario_name")),
        "issue_category": issue,
        "failure_type": failure,
        # Broad issue category is the split stratum; failure_type remains available below it.
        "category": issue,
        "faulty_devices": sorted(_set(truth.get("faulty_devices"))),
        "root_cause": sorted(_set(truth.get("root_cause_name"))),
        "submission_tier": evaluation["tier"],
        "submission_matches": evaluation["matches"],
    }
    identifier = f"nika_{stable_id(metadata['scenario'], issue, failure, incident)}"
    return make_row(identifier, "nika", "agentic", messages, _tools(observed), metadata)
