"""Convert NetOpsBench ATIF tool observations plus evaluator ground truth.

Agent reasoning, evidence summaries, and final diagnoses are never used.
"""

import json
from pathlib import Path
from typing import Any

from data_rca import config
from data_rca.io import read_json, stable_hash
from data_rca.schema import validate_record

IGNORED_TOOLS = {"read_file"}


def _observation_output(step: dict[str, Any]) -> str:
    results = step.get("observation", {}).get("results", [])
    values: list[Any] = []
    for result in results:
        content = result.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            values.append(content)
            continue
        artifact = parsed.get("artifact") or {}
        structured = artifact.get("structured_content")
        if structured is not None:
            values.append(structured)
        else:
            values.append(parsed)
    if not values:
        return ""
    output = json.dumps(
        values[0] if len(values) == 1 else values, ensure_ascii=False, sort_keys=True
    )
    if len(output) > config.MAX_OUTPUT_CHARS:
        output = output[: config.MAX_OUTPUT_CHARS] + "\n[output truncated by data_rca]"
    return output


def _tool_evidence(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for position, step in enumerate(trajectory.get("steps", [])):
        if step.get("extra", {}).get("type") != "tool_call":
            continue
        name = str(step.get("extra", {}).get("name", ""))
        if not name or name in IGNORED_TOOLS:
            continue
        calls = step.get("tool_calls") or []
        arguments = calls[0].get("arguments", {}) if calls else {}
        if not isinstance(arguments, dict):
            arguments = {}
        output = _observation_output(step)
        if not output:
            continue
        device = arguments.get("device") or arguments.get("src") or ""
        evidence.append(
            {
                "type": (
                    "config"
                    if name == "get_device_config"
                    else "log"
                    if name in {"get_device_logs", "query_bgp_events"}
                    else "cli"
                ),
                "device": str(device),
                "command": f"{name} {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}",
                "output": output,
                "tool": name,
                "arguments": arguments,
                "trace_position": position,
            }
        )
    return evidence


def _problem(trajectory: dict[str, Any], scenario_id: str) -> str:
    initial = next(
        (
            step.get("message")
            for step in trajectory.get("steps", [])
            if step.get("extra", {}).get("is_initial_context")
        ),
        None,
    )
    if isinstance(initial, str):
        try:
            payload = json.loads(initial)
            observations = payload.get("symptoms", {}).get("observations", {})
            if observations.get("anomalies_detected") is True:
                return (
                    "Runtime telemetry detected a network anomaly in "
                    f"NetOpsBench scenario {scenario_id}."
                )
            if observations:
                return (
                    "Runtime telemetry reported degraded network behavior in "
                    f"NetOpsBench scenario {scenario_id}."
                )
        except json.JSONDecodeError:
            pass
    return f"A network anomaly was reported in NetOpsBench scenario {scenario_id}."


def parse_netopsbench(
    result: dict[str, Any], trajectory: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    details = result.get("details", {})
    truth = details.get("ground_truth")
    if not isinstance(truth, dict) or not truth.get("fault_type"):
        raise ValueError("no_anomalous_ground_truth")
    fault_type = str(truth["fault_type"])
    if fault_type not in catalog:
        raise ValueError("fault_type_not_in_catalog")
    location = truth.get("location") or {}
    device = str(location.get("device", ""))
    interface = str(location.get("interface", ""))
    rule = catalog[fault_type]
    preferred = set(rule["preferred_tools"])

    evidence = _tool_evidence(trajectory)

    def relevance(item: dict[str, Any]) -> tuple[int, int, int, int]:
        searchable = json.dumps(
            {"arguments": item["arguments"], "output": item["output"]}, ensure_ascii=False
        )
        return (
            int(bool(device) and device in searchable),
            int(bool(interface) and interface in searchable),
            int(item["tool"] in preferred),
            item["trace_position"],
        )

    relevant = [
        item
        for item in evidence
        if item["tool"] in preferred
        and (not device or device in json.dumps({"a": item["arguments"], "o": item["output"]}))
    ]
    if not relevant:
        raise ValueError("no_ground_truth_relevant_tool_observation")
    selected = sorted(evidence, key=relevance, reverse=True)[: config.MAX_EVIDENCE_ITEMS]
    selected.sort(key=lambda item: item["trace_position"])

    location_text = device
    if interface:
        location_text += f" interface {interface}"
    cause = rule["cause"] + (f" on {location_text}" if location_text else "")
    identify_command = ", ".join(rule["preferred_tools"][:3])
    scenario_id = str(result.get("scenario_id") or details.get("scenario_id") or "unknown")
    trace_id = str(result.get("trace_id") or trajectory.get("trajectory_id"))
    selected = [
        {key: item[key] for key in ("type", "device", "command", "output")}
        for item in selected
    ]
    row = {
        "id": "netopsbench_" + stable_hash(trace_id),
        "source": "netopsbench",
        "platform": "sonic",
        "network_domain": rule["domain"],
        "protocol": rule["protocol"],
        "problem": _problem(trajectory, scenario_id),
        "evidence": selected,
        "diagnoses": [
            {
                "type": "confirmed_rca",
                "cause": cause,
                "reason": rule["reason"],
                "identify": [
                    {
                        "command": f"{identify_command} on {location_text or 'the affected path'}",
                        "expected_evidence": (
                            f"Evidence of {rule['cause'].lower()} at the benchmark "
                            "ground-truth location."
                        ),
                    }
                ],
                "resolution": rule["resolution"],
                "verification": rule["verification"],
            }
        ],
        "grounding": "runtime_trace",
        # Cross-model repetitions of one benchmark scenario cannot cross splits.
        "group_id": "netopsbench_" + stable_hash(scenario_id),
        "metadata": {
            "fault_type": fault_type,
            "scenario_id": scenario_id,
            "trace_id": trace_id,
        },
    }
    errors = validate_record(row)
    if errors:
        raise ValueError("; ".join(errors))
    return row


def _trajectory_map(run_dir: Path) -> dict[str, Path]:
    index_path = run_dir / "traces" / "index.jsonl"
    mapping: dict[str, Path] = {}
    if not index_path.exists():
        return mapping
    for line in index_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        mapping[str(item["trace_id"])] = run_dir / item["atif_path"]
    return mapping


def build_netopsbench() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not config.NETOPSBENCH_EXTRACTED.exists():
        return [], [{"source": "netopsbench", "reason": "snapshot_not_extracted"}]
    catalog = read_json(config.NETOPSBENCH_CATALOG_PATH)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for results_path in sorted(config.NETOPSBENCH_EXTRACTED.rglob("results.jsonl")):
        run_dir = results_path.parent.parent
        trajectories = _trajectory_map(run_dir)
        lines = results_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            result = json.loads(line)
            trace_id = str(result.get("trace_id", ""))
            trajectory_path = trajectories.get(trace_id)
            try:
                if trajectory_path is None or not trajectory_path.exists():
                    raise ValueError("trajectory_not_found")
                rows.append(parse_netopsbench(result, read_json(trajectory_path), catalog))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                rejected.append(
                    {
                        "source": "netopsbench",
                        "run": run_dir.name,
                        "line": line_number,
                        "trace_id": trace_id,
                        "reason": str(error),
                    }
                )
    return rows, rejected
