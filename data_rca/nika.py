"""Convert NIKA ground truth plus tool results into canonical RCA records.

Agent reasoning, final answers, and submission.json are intentionally never read.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any

from data_rca import config
from data_rca.io import read_json, stable_hash, write_jsonl
from data_rca.schema import validate_record
from data_rca.taxonomy import nika_domain

OUTPUT_NAME = re.compile(r"\bname=['\"]([^'\"]+)['\"]")
OUTPUT_CONTENT = re.compile(r"^content=(?P<value>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")", re.DOTALL)
DEVICE_KEYS = ("router_name", "host_name", "switch_name", "host_a", "host_b", "host1", "host2")


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(value)
    return parsed if isinstance(parsed, dict) else {}


def _output(value: Any) -> str:
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    match = OUTPUT_CONTENT.match(value)
    if match:
        try:
            return str(ast.literal_eval(match.group("value"))).strip()
        except (SyntaxError, ValueError):
            pass
    return value.strip()


def _device(arguments: dict[str, Any]) -> str:
    devices = [str(arguments[key]) for key in DEVICE_KEYS if arguments.get(key)]
    return ", ".join(dict.fromkeys(devices))


def _command(tool: str, arguments: dict[str, Any]) -> str:
    if arguments.get("command"):
        return str(arguments["command"])
    if tool == "frr_get_bgp_conf":
        return "show BGP configuration"
    if tool == "frr_get_ospf_conf":
        return "show OSPF configuration"
    if tool == "frr_show_running_config":
        return "show running-config"
    if tool == "frr_show_ip_route":
        return "show ip route"
    if tool == "get_host_net_config":
        return "inspect interface addresses and routes"
    if tool == "systemctl_ops":
        operation = arguments.get("operation", "status")
        service = arguments.get("service_name", "<service>")
        return f"systemctl {operation} {service}"
    if tool == "ethtool":
        return f"ethtool {arguments.get('interface', '<interface>')}"
    if tool == "bmv2_table_dump":
        return f"dump P4 table {arguments.get('table_name', '<table>')}"
    details = ", ".join(f"{key}={value}" for key, value in arguments.items())
    return f"{tool}({details})" if details else tool


def _evidence_type(tool: str) -> str:
    config_tools = {
        "frr_get_bgp_conf",
        "frr_get_ospf_conf",
        "frr_show_running_config",
        "bmv2_table_dump",
    }
    if tool in config_tools:
        return "config"
    if tool in {"systemctl_ops", "bmv2_get_log"}:
        return "log"
    return "cli"


def extract_tool_evidence(log_path: Path) -> list[dict[str, Any]]:
    """Pair tool_start/tool_end events while tolerating malformed unrelated log lines."""
    active: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("event")
        if kind == "tool_start":
            tool = event.get("tool") or {}
            active.append(
                {
                    "tool": str(tool.get("name", "")),
                    "arguments": _arguments(event.get("input")),
                }
            )
        elif kind == "tool_end" and active:
            raw_output = event.get("output")
            name_match = (
                OUTPUT_NAME.search(raw_output or "") if isinstance(raw_output, str) else None
            )
            name = name_match.group(1) if name_match else ""
            positions = [i for i, call in enumerate(active) if call["tool"] == name]
            position = positions[0] if len(positions) == 1 else 0
            call = active.pop(position)
            output = _output(raw_output)
            if not call["tool"] or not output:
                continue
            truncated = len(output) > config.MAX_OUTPUT_CHARS
            if truncated:
                output = output[: config.MAX_OUTPUT_CHARS] + "\n[output truncated by data_rca]"
            evidence.append(
                {
                    "type": _evidence_type(call["tool"]),
                    "device": _device(call["arguments"]),
                    "command": _command(call["tool"], call["arguments"]),
                    "output": output,
                    "tool": call["tool"],
                    "arguments": call["arguments"],
                    "truncated": truncated,
                }
            )
    return evidence


def _problem_and_group(
    session: dict[str, Any], fault_type: str, affected: set[str]
) -> tuple[str, str]:
    task = str(session.get("task_description", ""))
    pattern = r"Network Description:\s*(.*?)(?:\n(?:Switches|Hosts|Routers|Your goal):)"
    match = re.search(pattern, task, re.DOTALL)
    description = " ".join(match.group(1).split()) if match else ""
    context_match = re.search(r"Network Description:\s*(.*?)(?:\n\nYour goal)", task, re.DOTALL)
    topology_context = " ".join(context_match.group(1).split()) if context_match else description
    scenario = str(session.get("scenario_name", "network"))
    problem = f"A network anomaly was reported in the {scenario} scenario."
    if description:
        problem += f" Network context: {description}"
    # The same topology and affected side stay together, while different fault
    # locations in a repeated benchmark scenario can be split independently.
    group_id = "nika_" + stable_hash(
        fault_type,
        scenario,
        topology_context,
        ",".join(sorted(affected)),
    )
    return problem, group_id


def _platform(fault_type: str, tools: set[str]) -> str:
    if fault_type.startswith("p4_") or any(tool.startswith("bmv2_") for tool in tools):
        return "bmv2"
    frr_tool_seen = any(tool.startswith("frr_") for tool in tools)
    if fault_type.startswith(("bgp_", "ospf_", "frr_")) or frr_tool_seen:
        return "frr"
    return "linux"


def parse_case(case_dir: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    truth = read_json(case_dir / "ground_truth.json")
    causes = truth.get("root_cause_name") or []
    if not truth.get("is_anomaly") or len(causes) != 1:
        raise ValueError("requires_one_anomalous_ground_truth_cause")
    fault_type = str(causes[0])
    if fault_type not in catalog:
        raise ValueError("fault_type_not_in_catalog")
    log_path = case_dir / "conversation_diagnosis_agent.log"
    if not log_path.exists():
        raise ValueError("missing_diagnosis_log")

    all_evidence = extract_tool_evidence(log_path)
    rule = catalog[fault_type]
    allowed_tools = {tool for group in rule["evidence_groups"] for tool in group}
    affected = {str(device) for device in truth.get("faulty_devices") or []}

    def on_affected_side(item: dict[str, Any]) -> bool:
        return any(device in item["device"].split(", ") for device in affected)

    relevant = [item for item in all_evidence if item["tool"] in allowed_tools]
    confirmation_evidence = relevant
    if rule.get("affected_only"):
        confirmation_evidence = [item for item in relevant if on_affected_side(item)]
    present_tools = {item["tool"] for item in confirmation_evidence}
    groups_present = all(
        present_tools.intersection(group) for group in rule["evidence_groups"]
    )
    affected_side_present = not affected or any(
        on_affected_side(item) for item in confirmation_evidence
    )
    confirmed = groups_present and affected_side_present

    # A ground-truth label without enough trace evidence becomes a possible
    # cause instead of being discarded or overstated as confirmed.
    selected = confirmation_evidence if confirmed else (relevant or all_evidence)
    if not selected:
        raise ValueError("no_tool_evidence")
    selected.sort(key=lambda item: (not on_affected_side(item), item["tool"], item["command"]))
    selected = selected[: config.MAX_EVIDENCE_ITEMS]

    session_path = case_dir / "session_meta.json"
    session = read_json(session_path) if session_path.exists() else {}
    problem, group_id = _problem_and_group(session, fault_type, affected)
    tools = {item["tool"] for item in selected}
    selected = [
        {key: item[key] for key in ("type", "device", "command", "output")}
        for item in selected
    ]
    affected_label = ", ".join(sorted(affected))
    reason = rule["reason"]
    if not confirmed:
        reason = (
            f"The observed evidence is consistent with {rule['cause'].lower()}, but this trace "
            "does not contain every catalog-required confirmation check. Use the identification "
            "steps below before treating it as confirmed."
        )
    diagnosis = {
        "cause": f"{rule['cause']} on {affected_label}" if affected_label else rule["cause"],
        "type": "confirmed_rca" if confirmed else "possible_rca",
        "reason": reason,
        "identify": rule["identify"],
        "resolution": rule["resolution"],
        "verification": rule["verification"],
    }
    domain, protocol = nika_domain(fault_type)
    platform = _platform(fault_type, tools)
    row = {
        "id": "nika_" + stable_hash(*case_dir.parts[-3:]),
        "source": "nika",
        "platform": platform,
        "network_domain": domain,
        "protocol": protocol,
        "problem": problem,
        "evidence": selected,
        "diagnoses": [diagnosis],
        "grounding": "runtime_trace",
        "group_id": group_id,
        "metadata": {
            "fault_type": fault_type,
            "source_case": "/".join(case_dir.parts[-3:]),
        },
    }
    errors = validate_record(row)
    if errors:
        raise ValueError("; ".join(errors))
    return row


def build_nika() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog = read_json(config.NIKA_CATALOG_PATH)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    case_dirs = sorted(path.parent for path in config.NIKA_ROOT.rglob("ground_truth.json"))
    for case_dir in case_dirs:
        try:
            rows.append(parse_case(case_dir, catalog))
        except (ValueError, json.JSONDecodeError) as error:
            rejected.append(
                {
                    "source": "nika",
                    "case": "/".join(case_dir.parts[-3:]),
                    "reason": str(error),
                }
            )
    return rows, rejected


if __name__ == "__main__":
    records, rejects = build_nika()
    write_jsonl(config.CANONICAL_DIR / "nika.jsonl", records)
    write_jsonl(config.REJECTED_PATH, rejects)
    print(f"NIKA: kept={len(records)} rejected={len(rejects)}")
