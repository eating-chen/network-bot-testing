"""Convert the network-related Cloud-OpsBench snapshots without using trajectories."""

import json
from pathlib import Path

from data_sft_rca import config
from data_sft_rca.io_utils import compact_text, read_json

FAULTS = {
    "service_selector_mismatch": (
        "Kubernetes Service selector mismatch",
        "service",
        "Correct the Service selector so it selects the intended pods.",
    ),
    "service_port_mapping_mismatch": (
        "Kubernetes Service target port mismatch",
        "service",
        "Align the Service port and targetPort with the application listener.",
    ),
    "service_protocol_mismatch": (
        "Kubernetes Service protocol mismatch",
        "service",
        "Configure the Service and workload to use the same transport protocol.",
    ),
    "service_env_var_address_mismatch": (
        "Incorrect service address in an application environment variable",
        "service",
        "Set the environment variable to the intended Service address.",
    ),
    "gateway_misrouted": (
        "Gateway routes traffic to the wrong backend",
        "gateway",
        "Correct the gateway route and backend mapping.",
    ),
    "service_dns_resolution_failure": (
        "Kubernetes Service DNS resolution failure",
        "dns",
        "Restore the Service DNS record or correct the queried service name.",
    ),
    "pod_network_delay": (
        "Excessive network delay affecting a pod",
        "ip",
        "Remove the pod network delay impairment and restore the expected path latency.",
    ),
    "node_network_delay": (
        "Excessive network delay affecting a node",
        "ip",
        "Remove the node network delay impairment and restore expected latency.",
    ),
    "node_network_packet_loss": (
        "Packet loss affecting a Kubernetes node",
        "ip",
        "Repair the lossy node network path.",
    ),
    "kube_proxy_unavailable": (
        "kube-proxy is unavailable",
        "service",
        "Restore kube-proxy on the affected node and reconcile Service rules.",
    ),
}


def _cache_value(cache: dict, tool: str, arguments: dict) -> str:
    key = (
        f"{tool}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
    )
    value = cache.get(key)
    if value is None:
        prefix = f"{tool}:"
        for candidate, candidate_value in cache.items():
            if not candidate.startswith(prefix):
                continue
            try:
                if json.loads(candidate[len(prefix) :]) == arguments:
                    value = candidate_value
                    break
            except json.JSONDecodeError:
                continue
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return compact_text(value, config.MAX_EVIDENCE_CHARS)


def _pick_use(milestone: dict, cache: dict) -> tuple[dict, str] | None:
    candidates = []
    for use in milestone.get("admissible_tool_uses", []):
        tool = str(use.get("tool_name", ""))
        arguments = use.get("arguments") or {}
        output = _cache_value(cache, tool, arguments)
        if not output:
            continue
        patterns = [str(item.get("value", "")) for item in use.get("evidence_patterns", [])]
        hits = sum(pattern and pattern in output for pattern in patterns)
        candidates.append((hits, bool(patterns), -len(output), tool, arguments, output))
    if not candidates:
        return None
    _, _, _, tool, arguments, output = max(candidates, key=lambda item: item[:4])
    return {"tool": tool, "arguments": arguments}, output


def parse_case(metadata_path: Path) -> dict | None:
    case_dir = metadata_path.parent
    metadata = read_json(metadata_path)
    result = metadata.get("result", {})
    cause_id = str(result.get("root_cause", ""))
    if cause_id not in config.CLOUDOPS_FAULTS:
        return None
    relative = metadata_path.relative_to(config.CLOUDOPS_ROOT / "benchmark")
    system, category, case_id = relative.parts[:3]
    milestone_path = (
        config.CLOUDOPS_ROOT / "process-label" / system / category / case_id / "milestone.json"
    )
    cache_path = case_dir / "tool_cache.json"
    if not milestone_path.exists() or not cache_path.exists():
        return None
    milestones = read_json(milestone_path).get("milestones", [])
    cache = read_json(cache_path)
    evidence = []
    for milestone in milestones:
        selected = _pick_use(milestone, cache)
        if not selected:
            continue
        call, output = selected
        role_name = str(milestone.get("role", ""))
        role = (
            "decisive"
            if "root_cause" in role_name
            else "symptom"
            if "symptom" in role_name
            else "supporting"
        )
        arguments = call["arguments"]
        device = next(
            (
                str(arguments[key])
                for key in ("service_name", "app_name", "node_name", "name")
                if arguments.get(key)
            ),
            "",
        )
        evidence.append(
            {
                "id": f"ev{len(evidence) + 1}",
                "type": "config"
                if call["tool"] == "GetAppYAML"
                else "log"
                if "Logs" in call["tool"]
                else "cli",
                "device": device,
                "command": (
                    f"{call['tool']} {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
                ),
                "output": output,
                "role": role,
            }
        )
    if not evidence or not any(item["role"] == "decisive" for item in evidence):
        return None
    cause, protocol, resolution = FAULTS[cause_id]
    affected = str(result.get("fault_object", "affected component"))
    root_case_id = f"cloudops_{system}_{category}_{case_id}"
    decisive = [item for item in evidence if item["role"] == "decisive"]
    return {
        "id": root_case_id,
        "root_case_id": root_case_id,
        "group_id": root_case_id,
        "source": "cloudopsbench",
        "platform_family": "kubernetes",
        "network_domain": "service_networking",
        "protocol": protocol,
        "task_type": "confirmed_rca",
        "fault_family": "service_endpoint_unreachable"
        if category == "service"
        else "network_performance_or_infrastructure",
        "problem": str(
            metadata.get("query") or "A Kubernetes service networking fault was reported."
        ),
        "evidence": evidence,
        "diagnoses": [
            {
                "status": "confirmed",
                "cause_id": cause_id,
                "cause": f"{cause} affecting {affected}",
                "evidence_ids": [item["id"] for item in decisive],
                "reason": (
                    "The official snapshot and diagnostic milestone identify "
                    f"{cause.lower()} at {affected}."
                ),
                "resolution": [resolution],
                "verification": [
                    {
                        "command": decisive[-1]["command"],
                        "expected_result": (
                            "The corrected state is visible and the reported service "
                            "symptom is cleared."
                        ),
                    }
                ],
            }
        ],
        "provenance": {
            "label_source": "metadata.json",
            "evidence_source": "tool_cache.json_and_milestone.json",
            "llm_generated": False,
        },
    }


def build_cloudops() -> list[dict]:
    benchmark = config.CLOUDOPS_ROOT / "benchmark"
    if not benchmark.exists():
        return []
    rows = [parse_case(path) for path in sorted(benchmark.glob("*/*/*/metadata.json"))]
    return [row for row in rows if row]


if __name__ == "__main__":
    rows = build_cloudops()
    print({"rows": len(rows), "root_cases": len({row["root_case_id"] for row in rows})})
