"""Adapt the existing NIKA, NetOpsBench and ANTA converters to the v2 schema."""

import json
import re

from data_sft_rca import config
from data_sft_rca.io_utils import read_jsonl, stable_id

V1_DIR = config.ROOT / "data" / "rca" / "01_canonical"


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value[:80] or "unknown"


def _fault_family(cause_id: str, domain: str, protocol: str) -> str:
    if cause_id.startswith("bgp_") or protocol == "bgp":
        return "bgp_session_or_route_failure"
    if cause_id.startswith("ospf_") or protocol == "ospf":
        return "ospf_adjacency_or_route_failure"
    if any(word in cause_id for word in ("link", "mtu", "packet", "interface")):
        return "interface_or_path_degradation"
    if any(word in cause_id for word in ("dns", "dhcp", "service", "port")):
        return "service_endpoint_unreachable"
    if domain == "routing":
        return "routing_failure"
    return f"{domain}_abnormal_state"


def _decisive_index(items: list[dict]) -> int:
    markers = (
        "config",
        "status",
        "interface",
        "neighbor",
        "acl",
        "route",
        "table",
        "ethtool",
        "systemctl",
        "log",
        "metrics",
    )
    scored = []
    for index, item in enumerate(items):
        command = item.get("command", "").lower()
        score = sum(marker in command for marker in markers)
        score += 2 if item.get("type") in {"config", "log", "expected_observed"} else 0
        scored.append((score, index))
    return max(scored)[1]


def _evidence(v1: dict, task_type: str) -> list[dict]:
    items = v1["evidence"]
    decisive = _decisive_index(items) if task_type == "confirmed_rca" else -1
    result = []
    for index, item in enumerate(items):
        role = "decisive" if index == decisive else "symptom" if index == 0 else "supporting"
        if (
            task_type == "state_diagnosis"
            and v1["diagnoses"][0]["type"] == "confirmed_state_mismatch"
        ):
            role = "decisive"
        result.append(
            {
                "id": f"ev{index + 1}",
                "type": item.get("type", "cli"),
                "device": item.get("device", ""),
                "command": item.get("command", "observation"),
                "output": item["output"],
                "role": role,
            }
        )
    return result


def adapt_v1(v1: dict) -> dict:
    old = v1["diagnoses"][0]
    old_type = old["type"]
    task_type = {
        "confirmed_rca": "confirmed_rca",
        "possible_rca": "probable_cause_analysis",
        "confirmed_state_mismatch": "state_diagnosis",
        "diagnostic_guidance": "state_diagnosis",
    }[old_type]
    status = {
        "confirmed_rca": "confirmed",
        "possible_rca": "possible",
        "confirmed_state_mismatch": "state_only",
        "diagnostic_guidance": "state_only",
    }[old_type]
    meta = v1.get("metadata", {})
    cause_id = str(meta.get("fault_type") or meta.get("test_name") or _slug(old["cause"]))
    evidence = _evidence(v1, task_type)
    diagnosis = {
        "status": status,
        "cause_id": _slug(cause_id),
        "cause": old["cause"],
        "evidence_ids": [item["id"] for item in evidence],
        "reason": old["reason"],
        "resolution": old.get("resolution", []),
        "verification": old.get("verification", []),
    }
    if v1["source"] == "nika":
        root_case_id = "nika_case_" + stable_id(meta.get("source_case", v1["id"]))
        group_id = root_case_id
    elif v1["source"] == "netopsbench":
        root_case_id = "netopsbench_case_" + stable_id(meta.get("scenario_id", v1["id"]))
        group_id = root_case_id
    else:
        root_case_id = "anta_case_" + stable_id(
            meta.get("source_path", ""),
            meta.get("test_name", ""),
            meta.get("fixture_case", v1["id"]),
        )
        group_id = "anta_family_" + stable_id(
            meta.get("source_path", ""), meta.get("test_name", "")
        )
    return {
        "id": v1["id"],
        "root_case_id": root_case_id,
        "group_id": group_id,
        "source": v1["source"],
        "platform_family": v1["platform"],
        "network_domain": v1["network_domain"],
        "protocol": v1.get("protocol", ""),
        "task_type": task_type,
        "fault_family": _fault_family(cause_id, v1["network_domain"], v1.get("protocol", "")),
        "problem": v1["problem"],
        "evidence": evidence,
        "diagnoses": [diagnosis],
        "provenance": {
            "label_source": "benchmark_ground_truth_or_official_fixture",
            "evidence_source": v1["grounding"],
            "llm_generated": False,
        },
    }


def _v1_rows(source: str) -> list[dict]:
    path = V1_DIR / f"{source}.jsonl"
    if path.exists():
        return read_jsonl(path)
    if source == "nika":
        from data_rca.nika import build_nika

        return build_nika()[0]
    if source == "netopsbench":
        from data_rca.netopsbench import build_netopsbench

        return build_netopsbench()[0]
    from data_rca.anta import build_anta

    return build_anta()[0]


def build_legacy_sources() -> list[dict]:
    rows = [adapt_v1(row) for source in ("nika", "netopsbench") for row in _v1_rows(source)]
    anta = [adapt_v1(row) for row in _v1_rows("anta")]
    confirmed = [
        row
        for row in anta
        if row["diagnoses"][0]["status"] == "state_only"
        and any(item["role"] == "decisive" for item in row["evidence"])
    ]
    guidance = [row for row in anta if row not in confirmed]
    preferred_domains = {"routing", "interfaces", "switching", "overlay", "connectivity"}
    guidance.sort(
        key=lambda row: (
            row["network_domain"] not in preferred_domains,
            stable_id(config.SEED, row["id"], length=40),
        )
    )
    return rows + confirmed + guidance[: config.ANTA_GUIDANCE_LIMIT]


if __name__ == "__main__":
    built = build_legacy_sources()
    print(
        json.dumps(
            {
                source: sum(row["source"] == source for row in built)
                for source in {row["source"] for row in built}
            },
            indent=2,
        )
    )
