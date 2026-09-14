"""Quality filter, deduplicate/group, then make the traceable V1 selection."""

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from network_sft import config
from network_sft.io import (
    jsonl_rows,
    setup_logging,
    stable_id,
    stable_score,
    write_json,
    write_jsonl,
)
from network_sft.schema import validate_row

LOGGER = logging.getLogger(__name__)
WORDS = re.compile(r"[a-z0-9]+")
TOPICS = {
    "routing_protocols": ("bgp", "ospf", "eigrp", "rip", "routing protocol", "neighbor"),
    "vlan_stp_l2": ("vlan", "stp", "spanning tree", "etherchannel", "switchport", "layer 2"),
    "acl_nat_security": ("acl", "access list", "nat", "firewall", "security", "vpn", "ipsec"),
    "network_services": ("dhcp", "dns", "ntp", "snmp", "syslog", "aaa", "radius", "tacacs"),
    "ip_routing": ("ipv4", "ipv6", "subnet", "route", "gateway", "icmp", "arp", "ping"),
}
MINHASH_SALTS = tuple(
    int(hashlib.sha256(f"salt-{i}".encode()).hexdigest()[:16], 16) for i in range(12)
)


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(WORDS.findall(value))


def _first_user(row: dict[str, Any]) -> str:
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def network_topic(value: str) -> str:
    lowered = value.casefold()
    scores = {topic: sum(term in lowered for term in terms) for topic, terms in TOPICS.items()}
    topic, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    return topic if score else "management_cli_misc"


def _fingerprint(row: dict[str, Any]) -> str:
    semantic = {"messages": row["messages"], "tools": row["tools"]}
    compact = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized_text(compact).encode()).hexdigest()


def _shingles(value: str) -> set[int]:
    tokens = normalized_text(value).split()
    pieces = (" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2)))
    return {int(hashlib.blake2b(piece.encode(), digest_size=8).hexdigest(), 16) for piece in pieces}


def _signature(shingles: set[int]) -> tuple[int, ...]:
    return tuple(min((value ^ salt for value in shingles), default=0) for salt in MINHASH_SALTS)


def _find(parent: dict[int, int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _near_groups(rows: list[dict[str, Any]]) -> int:
    """LSH proposes pairs; exact Jaccard decides links. Near rows stay but share group_id."""
    links = 0
    by_stratum: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_stratum[str(row["metadata"]["category"])].append(index)

    for indices in by_stratum.values():
        parent = {index: index for index in indices}

        sets, buckets = {}, defaultdict(list)
        for index in indices:
            shingles = _shingles(_first_user(rows[index]))
            sets[index] = shingles
            signature = _signature(shingles)
            candidates = set()
            for band in range(4):
                key = (band, signature[band * 3 : band * 3 + 3])
                candidates.update(buckets[key])
                buckets[key].append(index)
            for other in candidates:
                overlap = len(shingles & sets[other])
                union_size = len(shingles | sets[other])
                if union_size and overlap / union_size >= config.NEAR_DUP_JACCARD:
                    parent[_find(parent, other)] = _find(parent, index)
                    links += 1
        members: defaultdict[int, list[int]] = defaultdict(list)
        for index in indices:
            members[_find(parent, index)].append(index)
        for cluster in members.values():
            group_id = (
                f"near_{stable_id(rows[min(cluster)]['source'], *(rows[i]['id'] for i in cluster))}"
            )
            for index in cluster:
                rows[index]["metadata"]["group_id"] = group_id
    return links


def _curate_sources() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    accepted, rejected, exact_seen = {}, [], {}
    report = {}
    for source in config.SOURCE_ORDER:
        rows = []
        for row in jsonl_rows(config.normalized_path(source)):
            if source in {"ccna", "5g_faults"}:
                row["metadata"]["category"] = network_topic(_first_user(row))
            errors = validate_row(row)
            if source == "toolace" and row["metadata"]["category"] == "no_tool":
                errors.append("ToolACE no-tool row is outside the V1 positive tool-use pool")
            fingerprint = _fingerprint(row)
            if fingerprint in exact_seen:
                errors.append(f"exact duplicate of {exact_seen[fingerprint]}")
            if errors:
                rejected.append({"id": row.get("id"), "source": source, "reasons": errors})
                continue
            exact_seen[fingerprint] = row["id"]
            rows.append(row)
        if source == "nika":
            for row in rows:
                meta = row["metadata"]
                key = (meta.get("scenario"), meta.get("failure_type"), meta.get("original_id"))
                meta["group_id"] = f"nika_{stable_id(*key)}"
            links = 0
        else:
            links = _near_groups(rows)
        accepted[source] = rows
        write_jsonl(config.curated_path(source), rows)
        report[source] = {"curated": len(rows), "near_duplicate_links": links}
    write_jsonl(config.REPORTS_DIR / "quality_rejected.jsonl", rejected)
    report["rejected"] = len(rejected)
    return accepted, report


def deterministic_sample(
    rows: list[dict[str, Any]], target: int, source: str
) -> list[dict[str, Any]]:
    """Take a reproducible simple sample without category quotas or oversampling."""
    ordered = sorted(rows, key=lambda row: stable_score(config.SEED, source, row["id"]))
    return ordered[: min(target, len(ordered))]


def run_curate() -> dict[str, Any]:
    sources, report = _curate_sources()
    selected = []
    selected.extend(sources["5g_faults"])
    selected.extend(sources["telelogs"])
    selected.extend(sources["ccna"])
    selected.extend(sources["nika"])
    selected.extend(
        deterministic_sample(sources["toolace"], config.SAMPLE_TARGETS["toolace"], "toolace")
    )
    selected.extend(
        deterministic_sample(
            sources["when2call"], config.SAMPLE_TARGETS["when2call"], "when2call"
        )
    )
    selected.sort(key=lambda row: stable_score(config.SEED, "mixed", row["id"]))
    write_jsonl(config.SELECTED_DIR / "selected.jsonl", selected)

    counts = Counter((row["source"], row["metadata"]["category"]) for row in selected)
    report["selected_rows"] = len(selected)
    report["selected_by_source"] = dict(Counter(row["source"] for row in selected))
    report["selected_by_capability"] = dict(
        Counter("diagnostic" if row["task_type"] == "diagnostic" else "agentic" for row in selected)
    )
    report["selected_by_source_category"] = {
        f"{source}/{category}": count for (source, category), count in sorted(counts.items())
    }
    report["sampling"] = {
        source: {
            "method": "deterministic_hash_sample",
            "available": len(sources[source]),
            "target": config.SAMPLE_TARGETS[source],
            "selected": sum(row["source"] == source for row in selected),
        }
        for source in config.SAMPLE_TARGETS
    }
    write_json(config.REPORTS_DIR / "curation.json", report)
    LOGGER.info("selected: %s", report["selected_by_source"])
    return report


if __name__ == "__main__":
    setup_logging()
    run_curate()
