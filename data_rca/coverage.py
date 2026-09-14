"""Create the OS/NOS by networking-domain coverage report."""

from collections import Counter
from typing import Any

PLATFORM_ORDER = ["linux", "frr", "sonic", "eos", "bmv2"]


def coverage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = Counter((row["network_domain"], row["platform"]) for row in rows)
    protocols = Counter((row["platform"], row["protocol"] or "none") for row in rows)
    diagnosis_types = Counter(row["diagnoses"][0]["type"] for row in rows)
    sources = Counter(row["source"] for row in rows)
    platforms = [name for name in PLATFORM_ORDER if any(key[1] == name for key in matrix)]
    platforms.extend(sorted({key[1] for key in matrix} - set(platforms)))
    domains = sorted({key[0] for key in matrix})
    return {
        "platforms": platforms,
        "domains": domains,
        "matrix": {
            domain: {platform: matrix[(domain, platform)] for platform in platforms}
            for domain in domains
        },
        "protocol_counts": {
            platform: {
                protocol: protocols[(platform, protocol)]
                for protocol in sorted({key[1] for key in protocols if key[0] == platform})
            }
            for platform in platforms
        },
        "diagnosis_type_counts": dict(sorted(diagnosis_types.items())),
        "source_counts": dict(sorted(sources.items())),
    }


def coverage_markdown(report: dict[str, Any]) -> str:
    platforms = report["platforms"]
    header = "| Network domain | " + " | ".join(platforms) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in platforms) + " |"
    lines = ["# OS / NOS × network domain coverage", "", header, separator]
    for domain in report["domains"]:
        counts = report["matrix"][domain]
        lines.append(
            "| "
            + domain
            + " | "
            + " | ".join(str(counts[platform]) for platform in platforms)
            + " |"
        )
    lines.extend(["", "Counts are canonical SFT rows, not unique fault labels.", ""])
    return "\n".join(lines)


def review_rows(rows: list[dict[str, Any]], per_source: int, seed: int) -> list[dict[str, Any]]:
    from data_rca.io import stable_hash

    selected = []
    for source in sorted({row["source"] for row in rows}):
        candidates = [row for row in rows if row["source"] == source]
        candidates.sort(key=lambda row: stable_hash(seed, source, row["id"], length=64))
        selected.extend(candidates[:per_source])
    return selected
