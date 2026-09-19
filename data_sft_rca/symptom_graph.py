"""Build a grounded symptom-family to cause catalog from canonical cases."""

import re
from collections import defaultdict

from data_sft_rca import config


def _generic_cause(text: str) -> str:
    text = re.split(r"\s+(?:on|affecting)\s+", text, maxsplit=1)[0]
    return text.rstrip(".")


def build_graph(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["task_type"] != "confirmed_rca":
            continue
        diagnosis = row["diagnoses"][0]
        grouped[(row["fault_family"], diagnosis["cause_id"])].append(row)

    graph: dict[str, list[dict]] = defaultdict(list)
    for (family, cause_id), cases in sorted(grouped.items()):
        root_cases = {row["root_case_id"] for row in cases}
        if len(root_cases) < config.MIN_GRAPH_SUPPORT:
            continue
        example = cases[0]
        diagnosis = example["diagnoses"][0]
        graph[family].append(
            {
                "cause_id": cause_id,
                "cause": _generic_cause(diagnosis["cause"]),
                "reason": (
                    f"This failure mechanism appears in {len(root_cases)} independent grounded "
                    "cases with the same symptom family. The visible evidence is "
                    "compatible with it, "
                    "but a discriminating check is still required."
                ),
                "verification": diagnosis.get("verification", []),
                "support": len(root_cases),
                "platforms": sorted({row["platform_family"] for row in cases}),
                "protocols": sorted({row.get("protocol", "") for row in cases}),
            }
        )
    return dict(graph)


def ordered_candidates(row: dict, graph: dict[str, list[dict]], limit: int = 4) -> list[dict]:
    candidates = graph.get(row["fault_family"], [])

    def cost(candidate: dict) -> int:
        text = " ".join(item.get("command", "") for item in candidate["verification"]).lower()
        return 0 if any(word in text for word in ("show", "ping", "get_", "inspect")) else 1

    return sorted(
        candidates,
        key=lambda item: (
            row.get("protocol", "") not in item["protocols"],
            row["platform_family"] not in item["platforms"],
            cost(item),
            -item["support"],
            item["cause_id"],
        ),
    )[:limit]
