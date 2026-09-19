"""Convert RCAEval DELAY/LOSS/SOCKET cases from the compact Parquet release."""

import math
import statistics
from pathlib import Path

import pyarrow.parquet as parquet

from data_sft_rca import config


def _median(values: list[object]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def _changes(path: Path, inject_time: int) -> list[tuple[str, float, float, float]]:
    table = parquet.read_table(path)
    times = table["time"].to_pylist()
    before_positions = [index for index, value in enumerate(times) if value < inject_time]
    after_positions = [index for index, value in enumerate(times) if value >= inject_time]
    result = []
    for name in table.column_names:
        if name == "time":
            continue
        values = table[name].to_pylist()
        before = _median([values[index] for index in before_positions])
        after = _median([values[index] for index in after_positions])
        if before is None or after is None:
            continue
        scale = max(abs(before), abs(after), 1e-9)
        score = abs(after - before) / scale
        result.append((name, before, after, score))
    return sorted(result, key=lambda item: (-item[3], item[0]))


def _service_match(column: str, service: str) -> bool:
    column_key = column.lower().replace("-", "").replace("_", "")
    service_key = service.lower().replace("-", "").replace("_", "")
    short_key = service_key.removesuffix("service")
    return service_key in column_key or (len(short_key) > 3 and short_key in column_key)


def _summary(changes: list[tuple[str, float, float, float]], limit: int = 4) -> str:
    return "\n".join(
        f"{name}: median before injection={before:.6g}; "
        f"after injection={after:.6g}; normalized change={score:.3f}"
        for name, before, after, score in changes[:limit]
    )


def parse_case(index_row: dict) -> dict | None:
    case = str(index_row["case"])
    fault = str(index_row["fault"]).lower()
    if fault not in config.RCAEVAL_FAULTS:
        return None
    metrics_path = config.RCAEVAL_ROOT / case / "metrics.parquet"
    if not metrics_path.exists():
        return None
    changes = _changes(metrics_path, int(index_row["inject_time"]))
    service = str(index_row["root_cause_service"])
    decisive_changes = [item for item in changes if _service_match(item[0], service)]
    symptom_changes = [item for item in changes if not _service_match(item[0], service)]
    if not decisive_changes or not symptom_changes:
        return None
    evidence = [
        {
            "id": "ev1",
            "type": "metrics",
            "device": str(index_row["system_name"]),
            "command": "compare system telemetry before and after fault injection",
            "output": _summary(symptom_changes, 3),
            "role": "symptom",
        },
        {
            "id": "ev2",
            "type": "metrics",
            "device": service,
            "command": f"compare {service} telemetry before and after fault injection",
            "output": _summary(decisive_changes, 4),
            "role": "decisive",
        },
    ]
    label = {
        "delay": "Network delay",
        "loss": "Network packet loss",
        "socket": "Socket connectivity failure",
    }[fault]
    cause_id = f"service_network_{fault}"
    group_id = f"rcaeval_{index_row['system']}_{service}_{fault}"
    return {
        "id": f"rcaeval_{case}",
        "root_case_id": f"rcaeval_{case}",
        "group_id": group_id,
        "source": "rcaeval",
        "platform_family": "microservices",
        "network_domain": "service_networking",
        "protocol": "socket" if fault == "socket" else "ip",
        "task_type": "confirmed_rca",
        "fault_family": "service_network_performance_or_connectivity",
        "problem": (
            f"{index_row['system_name']} shows a service-level network anomaly "
            "after the recorded injection time."
        ),
        "evidence": evidence,
        "diagnoses": [
            {
                "status": "confirmed",
                "cause_id": cause_id,
                "cause": f"{label} affecting {service}",
                "evidence_ids": ["ev2"],
                "reason": (
                    f"RCAEval labels {service} as the root-cause service and {fault} "
                    "as the injected fault; its telemetry changes after injection."
                ),
                "resolution": [
                    f"Remove the injected {fault} condition from {service} and "
                    "restore normal service communication."
                ],
                "verification": [
                    {
                        "command": f"Compare {service} and end-to-end telemetry after remediation.",
                        "expected_result": (
                            "The affected indicators and request behavior return to "
                            "their pre-injection range."
                        ),
                    }
                ],
            }
        ],
        "provenance": {
            "label_source": "cases.parquet",
            "evidence_source": "metrics.parquet",
            "llm_generated": False,
        },
    }


def build_rcaeval() -> list[dict]:
    index_path = config.RCAEVAL_ROOT / "cases.parquet"
    if not index_path.exists():
        return []
    rows = parquet.read_table(index_path).to_pylist()
    converted = [parse_case(row) for row in rows]
    return [row for row in converted if row]


if __name__ == "__main__":
    built = build_rcaeval()
    print({"rows": len(built), "groups": len({row["group_id"] for row in built})})
