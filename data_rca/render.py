"""Render canonical RCA records as model-neutral Hugging Face messages."""

from typing import Any

from data_rca import config


def _evidence_block(item: dict[str, Any]) -> str:
    device = item.get("device") or "unspecified device"
    command = item.get("command") or item.get("tool") or "observation"
    return f"[{device}]\n$ {command}\n{item['output']}"


def render_user(row: dict[str, Any]) -> str:
    evidence = "\n\n".join(_evidence_block(item) for item in row["evidence"])
    return f"Problem / Symptom:\n{row['problem']}\n\nEvidence:\n{evidence}"


def _steps(items: list[Any]) -> str:
    lines = []
    for item in items:
        if isinstance(item, str):
            lines.append(f"- {item}")
            continue
        command = item.get("command", "Inspect the relevant state")
        expected = item.get("expected_evidence") or item.get("expected_result", "")
        lines.append(f"- Run/inspect: {command}\n  Expected: {expected}")
    return "\n".join(lines)


def render_assistant(row: dict[str, Any]) -> str:
    sections = ["Troubleshooting Analysis:"]
    for number, diagnosis in enumerate(row["diagnoses"], 1):
        labels = {
            "confirmed_rca": "Confirmed Root Cause",
            "possible_rca": "Possible Cause",
            "confirmed_state_mismatch": "Confirmed State Mismatch",
            "diagnostic_guidance": "Observed Issue",
        }
        label = labels[diagnosis["type"]]
        if len(row["diagnoses"]) > 1:
            label += f" {number}"
        sections.extend(
            [
                f"{label}:\n{diagnosis['cause']}",
                f"Why:\n{diagnosis['reason']}",
                "Investigation / How to Identify:\n" + _steps(diagnosis["identify"]),
            ]
        )
        if diagnosis["resolution"]:
            sections.append("Resolution:\n" + _steps(diagnosis["resolution"]))
        sections.append("Verification / Expected Result:\n" + _steps(diagnosis["verification"]))
    return "\n\n".join(sections)


def to_sft(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source": row["source"],
        "platform": row["platform"],
        "network_domain": row["network_domain"],
        "protocol": row["protocol"],
        "diagnosis_type": row["diagnoses"][0]["type"],
        "messages": [
            {"role": "system", "content": config.SYSTEM_MESSAGE},
            {"role": "user", "content": render_user(row)},
            {"role": "assistant", "content": render_assistant(row)},
        ],
    }
