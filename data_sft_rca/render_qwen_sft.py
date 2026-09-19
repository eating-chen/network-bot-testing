"""Render canonical rows as plain system/user/assistant messages for Qwen SFT."""

from data_sft_rca.config import SYSTEM_MESSAGE


def _evidence_text(row: dict) -> str:
    blocks = []
    for item in row["evidence"]:
        where = f" on {item['device']}" if item.get("device") else ""
        blocks.append(f"$ {item['command']}{where}\n{item['output']}")
    return "\n\n".join(blocks)


def render_user(row: dict) -> str:
    request = {
        "confirmed_rca": (
            "Based on the evidence, diagnose the issue and explain how to resolve and verify it."
        ),
        "probable_cause_analysis": (
            "The evidence is incomplete. Identify the most relevant causes to investigate "
            "and explain how to verify each one."
        ),
        "next_diagnostic_step": (
            "The current evidence does not reveal one root cause. What should be checked "
            "next, and why?"
        ),
        "state_diagnosis": (
            "State what this evidence confirms, what it does not confirm, and the next "
            "useful checks."
        ),
        "premise_correction": (
            "Determine whether the reported fault is supported. Do not assume the ticket "
            "premise is correct."
        ),
    }[row["task_type"]]
    return f"{row['problem']}\n\nDiagnostic information:\n\n{_evidence_text(row)}\n\n{request}"


def _verification(items: list[dict]) -> str:
    if not items:
        return "Collect the missing discriminating evidence."
    lines = []
    for item in items:
        command = item.get("command") or item.get("action") or "Run the relevant check"
        expected = (
            item.get("expected_result")
            or item.get("expected_evidence")
            or "Confirm the expected healthy state."
        )
        lines.append(f"- {command}\n  Expected: {expected}")
    return "\n".join(lines)


def render_assistant(row: dict) -> str:
    task = row["task_type"]
    diagnoses = row["diagnoses"]
    first = diagnoses[0]
    if task == "confirmed_rca":
        resolution = (
            "\n".join(f"- {item}" for item in first.get("resolution", []))
            or "- Apply the source-defined correction."
        )
        return (
            f"Confirmed Root Cause:\n{first['cause']}\n\nWhy:\n{first['reason']}\n\n"
            f"Resolution:\n{resolution}\n\nVerification:\n"
            f"{_verification(first.get('verification', []))}"
        )
    if task == "probable_cause_analysis":
        parts = ["The available evidence is not sufficient to confirm a single root cause."]
        for index, diagnosis in enumerate(diagnoses, 1):
            parts.append(
                f"Possible Cause {index}: {diagnosis['cause']}\n\nWhy:\n{diagnosis['reason']}\n\n"
                f"How to verify:\n{_verification(diagnosis.get('verification', []))}"
            )
        return "\n\n".join(parts)
    if task == "next_diagnostic_step":
        parts = ["The current evidence does not confirm one root cause. Use this diagnostic order:"]
        for index, diagnosis in enumerate(diagnoses, 1):
            parts.append(
                f"{index}. Check for {diagnosis['cause']}\n"
                f"Why: {diagnosis['reason']}\n{_verification(diagnosis.get('verification', []))}"
            )
        return "\n\n".join(parts)
    if task == "premise_correction":
        resolution = "\n".join(f"- {item}" for item in first.get("resolution", []))
        resolution_block = f"\n\nResolution:\n{resolution}" if resolution else ""
        return (
            f"Conclusion:\n{first['cause']}\n\nWhy:\n{first['reason']}\n\n"
            "Recommended Next Step:\n"
            f"{_verification(first.get('verification', []))}{resolution_block}"
        )
    checks = first.get("verification", [])
    return (
        f"Observed Issue:\n{first['cause']}\n\nWhat This Confirms:\n{first['reason']}\n\n"
        "What This Does Not Confirm:\nThis evidence alone does not establish an "
        "underlying root cause.\n\n"
        f"Recommended Checks:\n{_verification(checks)}"
    )


def to_sft(row: dict) -> dict:
    return {
        "id": row["id"],
        "root_case_id": row["root_case_id"],
        "group_id": row["group_id"],
        "source": row["source"],
        "task_type": row["task_type"],
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": render_user(row)},
            {"role": "assistant", "content": render_assistant(row)},
        ],
    }
