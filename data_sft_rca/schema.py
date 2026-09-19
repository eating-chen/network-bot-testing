"""Validation for the intentionally compact canonical schema."""

from typing import Any

TASK_TYPES = {
    "confirmed_rca",
    "probable_cause_analysis",
    "next_diagnostic_step",
    "state_diagnosis",
    "premise_correction",
}
EVIDENCE_ROLES = {"symptom", "supporting", "decisive", "irrelevant"}
STATUSES = {"confirmed", "possible", "state_only", "no_fault"}


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "id",
        "root_case_id",
        "group_id",
        "source",
        "platform_family",
        "network_domain",
        "task_type",
        "fault_family",
        "problem",
        "evidence",
        "diagnoses",
        "provenance",
    }
    missing = required - row.keys()
    if missing:
        return [f"missing fields: {sorted(missing)}"]
    for key in required - {"evidence", "diagnoses", "provenance"}:
        if not isinstance(row[key], str) or not row[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if row["task_type"] not in TASK_TYPES:
        errors.append("invalid task_type")
    if not isinstance(row["evidence"], list) or not row["evidence"]:
        errors.append("evidence must be non-empty")
    else:
        ids = set()
        for index, item in enumerate(row["evidence"]):
            if not isinstance(item, dict):
                errors.append(f"evidence[{index}] must be an object")
                continue
            if item.get("role") not in EVIDENCE_ROLES:
                errors.append(f"evidence[{index}] has invalid role")
            if not str(item.get("output", "")).strip():
                errors.append(f"evidence[{index}] has no output")
            evidence_id = item.get("id")
            if not evidence_id or evidence_id in ids:
                errors.append(f"evidence[{index}] has missing or duplicate id")
            ids.add(evidence_id)
    if not isinstance(row["diagnoses"], list) or not row["diagnoses"]:
        errors.append("diagnoses must be non-empty")
    else:
        for index, diagnosis in enumerate(row["diagnoses"]):
            if diagnosis.get("status") not in STATUSES:
                errors.append(f"diagnoses[{index}] has invalid status")
            for key in ("cause_id", "cause", "reason"):
                if not str(diagnosis.get(key, "")).strip():
                    errors.append(f"diagnoses[{index}].{key} is empty")
            cited = set(diagnosis.get("evidence_ids", []))
            if not cited or not cited.issubset(ids):
                errors.append(f"diagnoses[{index}] cites invalid evidence")
    if row["task_type"] == "confirmed_rca" and not any(
        item.get("role") == "decisive" for item in row["evidence"]
    ):
        errors.append("confirmed_rca requires decisive evidence")
    if row.get("provenance", {}).get("llm_generated") is not False:
        errors.append("provenance.llm_generated must be false")
    return errors
