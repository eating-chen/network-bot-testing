"""Validation for the source-neutral troubleshooting representation."""

from typing import Any

DIAGNOSIS_TYPES = {
    "confirmed_rca",
    "possible_rca",
    "confirmed_state_mismatch",
    "diagnostic_guidance",
}
EVIDENCE_TYPES = {"cli", "config", "log", "expected_observed"}
GROUNDING_TYPES = {
    "runtime_trace",
    "official_test_fixture",
    "test_definition",
    "config_diff",
}


def validate_record(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "id",
        "source",
        "platform",
        "network_domain",
        "protocol",
        "problem",
        "evidence",
        "diagnoses",
        "grounding",
        "group_id",
    }
    missing = required - row.keys()
    if missing:
        return [f"missing fields: {sorted(missing)}"]

    for key in ("id", "source", "platform", "network_domain", "problem", "group_id"):
        if not isinstance(row[key], str) or not row[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if not isinstance(row["protocol"], str):
        errors.append("protocol must be a string")
    if row["grounding"] not in GROUNDING_TYPES:
        errors.append("grounding is invalid")

    if not isinstance(row["evidence"], list) or not row["evidence"]:
        errors.append("evidence must be a non-empty list")
    else:
        for index, evidence in enumerate(row["evidence"]):
            if not isinstance(evidence, dict):
                errors.append(f"evidence[{index}] must be an object")
                continue
            if evidence.get("type") not in EVIDENCE_TYPES:
                errors.append(f"evidence[{index}] has an invalid type")
            if not str(evidence.get("output", "")).strip():
                errors.append(f"evidence[{index}] has no output")

    if not isinstance(row["diagnoses"], list) or not row["diagnoses"]:
        errors.append("diagnoses must be a non-empty list")
    else:
        for index, diagnosis in enumerate(row["diagnoses"]):
            if not isinstance(diagnosis, dict):
                errors.append(f"diagnoses[{index}] must be an object")
                continue
            diagnosis_type = diagnosis.get("type")
            if diagnosis_type not in DIAGNOSIS_TYPES:
                errors.append(f"diagnoses[{index}] has an invalid type")
            for key in ("cause", "reason"):
                if not str(diagnosis.get(key, "")).strip():
                    errors.append(f"diagnoses[{index}].{key} is empty")
            for key in ("identify", "verification"):
                if not isinstance(diagnosis.get(key), list) or not diagnosis[key]:
                    errors.append(f"diagnoses[{index}].{key} must be a non-empty list")
            resolution = diagnosis.get("resolution")
            if not isinstance(resolution, list):
                errors.append(f"diagnoses[{index}].resolution must be a list")
            elif diagnosis_type != "diagnostic_guidance" and not resolution:
                errors.append(f"diagnoses[{index}].resolution must not be empty")

    return errors
