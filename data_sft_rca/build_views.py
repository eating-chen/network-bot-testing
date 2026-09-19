"""Create full, masked probable-cause and next-step views from grounded cases."""

from copy import deepcopy

from data_sft_rca import config
from data_sft_rca.io_utils import stable_id
from data_sft_rca.symptom_graph import ordered_candidates


def _candidate_diagnoses(row: dict, graph: dict, evidence_ids: list[str]) -> list[dict]:
    result = []
    for candidate in ordered_candidates(row, graph):
        result.append(
            {
                "status": "possible",
                "cause_id": candidate["cause_id"],
                "cause": candidate["cause"],
                "evidence_ids": evidence_ids,
                "reason": candidate["reason"],
                "resolution": [],
                "verification": candidate["verification"],
            }
        )
    return result


def _partial_view(row: dict, graph: dict, task_type: str) -> dict | None:
    masked = [item for item in row["evidence"] if item["role"] == "decisive"]
    visible = [deepcopy(item) for item in row["evidence"] if item["role"] != "decisive"]
    if not masked or not visible:
        return None
    evidence_ids = [item["id"] for item in visible]
    diagnoses = _candidate_diagnoses(row, graph, evidence_ids)
    if len(diagnoses) < 2:
        return None
    suffix = "probable" if task_type == "probable_cause_analysis" else "next"
    return {
        **{
            key: deepcopy(value)
            for key, value in row.items()
            if key not in {"id", "task_type", "evidence", "diagnoses"}
        },
        "id": f"{row['id']}__{suffix}",
        "task_type": task_type,
        "evidence": visible,
        "diagnoses": diagnoses,
        "evidence_mode": "partial",
        "masked_evidence_ids": [item["id"] for item in masked],
    }


def _focused_view(row: dict) -> dict | None:
    if len(row["evidence"]) < 3:
        return None
    decisive = [deepcopy(item) for item in row["evidence"] if item["role"] == "decisive"]
    context = [deepcopy(item) for item in row["evidence"] if item["role"] != "decisive"]
    if not decisive or not context:
        return None
    selected = context[:1] + decisive
    selected_ids = {item["id"] for item in selected}
    diagnosis = deepcopy(row["diagnoses"][0])
    diagnosis["evidence_ids"] = [item for item in diagnosis["evidence_ids"] if item in selected_ids]
    if not diagnosis["evidence_ids"]:
        diagnosis["evidence_ids"] = [item["id"] for item in decisive]
    return {
        **{
            key: deepcopy(value)
            for key, value in row.items()
            if key not in {"id", "evidence", "diagnoses"}
        },
        "id": f"{row['id']}__focused",
        "evidence": selected,
        "diagnoses": [diagnosis],
        "evidence_mode": "focused",
        "masked_evidence_ids": [
            item["id"] for item in row["evidence"] if item["id"] not in selected_ids
        ],
    }


def _limited(rows: list[dict], limit: int, label: str) -> list[dict]:
    rows.sort(key=lambda row: stable_id(config.SEED, label, row["id"], length=64))
    return rows[:limit]


def build_views(rows: list[dict], graph: dict) -> list[dict]:
    full, focused, probable_views, next_views = [], [], [], []
    for source_row in rows:
        row = deepcopy(source_row)
        row["evidence_mode"] = "full"
        row["masked_evidence_ids"] = []
        full.append(row)
        if row["task_type"] != "confirmed_rca":
            continue
        focused_view = _focused_view(row)
        probable = _partial_view(row, graph, "probable_cause_analysis")
        next_step = _partial_view(row, graph, "next_diagnostic_step")
        if focused_view:
            focused.append(focused_view)
        if probable:
            probable_views.append(probable)
        if next_step:
            next_views.append(next_step)
    return (
        full
        + _limited(focused, config.FOCUSED_CONFIRMED_LIMIT, "focused")
        + _limited(probable_views, config.PROBABLE_VIEW_LIMIT, "probable")
        + _limited(next_views, config.NEXT_STEP_VIEW_LIMIT, "next")
    )
