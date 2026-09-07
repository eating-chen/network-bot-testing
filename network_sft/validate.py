"""Validate canonical conversations and write all failures to one JSONL report."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from network_sft import config
from network_sft.io import jsonl_rows, setup_logging, write_json, write_jsonl
from network_sft.schema import validate_row

LOGGER = logging.getLogger(__name__)


def run_validate(
    paths: list[Path] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    paths = paths or [
        config.intermediate_path("general_sft"),
        config.intermediate_path("functiongemma_network"),
        config.intermediate_path("nika"),
    ]
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: dict[str, str] = {}

    for path in paths:
        if not path.exists():
            errors.append({"file": str(path), "id": None, "errors": ["file does not exist"]})
            continue
        for row_number, row in enumerate(jsonl_rows(path), start=1):
            counts[str(row.get("source") or "unknown")] += 1
            row_errors = validate_row(row)
            identifier = str(row.get("id") or "")
            if identifier in seen_ids:
                row_errors.append(f"duplicate id; first seen in {seen_ids[identifier]}")
            else:
                seen_ids[identifier] = str(path)
            if row_errors:
                errors.append(
                    {
                        "file": str(path),
                        "row_number": row_number,
                        "id": identifier or None,
                        "errors": row_errors,
                    }
                )

    write_jsonl(config.REPORTS_DIR / "validation_errors.jsonl", errors)
    report = {
        "files": [str(path) for path in paths],
        "rows": sum(counts.values()),
        "rows_by_source": dict(counts),
        "invalid_rows": len(errors),
    }
    write_json(config.REPORTS_DIR / "validation_summary.json", report)
    LOGGER.info("validation 完成：rows=%d invalid=%d", report["rows"], len(errors))
    if errors and strict:
        report_path = config.REPORTS_DIR / "validation_errors.jsonl"
        raise RuntimeError(
            f"Found {len(errors)} invalid rows; see {report_path}"
        )
    return report


if __name__ == "__main__":
    setup_logging()
    run_validate()
