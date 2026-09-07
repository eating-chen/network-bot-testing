"""Small IO helpers shared only by the SFT pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def stable_id(*parts: object, length: int = 24) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    # utf-8-sig accepts ordinary UTF-8 and FunctionGemma's leading BOM.
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def upstream_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Read an upstream Parquet shard; canonical outputs are always JSONL."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1_000):
        yield from batch.to_pylist(maps_as_pydicts="strict")
