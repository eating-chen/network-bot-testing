"""JSON helpers shared by the small build scripts."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(*parts: object, length: int = 20) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def compact_text(value: object, limit: int) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"
