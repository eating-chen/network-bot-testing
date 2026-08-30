"""Unified document schema shared by every stage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    source: str
    document: str
    title: str
    license: str
    language: str
    text: str
    snapshot: str
    section: str = ""
    estimated_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, metadata_as_json: bool = False) -> dict[str, Any]:
        row = asdict(self)
        if metadata_as_json:
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Document:
        values = dict(row)
        metadata = values.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw_metadata": metadata}
        values["metadata"] = metadata
        values["estimated_tokens"] = int(values.get("estimated_tokens") or 0)
        return cls(**{name: values.get(name, "") for name in cls.__dataclass_fields__})


# Download/parse 保留上游的空文字 row，讓 clean 階段統一記錄為 rejected；因此這裡只要求
# provenance/identity 欄位非空，不提前把 bad text 擋在 parse 階段。
REQUIRED_FIELDS = {"id", "source", "document", "license"}


def validate_document(document: Document) -> None:
    missing = [name for name in REQUIRED_FIELDS if not getattr(document, name)]
    if missing:
        raise ValueError(f"Document {document.id or '<no-id>'} 缺少欄位: {missing}")
