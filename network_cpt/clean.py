"""Normalize, quality-filter, and exact-deduplicate parsed documents."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any

from network_cpt import config
from network_cpt.io import (
    append_jsonl,
    estimate_tokens,
    open_text,
    read_jsonl,
    setup_logging,
    utc_now,
    write_json,
)
from network_cpt.schema import Document

LOGGER = logging.getLogger(__name__)

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EXCESS_BLANK_LINES = re.compile(r"\n{4,}")
REPEATED_PUNCTUATION = re.compile(r"([^\w\s])\1{15,}")
URL_ONLY = re.compile(r"^(?:\s*https?://\S+\s*)+$", re.IGNORECASE)
TOC_ENTRY = re.compile(r"^.{2,100}\.{5,}\s*\d+\s*$")
BOILERPLATE_LINES = (
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^skip to (?:main )?content$", re.IGNORECASE),
    re.compile(r"^(?:previous|next|edit on github)$", re.IGNORECASE),
    re.compile(r"^©\s*copyright\b", re.IGNORECASE),
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = CONTROL_CHARS.sub("", text)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if TOC_ENTRY.match(stripped):
            continue
        if any(pattern.match(stripped) for pattern in BOILERPLATE_LINES):
            continue
        # Keep indentation (CLI/config/code); only remove trailing whitespace.
        cleaned_lines.append(line.rstrip())
    return EXCESS_BLANK_LINES.sub("\n\n\n", "\n".join(cleaned_lines)).strip()


def quality_rejection_reason(document: Document) -> str | None:
    text = document.text
    if len(text) < config.MIN_TEXT_CHARS:
        return "too_short"
    if not document.license.strip():
        return "missing_license"
    if URL_ONLY.match(text):
        return "url_only"
    if REPEATED_PUNCTUATION.search(text):
        return "repeated_punctuation"
    replacement_ratio = text.count("�") / max(1, len(text))
    if replacement_ratio > config.MAX_REPLACEMENT_CHAR_RATIO:
        return "too_many_replacement_chars"
    alnum_ratio = sum(character.isalnum() for character in text) / max(1, len(text))
    if alnum_ratio < config.MIN_ALNUM_RATIO:
        return "low_alnum_ratio"
    return None


def exact_fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_clean(
    input_path: Path | None = None,
    destination: Path | None = None,
) -> dict[str, Any]:
    input_path = input_path or config.INTERIM_DIR / "01_parsed" / "documents.jsonl.gz"
    destination = destination or config.INTERIM_DIR / "02_cleaned" / "documents.jsonl.gz"
    rejected_path = destination.with_name("rejected.jsonl.gz")
    seen_hashes: dict[str, tuple[str, str]] = {}
    accepted: Counter[str] = Counter()
    accepted_documents: Counter[str] = Counter()
    input_documents: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    duplicate_documents: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()

    with open_text(destination, "w") as output, open_text(rejected_path, "w") as rejects:
        rows = read_jsonl(input_path)
        grouped_rows = groupby(rows, key=lambda row: (str(row["source"]), str(row["document"])))
        for group, document_rows in grouped_rows:
            source, document_id = group
            input_documents[source] += 1
            clean_records: list[Document] = []

            for row in document_rows:
                document = Document.from_dict(row)
                document.text = normalize_text(document.text)
                reason = quality_rejection_reason(document)
                if reason:
                    rejected[reason] += 1
                    append_jsonl(
                        rejects,
                        {
                            "id": document.id,
                            "source": source,
                            "document": document_id,
                            "reason": reason,
                        },
                    )
                    continue
                clean_records.append(document)

            if not clean_records:
                continue

            # 同一個 document 可能包含多個 semantic sections。先合併整份文件的文字再
            # fingerprint，避免其中一個 section 落在 train、重複文件落在 test。
            document_text = "\n\n".join(record.text for record in clean_records)
            fingerprint = exact_fingerprint(document_text)
            duplicate_of = seen_hashes.get(fingerprint)
            if duplicate_of is not None:
                duplicate_documents[source] += 1
                for document in clean_records:
                    rejected["exact_duplicate_document"] += 1
                    append_jsonl(
                        rejects,
                        {
                            "id": document.id,
                            "source": source,
                            "document": document_id,
                            "reason": "exact_duplicate_document",
                            "duplicate_of": {
                                "source": duplicate_of[0],
                                "document": duplicate_of[1],
                            },
                        },
                    )
                continue

            seen_hashes[fingerprint] = group
            accepted_documents[source] += 1
            for document in clean_records:
                document.estimated_tokens = estimate_tokens(document.text)
                document.metadata["document_exact_sha256"] = fingerprint
                append_jsonl(output, document.to_dict())
                accepted[source] += 1
                tokens[source] += document.estimated_tokens
                license_counts[document.license] += 1

    manifest = {
        "stage": "clean",
        "created_at": utc_now(),
        "input": str(input_path),
        "output": str(destination),
        "rejected_output": str(rejected_path),
        "input_documents_by_source": dict(input_documents),
        "accepted_documents": sum(accepted_documents.values()),
        "accepted_documents_by_source": dict(accepted_documents),
        "accepted_records": sum(accepted.values()),
        "accepted_by_source": dict(accepted),
        "estimated_tokens_by_source": dict(tokens),
        "rejected_by_reason": dict(rejected),
        "duplicate_documents_by_source": dict(duplicate_documents),
        "license_counts": dict(license_counts),
        "dedup": "document-level normalized exact SHA-256 before split",
        "near_dedup": "not enabled in v1",
    }
    write_json(destination.with_name("manifest.json"), manifest)
    LOGGER.info(
        "clean 完成：accepted=%d rejected=%d",
        manifest["accepted_records"],
        sum(rejected.values()),
    )
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_clean()
