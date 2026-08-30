"""Deterministic document-level train/validation/test split."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from network_cpt import config
from network_cpt.io import read_jsonl, setup_logging, utc_now, write_json
from network_cpt.schema import Document

LOGGER = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("document", pa.string()),
        ("title", pa.string()),
        ("license", pa.string()),
        ("language", pa.string()),
        ("text", pa.large_string()),
        ("snapshot", pa.string()),
        ("section", pa.string()),
        ("estimated_tokens", pa.int64()),
        ("metadata", pa.large_string()),
    ]
)


SPLIT_NAMES = ("train", "validation", "test")


def document_sort_key(source: str, document: str, seed: int) -> bytes:
    """Return a stable random-looking key used to order documents within a source."""
    return hashlib.sha256(f"{seed}\x1f{source}\x1f{document}".encode()).digest()


def allocate_split_counts(total: int) -> dict[str, int]:
    """Allocate an integer 90/5/5 document count using largest remainders."""
    ratios = (config.TRAIN_RATIO, config.VALIDATION_RATIO, config.TEST_RATIO)
    quotas = [total * ratio for ratio in ratios]
    counts = [math.floor(quota) for quota in quotas]
    remaining = total - sum(counts)
    priority = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in priority[:remaining]:
        counts[index] += 1
    return dict(zip(SPLIT_NAMES, counts, strict=True))


def build_source_stratified_assignments(
    input_path: Path,
) -> tuple[dict[tuple[str, str], str], dict[str, dict[str, int]]]:
    """First pass: collect document IDs and assign 90/5/5 independently per source.

    只保存 validation/test 的 mapping；沒有出現在 mapping 的 document 預設是 train，
    以降低大型 corpus 所需的常駐記憶體。
    """
    documents_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(input_path):
        documents_by_source[str(row["source"])].add(str(row["document"]))

    evaluation_assignments: dict[tuple[str, str], str] = {}
    document_counts: dict[str, dict[str, int]] = {}
    for source, documents in documents_by_source.items():
        ordered = sorted(
            documents,
            key=lambda document: document_sort_key(source, document, config.SPLIT_SEED),
        )
        counts = allocate_split_counts(len(ordered))
        document_counts[source] = counts

        train_end = counts["train"]
        validation_end = train_end + counts["validation"]
        for document in ordered[train_end:validation_end]:
            evaluation_assignments[(source, document)] = "validation"
        for document in ordered[validation_end:]:
            evaluation_assignments[(source, document)] = "test"

    return evaluation_assignments, document_counts


class SplitWriter:
    def __init__(self, output_dir: Path, batch_size: int = 1_000) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.buffers: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        self.writers: dict[str, pq.ParquetWriter | None] = {
            "train": None,
            "validation": None,
            "test": None,
        }

    def add(self, split: str, document: Document) -> None:
        self.buffers[split].append(document.to_dict(metadata_as_json=True))
        if len(self.buffers[split]) >= self.batch_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        if self.writers[split] is None:
            self.writers[split] = pq.ParquetWriter(
                self.output_dir / f"{split}.parquet",
                PARQUET_SCHEMA,
                compression="zstd",
            )
        self.writers[split].write_table(table)
        rows.clear()

    def close(self) -> None:
        for split in self.buffers:
            self.flush(split)
            writer = self.writers[split]
            if writer is not None:
                writer.close()
            else:
                # Always emit all three files, even for a tiny smoke-test corpus.
                pq.write_table(
                    pa.Table.from_pylist([], schema=PARQUET_SCHEMA),
                    self.output_dir / f"{split}.parquet",
                    compression="zstd",
                )

    def __enter__(self) -> SplitWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def run_split(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    input_path = input_path or config.INTERIM_DIR / "02_cleaned" / "documents.jsonl.gz"
    output_dir = output_dir or config.PROCESSED_DIR / "network_cpt_v1"
    record_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    source_split_counts: Counter[tuple[str, str]] = Counter()
    evaluation_assignments, document_counts = build_source_stratified_assignments(input_path)

    with SplitWriter(output_dir) as writer:
        for row in read_jsonl(input_path):
            document = Document.from_dict(row)
            group = (document.source, document.document)
            split = evaluation_assignments.get(group, "train")
            writer.add(split, document)
            record_counts[split] += 1
            token_counts[split] += document.estimated_tokens
            source_split_counts[(document.source, split)] += 1

    by_source: dict[str, dict[str, int]] = {}
    for (source, split), count in source_split_counts.items():
        by_source.setdefault(source, {})[split] = count
    manifest = {
        "stage": "document_level_split",
        "created_at": utc_now(),
        "input": str(input_path),
        "seed": config.SPLIT_SEED,
        "ratios": {
            "train": config.TRAIN_RATIO,
            "validation": config.VALIDATION_RATIO,
            "test": config.TEST_RATIO,
        },
        "allocation": "deterministic source-stratified document counts using largest remainders",
        "group_key": ["source", "document"],
        "unique_documents": sum(sum(counts.values()) for counts in document_counts.values()),
        "documents_by_source_and_split": document_counts,
        "records": dict(record_counts),
        "estimated_tokens": dict(token_counts),
        "records_by_source_and_split": by_source,
        "files": {
            split: str((output_dir / f"{split}.parquet").relative_to(config.PROJECT_ROOT))
            if (output_dir / f"{split}.parquet").is_relative_to(config.PROJECT_ROOT)
            else str(output_dir / f"{split}.parquet")
            for split in ("train", "validation", "test")
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text(json.dumps(manifest["records"], sort_keys=True) + "\n")
    LOGGER.info("split 完成：%s", dict(record_counts))
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_split()
