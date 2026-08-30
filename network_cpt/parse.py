"""Parse raw snapshots into the unified document schema."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from network_cpt import config
from network_cpt.io import (
    append_jsonl,
    estimate_tokens,
    open_text,
    read_jsonl,
    setup_logging,
    stable_id,
    utc_now,
    write_json,
)
from network_cpt.schema import Document, validate_document

LOGGER = logging.getLogger(__name__)

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RST_UNDERLINE = re.compile(r"^([=\-~^\"'`:+*#<>_])\1{2,}\s*$")
KEA_EXCLUDED_TITLES = re.compile(
    r"\b(install(?:ation|ing)?|build (?:requirements|dependencies)|release notes?|copyright)\b",
    re.IGNORECASE,
)


def split_markdown_sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    """Split on headings while keeping heading lines in section text."""
    sections: list[tuple[str, str]] = []
    current_title = fallback_title
    current_lines: list[str] = []
    for line in text.splitlines():
        match = MARKDOWN_HEADING.match(line)
        if match and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))
            current_lines = []
        if match:
            current_title = match.group(2).strip(" #") or fallback_title
        current_lines.append(line)
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_title, body))
    return sections


def split_markdown_chapters(text: str, fallback_title: str) -> list[tuple[str, str]]:
    """Split a manual at its repeated shallowest heading level.

    Docling 通常會輸出一個書名 heading，再輸出多個同層級章節。若最淺層 heading 只
    出現一次，就使用下一個至少出現兩次的層級，讓子標題留在所屬章節內。
    """
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    heading_counts: Counter[int] = Counter()
    for index, line in enumerate(lines):
        match = MARKDOWN_HEADING.match(line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip(" #") or fallback_title
        headings.append((index, level, title))
        heading_counts[level] += 1

    if not headings:
        body = text.strip()
        return [(fallback_title, body)] if body else []

    shallowest = min(heading_counts)
    repeated_levels = sorted(
        level for level, count in heading_counts.items() if level > shallowest and count >= 2
    )
    chapter_level = (
        repeated_levels[0] if heading_counts[shallowest] == 1 and repeated_levels else shallowest
    )
    boundaries = [heading for heading in headings if heading[1] == chapter_level]

    chapters: list[tuple[str, str]] = []
    for position, (heading_index, _, title) in enumerate(boundaries):
        # 把封面/書名前言併入第一章，不另外產生一個很短的 document。
        start = 0 if position == 0 else heading_index
        end = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if body:
            chapters.append((title, body))
    return chapters


def split_rst_sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    """Split reStructuredText on underlined headings and preserve directives/code."""
    lines = text.splitlines()
    boundaries = [0]
    titles: dict[int, str] = {0: fallback_title}
    for index in range(1, len(lines)):
        if RST_UNDERLINE.match(lines[index]) and lines[index - 1].strip():
            start = index - 1
            if start not in boundaries:
                boundaries.append(start)
            titles[start] = lines[start].strip()
    boundaries = sorted(set(boundaries))
    sections: list[tuple[str, str]] = []
    for position, start in enumerate(boundaries):
        end = boundaries[position + 1] if position + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if body:
            sections.append((titles.get(start, fallback_title), body))
    return sections


def _manifest_revision(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return str(manifest.get("revision") or manifest.get("commit") or manifest.get("sha256"))


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def parse_tele_data() -> Iterator[Document]:
    root = config.RAW_DIR / "huggingface" / "tele_data"
    revision = _manifest_revision(root.parent / "tele_data.manifest.json")
    files = sorted(root.glob("*/*.jsonl"))
    if not files:
        raise RuntimeError(f"{root} 找不到 Tele-Data JSONL")

    for path in files:
        subset = path.parent.name
        LOGGER.info("解析 Tele-Data/%s", subset)
        for index, row in enumerate(read_jsonl(path)):
            source_id = str(row.get("ID") or row.get("id") or f"{subset}-{index}")
            category = str(row.get("Category") or row.get("category") or subset)
            text = str(row.get("Content") or row.get("content") or "")
            metadata = _metadata_dict(row.get("Metadata") or row.get("metadata"))
            title = str(metadata.get("title") or metadata.get("Title") or source_id)
            document = f"{category}/{source_id}"
            yield Document(
                id=f"tele-{stable_id(document, text)}",
                source="tele_data",
                document=document,
                title=title,
                section=category,
                license="dataset-card:MIT; verify embedded source license",
                language="en",
                text=text,
                snapshot=revision,
                estimated_tokens=estimate_tokens(text),
                metadata={"category": category, "upstream": metadata},
            )


def _parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    """逐 batch 讀本地 Parquet，避免把整個 shard 放進記憶體。"""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1_000):
        yield from batch.to_pylist()


def parse_gsma_tcc() -> Iterator[Document]:
    root = config.RAW_DIR / "huggingface" / "gsma_tcc"
    revision = _manifest_revision(root.parent / "gsma_tcc.manifest.json")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"{root} 找不到 GSMA Parquet")

    row_number = 0
    for path in files:
        LOGGER.info("解析 GSMA shard: %s", path.relative_to(root))
        for row in _parquet_rows(path):
            identifier = str(row.get("identifier") or f"row-{row_number}")
            text = str(row.get("text") or "")
            collection = str(row.get("collection") or "unknown")
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"text", "identifier", "title", "license", "language", "token_count"}
            }
            yield Document(
                id=f"gsma-{stable_id(identifier, text)}",
                source="gsma_tcc",
                document=f"{collection}/{identifier}",
                title=str(row.get("title") or identifier),
                section=collection,
                license=str(row.get("license") or "UNKNOWN_REVIEW_REQUIRED"),
                language=str(row.get("language") or "en"),
                text=text,
                snapshot=revision,
                estimated_tokens=int(row.get("token_count") or estimate_tokens(text)),
                metadata=metadata,
            )
            row_number += 1


def _license_files(repo: Path) -> list[str]:
    patterns = ("LICENSE*", "COPYING*", "NOTICE*")
    paths = {
        str(path.relative_to(repo))
        for pattern in patterns
        for path in repo.glob(pattern)
        if path.is_file()
    }
    return sorted(paths)


def parse_repository(
    source: str,
    patterns: tuple[str, ...],
    section_format: str,
) -> Iterator[Document]:
    repo = config.RAW_DIR / "repos" / source
    manifest_path = repo.parent / f"{source}.manifest.json"
    revision = _manifest_revision(manifest_path)
    licenses = _license_files(repo)
    files = sorted({path for pattern in patterns for path in repo.glob(pattern) if path.is_file()})
    LOGGER.info("[%s] 找到 %d 個文件檔", source, len(files))
    splitter = split_markdown_sections if section_format == "markdown" else split_rst_sections

    for path in files:
        relative = str(path.relative_to(repo))
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        for section_index, (title, text) in enumerate(splitter(raw_text, path.stem)):
            document = relative
            item = Document(
                id=f"{source}-{stable_id(document, section_index, text)}",
                source=source,
                document=document,
                title=title,
                section=f"{relative}#{section_index}",
                license="SEE_REPOSITORY_LICENSE; review document notices",
                language="en",
                text=text,
                snapshot=revision,
                estimated_tokens=estimate_tokens(text),
                metadata={"path": relative, "repository_license_files": licenses},
            )
            yield item


def _docling_markdown(pdf_path: Path) -> tuple[Path, str]:
    output = config.INTERIM_DIR / "00_docling" / f"{pdf_path.stem}.md"
    if output.exists():
        LOGGER.info("沿用既有 Docling Markdown: %s", output)
        return output, output.read_text(encoding="utf-8")
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as error:
        raise RuntimeError("解析 Kea PDF 需要 Docling；請先執行 `uv sync --extra pdf`") from error

    LOGGER.info("Docling 轉換 PDF（首次執行可能下載模型）: %s", pdf_path)
    result = DocumentConverter().convert(pdf_path)
    markdown = result.document.export_to_markdown()
    # Docling 曾出現 NUL bytes；明確移除，避免下游 C-string/Parquet 工具截斷。
    nul_count = markdown.count("\x00")
    if nul_count:
        LOGGER.warning("Docling 輸出含 %d 個 NUL bytes，已移除", nul_count)
        markdown = markdown.replace("\x00", "")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return output, markdown


def parse_kea() -> Iterator[Document]:
    pdf = config.RAW_DIR / "documents" / "kea" / f"{config.KEA_VERSION}.pdf"
    revision = _manifest_revision(pdf.with_name("manifest.json"))
    markdown_path, markdown = _docling_markdown(pdf)
    for chapter_index, (title, text) in enumerate(
        split_markdown_chapters(markdown, "Kea Administrator Reference Manual")
    ):
        if KEA_EXCLUDED_TITLES.search(title):
            continue
        document = f"Kea-ARM-{config.KEA_VERSION}/{chapter_index:04d}-{stable_id(title)}"
        yield Document(
            id=f"kea-{stable_id(document, text)}",
            source="kea",
            document=document,
            title=title,
            section=f"chapter-{chapter_index}",
            license="SEE_ARM_COPYRIGHT_AND_LICENSE_NOTICES",
            language="en",
            text=text,
            snapshot=revision,
            estimated_tokens=estimate_tokens(text),
            metadata={
                "pdf": str(pdf.relative_to(config.PROJECT_ROOT)),
                "docling_markdown": str(markdown_path.relative_to(config.PROJECT_ROOT)),
                "version": config.KEA_VERSION,
                "document_boundary": "repeated_shallowest_markdown_heading",
            },
        )


def iter_all_documents() -> Iterator[Document]:
    parsers = {
        "tele_data": parse_tele_data,
        "gsma_tcc": parse_gsma_tcc,
        "frrouting": lambda: parse_repository("frrouting", ("doc/user/**/*.rst",), "rst"),
        "sonic": lambda: parse_repository("sonic", ("doc/**/*.md",), "markdown"),
        "openvswitch": lambda: parse_repository(
            "openvswitch",
            (
                "Documentation/topics/**/*.rst",
                "Documentation/howto/**/*.rst",
                "Documentation/tutorials/**/*.rst",
                "Documentation/ref/**/*.rst",
                "Documentation/faq/**/*.rst",
                "Documentation/intro/what-is-ovs.rst",
                "Documentation/intro/why-ovs.rst",
            ),
            "rst",
        ),
        "kea": parse_kea,
    }
    for source in config.ENABLED_SOURCES:
        LOGGER.info("解析來源 [%s]", source)
        yield from parsers[source]()


def run_parse() -> dict[str, Any]:
    config.validate_config()
    destination = config.INTERIM_DIR / "01_parsed" / "documents.jsonl.gz"
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    with open_text(destination, "w") as handle:
        for document in iter_all_documents():
            validate_document(document)
            append_jsonl(handle, document.to_dict())
            counts[document.source] += 1
            tokens[document.source] += document.estimated_tokens

    manifest = {
        "stage": "parse",
        "created_at": utc_now(),
        "input_manifest": str(
            (config.RAW_DIR / "download_manifest.json").relative_to(config.PROJECT_ROOT)
        ),
        "output": str(destination.relative_to(config.PROJECT_ROOT)),
        "records_by_source": dict(counts),
        "estimated_tokens_by_source": dict(tokens),
        "records": sum(counts.values()),
        "estimated_tokens": sum(tokens.values()),
    }
    write_json(destination.with_name("manifest.json"), manifest)
    LOGGER.info("parse 完成：%d records", manifest["records"])
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_parse()
