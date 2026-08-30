import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from network_cpt import config
from network_cpt.clean import run_clean
from network_cpt.io import write_json, write_jsonl
from network_cpt.parse import run_parse
from network_cpt.schema import Document
from network_cpt.split import run_split


def document(identifier: str, group: str, text: str) -> Document:
    return Document(
        id=identifier,
        source="fixture",
        document=group,
        title="Network protocol",
        license="CC-BY-4.0",
        language="en",
        text=text,
        snapshot="fixture-v1",
        estimated_tokens=len(text) // 4,
    )


def test_clean_and_split_pipeline_has_no_document_leakage(tmp_path: Path) -> None:
    base_a = "BGP chooses paths using attributes and routing policy. " * 8
    base_b = "OSPF floods link-state advertisements through an area. " * 8
    base_c = "DHCPv6 assigns addressing information to network clients. " * 8
    duplicate_a = document("duplicate-a", "other-document", base_a.lower())
    duplicate_a.source = "second-source"
    duplicate_b = document("duplicate-b", "other-document", base_b.lower())
    duplicate_b.source = "second-source"
    rows = [
        document("a-1", "same-document", base_a).to_dict(),
        document("a-2", "same-document", base_b).to_dict(),
        duplicate_a.to_dict(),
        duplicate_b.to_dict(),
        document("c-1", "third-document", base_c).to_dict(),
        document("short", "bad-document", "too short").to_dict(),
    ]
    parsed = tmp_path / "parsed.jsonl.gz"
    cleaned = tmp_path / "cleaned.jsonl.gz"
    output_dir = tmp_path / "processed"
    write_jsonl(parsed, rows)

    clean_manifest = run_clean(parsed, cleaned)
    split_manifest = run_split(cleaned, output_dir)

    assert clean_manifest["accepted_records"] == 3
    assert clean_manifest["accepted_documents"] == 2
    assert clean_manifest["rejected_by_reason"] == {
        "exact_duplicate_document": 2,
        "too_short": 1,
    }
    assert sum(split_manifest["records"].values()) == 3

    observed: dict[tuple[str, str], set[str]] = {}
    for split in ("train", "validation", "test"):
        table = pq.read_table(output_dir / f"{split}.parquet")
        for row in table.to_pylist():
            observed.setdefault((row["source"], row["document"]), set()).add(split)
            assert isinstance(json.loads(row["metadata"]), dict)
    assert all(len(splits) == 1 for splits in observed.values())
    assert (output_dir / "_SUCCESS").exists()


def test_split_allocates_each_source_independently(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned.jsonl.gz"
    output_dir = tmp_path / "processed"
    rows = []
    for source in ("frrouting", "sonic"):
        for index in range(40):
            item = document(
                f"{source}-{index}",
                f"document-{index}",
                "Network protocol documentation. " * 8,
            )
            item.source = source
            rows.append(item.to_dict())
    write_jsonl(cleaned, rows)

    manifest = run_split(cleaned, output_dir)

    expected = {"train": 36, "validation": 2, "test": 2}
    assert manifest["documents_by_source_and_split"] == {
        "frrouting": expected,
        "sonic": expected,
    }
    for split, expected_count in expected.items():
        table = pq.read_table(output_dir / f"{split}.parquet")
        observed = {(row["source"], row["document"]) for row in table.to_pylist()}
        assert sum(source == "frrouting" for source, _ in observed) == expected_count
        assert sum(source == "sonic" for source, _ in observed) == expected_count


def test_local_raw_data_runs_from_parse_to_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DIR", data_dir / "raw")
    monkeypatch.setattr(config, "INTERIM_DIR", data_dir / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", data_dir / "processed")
    monkeypatch.setattr(config, "ENABLED_SOURCES", ("tele_data",))

    hf_root = config.RAW_DIR / "huggingface"
    write_json(hf_root / "tele_data.manifest.json", {"revision": "fixture-revision"})
    write_jsonl(
        hf_root / "tele_data" / "standard" / "standard.jsonl",
        [
            {
                "ID": "bgp-1",
                "Category": "standard",
                "Content": "BGP route selection and routing policy behavior. " * 8,
                "Metadata": {"title": "BGP"},
            },
            {
                "ID": "ospf-1",
                "Category": "standard",
                "Content": "OSPF link-state flooding and area behavior. " * 8,
                "Metadata": {"title": "OSPF"},
            },
        ],
    )

    parse_manifest = run_parse()
    clean_manifest = run_clean()
    split_manifest = run_split()

    assert parse_manifest["records"] == 2
    assert clean_manifest["accepted_records"] == 2
    assert sum(split_manifest["records"].values()) == 2
    assert (config.PROCESSED_DIR / "network_cpt_v1" / "_SUCCESS").exists()
