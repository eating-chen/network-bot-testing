from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from network_cpt import config
from network_cpt.io import write_json, write_jsonl
from network_cpt.parse import (
    parse_gsma_tcc,
    parse_tele_data,
    split_markdown_chapters,
    split_markdown_sections,
    split_rst_sections,
)


def test_markdown_split_keeps_heading_and_code_block() -> None:
    text = """# BGP

Introduction to BGP.

```text
router bgp 65000
```

## Route reflector

Configure a route reflector here.
"""
    sections = split_markdown_sections(text, "fallback")

    assert [title for title, _ in sections] == ["BGP", "Route reflector"]
    assert "router bgp 65000" in sections[0][1]
    assert sections[1][1].startswith("## Route reflector")


def test_markdown_chapters_keep_subheadings_in_the_same_document() -> None:
    text = """# Kea ARM

Book preface.

## DHCPv4

DHCPv4 overview.

### Reservations

Reservation details.

## DHCPv6

DHCPv6 overview.
"""
    chapters = split_markdown_chapters(text, "Kea ARM")

    assert [title for title, _ in chapters] == ["DHCPv4", "DHCPv6"]
    assert "### Reservations" in chapters[0][1]
    assert "## DHCPv6" not in chapters[0][1]


def test_rst_split_keeps_directive_body() -> None:
    text = """BGP
===

Overview.

Example
-------

.. code-block:: shell

   show bgp summary
"""
    sections = split_rst_sections(text, "fallback")

    assert [title for title, _ in sections] == ["BGP", "Example"]
    assert "show bgp summary" in sections[1][1]


def test_huggingface_raw_files_are_parsed_locally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    hf_root = config.RAW_DIR / "huggingface"

    tele_file = hf_root / "tele_data" / "standard" / "standard.jsonl"
    write_jsonl(
        tele_file,
        [
            {
                "ID": "standard-1",
                "Category": "standard",
                "Content": "A complete 3GPP section.",
                "Metadata": {"title": "3GPP section"},
            }
        ],
    )
    write_json(
        hf_root / "tele_data.manifest.json",
        {"revision": "tele-revision"},
    )

    gsma_file = hf_root / "gsma_tcc" / "data" / "train-0000.parquet"
    gsma_file.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "identifier": "RFC1234",
                    "collection": "IETF-RFCs",
                    "license": "IETF Trust",
                    "language": "English",
                    "title": "Fixture RFC",
                    "token_count": 10,
                    "text": "A complete RFC section.",
                }
            ]
        ),
        gsma_file,
    )
    write_json(
        hf_root / "gsma_tcc.manifest.json",
        {"revision": "gsma-revision"},
    )

    tele_documents = list(parse_tele_data())
    gsma_documents = list(parse_gsma_tcc())

    assert tele_documents[0].document == "standard/standard-1"
    assert tele_documents[0].snapshot == "tele-revision"
    assert gsma_documents[0].document == "IETF-RFCs/RFC1234"
    assert gsma_documents[0].snapshot == "gsma-revision"
