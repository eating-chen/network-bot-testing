# `parse.py` 中文導讀

這份文件對應 `network_cpt/parse.py`，目的是方便逐段 trace raw data 如何被轉成統一的
`Document` schema。行號以目前版本為準；程式修改後可能位移，閱讀時應以函式名稱為主。

## 1. 先建立整體心智模型

`parse.py` 只負責「解析與格式統一」：

```text
data/raw
├─ Hugging Face JSONL / Parquet
├─ Git repository Markdown / RST
└─ Kea PDF
        │
        ▼
各來源 parser
        │
        ▼
Document dataclass
        │
        ▼
data/interim/01_parsed/documents.jsonl.gz
```

它不負責 quality cleaning、dedup、train/validation/test split，也不做 tokenizer、固定長度
chunk 或 4096-token packing。這些步驟必須在後續階段進行。

執行指令：

```bash
uv run --extra pdf python -m network_cpt.parse
```

實際呼叫順序：

```text
if __name__ == "__main__"
        ↓
setup_logging()
        ↓
run_parse()
        ↓
iter_all_documents()
        ↓
依 ENABLED_SOURCES 呼叫各來源 parser
        ↓
逐筆 yield Document
        ↓
validate_document()
        ↓
append_jsonl()
        ↓
manifest.json
```

## 2. Import 與全域規則（約第 1～35 行）

```python
from __future__ import annotations
```

延後解析 type annotation，讓 `Iterator[Document]` 等型別標註更單純。

```python
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
```

用途如下：

| 名稱 | 用途 |
|---|---|
| `json` | 讀 manifest 與字串形式的 metadata |
| `logging` | 顯示目前解析來源與進度 |
| `re` | 辨識 Markdown/RST heading |
| `Counter` | 統計 record 與 token 數 |
| `Iterator` | 表示 parser 逐筆產生資料 |
| `Path` | 操作路徑 |
| `Any` | metadata 可能是任意型別 |

```python
import pyarrow.parquet as pq
```

用 PyArrow 分批讀取本地 GSMA Parquet，不會一次把整個 shard 載入 RAM。

```python
from network_cpt import config
```

讀取 `RAW_DIR`、`INTERIM_DIR`、`ENABLED_SOURCES`、`KEA_VERSION` 等 hardcoded 設定。

`network_cpt.io` 提供的 helper：

| Helper | 用途 |
|---|---|
| `append_jsonl` | 向已開啟的 JSONL 寫一筆資料 |
| `estimate_tokens` | 在真正 tokenizer 前粗估 token 數 |
| `open_text` | 自動開普通文字或 gzip 文字檔 |
| `read_jsonl` | 逐行讀 JSONL |
| `setup_logging` | 設定 log 格式 |
| `stable_id` | 根據輸入內容產生穩定 SHA-256 ID |
| `utc_now` | 取得 UTC 時間 |
| `write_json` | 寫 manifest JSON |

```python
LOGGER = logging.getLogger(__name__)
```

建立本 module 專用 logger。

### Heading regex

```python
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
```

辨識：

```markdown
# BGP
## Route Reflector
### Configuration
```

- `match.group(1)` 是 `#`、`##` 等 heading level。
- `match.group(2)` 是標題文字。

```python
RST_UNDERLINE = re.compile(r"^([=\-~^\"'`:+*#<>_])\1{2,}\s*$")
```

辨識 RST underline heading：

```rst
BGP
===

Route Reflector
---------------
```

```python
KEA_EXCLUDED_TITLES = re.compile(..., re.IGNORECASE)
```

排除 Kea 的 installation、build requirements、release notes、copyright 等章節。

## 3. `split_markdown_sections()`（約第 38～56 行）

用途：遇到任何 Markdown heading 就建立新 section，主要供 SONiC 使用。

```python
sections = []
current_title = fallback_title
current_lines = []
```

- `sections`：最後要回傳的 `(title, text)`。
- `current_title`：目前 section 標題。
- `current_lines`：目前累積的文字行。

```python
for line in text.splitlines():
    match = MARKDOWN_HEADING.match(line)
```

逐行檢查是否為 Markdown heading。

```python
if match and current_lines:
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_title, body))
    current_lines = []
```

遇到新 heading 時，先把上一段寫入 `sections`，再清空 buffer。

```python
if match:
    current_title = match.group(2).strip(" #") or fallback_title
current_lines.append(line)
```

更新 title，並將 heading 本身保留在 section text 裡。保留 heading 對 CPT 很重要，因為它
提供內容結構與語意。

迴圈結束後還要把最後一段 flush，否則最後一個 section 不會被寫入。

範例輸出：

```python
[
    ("BGP", "# BGP\n..."),
    ("Route Reflector", "## Route Reflector\n..."),
]
```

## 4. `split_markdown_chapters()`（約第 59～98 行）

用途：替 Kea 找自然章節，不把每個子標題都切成獨立 document。

假設 Docling 輸出：

```markdown
# Kea ARM

## DHCPv4
### Reservations
### Client Classification

## DHCPv6
### Prefix Delegation
```

理想文件邊界是兩份：

```text
DHCPv4 document
├─ Reservations
└─ Client Classification

DHCPv6 document
└─ Prefix Delegation
```

```python
lines = text.splitlines()
headings = []
heading_counts = Counter()
```

`headings` 會保存 `(行號, heading level, title)`；`heading_counts` 統計各層級出現次數。

```python
level = len(match.group(1))
```

例如 `##` 長度是 2，因此代表 H2。

```python
headings.append((index, level, title))
heading_counts[level] += 1
```

假設統計結果是：

```python
{1: 1, 2: 12, 3: 67}
```

代表一個書名、十二個章節、六十七個子標題。

```python
if not headings:
    return [(fallback_title, body)] if body else []
```

完全沒有 heading 時，整份文字視為一個 document。

```python
shallowest = min(heading_counts)
```

數字最小的 heading 是最淺層，例如 H1。

```python
repeated_levels = sorted(
    level
    for level, count in heading_counts.items()
    if level > shallowest and count >= 2
)
```

找出比書名更深、而且至少出現兩次的層級。

```python
chapter_level = (
    repeated_levels[0]
    if heading_counts[shallowest] == 1 and repeated_levels
    else shallowest
)
```

- 如果最淺層只出現一次，它通常是書名，因此使用下一個重複層級。
- 如果最淺層本來就重複出現，它本身就是章節層級。

```python
boundaries = [heading for heading in headings if heading[1] == chapter_level]
```

只留下章節 heading 作為切分邊界。

```python
start = 0 if position == 0 else heading_index
```

第一章從第 0 行開始，讓封面或前言併入第一章，不產生一個只有書名的短 document。

```python
end = boundaries[position + 1][0] ...
```

一章的結束位置就是下一章的 heading；最後一章則到文件結尾。

這是一個 heuristic。第一次跑真實 Kea PDF 後，應人工抽查
`data/interim/00_docling/kea-3.0.4.md` 和 parsed manifest，確認 Docling heading 結構符合預期。

## 5. `split_rst_sections()`（約第 101～119 行）

用途：解析 FRRouting 與 Open vSwitch 的 RST。

```python
boundaries = [0]
titles = {0: fallback_title}
```

預設文件從第 0 行開始，尚未找到 heading 時使用 fallback title。

```python
for index in range(1, len(lines)):
```

從第二行開始，因為要同時檢查目前 underline 與上一行標題。

```python
if RST_UNDERLINE.match(lines[index]) and lines[index - 1].strip():
    start = index - 1
```

如果目前行是 `===` 或 `---`，上一行就是 section 起點與標題。

```python
boundaries = sorted(set(boundaries))
```

移除重複邊界並固定順序。

接下來使用目前 boundary 到下一個 boundary 的範圍切出 section，最後回傳
`list[tuple[title, body]]`。RST directive、CLI example 和 code indentation 都仍保留在文字中。

## 6. `_manifest_revision()`（約第 122～124 行）

```python
manifest = json.loads(path.read_text(encoding="utf-8"))
```

讀取下載階段建立的 manifest。

```python
manifest.get("revision") or manifest.get("commit") or manifest.get("sha256")
```

不同來源使用不同 snapshot identity：

| 來源 | 欄位 |
|---|---|
| Hugging Face | `revision` |
| Git repository | `commit` |
| Kea PDF | `sha256` |

最後統一放進 `Document.snapshot`。

## 7. `_metadata_dict()`（約第 127～136 行）

Tele-Data metadata 可能是 dictionary、JSON string 或普通文字。

```python
if isinstance(value, dict):
    return value
```

已是 dictionary 就直接回傳。

```python
parsed = json.loads(value)
```

JSON string 會嘗試轉成 dictionary。若不是合法 JSON，則保存成：

```python
{"raw": value}
```

這樣不會因 metadata 格式不一致而丟失 provenance。

## 8. `parse_tele_data()`（約第 139～168 行）

```python
root = config.RAW_DIR / "huggingface" / "tele_data"
```

輸入根目錄是 `data/raw/huggingface/tele_data`。

```python
files = sorted(root.glob("*/*.jsonl"))
```

尋找 `arxiv/*.jsonl`、`standard/*.jsonl`、`web/*.jsonl`、`wiki/*.jsonl`。`sorted()` 讓
每次 parse 順序一致。

```python
if not files:
    raise RuntimeError(...)
```

找不到 raw data 時直接停止，不默默產生空 corpus。

```python
for index, row in enumerate(read_jsonl(path)):
```

逐行讀 JSONL，不一次載入整個 dataset。

欄位 fallback：

```python
source_id = row["ID"] / row["id"] / subset-index
category = row["Category"] / row["category"] / subset
text = row["Content"] / row["content"] / ""
title = metadata["title"] / metadata["Title"] / source_id
```

```python
document = f"{category}/{source_id}"
```

這是 Tele-Data 的 document key，例如 `standard/3gpp-38.331`。後續 exact dedup 和 split
都以 `(source, document)` 為文件身份。

```python
yield Document(...)
```

逐筆產生統一 schema。空文字暫時保留，後續 `clean.py` 才統一記錄拒絕原因。

## 9. `_parquet_rows()`（約第 171～175 行）

```python
parquet = pq.ParquetFile(path)
```

開啟本地 Parquet。

```python
for batch in parquet.iter_batches(batch_size=1_000):
    yield from batch.to_pylist()
```

一次只讀 1,000 rows，再逐筆交給 GSMA parser。這是本地 batch IO，不是網路 streaming。

`yield from rows` 等同：

```python
for row in rows:
    yield row
```

## 10. `parse_gsma_tcc()`（約第 178～210 行）

```python
files = sorted(root.rglob("*.parquet"))
```

遞迴尋找 GSMA repository 裡所有 Parquet shards。

```python
identifier = row.get("identifier") or f"row-{row_number}"
collection = row.get("collection") or "unknown"
```

優先沿用上游 document ID；缺少時才建立 fallback。

```python
metadata = {
    key: value
    for key, value in row.items()
    if key not in {...canonical fields...}
}
```

已經放進 canonical schema 的欄位不重複保存，其餘 provenance 留在 metadata。

```python
document = f"{collection}/{identifier}"
```

例如 `IETF-RFCs/RFC4271`。

```python
license=row.get("license") or "UNKNOWN_REVIEW_REQUIRED"
```

缺少 license 時明確標示需要人工審查。

```python
estimated_tokens=int(row.get("token_count") or estimate_tokens(text))
```

優先使用 GSMA 上游 token count，沒有才用 `len(text) / 4` 粗估。

## 11. `_license_files()`（約第 213～221 行）

搜尋 Git repo 根目錄的：

```text
LICENSE*
COPYING*
NOTICE*
```

set comprehension 去除重複，`relative_to(repo)` 把完整路徑轉成 repo 內相對路徑，最後
`sorted()` 固定輸出順序。

這只保存「repository 有哪些 license 檔」，不等於自動完成商用法律審核。

## 12. `parse_repository()`（約第 224～255 行）

這是 FRRouting、SONiC、Open vSwitch 共用 parser。

三個參數：

| 參數 | 範例 |
|---|---|
| `source` | `frrouting` |
| `patterns` | `("doc/user/**/*.rst",)` |
| `section_format` | `rst` 或 `markdown` |

```python
repo = config.RAW_DIR / "repos" / source
revision = _manifest_revision(manifest_path)
licenses = _license_files(repo)
```

建立 repo 路徑，取得 Git commit，並記錄 license files。

```python
files = sorted({
    path
    for pattern in patterns
    for path in repo.glob(pattern)
    if path.is_file()
})
```

- 對每個 pattern 找檔案。
- `set` 避免不同 pattern 找到同一檔。
- `sorted` 固定 parse 順序。

```python
splitter = split_markdown_sections if section_format == "markdown" else split_rst_sections
```

依格式選擇 heading parser。

```python
relative = str(path.relative_to(repo))
```

例如把完整路徑轉成 `doc/user/bgp.rst`。

```python
raw_text = path.read_text(encoding="utf-8", errors="replace")
```

無法解碼的 byte 會轉成 `�`，後續 clean 再依 replacement-character ratio 判斷是否拒絕。

```python
for section_index, (title, text) in enumerate(splitter(...)):
    document = relative
```

一個檔案可以產生多筆 section records，但它們共用同一個 `document=relative`。

```text
source   = frrouting
document = doc/user/bgp.rst
section  = doc/user/bgp.rst#12
```

因此：

- `clean.py` 可以把 sections 合起來做完整 document exact dedup。
- `split.py` 會讓同一檔案的所有 sections 進入同一 split。

`parse_repository()` 會連續 yield 同一文件的所有 sections。這是重要 pipeline invariant，
因為 `clean.py` 使用連續的 `(source, document)` 做 `groupby()`。

## 13. `_docling_markdown()`（約第 258～278 行）

```python
output = config.INTERIM_DIR / "00_docling" / f"{pdf_path.stem}.md"
```

Docling 轉換結果保存到 `data/interim/00_docling/kea-3.0.4.md`。

```python
if output.exists():
    return output, output.read_text(...)
```

已有 Markdown 就直接重用，避免每次重新跑 PDF parsing。

```python
from docling.document_converter import DocumentConverter
```

採 lazy import：只有真的處理 Kea 時才需要 Docling。

```python
result = DocumentConverter().convert(pdf_path)
markdown = result.document.export_to_markdown()
```

將 PDF 轉成 Docling document，再匯出 Markdown。

```python
nul_count = markdown.count("\x00")
markdown = markdown.replace("\x00", "")
```

移除 NUL bytes，避免下游 C-string 或 Parquet 工具截斷文字。

最後把 Markdown 寫到磁碟並回傳 `(output_path, markdown_text)`。

## 14. `parse_kea()`（約第 281～308 行）

```python
pdf = config.RAW_DIR / "documents" / "kea" / f"{config.KEA_VERSION}.pdf"
```

找到固定版本的 Kea ARM PDF。

```python
revision = _manifest_revision(pdf.with_name("manifest.json"))
```

Kea snapshot identity 是 PDF SHA-256。

```python
markdown_path, markdown = _docling_markdown(pdf)
```

取得 Docling Markdown。

```python
for chapter_index, (title, text) in enumerate(split_markdown_chapters(...)):
```

逐一處理自然章節。

```python
if KEA_EXCLUDED_TITLES.search(title):
    continue
```

排除 installation、release notes 等章節。

```python
document = f"Kea-ARM-{config.KEA_VERSION}/{chapter_index:04d}-{stable_id(title)}"
```

每章建立穩定 document ID。`:04d` 會把 `3` 格式化成 `0003`。

現在 Kea 不再是「整本手冊一個 document」，因此不同自然章節可以做 per-source
document-level split；章節內子標題仍不會跨 split。

## 15. `iter_all_documents()`（約第 311～334 行）

`parsers` dictionary 把 source key 對應到 parser：

```python
"tele_data": parse_tele_data
"gsma_tcc": parse_gsma_tcc
"kea": parse_kea
```

Git parser 需要參數，所以使用 `lambda` 包裝：

```python
"frrouting": lambda: parse_repository(
    "frrouting",
    ("doc/user/**/*.rst",),
    "rst",
)
```

實際文件範圍：

| 來源 | Parse pattern |
|---|---|
| FRRouting | `doc/user/**/*.rst` |
| SONiC | `doc/**/*.md` |
| OVS | topics/howto/tutorials/ref/faq，加兩篇 intro |

```python
for source in config.ENABLED_SOURCES:
    yield from parsers[source]()
```

依 `config.py` 的順序逐來源執行。`yield from` 等同：

```python
for document in parsers[source]():
    yield document
```

## 16. `run_parse()`（約第 337～363 行）

```python
config.validate_config()
```

先檢查來源設定是否合法。

```python
destination = config.INTERIM_DIR / "01_parsed" / "documents.jsonl.gz"
```

指定 unified parsed output。

```python
counts = Counter()
tokens = Counter()
```

分來源統計 records 與粗估 tokens。

```python
with open_text(destination, "w") as handle:
```

以 gzip text write mode 開啟。重新執行會覆寫舊 parsed JSONL。

```python
for document in iter_all_documents():
    validate_document(document)
    append_jsonl(handle, document.to_dict())
```

逐筆取得 `Document`、檢查必要 identity/provenance 欄位、轉 dictionary，再寫成一行 JSON。

`validate_document()` 目前要求：

```text
id
source
document
license
```

空文字與短文字不在 parse 階段拒絕，因為要讓 `clean.py` 統一記錄 rejection reason。

```python
counts[document.source] += 1
tokens[document.source] += document.estimated_tokens
```

這裡的 count 是 record/section 數，不一定是 unique document 數。

最後建立 manifest：

```json
{
  "stage": "parse",
  "input_manifest": "data/raw/download_manifest.json",
  "output": "data/interim/01_parsed/documents.jsonl.gz",
  "records_by_source": {},
  "estimated_tokens_by_source": {},
  "records": 0,
  "estimated_tokens": 0
}
```

manifest 寫入 `data/interim/01_parsed/manifest.json`。

## 17. Module 進入點（約第 366～368 行）

```python
if __name__ == "__main__":
    setup_logging()
    run_parse()
```

使用 `python -m network_cpt.parse` 時會執行；如果只是從測試 import 某個 parser，則不會
自動跑完整 pipeline。

## 18. `record`、`section`、`document` 的差異

這是整支程式最重要的概念。

```text
FRRouting doc/user/bgp.rst          一個 document
├─ doc/user/bgp.rst#0               一筆 section record
├─ doc/user/bgp.rst#1               一筆 section record
└─ doc/user/bgp.rst#2               一筆 section record
```

統一 schema 範例：

```json
{
  "id": "frrouting-...",
  "source": "frrouting",
  "document": "doc/user/bgp.rst",
  "title": "Route Reflector",
  "section": "doc/user/bgp.rst#12",
  "text": "...",
  "license": "SEE_REPOSITORY_LICENSE; review document notices",
  "snapshot": "git-commit",
  "estimated_tokens": 1234,
  "metadata": {}
}
```

後續使用 `(source, document)`：

```text
clean.py → 將同一文件的 clean sections 合併後做 exact fingerprint
split.py → 將同一文件的所有 records 寫進同一 split
```

## 19. 建議的 trace 方法

先只啟用一個小來源，例如在 `config.py` 暫時設定：

```python
ENABLED_SOURCES = ("frrouting",)
```

執行：

```bash
uv run python -m network_cpt.parse
```

查看 manifest：

```bash
uv run python -c 'import json; print(json.load(open("data/interim/01_parsed/manifest.json")))'
```

查看第一筆 parsed record：

```bash
gzip -cd data/interim/01_parsed/documents.jsonl.gz | head -n 1
```

查看同一 document 的前幾筆 sections：

```bash
gzip -cd data/interim/01_parsed/documents.jsonl.gz \
  | uv run python -c 'import sys,json,itertools; rows=(json.loads(x) for x in itertools.islice(sys.stdin,10)); [print(r["document"], r["section"]) for r in rows]'
```

確認後記得把 `ENABLED_SOURCES` 改回：

```python
ENABLED_SOURCES = SUPPORTED_SOURCES
```

## 20. 目前需要知道的限制

- Markdown parser 遇到每個 heading 都切 section，沒有分析 heading hierarchy；但同檔案仍
  共用 `document`，所以不會造成 split leakage。
- RST parser 主要辨識 underline-style headings，複雜 Sphinx 結構可能需要後續抽查。
- Kea chapter boundary 是根據 Docling Markdown heading 分布推論，第一次正式轉換後應抽查。
- Tele-Data language 目前固定記為 `en`；內嵌來源 license 仍需額外審查。
- Token 數只是 pre-tokenizer estimate，CPT 前必須使用 Llama-3.1 tokenizer 重算。
- Parse 階段不做 near-dedup；v1 cleaning 也只做 exact document dedup。

掌握這支程式時，可以一直用以下一句話檢查自己的理解：

> 每個來源 parser 都只做兩件事：找到自然文件邊界，然後逐筆 `yield` 統一的 `Document`。
