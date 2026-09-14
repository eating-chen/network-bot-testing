# Network training data：CPT 與 Agentic SFT 前處理

本 repository 有兩條互相分離的 pipeline：

- `network_cpt/`：document/text 格式的持續預訓練資料，使用既有 `data/raw`、
  `data/interim`、`data/processed/network_cpt_v1`。
- `network_sft/`：六來源、約 22k 筆的 Diagnostic/Agentic Mixed SFT V1；使用
  model-neutral Hugging Face/TRL `messages + tools` JSONL，只讀寫 `data/sft/`。

本頁以下說明 CPT。SFT 的來源、filter、逐支 Python 執行方式與輸出 schema 請看
[Agentic SFT data guide](docs/sft_data_guide.md)；要逐檔理解函式與資料如何流動，請看
[Network SFT 程式導讀](docs/network_sft_code_guide.md)。

## CPT pipeline

這個專案把 Network CPT v1 的原始資料處理到 document-level
`train / validation / test` split。專案採用
[Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/)
的目錄觀念，以 `uv` 管理 Python 與套件；第一版把設定集中寫在
`network_cpt/config.py`，刻意不在每支程式塞滿 `argparse`，方便逐行 trace。

目前範圍到 Parquet split 為止，尚未做 Llama tokenizer、4096-token packing 或 CPT。

程式導讀：[parse.py 中文逐段說明](docs/parse_py_guide.md)

## Pipeline

```text
Tele-Data / GSMA TCC / FRR / SONiC / OVS / Kea PDF
                         │
                         ▼
data/raw          上游原始檔 + revision/commit/SHA-256
                         │
                         ▼
data/interim/01_parsed   unified schema，保留 heading/code/table text
                         │
                         ▼
data/interim/02_cleaned  boilerplate/bad text/document-level exact dedup
                         │
                         ▼
data/processed/network_cpt_v1
                  per-source document-level 90/5/5 Parquet
```

每一階段都會寫 `manifest.json`；被 quality filter 丟棄的資料只記 ID、來源與原因到
`rejected.jsonl.gz`，不複製整段內容。

## 目錄結構

```text
├── data
│   ├── external/       # 第三方手動資料（目前未使用）
│   ├── raw/            # 原始 snapshot，不可手改、不進 Git
│   ├── interim/        # parse 與 clean 的可檢查中間結果
│   └── processed/      # 最終 canonical split
├── docs/
├── models/
├── notebooks/
├── references/
├── reports/figures/
├── network_cpt
│   ├── config.py       # v1 所有 hardcoded 設定
│   ├── download.py     # HF repository、Git repository、Kea PDF 下載
│   ├── parse.py        # Docling/RST/Markdown/HF -> unified schema
│   ├── clean.py        # 格式、quality、exact dedup
│   ├── split.py        # document-level deterministic split
│   └── dataset.py      # 完整 pipeline 入口
├── tests/
└── pyproject.toml
```

## 快速開始

需求：Linux/macOS、Git、`uv`、足夠磁碟空間。Docling 支援 Python 3.11/3.12；本專案
預設 `.python-version` 是 3.12。

```bash
# 安裝基本套件、開發工具與 Docling PDF 支援
uv sync --dev --extra pdf

# 先跑不碰網路的測試
uv run pytest
```

正式執行前，先打開 `network_cpt/config.py` 確認來源。預設六個來源全部啟用：

```python
ENABLED_SOURCES = SUPPORTED_SOURCES
```

pipeline 不設定 token budget、row 上限、語言或 collection 預先篩選。Hugging Face 來源使用
`snapshot_download()` 把 repository 原始檔完整放進 `data/raw`；下載階段不逐 row 轉換。
`estimated_tokens` 到 parse 階段才計算，採 `len(text) / 4` 粗估（GSMA 有 `token_count` 時
優先使用）。真正訓練 token 數應在後續用 Llama-3.1 tokenizer 重算。

逐階段執行：

```bash
uv run python -m network_cpt.download
uv run --extra pdf python -m network_cpt.parse
uv run python -m network_cpt.clean
uv run python -m network_cpt.split
```

或一次跑完：

```bash
uv run --extra pdf python -m network_cpt.dataset
```

### 第一版怎麼 trace

建議第一次逐階段執行，每跑完一支就檢查它的輸出：

| 階段 | 讀取 | 寫出 |
|---|---|---|
| `download.py` | 網路上的 HF/Git/PDF | `data/raw/` 原始檔與 `download_manifest.json` |
| `parse.py` | `data/raw/` | `data/interim/01_parsed/documents.jsonl.gz` |
| `clean.py` | parsed JSONL | `data/interim/02_cleaned/documents.jsonl.gz`、`rejected.jsonl.gz` |
| `split.py` | cleaned JSONL | `data/processed/network_cpt_v1/{train,validation,test}.parquet` |

幾個不會修改資料的檢查指令：

```bash
# 看 raw 原始檔是否下載完成
find data/raw -maxdepth 4 -type f | head

# 看 parse/clean 的第一筆 JSON
gzip -cd data/interim/01_parsed/documents.jsonl.gz | head -n 1
gzip -cd data/interim/02_cleaned/documents.jsonl.gz | head -n 1

# 看 split 數量
uv run python -c 'import json; print(json.load(open("data/processed/network_cpt_v1/dataset_manifest.json"))["records"])'
```

## 來源與 v1 選擇

| source key | 下載與選擇方式 |
|---|---|
| `gsma_tcc` | 完整下載 HF repository；parse 時逐 batch 讀本地 Parquet |
| `tele_data` | 完整下載 HF repository；parse 時逐行讀本地 `*/*.jsonl`，包含 `arxiv`、`standard`、`web`、`wiki` |
| `frrouting` | 普通 shallow clone；只解析 `doc/user/**/*.rst` |
| `sonic` | 普通 shallow clone；只解析 `doc/**/*.md` |
| `openvswitch` | 普通 shallow clone；解析 topics/howto/tutorials/ref/faq RST，另收兩篇 OVS 概念介紹 |
| `kea` | 完整固定版 ARM PDF，经 Docling 轉 Markdown |

本專案使用 Tele-Data 的正確 subset/config 形式，而不是假設所有資料都在單一 split。
GSMA TCC 的 row-level `license`、`collection` 與 provenance 會保留下來。來源說明可查
[Tele-Data dataset card](https://huggingface.co/datasets/AliMaatouk/Tele-Data)、
[GSMA Telco Common Corpus](https://huggingface.co/datasets/GSMA/Telco-Common-Corpus) 與
[Docling quickstart](https://docling-project.github.io/docling/getting_started/quickstart/)。

OVS 刻意不使用過寬的 `Documentation/**/*.rst`：除了原本高價值的 deep-dive、how-to、
tutorial 與 reference，另外納入包含 VLAN、VXLAN、OpenFlow、QoS 和疑難排解的 `faq/`，
以及 `intro/what-is-ovs.rst`、`intro/why-ovs.rst`。`internals/` 的專案治理、貢獻流程，和
`intro/install/` 的大量建置安裝文件不納入 v1。

### Kea 與 Docling

Kea 固定下載指定版本的 Administrator Reference Manual PDF，再使用 Docling 轉成
Markdown；不提供其他解析模式。PDF 版本與 SHA-256 都會寫入 manifest。首次轉換可能
下載 Docling 模型且耗時；產出的 `data/interim/00_docling/*.md` 會被重用。若要重新解析，
請先人工檢查後移走該 Markdown；pipeline 不會自動破壞中間產物。

Kea 不把整本 ARM 當成單一 document。parser 會找 Docling Markdown 中重複出現的最淺
heading 層級，以自然章節建立 `document`；章節下的子標題仍留在同一份文件。這能避免
整本 Kea 只能全部進 train 或全部進 test。

## Unified schema

每一筆是「一個來源文件中的 logical section」：

```json
{
  "id": "frrouting-...",
  "source": "frrouting",
  "document": "doc/user/bgp.rst",
  "title": "BGP Route Reflector",
  "license": "SEE_REPOSITORY_LICENSE; review document notices",
  "language": "en",
  "text": "完整 section，含 heading 與 CLI example...",
  "snapshot": "git commit / HF revision / PDF SHA-256",
  "section": "doc/user/bgp.rst#12",
  "estimated_tokens": 1234,
  "metadata": {}
}
```

Parquet 中的 `metadata` 是 JSON string，避免不同來源的 nested schema 互相衝突。
真正 split group key 是 `(source, document)`，不是 `id` 或 section；同一份 BGP 文件的
不同 section 一定進同一個 split。

## 清洗規則

目前 v1 會：

- 統一換行與 Unicode NFC，移除控制字元與 Docling NUL byte。
- 移除常見 navigation、TOC dot leaders、過多空行，但保留 code indentation。
- 丟棄少於 200 字元、只有 URL、license 空白、亂碼比例過高、文字比例過低的資料。
- 將同一個 `(source, document)` 的 clean sections 合併，經 Unicode NFKC、casefold 與
  whitespace normalization 後做整份文件的 SHA-256 exact dedup。dedup 在 split 前跨來源
  執行，若 Tele-Data 與 GSMA 出現相同文件，只保留先出現的一份。
- 保留 heading、CLI/config/code block 與表格的 Markdown/RST 文字。

v1 **尚未做 near-duplicate MinHash/SimHash**；manifest 會明確記錄這件事。也不會先切成
RAG chunk，section 只用於保留語意單位，後續交給 tokenizer 與 TRL packing。

## Train / validation / test split

split 只看完整文件，不拿 section、token chunk 或 packed sequence 個別抽樣。程式會：

1. 先讀取 clean corpus，收集每個 source 的唯一 `document`。
2. 在每個 source 內，用 `SHA-256(seed + source + document)` 產生穩定排序。
3. 以 largest-remainder 整數分配，各來源獨立切成 90% train、5% validation、5% test。
4. 第二次讀 clean corpus，將同一 `(source, document)` 的所有 records 寫進同一 Parquet。

因此 GSMA、Tele-Data、FRRouting、SONiC、OVS 與 Kea 都各自切分，再合併成三份輸出；
不是把所有來源混在一起做一次 random split。`SPLIT_SEED = 42` 時結果可重現。比例依
document 數量計算，不是依 token 數；來源文件太少時，5% 經整數分配仍可能是 0，實際
數量會記錄在 `dataset_manifest.json` 的 `documents_by_source_and_split`。

Tokenizer、固定長度 chunk 與 4096-token packing 必須在三份 split 產生後分別執行，
不會放在這個 preprocessing pipeline 的 split 之前。TeleQnA 等 external network
benchmark 也不混入這三份 corpus，之後獨立用於 Base vs CPT 評估。

## 輸出與檢查

成功後會得到：

```text
data/processed/network_cpt_v1/
├── train.parquet
├── validation.parquet
├── test.parquet
├── dataset_manifest.json
└── _SUCCESS
```

快速查看統計與資料：

```bash
uv run python -c 'import json; print(json.load(open("data/processed/network_cpt_v1/dataset_manifest.json")))'

uv run python -c 'import pyarrow.parquet as pq; t=pq.read_table("data/processed/network_cpt_v1/train.parquet"); print(t.schema); print(t.slice(0, 1).to_pylist())'
```

重新執行時，只要對應 manifest 存在，既有 Git/HF/PDF raw snapshot 就會重用。不要直接改
`data/raw`；需要取得新版資料時，先把舊 `data/raw` 歸檔，再重新執行 download。這個規則
刻意保持簡單，第一版不另外提供 force、budget 或 selection mode。

## License / 商用注意事項

這個 pipeline 保留 provenance，**不代表自動完成法律審核**。

- GSMA TCC 使用 row-level license；缺值會標為 `UNKNOWN_REVIEW_REQUIRED`。
- Tele-Data dataset card 標示 MIT，但內嵌的 arXiv、Wikipedia、3GPP 原始內容可能有各自
  條款，所以標成「需核對 embedded source license」。
- Git repo 文件記錄 commit 與 LICENSE/COPYING 路徑，但 repository license 未必能直接
  代表每一份外部引用文件。
- Kea ARM 需一起核對文件內的 copyright/license notices。

商用訓練前應依 `license_counts` 做 allowlist/denylist，再由有權責的人完成審核；不要只
因為 `license` 欄非空就視為已核准。

## 測試與品質檢查

```bash
uv run ruff check .
uv run pytest
```

測試涵蓋 Markdown/RST section、Kea 自然章節、navigation 清理、document-level exact
dedup、per-source 90/5/5、三份 Parquet 建立，以及 `(source, document)` 不跨 split。
測試只用本地 fixture，不會下載外部 corpus。
