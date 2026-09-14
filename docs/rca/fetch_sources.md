# `data_rca/fetch_sources.py`

## 角色與運作方式

這是唯一會存取網路的 RCA 模組。下載 pinned NetOpsBench dataset、解壓 archives，並 clone pinned
ANTA revision。`build.py` 不會自動呼叫它。

## Functions

### `_extract_netopsbench()`

- Input：沒有參數；讀 `config.NETOPSBENCH_ROOT`。
- Output：`None`。
- Side effects：建立 `NETOPSBENCH_EXTRACTED`，對每個 `.tar.zst` 執行 system `tar --zstd -xf`。
- Error：任一 tar command 非零就由 `subprocess.run(check=True)` 拋錯。

### `fetch()`

- Input：沒有參數；revision/path 全由 config 提供。
- Output：`None`。
- Side effects：
  1. 用 `snapshot_download` 下載 `yyyyyt/netopsbench-trace` 指定 revision。
  2. 呼叫 `_extract_netopsbench()`。
  3. `ANTA_ROOT` 不存在時 clone ANTA 並 checkout pinned commit。
- Fallback：若 ANTA root 已存在，不覆蓋也不重新 checkout。

## 執行方式

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m data_rca.fetch_sources
```

## Input / output 目錄長相

Function 沒有 Python return data；它的 output 是 filesystem：

```text
data/rca/raw/
├── netopsbench/
│   ├── manifest.jsonl
│   ├── runs/**/*.tar.zst
│   └── extracted/
│       └── <run-id>/
│           ├── report.json
│           └── traces/
│               ├── index.jsonl
│               ├── results.jsonl
│               └── **/*.atif.json
└── anta/
    ├── anta/tests/**/*.py
    └── tests/units/anta_tests/**/test_*.py
```
