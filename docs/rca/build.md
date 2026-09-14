# `data_rca/build.py`

## 角色與運作順序

這是 offline pipeline entry point：

```text
build_nika / build_netopsbench / build_anta
  -> source canonical JSONL
  -> merged canonical + rejected audit
  -> group-aware split
  -> HF messages
  -> coverage / review / feasibility / manifest
```

## Function

### `build()`

- Input：沒有 function arguments。所有 paths/rules 來自 config 與 catalogs。
- Output：dataset manifest dictionary。
- Side effects：寫入：
  - `01_canonical/{nika,netopsbench,anta,all}.jsonl`
  - `rejected.jsonl`
  - `train.jsonl`、`val.jsonl`、`test.jsonl`
  - coverage、review sample、feasibility、manifest、`_SUCCESS`

運作細節：

1. 依固定順序呼叫三個 builders。
2. 每個 builder 回 `(kept_rows, rejected_rows)`。
3. Canonical rows 依 ID 排序。
4. `split_rows()` 依 group 切分。
5. `to_sft()` render 每筆 split row。
6. 統計 reject reason，另外輸出 coverage 與 candidate-source feasibility。
7. 寫完所有 artifacts 才寫 `_SUCCESS`。

## Failure behavior

Source parser 預期中的 row-level 錯誤會進 rejected；無法序列化、寫檔失敗等 pipeline-level exception
會終止 build，不會寫出假的成功狀態。

## 執行

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m data_rca.build
```

## Return manifest 長相

```python
manifest = build()
# -> {
#   "canonical_rows": 2140,
#   "rows": {"train": 1732, "val": 216, "test": 192},
#   "sources": {
#       "nika": {"kept": 901, "rejected": 5},
#       "netopsbench": {"kept": 596, "rejected": 159},
#       "anta": {"kept": 643, "rejected": 0},
#   },
#   "rejected_rows": 164,
#   "rejection_reasons": {...},
#   "llm_generated": False,
# }
```

Builder 共通 output tuple 長相：

```python
kept_rows = [{"id": "...", "source": "nika", "evidence": [...], "diagnoses": [...]}]
rejected_rows = [{"source": "nika", "case": "...", "reason": "no_tool_evidence"}]
```
