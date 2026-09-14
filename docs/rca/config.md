# `data_rca/config.py`

## 角色與運作方式

集中所有可調設定，不使用 argparse。其他模組 import 這些常數，因此改完設定後直接重新執行 build。

## Inputs

沒有 function input。隱含 input 是 `config.py` 自身所在位置，用它計算 repository root。

## Outputs

沒有 return value或直接寫檔；輸出是一組 constants：

| Constant | 用途 |
| --- | --- |
| `PROJECT_ROOT` | Repository root |
| `NETOPSBENCH_REVISION` | 固定 HF snapshot |
| `ANTA_REVISION` | 固定 ANTA commit |
| `NIKA_ROOT` | NIKA raw cases |
| `RAW_DIR` | RCA public raw root |
| `NETOPSBENCH_ROOT` / `NETOPSBENCH_EXTRACTED` | Download/extracted traces |
| `ANTA_ROOT` | ANTA repository snapshot |
| `ITU_TRACK_B_ROOT` / `FAULTBENCH_ROOT` | 尚未啟用來源的位置 |
| `ANTA_INPUT` | Optional real ANTA JSONL export |
| `CANONICAL_DIR` | Source-neutral JSONL output |
| `FINAL_DIR` | Train/val/test 與 reports |
| `REJECTED_PATH` | Reject/exclusion audit log |
| `*_CATALOG_PATH` | Source diagnosis rules |
| `SEED` | Deterministic split/review seed |
| `SPLIT_RATIOS` | 80/10/10 target |
| `MAX_EVIDENCE_ITEMS` | 每筆最多 6 observations |
| `MAX_OUTPUT_CHARS` | 每個 output 最多 12,000 chars |
| `REVIEW_ROWS_PER_SOURCE` | 每來源抽 8 筆 review |
| `SYSTEM_MESSAGE` | SFT system message |

## Failure behavior

路徑不存在通常由各 builder 轉成 rejection；此模組本身不檢查檔案。

## Constants 的實際長相

```python
PROJECT_ROOT
# Path("/home/eating/agentic-research")

SPLIT_RATIOS
# {"train": 0.80, "val": 0.10, "test": 0.10}

NETOPSBENCH_EXTRACTED
# Path(".../data/rca/raw/netopsbench/extracted")

SYSTEM_MESSAGE
# "You are a network troubleshooting assistant. ..."
```
