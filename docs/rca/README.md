# RCA pipeline 文件索引

這個目錄和 `data_rca/*.py` 一對一對應。每份文件著重程式的運作方式、function input/output、
filter 與 failure behavior，並附上實際的 JSON、JSONL、Python 或目錄 input/output 範例。

## 建議閱讀順序

1. [`build.md`](build.md)：整條 pipeline 的入口。
2. [`schema.md`](schema.md)：canonical row 必須長什麼樣子。
3. [`nika.md`](nika.md)、[`netopsbench.md`](netopsbench.md)、[`anta.md`](anta.md)：三個來源的 converter/filter。
4. [`split.md`](split.md) 與 [`render.md`](render.md)：切分與 SFT messages。
5. 其他 supporting modules。

## Python file 對照

| Python | 文件 | 角色 |
| --- | --- | --- |
| `data_rca/__init__.py` | [`__init__.md`](__init__.md) | Package marker |
| `data_rca/config.py` | [`config.md`](config.md) | 路徑與常數 |
| `data_rca/io.py` | [`io.md`](io.md) | JSON/JSONL 與 stable hash |
| `data_rca/fetch_sources.py` | [`fetch_sources.md`](fetch_sources.md) | 下載 pinned public artifacts |
| `data_rca/nika.py` | [`nika.md`](nika.md) | NIKA trace + GT converter |
| `data_rca/netopsbench.py` | [`netopsbench.md`](netopsbench.md) | SONiC ATIF + evaluator GT converter |
| `data_rca/anta.py` | [`anta.md`](anta.md) | EOS fixture/export converter |
| `data_rca/taxonomy.py` | [`taxonomy.md`](taxonomy.md) | Domain/protocol mapping |
| `data_rca/schema.py` | [`schema.md`](schema.md) | Canonical validation |
| `data_rca/render.py` | [`render.md`](render.md) | Canonical -> HF messages |
| `data_rca/split.py` | [`split.md`](split.md) | Group-aware split |
| `data_rca/coverage.py` | [`coverage.md`](coverage.md) | Coverage/review reports |
| `data_rca/feasibility.py` | [`feasibility.md`](feasibility.md) | Candidate-source checks |
| `data_rca/build.py` | [`build.md`](build.md) | Offline orchestration |

## 三個來源的 filter 摘要

```text
NIKA
  invalid GT / missing catalog / missing log / no evidence -> reject
  required tool groups + affected side present             -> confirmed_rca
  有 evidence 但 confirmation 不完整                        -> possible_rca

NetOpsBench
  no anomalous GT / missing trajectory                      -> exclude
  no preferred-tool observation mentioning GT device       -> exclude
  otherwise                                                 -> confirmed_rca

ANTA official fixture
  non-failure / empty literal messages                      -> skip
  every message has Expected + Actual                       -> confirmed_state_mismatch
  otherwise                                                 -> diagnostic_guidance
```

更細的逐行解釋仍保留在 [`data_rca/CODE_WALKTHROUGH.md`](../../data_rca/CODE_WALKTHROUGH.md)。
