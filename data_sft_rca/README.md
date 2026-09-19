# Network-RCA-SFT-v2

這個目錄實作一個乾淨的 Qwen 27B RCA LoRA ablation data pipeline：不做 CPT、不加入
ToolACE / When2Call、不保留 tool calling trajectory、不呼叫 LLM 產資料，也不訓練 hidden CoT。

## 已產生的資料

目前 workspace 內的 `raw/` 與 `output/` 已完整建立；兩者被本目錄的 `.gitignore` 排除，避免把
大型 public artifacts 和衍生 JSONL commit 進 Git。

Canonical root cases 共 2,564 rows：

| Source | Canonical rows | 用途 |
| --- | ---: | --- |
| NIKA | 901 | Linux / FRR |
| NetOpsBench | 596 | SONiC |
| ANTA selected | 419 | EOS state diagnosis |
| Cloud-OpsBench network subset | 164 | Kubernetes service networking |
| FaulT-Bench core | 200 | fault / false premise / wrong ticket |
| RCAEval network subset | 284 | DELAY / LOSS / SOCKET telemetry |

經 full、focused、masked probable-cause、next-step views 展開及去重後為 5,012 SFT rows：

| Task | Rows |
| --- | ---: |
| `confirmed_rca` | 2,508 |
| `probable_cause_analysis` | 1,269 |
| `next_diagnostic_step` | 702 |
| `state_diagnosis` | 413 |
| `premise_correction` | 120 |

Split 是 `group_id` aware：train 4,004、validation 509、test 499，目前 leakage 為 0。
`unseen_fault_family_test.jsonl` 是 test 中 cause 未出現在 train 的額外 slice，不是重複加入
training data。

## 目錄

```text
data_sft_rca/
├── raw/                         # public source snapshots，已下載、gitignored
├── output/
│   ├── canonical/               # source-neutral root cases
│   ├── views/all.jsonl          # full + derived evidence views
│   ├── qwen_sft/                # train / val / test messages JSONL
│   ├── symptom_cause_graph.json
│   └── dataset_manifest.json
├── *_converter.py               # source converters
├── symptom_graph.py
├── build_views.py
├── validate_views.py
├── split.py
├── render_qwen_sft.py
└── build.py
```

NIKA、NetOpsBench、ANTA 直接重用 repo 內已驗證的 v1 source conversion，再轉成 v2 schema；
不複製既有 1.5 GB raw snapshots。新增的三個 sources 放在本目錄的 `raw/`。RCAEval 只下載
285 個 network-related metrics files，不下載不相關的 CPU / MEM / DISK cases 或 agent traces；
其中 284 個通過 evidence gate。

## 執行

設定都直接放在 `config.py`，沒有 argparse。

```bash
# 新環境第一次抓 public sources
.venv/bin/python -m data_sft_rca.fetch_sources

# 離線、deterministic 重建
.venv/bin/python -m data_sft_rca.build

# 驗證
.venv/bin/ruff check data_sft_rca tests/test_data_sft_rca.py
.venv/bin/pytest
```

在訓練 B 前，把 A 用過的 eval case id、`root_case_id` 或 `group_id` 一行一個放進
`frozen_eval_ids.txt`。Builder 會在建立 graph 和 views 前排除它們，manifest 會記錄排除數量。

## Canonical 與 views

Canonical 只保留必要欄位：source/platform/domain、task/fault family、problem、帶
`symptom|supporting|decisive|irrelevant` role 的 evidence、diagnosis、grouping 與簡短 provenance。

Probable-cause candidate 只來自至少兩個 independent grounded root cases 支持的
`fault_family -> cause` graph。排序只看 visible protocol/platform、檢查成本、support count 與
stable tie-break，不用被 mask 的 ground truth 決定第一名。ANTA 不做 evidence masking。

`qwen_sft/*.jsonl` 只有三個 messages roles：`system`、`user`、`assistant`；沒有 `tools`、
`tool_calls`、`role=tool` 或 `<think>`。訓練時仍應由 trainer 對 system/user tokens 設 `-100`，
只計 assistant completion loss。

## Source 與授權

下載版本固定在 `config.py` 的 revisions。Raw source 的再散布與使用須遵守各上游授權；尤其
FaulT-Bench dataset 為 CC BY 4.0。此 pipeline 不把上游 raw data commit 進本 repo。
