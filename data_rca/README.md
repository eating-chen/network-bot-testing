# Cross-NOS network troubleshooting SFT

這個資料夾把既有 public artifacts deterministic 地轉成跨 OS / NOS troubleshooting SFT：

```text
NIKA trace + ground truth             -> Linux / FRR RCA
NetOpsBench ATIF + evaluator results  -> SONiC RCA
ANTA official failure fixtures        -> Arista EOS state diagnosis
                                      -> canonical JSONL
                                      -> group-aware train / val / test
```

不建 lab、不呼叫 LLM、不用 LLM judge，也不讀 agent reasoning 或 final answer。程式不用
`argparse`；路徑、seed、split ratio 都直接放在 [`config.py`](config.py)。

逐檔的運作方式與每個 function input/output 請從
[`docs/rca/README.md`](../docs/rca/README.md) 開始；逐行補充則保留在
[`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md)。

## 為什麼這版 schema 比全叫 confirmed RCA 合理

每筆 diagnosis 明確分成四種 certainty：

- `confirmed_rca`：benchmark ground truth 加上 affected-side runtime observation。
- `possible_rca`：有 ground-truth label，但 trace 不足以支撐 confirmed；目前來自 NIKA。
- `confirmed_state_mismatch`：ANTA fixture 明列 `Expected` 與 `Actual`，只確認 state mismatch，
  不宣稱知道底層 root cause。
- `diagnostic_guidance`：官方 ANTA test failure 能指出調查方向，但 evidence 不足以寫 RCA。

canonical row 主要欄位：

```json
{
  "source": "netopsbench",
  "platform": "sonic",
  "network_domain": "routing",
  "protocol": "bgp",
  "problem": "...",
  "evidence": [],
  "diagnoses": [
    {
      "type": "confirmed_rca",
      "cause": "...",
      "reason": "...",
      "identify": [],
      "resolution": [],
      "verification": []
    }
  ],
  "grounding": "runtime_trace"
}
```

`platform` 直接使用 `linux`、`frr`、`sonic`、`eos` 或 `bmv2`，不再另外保存重複的
`platform_family` / `metadata.platform_detail`。

## 各資料來源範例

以下都取自目前實際生成的 canonical rows，為了讓 README 容易閱讀，只保留一筆關鍵 evidence
並縮短 output。完整內容可在 `data/rca/01_canonical/` 依 `id` 查回。

### NIKA：Linux / FRR confirmed RCA

來源 join：

```text
ground_truth.json
  + conversation_diagnosis_agent.log 的 tool output
  + fault_catalog.json
```

實際案例 `nika_2e97d474a204071c1bc4af94`：

```json
{
  "source": "nika",
  "platform": "frr",
  "network_domain": "routing",
  "protocol": "bgp",
  "evidence": [
    {
      "type": "cli",
      "device": "leaf_router_0_0",
      "command": "show ip bgp summary",
      "output": "local AS number 65800 ... peers ... Idle"
    }
  ],
  "diagnoses": [
    {
      "type": "confirmed_rca",
      "cause": "BGP AS mismatch on leaf_router_0_0",
      "reason": "The configured BGP AS is inconsistent with the intended peer relationship.",
      "resolution": [
        "Correct the BGP AS configuration."
      ],
      "verification": [
        {
          "command": "show ip bgp summary",
          "expected_result": "The BGP session reaches Established."
        }
      ]
    }
  ],
  "grounding": "runtime_trace"
}
```

若同一個 ground-truth case 的 trace 沒有收齊 catalog 指定的 confirmation evidence，格式相同，
但 `diagnoses[0].type` 會降成 `possible_rca`。

### NetOpsBench：SONiC confirmed RCA

來源 join：

```text
traces/results.jsonl 的 evaluator ground_truth
  + traces/**/*.atif.json 的 tool_call observation
  + netopsbench_catalog.json
```

實際案例 `netopsbench_0e8c3e11a8e7cd13c0167eeb`：

```json
{
  "source": "netopsbench",
  "platform": "sonic",
  "network_domain": "interfaces",
  "protocol": "ethernet",
  "problem": "A network anomaly was reported in generated_mtu_mismatch_fat-tree-k12_003.",
  "evidence": [
    {
      "type": "cli",
      "device": "edge59",
      "command": "get_device_interfaces {\"device\": \"edge59\"}",
      "output": "{\"interfaces\": [{\"name\": \"Ethernet0\", \"admin\": \"up\", \"oper\": \"up\", \"mtu\": 1400}, {\"name\": \"Ethernet4\", \"admin\": \"up\", \"oper\": \"up\", \"mtu\": 9100}]}"
    }
  ],
  "diagnoses": [
    {
      "type": "confirmed_rca",
      "cause": "Interface MTU mismatch on edge59 interface Ethernet0",
      "reason": "The evaluator ground truth identifies an inconsistent MTU on the affected interface.",
      "resolution": [
        "Configure a consistent intended MTU across the affected path."
      ],
      "verification": [
        {
          "command": "get_device_interfaces",
          "expected_result": "The affected interface reports the intended MTU."
        },
        {
          "command": "ping_test with the intended DF payload",
          "expected_result": "Large packets traverse the path successfully."
        }
      ]
    }
  ],
  "grounding": "runtime_trace"
}
```

這裡不會讀 `details.agent_output`、agent evidence summary、reasoning 或 final diagnosis。若 trace
沒有 ground-truth affected side 的 relevant observation，該筆會進 `rejected.jsonl`。

### ANTA：Arista EOS confirmed state mismatch

來源 join：

```text
tests/units/anta_tests/**/test_*.py 的 failure fixture
  + anta/tests/**/*.py 的 test description / categories / EOS commands
```

實際案例 `anta_011e203a13d0f0b6d1b7831c`：

```json
{
  "source": "anta",
  "platform": "eos",
  "network_domain": "interfaces",
  "protocol": "ethernet",
  "problem": "Verifies the interfaces counter details.",
  "evidence": [
    {
      "type": "expected_observed",
      "device": "EOS fixture device",
      "command": "show interfaces",
      "output": "{\"test_inputs\": {\"counters_threshold\": 10}, \"failure_messages\": [\"Interface: Ethernet4 - Input discards above threshold - Expected: <= 10 Actual: 30\"]}"
    }
  ],
  "diagnoses": [
    {
      "type": "confirmed_state_mismatch",
      "cause": "VerifyInterfacesCounterDetails confirmed an EOS state mismatch.",
      "reason": "The official fixture explicitly records different expected and actual values.",
      "resolution": [
        "Correct the EOS configuration or operational state so it matches the ANTA test input."
      ],
      "verification": [
        {
          "command": "Run VerifyInterfacesCounterDetails again.",
          "expected_result": "The test passes without the recorded failure messages."
        }
      ]
    }
  ],
  "grounding": "official_test_fixture"
}
```

注意這裡確認的是「counter 超過 test threshold」，不是宣稱已知道 counters 增加的 root cause。

### ANTA：Arista EOS diagnostic guidance

實際案例 `anta_01d532a8805a389fdc364fcc`：

```json
{
  "source": "anta",
  "platform": "eos",
  "network_domain": "routing",
  "protocol": "bgp",
  "problem": "Verifies the health of all the BGP peers.",
  "evidence": [
    {
      "type": "expected_observed",
      "command": "show ip bgp neighbors vrf all",
      "output": "{\"failure_messages\": [\"Peer: 10.100.0.8 VRF: default - Session has non-empty message queues - InQ: 5 OutQ: 10\"]}"
    }
  ],
  "diagnoses": [
    {
      "type": "diagnostic_guidance",
      "cause": "VerifyBGPPeersHealthRibd did not reach its expected EOS operational state.",
      "reason": "The fixture records a failed test, but it does not establish the underlying root cause.",
      "identify": [
        {
          "command": "show ip bgp neighbors vrf all",
          "expected_evidence": "The affected peer has non-empty InQ or OutQ queues."
        }
      ],
      "resolution": [],
      "verification": [
        {
          "command": "Run VerifyBGPPeersHealthRibd again.",
          "expected_result": "The test passes after the underlying issue is corrected."
        }
      ]
    }
  ]
}
```

因為 fixture 只證明 BGP message queue 異常，沒有證明是 MTU、policy、CPU 或 peer-side 問題，
所以沒有生成假的 resolution 或 exact RCA。

## 目前實際產出

以目前 pinned snapshots 執行後：

| Source | 掃描結果 | 納入 v1 |
| --- | ---: | ---: |
| NIKA | 906 cases | 901 |
| NetOpsBench | 755 trajectories | 596 |
| ANTA | 643 official failure fixtures | 643 |
| **Total** |  | **2,140** |

NetOpsBench 沒有把 755 筆硬塞進 SFT：77 筆沒有 anomalous ground truth（healthy / negative），
82 筆 trace 沒有 ground-truth affected side 的 relevant tool observation，因此排除。596 筆納入資料
全部只使用 `results.jsonl` 的 evaluator ground truth 與 ATIF `tool_call` observation；跨模型重跑的
同一 `scenario_id` 共用 `group_id`，不會跨 split。

ANTA 靜態讀官方 unit-test `DATA` fixtures，共涵蓋 204 種 tests。219 筆是明確
expected/actual mismatch，424 筆只輸出 diagnostic guidance。fixture 的 expected failure message
是 test oracle，不是現場設備 trace，因此 grounding 強度刻意標成 `official_test_fixture`。

目前 split：

```text
train = 1732
val   = 216
test  = 192
```

## 執行

第一次下載 pinned public snapshots（約數十 MB，需要網路）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m data_rca.fetch_sources
```

離線生成 canonical、SFT split、coverage 與 review sample：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m data_rca.build
```

測試：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check data_rca tests/test_data_rca.py
```

## 可 trace 的輸出

```text
data/rca/
├── raw/                                  # pinned public snapshots；git ignored
├── 01_canonical/
│   ├── nika.jsonl
│   ├── netopsbench.jsonl
│   ├── anta.jsonl
│   └── all.jsonl
├── rejected.jsonl                        # 每筆排除原因
└── network_troubleshooting_sft_v1/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── dataset_manifest.json
    ├── coverage_matrix.json
    ├── coverage_matrix.md
    ├── review_sample.jsonl               # 每來源 deterministic 抽 8 筆
    ├── source_feasibility.json
    └── _SUCCESS
```

建議 review 順序：先看 `rejected.jsonl` 的原因統計，再看 `review_sample.jsonl` 的 user evidence
是否真的支撐 assistant certainty，最後看 `coverage_matrix.md` 決定下一個缺口。

## Source-specific 規則

### NIKA

只讀 `ground_truth.json`、`session_meta.json`、`conversation_diagnosis_agent.log` 的 tool events。
不讀 `submission.json`、agent reasoning 或 final。catalog 的 evidence groups 沒收齊時保留為
`possible_rca`，不 overclaim。

### NetOpsBench

只讀 archive 裡的：

```text
traces/index.jsonl
traces/results.jsonl
traces/**/*.atif.json -> tool_call observation only
```

`details.agent_output`、ATIF LLM steps、`extra.final_diagnosis` 全部不用。每個 fault type 的
resolution/verification 只在 `netopsbench_catalog.json` 寫一次。

### ANTA

從 `tests/units/anta_tests/**/test_*.py` 靜態抽 failure fixture，並從 `anta/tests/**/*.py`
抽 test description、categories 與 EOS commands。不 import 或執行 ANTA tests，所以不需要把
ANTA 加進本專案 dependencies。`input/anta.jsonl` 仍可額外加入真實 ANTA failed export；example
只示範格式，不會被正式 build 自動使用。

## 後續來源

`source_feasibility.json` 目前把 ITU Track B 與 FaulT-Bench 保持 production disabled：

- ITU Track B：只有同時找到 device outputs 與 official ground-truth candidates，才進一步做 ID
  join；目前未下載，所以不人工補答案。
- FaulT-Bench：定位為 fault diversity，不算新 NOS。等前三個來源 review 完再接 scenario 的
  `[GROUND-TRUTH] / [FIX] / [POST-INJECT-CHECK]`。

這兩個來源的存在不影響 v1 build，也不會用空資料偽裝成 coverage。
