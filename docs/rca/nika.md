# `data_rca/nika.py`

## 角色與運作方式

將每個 NIKA case 的 `ground_truth.json` 與 diagnosis log tool events join 成 canonical row。
不讀 `submission.json`、agent reasoning 或 final answer。

```text
case directory
  ├── ground_truth.json
  ├── session_meta.json                 optional
  └── conversation_diagnosis_agent.log
          -> pair tool_start/tool_end
          -> catalog evidence groups
          -> confirmed or possible RCA
```

## Function I/O

### `_arguments(value)`

- Input：tool input，可能是 dict、JSON string、Python-literal string 或空值。
- Output：dictionary；無法得到 dict 時回 `{}`。
- JSON parse 失敗後才用 `ast.literal_eval`，不使用 `eval`。

### `_output(value)`

- Input：NIKA tool-end output，可能是 `content='...' name='...'` wrapper 或其他 value。
- Output：clean string。非字串轉 sorted JSON；wrapper 會取出 content。

### `_device(arguments)`

- Input：tool arguments dictionary。
- Output：從 router/host/switch/device-pair keys 收集的去重 device string。

### `_command(tool, arguments)`

- Input：tool name 與 arguments。
- Output：human-readable command string。
- 有真 `command` 就直接保留；已知 structured tools 做固定 mapping；未知 tool render 成 function call。

### `_evidence_type(tool)`

- Input：tool name。
- Output：`config`、`log` 或 `cli`。

### `extract_tool_evidence(log_path)`

- Input：diagnosis JSONL log `Path`。
- Output：`evidence_list`。
- Evidence fields：type、device、command、output、tool、arguments、truncated。
- 運作：
  1. 保存每個 `tool_start` 到 active list。
  2. `tool_end` 依 wrapper tool name 配對；不明確時 fallback 最早 active call。
  3. 空 tool/output 跳過。
  4. output 超過 12,000 chars 截斷並標記。
- Malformed JSON line 直接跳過，不 reject 整個 case。

### `_problem_and_group(session, fault_type, affected)`

- Input：session metadata、fault label、faulty-device set。
- Output：`(problem_string, group_id)`。
- Problem 只用 scenario/network description，不放答案。
- Group hash：fault + scenario + topology context + affected devices。

### `_platform(fault_type, tools)`

- Input：fault label 與 selected tool names。
- Output：`bmv2`、`frr` 或 `linux`。

### `parse_case(case_dir, catalog)`

- Input：
  - 一個 NIKA case directory。
  - 完整 fault catalog dictionary。
- Output：一個 validated canonical row。
- Raises：`ValueError` 或 JSON error；caller 會轉成 rejected record。

Filter 決策：

```text
is_anomaly != true OR causes != 1             -> reject
fault type 不在 catalog                        -> reject
缺 conversation_diagnosis_agent.log           -> reject

relevant = tool name 出現在 catalog evidence_groups
confirmation = relevant；affected_only 時再限 faulty device

catalog 每組至少命中一個 tool
AND confirmation evidence 命中 affected side  -> confirmed_rca

否則但仍有 relevant/all tool evidence          -> possible_rca
完全沒有 tool evidence                         -> reject
```

Evidence 排序：affected side 優先，再依 tool/command；最多六筆。

重要限制：confirmed gate 驗 tool presence/device，不解析 CLI output 是否真的包含錯誤值。RCA label
正確性由 ground truth 提供。

### `build_nika()`

- Input：沒有參數；讀 config root/catalog。
- Output：`(kept_rows, rejected_rows)`。
- 掃描方式：所有 `NIKA_ROOT/**/ground_truth.json` 的 parent directory。
- Row-level parse errors 不終止 batch，而是寫 source/case/reason。

## Input / output 實際長相

### Raw ground truth input

```json
{
  "is_anomaly": true,
  "faulty_devices": ["leaf_router_0_0"],
  "root_cause_name": ["bgp_asn_misconfig"]
}
```

### Raw log input

```jsonl
{"event":"tool_start","tool":{"name":"frr_exec"},"input":{"router_name":"leaf_router_0_0","command":"show ip bgp summary"}}
{"event":"tool_end","output":"content='BGP state = Idle' name='frr_exec'"}
```

### Helper output

```python
_arguments('{"router_name":"r1","command":"show ip bgp summary"}')
# -> {"router_name": "r1", "command": "show ip bgp summary"}

_device({"router_name": "r1"})
# -> "r1"

_command("frr_exec", {"router_name": "r1", "command": "show ip bgp summary"})
# -> "show ip bgp summary"

extract_tool_evidence(log_path)
# -> [{
#      "type": "cli",
#      "device": "leaf_router_0_0",
#      "command": "show ip bgp summary",
#      "output": "BGP state = Idle",
#      "tool": "frr_exec",
#      "arguments": {...},
#      "truncated": False,
#    }]
```

### `parse_case()` canonical output

```json
{
  "id": "nika_2e97d474a204071c1bc4af94",
  "source": "nika",
  "platform": "frr",
  "network_domain": "routing",
  "protocol": "bgp",
  "problem": "A network anomaly was reported in the dc_clos_service scenario.",
  "evidence": [
    {
      "type": "cli",
      "device": "leaf_router_0_0",
      "command": "show ip bgp summary",
      "output": "local AS number 65800 ... Idle"
    }
  ],
  "diagnoses": [
    {
      "type": "confirmed_rca",
      "cause": "BGP AS mismatch on leaf_router_0_0",
      "reason": "The BGP AS configuration is inconsistent with the intended peer relationship.",
      "identify": [{"command": "show ip bgp summary", "expected_evidence": "Neighbor is not Established"}],
      "resolution": ["Correct the BGP AS configuration."],
      "verification": [{"command": "show ip bgp summary", "expected_result": "Neighbor is Established"}]
    }
  ],
  "grounding": "runtime_trace",
  "group_id": "nika_<stable-hash>",
  "metadata": {
    "fault_type": "bgp_asn_misconfig",
    "source_case": "misconfigurations/bgp_asn_misconfig/123"
  }
}
```

### Rejected output

```json
{
  "source": "nika",
  "case": "misconfigurations/bgp_asn_misconfig/123",
  "reason": "no_tool_evidence"
}
```

## Standalone output

`python -m data_rca.nika` 只寫 NIKA canonical 與 rejected，不做 merged split。
