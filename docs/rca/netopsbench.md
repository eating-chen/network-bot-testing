# `data_rca/netopsbench.py`

## 角色與運作方式

將 NetOpsBench evaluator ground truth 與 ATIF tool observations join 成 SONiC confirmed RCA。

```text
traces/index.jsonl    trace_id -> ATIF path
traces/results.jsonl  evaluator ground truth
ATIF steps            tool_call observations only
catalog               cause/domain/fix/verification
```

明確不用：`details.agent_output`、agent evidence summary、reasoning、final diagnosis、LLM steps。

## Function I/O

### `_observation_output(step)`

- Input：一個 ATIF step dictionary。
- Output：observation string；沒有 usable result 時回 `""`。
- 優先抽 `artifact.structured_content`；沒有時保留 parsed wrapper；非 JSON content 保留原文。
- Output 超過 12,000 chars 時截斷。

### `_tool_evidence(trajectory)`

- Input：完整 ATIF trajectory dictionary。
- Output：canonical-like evidence list。
- 只收 `extra.type == "tool_call"`。
- `read_file` 被忽略，空 observation 被忽略。
- Device 從 arguments 的 `device` 或 `src` 取得。
- `get_device_config` -> config；logs/BGP events -> log；其他 -> cli。

### `_problem(trajectory, scenario_id)`

- Input：ATIF trajectory 與 scenario ID。
- Output：不含答案的 problem string。
- 只讀 initial context 的 symptoms；無法解析時使用 generic fallback。

### `parse_netopsbench(result, trajectory, catalog)`

- Input：
  - 一筆 `results.jsonl` evaluator result。
  - 對應的 ATIF trajectory。
  - fault catalog dictionary。
- Output：validated SONiC canonical row。
- Raises：GT、catalog、evidence 或 schema 不合時 `ValueError`。

Keep filter：

```text
details.ground_truth.fault_type 存在
AND fault type 在 catalog
AND 至少一筆 evidence：
      tool in preferred_tools
      AND GT device 出現在 arguments 或 output
```

否則分別得到 `no_anomalous_ground_truth`、`fault_type_not_in_catalog` 或
`no_ground_truth_relevant_tool_observation`。

Evidence ranking 由高至低比較：

```text
(GT device hit, GT interface hit, preferred tool, later trace position)
```

取前六筆後恢復原始 trace 順序。Diagnosis 一律 `confirmed_rca`，label/location 來自 evaluator GT；
reason/resolution/verification 來自 catalog。

重要限制：keep gate 是字串與 tool metadata matching，不重新解析 observation 推導 fault。

### `_trajectory_map(run_dir)`

- Input：解壓後單一 run directory。
- Output：`dict[trace_id, Path]`。
- `traces/index.jsonl` 不存在時回空 map。

### `build_netopsbench()`

- Input：沒有參數；讀 extracted root 與 catalog。
- Output：`(kept_rows, rejected_rows)`。
- 若 snapshot 未解壓，回零 kept 加一筆 `snapshot_not_extracted`。
- 對每個 `results.jsonl` 用 trace ID join trajectory；缺 trajectory 或 parse error寫入 run/line/trace/reason。

## Input / output 實際長相

### `results.jsonl` input

程式只會使用 `trace_id/scenario_id/run_id/details.ground_truth`：

```json
{
  "trace_id": "run-1:worker-1:case-1:trace-1",
  "run_id": "run-1",
  "scenario_id": "generated_mtu_mismatch_xs_001",
  "details": {
    "ground_truth": {
      "fault_type": "mtu_mismatch",
      "location": {"device": "spine1", "interface": "Ethernet0"}
    },
    "agent_output": "存在於 raw，但 converter 不讀"
  }
}
```

### ATIF tool step input

```json
{
  "extra": {"type": "tool_call", "name": "get_device_interfaces"},
  "tool_calls": [
    {"arguments": {"device": "spine1"}, "function_name": "get_device_interfaces"}
  ],
  "observation": {
    "results": [
      {
        "content": "{\"artifact\":{\"structured_content\":{\"device\":\"spine1\",\"interfaces\":[{\"name\":\"Ethernet0\",\"mtu\":1400}]}}}"
      }
    ]
  }
}
```

### Helper outputs

```python
_observation_output(step)
# -> '{"device":"spine1","interfaces":[{"mtu":1400,"name":"Ethernet0"}]}'

_tool_evidence(trajectory)
# -> [{
#   "type": "cli",
#   "device": "spine1",
#   "command": "get_device_interfaces {\"device\": \"spine1\"}",
#   "output": "{...}",
#   "tool": "get_device_interfaces",
#   "arguments": {"device": "spine1"},
#   "trace_position": 4,
# }]

_trajectory_map(run_dir)
# -> {"run-1:worker-1:case-1:trace-1": Path(".../trajectory.atif.json")}
```

### `parse_netopsbench()` canonical output

```json
{
  "id": "netopsbench_<trace-hash>",
  "source": "netopsbench",
  "platform": "sonic",
  "network_domain": "interfaces",
  "protocol": "ethernet",
  "problem": "Runtime telemetry detected a network anomaly in NetOpsBench scenario generated_mtu_mismatch_xs_001.",
  "evidence": [
    {
      "type": "cli",
      "device": "spine1",
      "command": "get_device_interfaces {\"device\": \"spine1\"}",
      "output": "{\"interfaces\":[{\"name\":\"Ethernet0\",\"mtu\":1400}]}"
    }
  ],
  "diagnoses": [
    {
      "type": "confirmed_rca",
      "cause": "Interface MTU mismatch on spine1 interface Ethernet0",
      "reason": "The benchmark ground truth identifies an inconsistent MTU on the affected interface.",
      "identify": [{"command": "get_device_interfaces on spine1", "expected_evidence": "MTU mismatch"}],
      "resolution": ["Configure a consistent intended MTU across the affected path."],
      "verification": [{"command": "get_device_interfaces", "expected_result": "The intended MTU is reported."}]
    }
  ],
  "grounding": "runtime_trace",
  "group_id": "netopsbench_<scenario-hash>",
  "metadata": {
    "fault_type": "mtu_mismatch",
    "scenario_id": "generated_mtu_mismatch_xs_001",
    "trace_id": "run-1:worker-1:case-1:trace-1"
  }
}
```

### Rejected output

```json
{
  "source": "netopsbench",
  "run": "run-1",
  "line": 12,
  "trace_id": "run-1:worker-1:case-1:trace-1",
  "reason": "no_ground_truth_relevant_tool_observation"
}
```

## Group behavior

`group_id = hash(scenario_id)`；相同 scenario 的跨模型 trajectories 不會跨 split。
