# `data_rca/schema.py`

## 角色與運作方式

定義所有來源共用的 canonical contract。Builder 只有在 `validate_record()` 回傳空 list 時才保留 row。

## 合法 enum

- Diagnosis：`confirmed_rca`、`possible_rca`、`confirmed_state_mismatch`、`diagnostic_guidance`
- Evidence：`cli`、`config`、`log`、`expected_observed`
- Grounding：`runtime_trace`、`official_test_fixture`、`test_definition`、`config_diff`

## Function

### `validate_record(row)`

- Input：canonical candidate `dict[str, Any]`。
- Output：`list[str]`。空 list 表示通過；每個字串是一個 validation error。
- 不會修改 input。

必要 top-level fields：

```text
id, source, platform, network_domain, protocol, problem,
evidence, diagnoses, grounding, group_id
```

Validation：

1. Identity/problem fields 必須是非空字串；protocol 可為空但必須是字串。
2. Evidence 必須非空，每項必須有合法 type 與非空 output。
3. Diagnoses 必須非空，每項要有合法 type、cause、reason、identify、verification。
4. Resolution 必須是 list；只有 `diagnostic_guidance` 可以是空 list。
5. `grounding` 必須是合法值。

此 validator 驗結構，不驗 network semantics。

## Input / output 長相

```python
row = {
    "id": "netopsbench_abc",
    "source": "netopsbench",
    "platform": "sonic",
    "network_domain": "interfaces",
    "protocol": "ethernet",
    "problem": "Runtime telemetry detected an anomaly.",
    "evidence": [
        {
            "type": "cli",
            "device": "leaf1",
            "command": "get_device_interfaces",
            "output": "{\"interfaces\": []}",
        }
    ],
    "diagnoses": [
        {
            "type": "confirmed_rca",
            "cause": "Interface MTU mismatch",
            "reason": "Ground truth identifies the mismatch.",
            "identify": [{"command": "show interfaces", "expected_evidence": "MTU differs"}],
            "resolution": ["Correct the MTU."],
            "verification": [{"command": "show interfaces", "expected_result": "MTU matches"}],
        }
    ],
    "grounding": "runtime_trace",
    "group_id": "netopsbench_scenario_1",
}

validate_record(row)
# -> []

validate_record({"id": "broken"})
# -> ["missing fields: ['diagnoses', 'evidence', ...]"]
```
