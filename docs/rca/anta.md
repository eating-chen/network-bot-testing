# `data_rca/anta.py`

## 角色與兩種 input

1. 靜態解析官方 ANTA Python unit-test failure fixtures。
2. 讀 optional `input/anta.jsonl` real failure exports。

使用 AST，不 import 或執行 ANTA，因此 upstream test code 不會在資料 pipeline 中運行。

## Function I/O

### `_class_metadata(source_root)`

- Input：ANTA `anta/tests` source root `Path`。
- Output：`dict[test_class_name, metadata]`。
- Metadata：categories、literal EOS commands/templates、docstring description、source path。
- 只讀 top-level `Verify*` classes；動態 command/category expressions 不執行。

### `_dict_fields(node)`

- Input：AST node。
- Output：constant-key AST dictionary；不是 `ast.Dict` 時回 `{}`。

### `_literal(node, default)`

- Input：AST node 與 fallback。
- Output：`ast.literal_eval(node)` 結果，不能安全解析時回 default。

### `_fixture_rows(fixtures_root)`

- Input：官方 `tests/units/anta_tests` root。
- Output：中間 fixture dictionaries list。
- 每筆包含 test name、case name、literal inputs、failure messages、source path。

Filter：

```text
top-level annotated DATA dict
  -> key 必須是 (VerifyClass, constant case name)
  -> case name 必須包含 "failure"
  -> expected.messages 必須是非空 literal list
```

Success/skipped cases、動態或空 messages 不輸出。

### `parse_official_fixture(raw, metadata)`

- Input：一筆 `_fixture_rows()` output 與對應 class metadata。
- Output：validated Arista EOS canonical row。

Diagnosis decision：

```text
all failure messages match "Expected: ... Actual: ..."
    -> confirmed_state_mismatch
otherwise
    -> diagnostic_guidance
```

State mismatch 有 generic corrective resolution；guidance 的 resolution 強制為空。Verification 都是重新
執行相同 ANTA test。Evidence output 是 `test_inputs + failure_messages`，不是完整 `eos_data`。

### `parse_anta(raw)`

- Input：一筆 optional real export dictionary，必要內容：expected、observed、non-empty diagnoses。
- Output：validated canonical row。
- Reject：`expected == observed` 或 diagnoses 缺失。
- `direct_configuration_check=true` 且只有一個 diagnosis → state mismatch；其他 → guidance。
- Guidance 即使 input diagnosis 有 resolution，也會清空，避免 overclaim。

### `build_anta()`

- Input：沒有參數；讀 config ANTA root 與 optional JSONL。
- Output：`(kept_rows, rejected_rows)`。
- 官方 paths 存在時先抽 metadata/fixtures；缺 snapshot 產 `official_snapshot_not_found`。
- 再 append optional export rows。
- 每個錯誤保留 test/case 或 input line/reason。

## Input / output 實際長相

### Official Python fixture input

```python
DATA = {
    (VerifyL3MTU, "failure"): {
        "inputs": {"mtu": 1500},
        "eos_data": [{"interfaces": {"Ethernet2": {"mtu": 1600}}}],
        "expected": {
            "result": AntaTestStatus.FAILURE,
            "messages": [
                "Interface: Ethernet2 - Incorrect MTU - Expected: 1500 Actual: 1600"
            ],
        },
    }
}
```

### `_fixture_rows()` intermediate output

```python
[{
    "test_name": "VerifyL3MTU",
    "case": "failure",
    "inputs": {"mtu": 1500},
    "messages": ["Interface: Ethernet2 - Incorrect MTU - Expected: 1500 Actual: 1600"],
    "source_path": "tests/units/anta_tests/test_interfaces.py",
}]
```

### `_class_metadata()` output

```python
{
    "VerifyL3MTU": {
        "categories": ["interfaces"],
        "commands": ["show interfaces"],
        "description": "Verifies the global L3 MTU of all L3 interfaces.",
    }
}
```

### `parse_official_fixture()` canonical output

```json
{
  "id": "anta_<fixture-hash>",
  "source": "anta",
  "platform": "eos",
  "network_domain": "interfaces",
  "protocol": "ethernet",
  "problem": "Verifies the global L3 MTU of all L3 interfaces.",
  "evidence": [
    {
      "type": "expected_observed",
      "device": "EOS fixture device",
      "command": "show interfaces",
      "output": "{\"test_inputs\":{\"mtu\":1500},\"failure_messages\":[\"... Expected: 1500 Actual: 1600\"]}"
    }
  ],
  "diagnoses": [
    {
      "type": "confirmed_state_mismatch",
      "cause": "VerifyL3MTU confirmed an EOS state mismatch: ...",
      "reason": "The official ANTA failure fixture explicitly records different expected and actual values.",
      "identify": [{"command": "show interfaces", "expected_evidence": "..."}],
      "resolution": ["Correct the EOS configuration or operational state so it matches the ANTA test input."],
      "verification": [{"command": "Run VerifyL3MTU again.", "expected_result": "The ANTA test passes."}]
    }
  ],
  "grounding": "official_test_fixture",
  "group_id": "anta_<semantic-hash>",
  "metadata": {
    "test_name": "VerifyL3MTU",
    "fixture_case": "failure",
    "source_path": "tests/units/anta_tests/interfaces/test_mtu.py"
  }
}
```

### Optional `anta.jsonl` input

```json
{
  "id": "leaf1-mtu",
  "test_name": "VerifyL3MTU",
  "device": "leaf1",
  "command": "show interfaces Ethernet1",
  "expected": 1500,
  "observed": 9000,
  "direct_configuration_check": true,
  "diagnoses": [
    {
      "cause": "Interface MTU mismatch",
      "reason": "Observed MTU differs from expected.",
      "identify": [{"command": "show interfaces Ethernet1", "expected_evidence": "MTU 9000"}],
      "resolution": ["Configure MTU 1500."],
      "verification": [{"command": "show interfaces Ethernet1", "expected_result": "MTU 1500"}]
    }
  ]
}
```

### Rejected output

```json
{
  "source": "anta",
  "case": "VerifyL3MTU:failure-dynamic-message",
  "reason": "diagnoses[0].verification must be a non-empty list"
}
```

## Grounding 限制

官方 rows 的 grounding 是 `official_test_fixture`。Failure messages 是 test oracle，可確認 test state，
但不是現場 EOS runtime trace；`diagnostic_guidance` 不會捏造真正 RCA。
