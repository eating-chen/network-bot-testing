# `data_rca/render.py`

## 角色與資料流

將 canonical row deterministic render 成 Hugging Face `messages`。不做 filtering、不呼叫 LLM。

## Functions

### `_evidence_block(item)`

- Input：一個 evidence dictionary。
- Output：string：`[device]`、`$ command`、`output` 三段。
- Fallback：缺 device 用 `unspecified device`；缺 command 用 tool name 或 `observation`。

### `render_user(row)`

- Input：validated canonical row。
- Output：user prompt string，包含 `Problem / Symptom` 與所有 evidence blocks。

### `_steps(items)`

- Input：resolution string list，或 identify/verification object list。
- Output：Markdown bullet string。
- Object 會 render `command` 與 `expected_evidence`/`expected_result`。

### `render_assistant(row)`

- Input：validated canonical row。
- Output：assistant answer string。
- Label mapping：
  - confirmed RCA → `Confirmed Root Cause`
  - possible RCA → `Possible Cause`
  - state mismatch → `Confirmed State Mismatch`
  - guidance → `Observed Issue`
- Resolution 空時不 render resolution section。

### `to_sft(row)`

- Input：validated canonical row。
- Output：HF-style dictionary：`id/source/platform/network_domain/protocol/diagnosis_type/messages`。
- `group_id` 與 source-specific metadata 只用於 canonical audit/split，不複製到訓練檔。

## Input / output 長相

縮短 canonical input：

```python
row = {
    "id": "anta_001",
    "source": "anta",
    "problem": "The MTU validation failed.",
    "evidence": [{
        "device": "EOS fixture device",
        "command": "show interfaces",
        "output": "Expected: 1500 Actual: 9000",
    }],
    "diagnoses": [{
        "type": "confirmed_state_mismatch",
        "cause": "Interface MTU does not match.",
        "reason": "Expected and actual values differ.",
        "identify": [{"command": "show interfaces", "expected_evidence": "MTU mismatch"}],
        "resolution": ["Configure the intended MTU."],
        "verification": [{"command": "show interfaces", "expected_result": "MTU is 1500"}],
    }],
    "platform": "eos",
    "network_domain": "interfaces",
    "protocol": "ethernet",
    "grounding": "official_test_fixture",
    "group_id": "anta_mtu",
}
```

`to_sft(row)` output：

```json
{
  "id": "anta_001",
  "source": "anta",
  "platform": "eos",
  "network_domain": "interfaces",
  "protocol": "ethernet",
  "diagnosis_type": "confirmed_state_mismatch",
  "messages": [
    {"role": "system", "content": "You are a network troubleshooting assistant..."},
    {"role": "user", "content": "Problem / Symptom:\nThe MTU validation failed..."},
    {"role": "assistant", "content": "Troubleshooting Analysis:\n\nConfirmed State Mismatch:..."}
  ]
}
```
