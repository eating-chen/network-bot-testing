# Agentic SFT data preparation v1

這條 pipeline 只建立 model-agnostic canonical dataset。它不處理 training YAML、LoRA、
FSDP、GPU ranks、evaluation 或 RL，也不保存 rendered messages、special tokens 或 tokens。

## 資料流與 CPT 邊界

```text
NIKA / UltraChat / FunctionGemma
              ↓ converters
       canonical JSONL
      system + tools + turns
              ↓ model adapter
       Llama / Qwen / Gemma
              ↓ tokenizer.apply_chat_template(...)
          training tokens
```

```text
network_cpt/                    network_sft/
document + text                semantic conversations
data/raw/...                   data/sft/raw/...
data/interim/...               data/sft/intermediate/...
data/processed/network_cpt_v1  data/sft/canonical/...
```

SFT 不讀寫 CPT 的中間檔。上游資料可以是 JSONL 或 Parquet，但 canonical master 與 split
永遠是 JSONL。

## Canonical schema

```python
{
    "id": "nika_123",
    "source": "nika",
    "task_type": "network_agent",
    "system": "You are a network troubleshooting assistant.",
    "tools": [{
        "name": "ping",
        "description": "Ping a network device.",
        "parameters": {
            "type": "object",
            "properties": {"host": {"type": "string"}},
            "required": ["host"]
        }
    }],
    "turns": [
        {"role": "user", "content": "Diagnose why r1 is unreachable."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "call_id": "call_0",
                "name": "ping",
                "arguments": {"host": "r1"}
            }]
        },
        {
            "role": "tool",
            "call_id": "call_0",
            "name": "ping",
            "content": "timeout"
        },
        {"role": "assistant", "content": "r1 is unreachable."}
    ],
    "metadata": {
        "scenario": "simple_bgp",
        "failure_type": "link_down",
        "success": True
    }
}
```

Canonical 層禁止以下 representation：

- `messages`
- `type: function` / `function: {...}` wrapper
- `tool_call_id`
- JSON string 形式的 arguments
- `<|start_header_id|>`、`<tool_call>` 等 special tokens

這些若有需要，只能由 `network_sft/adapters/` 在記憶體中產生。

## 程式結構

```text
network_sft/
├── general.py                 UltraChat converter
├── functiongemma.py           FunctionGemma converter
├── nika.py                    NIKA event parser
├── schema.py                  canonical constructor + validator
├── split_merge.py             source-first split + merge
├── validate.py
├── stats.py                   render → chat template → length only
└── adapters/
    ├── huggingface.py         canonical → HF input convention
    ├── llama.py
    ├── qwen.py
    └── gemma.py
```

目前三個 model adapter 都透過 Hugging Face 的共同 input contract；若某個 tokenizer 有
額外要求，只改對應 adapter，不改 converter、canonical JSONL 或其他 model adapter。

## 安裝與設定

```bash
uv sync --dev --extra sft
```

第一版設定集中在 `network_sft/config.py`：

- General SFT 預設 deterministic sample 2,000 筆。
- General conversation 上限預設 32,000 characters。
- Split seed 預設 42。
- Token statistics 預設使用 `meta-llama/Llama-3.1-8B-Instruct`。

若 CPT 未來也使用 UltraChat replay，把 CPT 使用的 upstream `prompt_id` 一行一個放在：

```text
data/sft/general_cpt_replay_ids.txt
```

General converter 會排除它們。現在的 CPT 沒有 UltraChat，因此檔案可以不存在。

## 逐階段執行

```bash
# 1. 下載 raw snapshots
uv run python -m network_sft.download

# 2. 各來源轉成 canonical intermediate JSONL
uv run python -m network_sft.general
uv run python -m network_sft.functiongemma
uv run python -m network_sft.nika

# 3. 驗證 canonical schema 和 trajectory
uv run python -m network_sft.validate

# 4. 各來源先 split，再 merge/shuffle 成 canonical master
uv run python -m network_sft.split_merge

# 5. 經 Llama adapter 套 template，只記錄 token length
uv run --extra sft python -m network_sft.stats
```

一次跑完：

```bash
uv run --extra sft python -m network_sft.pipeline
```

Llama tokenizer 可能要求先接受模型條款並執行 `hf auth login`。

## 各來源處理規則

### General SFT

讀 UltraChat `train_sft`，只保留非空、user/assistant 交替、以 assistant 結尾且未超過
上限的 conversation。先 exact dedup，再依 seed + prompt_id 做 deterministic top-K。
輸出 `system=""`、`tools=[]`，原始 messages 只轉成 canonical turns。

### FunctionGemma Network

只下載 `data/en/{train,validation,test}.jsonl`。converter 會拆除 FunctionGemma/HF 的
developer message、function wrapper、tool call wrapper，轉成 canonical system/tools/turns。
停在 tool call、沒有 tool result/final response 的 first-turn-only rows 會丟棄；完整 tool
cycle 與正常 direct response 會保留，官方 split 也會保留。

### NIKA

逐行讀 `conversation_diagnosis_agent.log` JSON events，以 `tool_start/tool_end` 重建語意
trajectory。一次 assistant turn 可以含多個平行 calls，後面必須有同數量 results。

只有以下條件全部成立才保留：

- submission 存在且可解析。
- anomaly、root cause、faulty devices 與 ground truth 全部一致。
- log 沒有 fatal/error event 或 invalid tool call。
- 每個 call 有 result，最後有非空 assistant diagnosis。

NIKA 以完整 failure type 為 split group；test failure types 不會出現在 train。

## 輸出

```text
data/sft/
├── raw/
├── intermediate/
│   ├── general_sft.jsonl
│   ├── functiongemma_network.jsonl
│   └── nika.jsonl
├── canonical/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── dataset_manifest.json
├── rendered/                   本階段不產生；預留給未來 model-specific artifacts
└── reports/
    ├── dataset_stats.json
    ├── validation_errors.jsonl
    └── *_stats.json
```

查看 canonical sample：

```bash
uv run python -c '
from pathlib import Path
from network_sft.io import jsonl_rows
row = next(jsonl_rows(Path("data/sft/canonical/train.jsonl")))
print(row["id"], row["source"], row["system"])
for turn in row["turns"]:
    print(turn)
'
```

真正訓練某個 model 時才執行：

```python
messages, tools = model_adapter(example)
input_ids = tokenizer.apply_chat_template(
    messages,
    tools=tools or None,
    tokenize=True,
    add_generation_prompt=False,
)
```

因此換 model 或 chat template 時，canonical JSONL 完全不需要重做。
