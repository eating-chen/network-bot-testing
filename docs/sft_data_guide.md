# Network Mixed SFT V1

第一版只做一件事：把三個 network diagnostic sources 和三個 agentic sources 整理成
約 22k 筆、可以逐階段人工追蹤的 model-neutral corpus。所有設定都直接放在
`network_sft/config.py`，沒有 `argparse`、YAML 或 plugin framework。

若你想逐支理解每個函式、輸入輸出與 debug 位置，請看
[`network_sft_code_guide.md`](network_sft_code_guide.md)。

## 鎖定的資料配方

| source | V1 selection | 說明 |
|---|---:|---|
| `greenwich157/5G-Faults-Full-v2` | 全部 valid | 5G diagnostics |
| `tecnicolaude/Telelogs-CoT` | 全部 valid | 5G RCA + long CoT |
| `Rzkoohi/CCNA_small` | 全部 valid | 不做 sampling；仍做 quality 與 dedup/group |
| NIKA Traces | 全部 valid/correct | submission 正確且 trajectory 完整 |
| `Team-ACE/ToolACE` | 3,000 | valid tool-use rows 做 deterministic simple sample |
| `nvidia/When2Call/train_sft` | 2,000 | valid rows 做 deterministic simple sample |

「全部」代表不做人為 sampling；invalid、exact duplicate 或 leakage risk 仍會處理。實際總數
會依 dedup 與 NIKA validity 結果浮動；NIKA 只保留答案正確、call/result 對得上、最後有診斷的
trace。

ToolACE 下載的是 `minpeter/toolace-parsed`（11,072 rows），因為它是原始
`Team-ACE/ToolACE` 的公開標準化轉換，可避免在這個 PoC 重新維護脆弱的 bracket-call
parser。每筆 metadata 仍明記原始 dataset provenance。

## 目錄與逐步執行

```text
data/sft/raw/                 原始 snapshot，不手改
data/sft/01_normalized/       六份 model-neutral JSONL
data/sft/02_curated/          validity + exact dedup 後，已有 group_id
data/sft/03_selected/         selected.jsonl，約 22k mixed corpus
data/sft/network_sft_v1/      train.jsonl / val.jsonl / test.jsonl
data/sft/experiments/controlled_llama_qwen/
                              Llama/Qwen 共用 sample IDs 的實驗 split
data/sft/reports/             rejection、sampling、review、token/template reports
```

NIKA 另有 `01_normalized/nika_partial.jsonl`（保存但不進 V1）與
`reports/nika_submission_audit.jsonl`（每個 incident 的 exact/partial/wrong/missing 判定）。

```bash
uv sync --dev --extra sft

uv run python -m network_sft.download
uv run python -m network_sft.normalize
uv run python -m network_sft.curate
uv run python -m network_sft.split
uv run --extra sft python -m network_sft.stats
uv run python -m network_sft.controlled
```

或一次執行：

```bash
uv run --extra sft python -m network_sft.pipeline
```

每一步都可以停下來看 JSONL。`curate.py` 雖然同時做 quality、dedup/group 和 sampling，仍
分別寫出 `02_curated` 與 `03_selected`，避免為了形式多拆三支小程式。

## Normalize 與 canonical schema

```json
{
  "id": "nika_...",
  "source": "nika",
  "task_type": "agentic",
  "messages": [
    {"role": "user", "content": "Diagnose r1."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "ping", "arguments": {"host": "r1"}}
      }]
    },
    {"role": "tool", "name": "ping", "tool_call_id": "call_0", "content": "timeout"},
    {"role": "assistant", "content": "r1 is unreachable."}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "ping",
      "description": "Ping a host.",
      "parameters": {"type": "object", "properties": {"host": {"type": "string"}}}
    }
  }],
  "metadata": {
    "original_id": "...",
    "category": "link_down",
    "group_id": "nika_..."
  }
}
```

Corpus 不含 Llama/Qwen/DeepSeek special tokens。When2Call 上游把 calls 放在
`<TOOLCALL>...</TOOLCALL>` 內，normalize 時會拆成上面的 structured `tool_calls`。

## Quality、dedup 與 group

`curate.py` 依序：

1. 驗證 roles、空內容、tool JSON schema、call ID、result pairing 和 final answer。
2. 跨來源做 exact semantic dedup；來源優先序是 network-specific sources 在前。
3. 在同 source/category 內，以 user text 的 MinHash LSH 找 candidates，再用 Jaccard
   `>= 0.86` 建 near-duplicate cluster。near rows 不硬刪，而是共用 `group_id`。
4. NIKA 的 group key 固定是 `scenario + failure_type + incident_id`。

可檢查：

```text
data/sft/reports/normalize_rejected.jsonl
data/sft/reports/quality_rejected.jsonl
data/sft/reports/curation.json
```

## Selection：CCNA 全留，另外兩份簡單抽樣

CCNA 不再做 troubleshooting filter 或 topic quota；通過 validity、exact dedup 和 near-duplicate
grouping 的 rows 全部進 selected corpus。`metadata.category` 仍保留 topic，僅供統計與 split
分層，不會決定某筆 CCNA 是否被選中。

ToolACE 先排除完全沒有 tool call 的 rows，When2Call 保留所有 valid rows；兩者都只使用同一個
簡單規則：

```python
ordered = sorted(rows, key=stable_hash(seed, source, row_id))
selected = ordered[:target]
```

也就是 ToolACE 取 3,000、When2Call 取 2,000，不再按 category quota、補缺口或 oversample。
相同 raw revision 與 `SEED=42` 每次會選到相同資料。兩份資料推導出的 category 仍寫在 metadata
和 `curation.json`，方便觀察抽樣後的自然分布，但不參與選擇。

目前 When2Call snapshot 實測沒有 `<TOOLCALL>` positive；simple sample 不會合成假的 positive，
因此這 2,000 筆主要提供 direct/clarification/cannot decision examples。

## Split 與 internal test 邊界

`split.py` 先按 `source + category` 分層，再以完整 `group_id` 分配 80/10/10。比例是目標，
不會拆 group 來湊漂亮數字。`dataset_manifest.json` 會列實際 rows/groups/source counts，並
檢查 `leaked_group_ids` 必須為空。

公司圖片真實題目不放在 `data/sft/`，也沒有任何程式會把 `internal_test` 併進 split。

## 不同 model 的 chat template

Canonical 使用 Hugging Face/TRL 建議的 conversational tool-calling contract，所以訓練時
直接傳：

```python
input_ids = tokenizer.apply_chat_template(
    example["messages"],
    tools=example["tools"],
    tokenize=True,
    add_generation_prompt=False,
)
```

但「有 `chat_template`」不等於「支援 tool trajectory」。一般 diagnostic rows 幾乎所有
chat tokenizer 都能 render；agentic rows 還要求 template 認得 `tools`、assistant
`tool_calls` 和 `tool` role。有些模型只支援 single call，有些會忽略 tools，有些直接報錯。

`stats.py` 會真的逐筆套用 `config.TOKENIZER_IDS`，把每個 tokenizer 的 compatible rows、
failure examples、total tokens 和 assistant loss tokens 寫到 `reports/token_stats.json`。
完整的不相容 row/model/reason 另寫到 `reports/template_incompatible.jsonl`，不只保留報表中的
20 筆 examples。
模型換掉時只改 tokenizer 清單並重跑 stats，不重建 corpus。若 template 不支援 tool use，
應換原生 tool-calling model/template，而不是把 model special tokens塞回 dataset。

assistant token 優先使用 template 的 `{% generation %}` mask；template 沒提供時，report 會
明記 `assistant_content_estimate`，避免把估算值說成精確 loss mask。

## 2026-09-14 實跑 sanity check

在目前 raw revisions 上，CCNA 全留且啟用嚴格 NIKA tool-result pairing 後，實際 selected
corpus 是 22,195 rows：

```text
5G-Faults  3,113     Telelogs 2,400     CCNA 11,662
NIKA          20     ToolACE  3,000     When2Call 2,000

Diagnostic 17,175 (77.4%)     Agentic 5,020 (22.6%)
Train 17,758 / Val 2,222 / Test 2,215
```

差異有明確原因：5G 有 50 筆 exact duplicates；CCNA 約一半 raw rows 是 exact duplicate；
NIKA 906 raw incidents 的 submission tiers 是 158 exact、44 partial、453 wrong、251 missing。
158 筆 exact 中，125 筆有無法唯一配對的同名 parallel tool results，13 筆有 fatal/invalid event，
最後 20 筆是沒有猜測 call/result 關係的 clean positive trajectories。另有 8 筆 clean partial
trajectories 被保存，但不進 V1。

Simple sample 後的自然分布是 ToolACE：single 1,447 / parallel 1,291 / multi-turn 147 /
sequential 115；When2Call：clarification 960 / cannot 1,019 / direct 21 / tool-call 0。這些只是抽中
結果，不再是 quota。

Qwen 2.5 template 實測可保留全部 22,195 rows。Llama 3.1 可保留 20,904 rows；不相容的
1,291 rows 全部是 ToolACE single-turn parallel calls，因為官方 template 每個 assistant turn 只
允許一個 tool call。DeepSeek-R1-Distill-Qwen 可保留 17,469 rows，但其 template 會忽略多數 tool
definitions/calls，因此不是這份 mixed tool SFT 的直接替代 template。

## Controlled Llama/Qwen comparison

`controlled.py` 取 `CONTROLLED_TOKENIZER_IDS` 中所有模型的 compatible row-ID 交集。V1 明確
比較 Qwen 2.5 7B Instruct 與 Llama 3.1 8B Instruct；DeepSeek 目前只做 audit，不影響交集。
master corpus 完整保留，公平比較時改讀：

```text
data/sft/experiments/controlled_llama_qwen/
├── train.jsonl       16,725
├── val.jsonl          2,093
├── test.jsonl         2,086
├── excluded.jsonl     1,291
└── manifest.json
```

兩個模型都使用這三份相同 row IDs；訓練時才各自套原生 chat template。`excluded.jsonl` 逐筆
記錄不相容 tokenizer 與原始 error。要更換比較模型時，修改 `CONTROLLED_TOKENIZER_IDS`，
重跑 `stats` 再重跑 `controlled`。

Qwen 的 assistant-content token estimate 顯示 Diagnostic 約佔 94.4%，高於 row ratio 的
77.4%；長 Telelogs CoT、CCNA 全留與嚴格縮減後的 NIKA 都是原因。正式 Mixed SFT 前應在
dataloader 設 sampling weight，或先審核長 reasoning 的安全縮短策略；不要僅看 row counts。

## 測試

```bash
uv run ruff check network_sft tests/test_sft.py
uv run pytest
```

測試全用本地 fixture，不下載 corpus；真正的 model template 相容性由 `stats.py` 驗證。
