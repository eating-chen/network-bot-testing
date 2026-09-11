# Network SFT 程式導讀

這份文件回答的是「每支程式在做什麼」。資料配方、實跑數量與使用指令請搭配
[`sft_data_guide.md`](sft_data_guide.md) 閱讀。

## 先看整體資料流

```text
config.py
   │ 共用路徑、來源、quota、seed
   ▼
download.py ────────────────> data/sft/raw/
   ▼
normalize.py + nika.py ─────> data/sft/01_normalized/
   │              使用 schema.py 統一格式
   ▼
curate.py ──────────────────> data/sft/02_curated/
   │                         data/sft/03_selected/selected.jsonl
   ▼
split.py ───────────────────> data/sft/network_sft_v1/{train,val,test}.jsonl
   ▼
stats.py ───────────────────> data/sft/reports/token_stats.json

io.py       每個 stage 共用的讀寫、stable ID 與 deterministic score
pipeline.py 依上面順序呼叫全部 stage
```

每一階段都只讀前一階段的輸出，方便定位問題。例如 `selected.jsonl` 的內容不對，先看
`02_curated`；若 `02_curated` 已經不對，再回頭看 `01_normalized`，不必從下載重新猜。

## 一筆資料如何走完整條 pipeline

以 CCNA row 為例：

1. `download.py` 把 Hugging Face snapshot 放到 `raw/ccna`，並留下 revision。
2. `normalize.py::_diagnostic()` 將原始 `question/answer` 轉成 `messages`，此時 category 暫時是
   `unclassified`。
3. `curate.py::_curate_sources()` 先用 `network_topic()` 補 category，再驗證 schema、移除 exact
   duplicate，並把 near duplicates 綁成同一個 `group_id`。
4. `curate.py::ccna_candidate_tier()` 判斷它是否屬於 troubleshooting candidate；通過後才進
   `stratified_sample()` 爭取 3,500 個名額。
5. `split.py::assign_groups()` 以整個 `group_id` 為單位放入 train、val 或 test。
6. `stats.py` 用每個 tokenizer 的原生 `chat_template` 實際 render，檢查內容和 tool schema
   有沒有被吃掉，並計算 token 統計。

## `config.py`：唯一的設定入口

這支檔案故意只有常數，沒有 argparse、YAML 或設定 class。

- `*_DIR`：定義 raw、normalized、curated、selected、final 和 reports 的固定位置。
- `HF_SOURCES`：五個 Hugging Face dataset 的 repo ID 與要下載的副檔名；NIKA 另外使用
  `NIKA_URL` 與 `NIKA_MD5`。
- `SOURCE_ORDER`：quality/dedup 的處理順序。因為 exact dedup 是先到先保留，所以
  network-specific sources 放前面。
- `FULL_SOURCES`：語意是「不做人為抽樣」；仍然會過 validity 和 dedup。
- `SAMPLE_TARGETS`：CCNA、ToolACE、When2Call 的最終 row 目標。
- `CCNA_QUOTAS`、`TOOLACE_QUOTAS`、`WHEN2CALL_QUOTAS`：各 source 的分層抽樣目標。
- `SEED`：所有排序和抽樣都依賴它，讓重跑結果一致。
- `SPLIT_RATIOS`、`NEAR_DUP_JACCARD`、`REVIEW_ROWS`：split、near dedup 與人工 review 設定。
- `TOKENIZER_IDS`：`stats.py` 要實測的 model tokenizer；改模型只需改這裡，不需重建 corpus。
- `normalized_path()`、`curated_path()`：避免各 stage 重複拼錯檔名。

第一版最常改的是 quota、seed、tokenizer 清單。若改 near-duplicate threshold，應重新執行
`curate.py`、`split.py` 和 `stats.py`。

## `io.py`：小型共用工具

這支檔案不含資料策略，只處理可靠、可重現的 IO。

- `setup_logging()`：讓每支可直接執行的 module 使用一致 log 格式。
- `stable_id(*parts)`：把來源欄位做 SHA-256 後截短，用來產生不受 Python process 影響的 ID。
- `stable_score(*parts)`：產生 deterministic 排序鍵。它取代 `random.shuffle()`，因此相同 seed
  與相同輸入會選到完全相同的 rows。
- `utc_now()`：manifest/report 的 UTC timestamp。
- `write_json()`：寫有縮排、適合閱讀的 report 或 manifest。
- `write_jsonl()`：逐筆寫 corpus，回傳寫入筆數。
- `jsonl_rows()`：逐行讀 JSONL，並拒絕不是 JSON object 的 row。
- `data_rows()`：把 Parquet、JSONL、CSV、JSON array 統一成 iterator；Parquet 以 batch 讀取，
  避免一次載入全部資料。
- `source_files()`：找支援的資料檔，刻意排除 Hugging Face `.cache` 內的 metadata。

## `schema.py`：model-neutral 格式與品質底線

### 正規化 helpers

- `text()`：把 `None` 變空字串、字串去除頭尾空白，其他型別穩定地轉成 JSON 字串。
- `json_object()`：確保 tool arguments 最後一定是 Python dict；不是 JSON object 就報錯。
- `_fix_schema()`：把上游常見的 `str/int/list/dict` 等型別名稱修成 JSON Schema 型別。
- `normalize_tool()`：把不同來源的 function 定義統一成 OpenAI/Hugging Face 常見的
  `{"type":"function","function":...}` 外框。
- `make_row()`：只負責組 canonical row，不偷偷做 validation。

### `validate_row()` 驗證什麼

它是 quality filter 的共同入口，依序檢查：

1. `id/source/task_type/messages/tools/metadata` 六個頂層欄位是否齊全。
2. tools 是否為不重名、合法的 object JSON schema。
3. conversation 是否由 system 或 user 開始，system 是否只在第一個位置出現一次。
4. user/system 是否有文字，資料裡是否混入 Llama/Qwen 等 model-specific tokens。
5. `tool_calls` 是否只出現在 assistant，function 名稱是否存在於 tools，arguments 是否為 object。
6. 每個 tool result 的 `tool_call_id` 和 name 是否能配對前面的 call。
7. trajectory 是否完整並以 assistant 結束。

ToolACE/When2Call 可以用「assistant 最後產生 tool call」作為訓練 target，所以特定
`task_type` 允許最後一個 call 尚無 result；NIKA 則必須有完整 observation 和 final answer。

## `download.py`：只負責取得可追溯的 raw snapshot

### `_hf_source()`

先檢查本地是否已有真正的資料檔；有就 reuse，不重複下載。沒有時：

1. 向 Hub 取得目前 commit SHA。
2. 以該 SHA 做 `snapshot_download()`，避免下載期間資料版本漂移。
3. 某些 repo 的 main branch 只有 README，這時 fallback 到 `refs/convert/parquet`。
4. 若仍找不到 CSV/JSON/JSONL/Parquet，直接失敗，不留下看似成功的空 snapshot。

回傳的 revision 和路徑最後會進 `raw/download_manifest.json`。

### `_nika()`

NIKA 不是 HF dataset，因此下載 zip 後：

- 用 publisher 提供的 MD5 驗證檔案；這裡 MD5 是 integrity check，不是安全簽章。
- 解壓前檢查每個 member 都仍位於目的目錄下，避免 zip path traversal。
- 使用 `.extracted` marker，重跑時不重複解壓。

### `run_download()`

依序處理五個 HF sources 和 NIKA，最後集中寫 download manifest。這是此 stage 的公開入口。

## `normalize.py`：六種 raw schema 轉成一種 schema

### Diagnostic sources

`_diagnostic()` 依 source 做最少量 mapping：

- 5G Faults：`instruction → system`、`input → user`、`output → assistant`。
- Telelogs：`q → user`、`CoT → assistant`，並把 `c/RCA` 留在 metadata。
- CCNA：`question → user`、`answer → assistant`。

這一步不判斷 CCNA 是否是 troubleshooting，也不做 dedup；責任留給 `curate.py`。

### ToolACE

- `_tools()`：逐個呼叫 `schema.normalize_tool()`。
- `_openai_messages()`：統一 `human/gpt/function` 和 `user/assistant/tool` role，補缺少的 call ID，
  並把 tool result 對回等待中的 call。
- `_complexity()`：由 conversation 本身計算 calls、results、user turns、單輪最多 calls，然後分成
  `single_tool`、`sequential`、`multi_turn`、`parallel` 或 `no_tool`。
- `normalize_toolace()`：組成 `task_type=tool_calling` 的 canonical row，保留原始 provenance。

分類優先順序很重要：同一輪多 call 先算 parallel；多個 user turn 再算 multi-turn；單一 user
turn 中分次 call 才算 sequential。

### When2Call

- `_decision_answer()`：先解析完整 `<TOOLCALL>...</TOOLCALL>`；否則用明確 regex 分成
  clarification、cannot solve 或 direct answer。
- `normalize_when2call()`：要求一問一答，輸出 `task_type=tool_decision`，並把規則名稱記在
  `label_method`，方便日後替換 heuristic。

這些分類是可追蹤規則，不等於人工 ground truth；真實分布與 quota shortfall 要看
`reports/curation.json`。

### `run_normalize()`

逐 source、逐 row normalize。單筆格式錯誤不會讓整批停止，而是記到
`normalize_rejected.jsonl`；每個 source 的 input/normalized/rejected 數量則寫到
`normalize.json`。NIKA 由 `nika.parse_incident()` 另行處理，submission 分層則另外寫入
`nika_submission_audit.jsonl`。

## `nika.py`：把 incident event log 重建成 tool trajectory

NIKA 最複雜，所以從 `normalize.py` 拆成一支專用 parser。

- `_object()`：讀入必須為 JSON object 的 ground truth、submission 和 session metadata。
- `_set()`、`evaluate_submission()`：以原始、大小寫敏感的 set 比較 anomaly flag、faulty devices
  和 root cause，分成 `exact_correct`、`partial_correct`、`wrong`，不把近似 label 偷算成正確。
- `audit_incident()`：把 truth、submission、三項 match 狀態與交集寫成可追蹤的 audit record。
- `_arguments()`：tool input 先試 JSON，再兼容 Python literal；最後必須為 object。
- `_output()`：把 log 裡包裝過的 tool output 還原成純文字。
- `_json_type()`、`_tools()`：NIKA log 沒有完整統一 tool schema，因此從實際 calls 推導參數名稱、
  型別與 required 欄位。
- `_events()`：逐行解析 event JSON，並在錯誤中保留 line number。

`parse_incident()` 是核心 state machine：

```text
llm_end(tool_calls)  暫存 assistant content
tool_start           建 call、配置 call_id、記錄觀察到的參數
tool_end             寫 assistant tool_calls，再寫配對的 tool result
llm_end(final)       寫最終 assistant diagnosis
```

遇到 error event、invalid call、未配對 result、工具還沒跑完就回答，或結尾沒有 assistant final
answer，都直接 reject。tool result 必須唯一對應到 active call；找不到 tool name 或 parallel calls
中有同名工具時，不再預設配給第一個 call，而是回報 `ambiguous_tool_result`。最後 metadata 保留
scenario、issue、failure、incident、faulty device 和 root cause，供後續 grouping 與分析使用。

NIKA submission 分層定義如下：

```text
Tier A / exact_correct
  anomaly、device set、root-cause set 全部 exact match
  → 通過 parser 後進 nika.jsonl，再由 curate 做完整 schema quality filter

Tier B / partial_correct
  anomaly exact，device/root-cause 其中一組 exact，另一組至少有交集
  → clean trajectory 另存 nika_partial.jsonl，不進 V1 selected corpus

Tier C / wrong
  其餘情況
  → 不解析成 positive trajectory，但 truth/submission 比較仍留在 audit
```

沒有 `submission.json` 的 incident 獨立記為 `missing_submission`，不混進 wrong。所有 906 筆的
去向可在 `reports/nika_submission_audit.jsonl` 逐筆追查；raw log 永遠保留。

## `curate.py`：quality、dedup/group 與 sampling

這是策略最集中的一支程式，但中間結果仍拆成 curated 和 selected 兩層。

### CCNA filter 與 topic

- `normalized_text()`：NFKC、case folding，只保留英數 token，供 dedup 使用。
- `network_topic()`：計算各 topic keyword 命中數；完全沒命中時放入
  `management_cli_misc`。
- `is_ccna_troubleshooting()`：嚴格 troubleshooting regex 的布林 helper，主要供測試與檢查。
- `ccna_candidate_tier()`：先收 strict troubleshooting；沒有 strict keyword 但有 operational/CLI
  語意時收進 high-recall tier；兩者都沒有就不參與 3,500 抽樣。

### Exact 與 near duplicate

- `_fingerprint()`：只對語意內容 `messages + tools` 做標準化 hash，不把不同 source ID 誤當差異。
- `_shingles()`：把第一個 user question 切成連續 3-token shingles。
- `_signature()`：用 12 個固定 salts 建近似 MinHash signature。
- `_near_groups()`：每 3 個 signature 值組一個 LSH band 找 candidates，再用真正的 Jaccard
  `>= 0.86` 決定是否連結。near duplicate 不刪除，只共用 `group_id`，確保 split 不洩漏。
- `_find()`：union-find 的 root lookup，用來把連鎖相似 rows 合成同一群。

### `_curate_sources()`

按 `SOURCE_ORDER` 處理，因此跨來源 exact duplicate 會保留先出現的 network-specific row。每筆
先分類、validate、exact dedup，再對同 source/category 做 near grouping。NIKA 不跑文字 near
dedup，而以 `scenario + failure_type + incident` 建 group。

輸出是每個 source 的 `02_curated/*.jsonl`、`quality_rejected.jsonl` 和初步 report。

### `stratified_sample()` 與 `run_curate()`

`stratified_sample()` 先用 `stable_score(seed, source, id)` 做可重現排序，再依 quota 取各 category；
某類不足時不複製資料，而從尚未選中的其他類別補滿總 target。

`run_curate()` 的選擇順序就是 V1 配方：完整加入 5G、Telelogs、valid NIKA，對 CCNA、ToolACE、
When2Call 分層抽樣。最後輸出：

- `03_selected/selected.jsonl`：要進 split 的完整 mixed corpus。
- `reports/ccna_review_200.jsonl`：固定且可重現的 200 筆人工 review 樣本。
- `reports/curation.json`：candidate、selected、category 和 quota shortfall 統計。

## `split.py`：group-aware 80/10/10

`assign_groups()` 先建立兩層結構：

```text
(source, category)
└── group_id
    └── 此 group 的所有 rows
```

對每個 stratum 分別算 80/10/10 目標。每次配置一整個 group 到目前缺口最大的 split，因此不會
拆 group；代價是 row 數不一定剛好等於理論比例。排序仍使用 stable score，結果可重現。

`run_split()` 將 split 寫回 `metadata.split`，輸出三份 JSONL，然後再次掃描所有 row 驗證同一
`group_id` 沒出現在兩個 split。發現 leakage 會直接 raise，而不是只在 report 警告。

## `stats.py`：真的套 template，不只檢查欄位存在

- `_summary()`：回報 rows、total、p50/p90/p95/p99/max token length。
- `apply_template()`：將 canonical `messages` 和非空 `tools` 傳給 tokenizer；corpus 本身不加
  model special tokens。
- `_check_render()`：以文字模式 render，再確認每段 message content、每個 called function、tool
  definition 與 description probe 都存在。這可抓出「函式沒報錯，但 template 默默忽略 tools」的
  情況。
- `_assistant_tokens()`：優先要求 template 回傳 `{% generation %}` assistant mask；不支援時只
  token 化 assistant content/tool calls 做 estimate，並把採用的方法寫入 report。
- `_one_tokenizer()`：逐 row 驗證、tokenize、累積 task/source 統計、model max length 超限數、
  diagnostic/agentic token share 和最多 20 筆 failure examples。
- `run_stats()`：讀回 train/val/test 的全部 rows，依 `TOKENIZER_IDS` 逐個載入 tokenizer；單一模型
  載入失敗不會遮掉其他模型結果，每測完一個就更新 `token_stats.json`。

所以 model compatibility 的判準不是「模型名稱看起來支援聊天」，而是該 tokenizer template
實際保留這份 corpus 所需的 messages、tools、tool calls 和 tool results。

## `pipeline.py` 與 `__init__.py`

`pipeline.py::run_pipeline()` 沒有額外邏輯，只把公開 stage 入口照鎖定順序串起來：

```python
run_download()
run_normalize()
run_curate()
run_split()
run_stats()
```

因此 debug 時建議單獨執行有問題的 stage；確定整條流程可重跑後，才使用 pipeline 一次跑完。
`__init__.py` 只提供 package 說明和版本號。

## `tests/test_sft.py`：用小 fixture 鎖住關鍵行為

測試不下載完整 dataset，而是建立幾筆最小資料，確認最容易回歸的邊界：

- canonical row 能被 Hugging Face/TRL conversational dataset 接受。
- When2Call 的 tool-call markup 會變成 structured call，文字回答的規則分類可預期。
- ToolACE complexity 確實由 messages 計算，terminal tool call 也是合法 SFT target。
- CCNA troubleshooting/topic filter 和 quota redistribution 不會因重構改變。
- 同一個 group 永遠不會跨 split。
- NIKA 只接受正確且完整的 trace。
- template boundary 直接傳 messages/tools，並兼容 Transformers v5 的 mapping result。

這些是單元測試；真正的 dataset 數量、人工品質和模型 template 相容性仍以 reports 與人工
review 為準。

## 最實用的 trace 方法

假設你看到 final row `id=ccna_xxx` 有問題，可依序查：

```bash
rg '"id": "ccna_xxx"' data/sft/network_sft_v1
rg '"id": "ccna_xxx"' data/sft/03_selected/selected.jsonl
rg '"id": "ccna_xxx"' data/sft/02_curated/ccna.jsonl
rg '"id": "ccna_xxx"' data/sft/01_normalized/ccna.jsonl
```

若 row 消失，查 rejection report：

```bash
rg 'ccna_xxx' data/sft/reports/quality_rejected.jsonl
rg 'ccna' data/sft/reports/normalize_rejected.jsonl
```

要理解「為何沒抽到」時，先看 row 的 `metadata.category/candidate_tier`，再對照
`curation.json` 的 available、selected 與 quota shortfalls。要理解「為何被分到 test」時，看
`group_id`；split 是 deterministic allocation，不是人工指定。

## 建議閱讀順序

第一次讀 code 不需要從最長的檔案開始。建議依序看：

1. `config.py`：先知道所有資料和數量設定。
2. `pipeline.py`：記住 stage 順序。
3. `schema.py`：理解最終 row 的契約。
4. `normalize.py`：看六種來源如何進入契約。
5. `nika.py`：只在要追 NIKA trajectory 時深入。
6. `curate.py`：理解 filter、dedup 和 sampling 決策。
7. `split.py`：理解 leakage protection。
8. `stats.py`：理解模型 template 與 token report。
9. `io.py`：遇到檔案、ID 或 reproducibility 問題時再查。
10. `tests/test_sft.py`：從最小案例確認你對行為的理解。
