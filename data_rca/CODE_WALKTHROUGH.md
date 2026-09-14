# `data_rca` 程式逐行導讀與 filtering 邏輯

每個 Python file 的獨立 function I/O 文件位於 [`docs/rca/`](../docs/rca/README.md)。本文件保留為
逐行與跨模組 filter 補充。

這份文件解釋 `data_rca` 每個 Python 程式的用途、資料流，以及每個有意義程式行／連續語句在做什麼。
行號以目前版本為準。空白行、只有 `}`、`]`、`)` 的結束行不單獨解釋；它們屬於前一個資料結構或
函式呼叫。若程式修改造成行號位移，應以函式名稱及旁邊的程式內容為主。

## 1. 一次看懂整條 pipeline

```text
fetch_sources.py
    │
    ├── NetOpsBench pinned Hugging Face snapshot
    └── ANTA pinned Git revision
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ nika.py          netopsbench.py          anta.py     │
│ trace + GT       ATIF + evaluator GT     fixtures    │
└───────────────┬──────────────────────────────────────┘
                ▼
            schema.py
                ▼
       canonical source JSONL
                ▼
          split.py + render.py
                ▼
       train / val / test messages
                │
                ├── coverage.py
                ├── feasibility.py
                ├── review_sample.jsonl
                └── dataset_manifest.json
```

`build.py` 是 orchestrator。它不下載資料，只呼叫三個 converter、寫 canonical rows、切分、render，
最後產生報表。

## 2. Filter 邏輯總表

| Source | 一開始掃描什麼 | Keep 條件 | 降級條件 | Reject / exclude 條件 |
| --- | --- | --- | --- | --- |
| NIKA | 每個 `ground_truth.json` 的 case directory | anomaly、正好一個 GT cause、cause 在 catalog、有 diagnosis log、有 tool evidence | catalog confirmation tool groups 或 affected side 不完整時，改成 `possible_rca` | 非 anomaly／多 cause、catalog 不認得、缺 log、完全沒有可用 tool evidence |
| NetOpsBench | 每個 `results.jsonl` row + `trace_id` 對應 ATIF | 有 anomalous evaluator GT、fault type 在 catalog、trajectory 存在、至少一筆 preferred tool observation 命中 GT device | 不降級；此 source 只產 confirmed RCA | healthy/negative、缺 GT、未知 fault、缺 trajectory、沒有 affected-device relevant observation |
| ANTA official | unit-test `DATA` dictionary 內的 fixture | case name 含 `failure`，且 expected messages 是非空 literal list | 所有 message 都含 `Expected: ... Actual: ...` 才是 `confirmed_state_mismatch`；否則為 `diagnostic_guidance` | 非 failure case、沒有 messages、不是可靜態解析的 fixture shape |
| ANTA optional export | `input/anta.jsonl` | `expected != observed` 且有 curated diagnoses | 只有 `direct_configuration_check=true` 且恰好一個 diagnosis 才是 state mismatch；其他為 guidance | expected 等於 observed、缺 diagnosis、schema 不合法 |

目前實際結果：

```text
NIKA          scanned 906  -> kept 901, rejected 5
NetOpsBench   scanned 755  -> kept 596, excluded 159
ANTA          extracted 643 -> kept 643, rejected 0
Total canonical rows        -> 2,140
```

NetOpsBench 的 159 筆包含：

```text
77  no_anomalous_ground_truth
82  no_ground_truth_relevant_tool_observation
```

NIKA 的 5 筆包含：

```text
1  missing_diagnosis_log
4  no_tool_evidence
```

## 3. Filter 到底證明了什麼

這是最重要的界線：目前 filter 是 deterministic structural grounding，不是 CLI semantic judge。

### NIKA confirmed 的實際定義

```text
confirmed =
    catalog 每一個 evidence group 都至少命中一個 tool
    AND
    至少一筆 confirmation evidence 位於 faulty device
```

`evidence_groups` 是 AND-of-OR。例如：

```json
{
  "evidence_groups": [
    ["frr_get_bgp_conf", "frr_show_running_config"],
    ["frr_exec"]
  ]
}
```

意思是第一組 config tools 至少出現一個，第二組 `frr_exec` 也必須出現。程式只確認 tool 被呼叫、
有非空 output、affected device 對得上；它沒有解析 output 內容以證明 AS 數字真的錯。因此 label 的
正確性仍來自 `ground_truth.json`，trace filter 只避免完全沒有相關查證行為也被標成 confirmed。

### NetOpsBench confirmed 的實際定義

```text
confirmed =
    results.jsonl 有 evaluator ground_truth.fault_type
    AND fault type 在 catalog
    AND ATIF trajectory 存在
    AND 至少一筆 observation：
          tool 屬於該 fault 的 preferred_tools
          AND arguments/output 含 ground-truth device 名稱
```

介面名稱會影響 evidence ranking，但不是 keep gate。程式沒有解析 MTU、ASN、route 或 ACL 的值來
重新判定 ground truth；RCA label 仍由 evaluator ground truth 提供。

### ANTA certainty 的實際定義

官方 fixture 只有在「所有 failure messages」都符合以下 regex 時才是 state mismatch：

```regex
\bExpected:\s*.+?\s+Actual:\s*.+
```

只要其中一條 message 沒有 explicit expected/actual，就整筆降成 `diagnostic_guidance`。即使是
`confirmed_state_mismatch`，它也只代表 test oracle 確認狀態不符，不代表知道底層 RCA。

## 4. Evidence selection 與截斷

### NIKA

1. 先抽所有成功配對的 `tool_start` / `tool_end`。
2. 只保留 catalog `evidence_groups` 提到的 relevant tools；若完全沒有 relevant tool，fallback
   到所有 tool evidence，但 diagnosis 只能是 possible。
3. affected-device evidence 排前面，再依 tool、command 排序。
4. 最多留下 `MAX_EVIDENCE_ITEMS=6`。
5. 每個 output 最多 `MAX_OUTPUT_CHARS=12000`，超過時保留前 12,000 字並加 truncation marker。

### NetOpsBench

1. 只接受 `step.extra.type == "tool_call"`。
2. 明確忽略 `read_file`；LLM request、assistant message、reasoning、final diagnosis 都不會進 evidence。
3. 沒有 observation output 的 tool call 被跳過。
4. ranking tuple 依序是：GT device 命中、GT interface 命中、preferred tool、trace position。
5. 由大到小選前 6 筆，再恢復原始 trace 順序。
6. 實際 audit 顯示目前 596 rows 的 selected evidence 都至少保留一筆 required relevant observation。

注意：ranking 是字串／metadata heuristic，不是語意重要性模型。關鍵文字若只出現在超長 output 尾端，
也可能因 12,000 字截斷而遺失。

### ANTA

ANTA 每筆只產一個 `expected_observed` evidence：

```json
{
  "test_inputs": {},
  "failure_messages": []
}
```

目前沒有把完整 `eos_data` fixture 塞進 prompt，因為它通常很大；使用的是官方 test oracle 的 failure
messages。這讓資料容易 trace，但它是 fixture-grounded，不是 runtime CLI trace。

## 5. Split filter 與 leakage 防護

切分單位不是 row，而是 `group_id`：

- NIKA：`fault_type + scenario + topology context + affected devices`。
- NetOpsBench：`scenario_id`，所以同一 scenario 的跨模型重跑一定同 split。
- ANTA：`test_name + inputs + messages`，相同 fixture semantics 綁在一起。

每個 `(source, network_domain)` 是一個 stratum。stratum 內 group 先用 seeded SHA-256 排序，再逐 group
放進目前最缺 rows 的 split。這不是 Python `random.shuffle`，所以每次 build 結果固定。

## 6. `config.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1 | 模組說明：所有設定直接改檔案，不使用 argparse。 |
| 3 | 匯入 `Path` 組合跨平台路徑。 |
| 5 | 由 `config.py` 往上兩層取得 repository root。 |
| 6 | 固定 NetOpsBench Hugging Face snapshot revision，避免上游更新後資料漂移。 |
| 7 | 固定 ANTA Git commit。 |
| 9–10 | 指向 repo 既有的 NIKA raw snapshot；build 不會啟動 lab。 |
| 12–19 | 定義 public raw root、NetOpsBench download/extract path、ANTA repo path，以及尚未啟用的 ITU/FaulT-Bench path。 |
| 21–24 | 定義 optional real ANTA export `input/anta.jsonl`。檔案不存在時 reader 產生零筆，不報錯。 |
| 26–30 | 定義 canonical、final SFT 和 rejected JSONL 的輸出位置。 |
| 32–33 | 指向 NIKA 與 NetOpsBench 的人工 catalog。 |
| 35 | split/review 的 deterministic seed。 |
| 36 | train/val/test 目標比例。實際比例會受 group size 影響。 |
| 37 | 每筆最多六個 evidence items。 |
| 38 | 單一 observation 最大字元數。 |
| 39 | review sample 每個 source 抽八筆。 |
| 40–43 | 訓練 messages 的 system prompt，要求區分 RCA、state mismatch、guidance 並只使用 supplied evidence。 |

## 7. `io.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1 | 模組用途說明。 |
| 3–7 | 匯入 SHA-256、JSON、iterator type、Path 和通用 type。 |
| 10–11 | `read_json`：整檔 UTF-8 讀取後 `json.loads`。 |
| 14–16 | `read_jsonl`：檔案不存在就結束 generator，讓 optional input 可以缺席。 |
| 17–20 | 用 `utf-8-sig` 相容 BOM；逐行讀取並跳過空行。 |
| 21–24 | parse 每行；若不是 JSON object 立即報含 path/line 的錯；合法 object yield 出去。 |
| 27–32 | `write_json`：先建 parent directory，再用縮排與 sorted keys 寫 UTF-8 JSON。 |
| 35–42 | `write_jsonl`：逐 row 寫一行 compact JSON，同時計數並回傳寫入筆數。 |
| 45–47 | `stable_hash`：用不可見 separator 串接欄位，SHA-256 後取前 N 字；用於 stable ID、group 和排序。 |

## 8. `fetch_sources.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1–4 | 說明下載與 build 分離，build 本身可離線。 |
| 6 | 匯入 `subprocess` 呼叫 system `tar` 和 `git`。 |
| 8 | 匯入 Hugging Face snapshot downloader。 |
| 10 | 匯入集中設定。 |
| 13–14 | `_extract_netopsbench` 建立 extraction directory。 |
| 15 | 遞迴找所有 `.tar.zst`，排序確保固定處理順序。 |
| 16–20 | 用 `tar --zstd -xf` 解壓；`check=True` 讓任何 archive failure 立即終止。 |
| 23–29 | `fetch` 下載固定 revision 的 `yyyyyt/netopsbench-trace` 到 local raw path。 |
| 30 | 下載後解壓所有 run/composite archives。 |
| 31 | 只有 ANTA root 不存在時才 clone；避免覆蓋既有 snapshot。 |
| 32–41 | partial clone ANTA；`check=True` 確保失敗不會靜默。 |
| 42–45 | checkout 固定 commit，而不是跟著 main 漂移。 |
| 48–50 | 直接執行模組時呼叫 fetch 並顯示 raw root。import 此模組時不會下載。 |

## 9. `nika.py` 逐行說明

### 9.1 Imports 與 parser constants

| 行 | 說明 |
| --- | --- |
| 1–4 | 明示只用 GT 與 tool result，故意不讀 reasoning、final、submission。 |
| 6–10 | `ast/json/re` 處理 log 中混合格式；`Path/Any` 做路徑與 type annotation。 |
| 12–15 | 匯入設定、IO、schema validator 和 taxonomy mapping。 |
| 17 | 從 tool output wrapper 抽 `name='tool_name'`。 |
| 18 | 從 `content='...'` 或 `content="..."` 抽真正 output，支援 escaped characters 和 multiline。 |
| 19 | 列出 NIKA tool arguments 中可能代表 device 的 keys。 |

### 9.2 `_arguments`、`_output`、device/command/type normalizer

| 行 | 說明 |
| --- | --- |
| 22–24 | argument 已經是 dict 就直接使用。 |
| 25–26 | 空 argument 轉成空 dict。 |
| 27–31 | 先嘗試 JSON，再 fallback `ast.literal_eval`；最後只接受 dict。 |
| 34–36 | 非字串 output 轉 deterministic JSON string。 |
| 37–43 | 若符合 wrapper，安全 literal-decode content；失敗則保留原字串並 strip。 |
| 46–48 | 從所有 device keys 收集 device 名稱、去重並用逗號連接。 |
| 51–53 | tool arguments 本身有 command 時優先保留真 command。 |
| 54–71 | 沒有 command 時，將已知 NIKA tools 映射成人類可讀 command。 |
| 72–73 | 未知 tool 以 `tool(key=value)` 呈現，不丟失 arguments。 |
| 76–82 | 定義哪些 tools 的 output 算 config。 |
| 83–87 | config tools -> `config`；service/log tools -> `log`；其他 -> `cli`。 |

### 9.3 `extract_tool_evidence`

| 行 | 說明 |
| --- | --- |
| 90–94 | 初始化尚未配對的 active calls 與完成的 evidence。 |
| 95–97 | 逐行讀 diagnosis log，容忍 encoding error，空行跳過。 |
| 97–100 | JSON parse 失敗直接跳過，不讓 unrelated malformed log 毀掉 case。 |
| 103–111 | 遇到 `tool_start`，保存 tool name 與 normalized arguments 到 active list。 |
| 112 | 只有 `tool_end` 且 active 非空才配對。 |
| 113–117 | 從 output wrapper 找 tool name；找不到時 name 是空字串。 |
| 118–120 | 若 active 中只有一個同名 tool 就精準配對；否則 fallback 第一個 active call。 |
| 121–123 | normalize output；缺 tool name 或空 output 就不成為 evidence。 |
| 124–126 | 超過 12,000 字截前段並標記 truncation。 |
| 127–137 | 組成 canonical evidence，保留 type/device/command/output/tool/raw arguments/truncated。 |
| 136 | 回傳 evidence list。 |

### 9.4 Problem、group 與 platform

| 行 | 說明 |
| --- | --- |
| 141–149 | 從 `task_description` 擷取 network description 與更完整 topology context。 |
| 150–153 | 取得 scenario name，產生不含答案的 generic problem，再附 network context。 |
| 154–161 | group hash 使用 fault、scenario、topology、affected devices，避免近重複跨 split。 |
| 162 | 回傳 problem/group。 |
| 165–167 | P4 fault 或 BMv2 tool -> `bmv2`。 |
| 168–170 | BGP/OSPF/FRR fault 或 FRR tool -> `frr`。 |
| 171 | 其他 NIKA case -> `linux`。 |

### 9.5 `parse_case` 的核心 filter

| 行 | 說明 |
| --- | --- |
| 174–176 | 讀 ground truth，取得 root cause labels。 |
| 177–178 | 只接受 anomaly 且正好一個 cause；其他 reject。 |
| 179–181 | fault type 必須存在人工 catalog。 |
| 182–184 | diagnosis log 必須存在；submission 不在讀取路徑中。 |
| 184–187 | 抽所有 tool evidence、取得 catalog rule、展平 allowed tools、取得 faulty devices。 |
| 189–190 | local helper：evidence device 是否精確出現在 affected set。 |
| 192 | relevant evidence 只靠 tool name 是否在 catalog，不解析 output 語意。 |
| 193–195 | 若 rule 設 `affected_only`，confirmation pool 再限制到 faulty device。 |
| 196–199 | `groups_present` 實作 AND-of-OR：每個 evidence group 都要至少命中一個 tool。 |
| 200–203 | affected side 與 tool groups 同時通過才 confirmed。 |
| 205–211 | confirmed 選 confirmation pool；否則用 relevant/all evidence；無 evidence 才 reject，最多留六筆。 |
| 213–220 | 取得 problem/group/platform 判斷需要的 tool 後，將 evidence 縮成 type/device/command/output。 |
| 221–228 | confirmed 用 catalog reason；possible 改成明示 evidence incomplete 的保守 reason。 |
| 229–236 | diagnosis type 在這裡決定；identify/resolution/verification 全來自 catalog。 |
| 237–238 | 由 fault/tool 推 domain、protocol、精確 platform。 |
| 239–254 | 組精簡 canonical row；metadata 只留 fault type 與原始 case locator。 |
| 255–258 | schema 不合法則 reject；合法才回傳。 |

### 9.6 `build_nika`

| 行 | 說明 |
| --- | --- |
| 261–265 | 讀 catalog、建 output lists、以所有 `ground_truth.json` 的 parent 找 case。 |
| 266–276 | 每個 case 呼叫 parser；已知 parse/schema error 寫入 rejected record，不中止全批。 |
| 277 | 回傳 kept/rejected。 |
| 280–284 | 單獨執行 `python -m data_rca.nika` 時只建 NIKA canonical 與 rejected。 |

## 10. `netopsbench.py` 逐行說明

### 10.1 Observation extractor

| 行 | 說明 |
| --- | --- |
| 1–4 | 明示只轉 tool observations + evaluator GT，不使用 agent diagnosis。 |
| 6–12 | 匯入 JSON/path/types、設定、IO、schema。 |
| 14 | `read_file` 是 agent framework 行為，不是 network evidence，因此忽略。 |
| 17–20 | 從一個 ATIF tool step 取得 observation results 並逐 result 處理。 |
| 21–23 | content 不是字串就跳過。 |
| 24–28 | content 若是 JSON 就 parse；不是 JSON 則原文保留。 |
| 29–34 | 優先抽 `artifact.structured_content`；沒有才保留 parsed wrapper。 |
| 35–36 | 全部 result 都沒內容就回空字串，caller 會略過此 tool call。 |
| 37–39 | 單 result 直接 serialize，多 result serialize list；sorted keys 保持 deterministic。 |
| 40–42 | output 截斷與回傳。 |

### 10.2 `_tool_evidence` 與 problem

| 行 | 說明 |
| --- | --- |
| 45–49 | 逐 ATIF step，只接受 `type=tool_call`。 |
| 50–52 | 抽 tool name；空 name 或 `read_file` 跳過。 |
| 53–56 | 取第一個 tool call arguments；非 dict 正規化為空 dict。 |
| 57–59 | 抽 observation；空 observation 跳過。 |
| 60 | device 優先取 `device`，其次 `src`。 |
| 61–77 | 組 evidence；config/log tools 特別分類，其餘當 cli；保存 trace position 供 ranking。 |
| 78 | 回傳所有 network tool evidence。 |
| 81–89 | 找 initial context message。 |
| 90–105 | 若可 parse symptoms，就依 `anomalies_detected`/observations 產 generic problem。 |
| 106 | 無法 parse 時 fallback generic problem，不讀 agent final 補題目。 |

### 10.3 `parse_netopsbench` 的核心 filter

| 行 | 說明 |
| --- | --- |
| 109–115 | 從 evaluator result 取 GT；沒有 fault type 視為 non-anomalous/unsupported，exclude。 |
| 116–118 | fault type 必須存在 catalog。 |
| 119–123 | 取 GT device/interface、catalog rule、preferred tools。 |
| 125 | 從 ATIF 抽 tool evidence。 |
| 127–136 | ranking key：device hit、interface hit、preferred tool、較晚 trace position。 |
| 138–143 | keep gate 的 relevant pool：preferred tool 且 arguments/output 含 GT device。 |
| 144–145 | relevant pool 空就 exclude，不把只有 agent 結論的 trace 當 grounded RCA。 |
| 146 | 所有 evidence 依 ranking 取前六筆。 |
| 147 | 選完恢復原始 investigation 時序。 |
| 149–155 | 組 location/cause/identify command/scenario/trace。 |
| 156–159 | ranking 用完後，evidence 只保留 type/device/command/output。 |
| 160–186 | 組精簡 SONiC canonical row 與 confirmed RCA；resolution/verification 來自 catalog。 |
| 187–188 | group 只 hash scenario ID，跨模型相同 scenario 不跨 split。 |
| 189–194 | metadata 只保留 fault、scenario、trace 三個 locator。 |
| 195–198 | schema 驗證，不合法就 reject。 |

### 10.4 Join 與 batch build

| 行 | 說明 |
| --- | --- |
| 201–205 | 準備 `trace_id -> ATIF path` map；缺 index 回空 map。 |
| 206–209 | 逐 index JSONL 將 relative path接到 run directory。 |
| 212–214 | snapshot 未解壓時不 crash，回零 rows 加明確 rejection。 |
| 215–217 | 讀 fault catalog，初始化 kept/rejected。 |
| 218–225 | 找 results、建立 trajectory map，逐 evaluator row 找 ATIF path。 |
| 226–229 | trajectory 必須存在，才讀 JSON 並 parse。 |
| 230–239 | join/parse/schema error 寫 source/run/line/trace/reason，繼續下一筆。 |
| 240 | 回傳 kept/rejected。 |

## 11. `anta.py` 逐行說明

### 11.1 Metadata AST extractor

| 行 | 說明 |
| --- | --- |
| 1 | 模組用途。 |
| 3–13 | `ast` 靜態讀 Python source；不 import/execute ANTA；其餘是 parser、IO、schema、taxonomy。 |
| 15 | explicit expected/actual regex，大小寫不敏感。 |
| 18–21 | 遞迴 parse ANTA source Python files。 |
| 22–24 | 只看 top-level、class name 以 `Verify` 開頭的 tests。 |
| 25–34 | 初始化 categories/commands，兼容 annotated assignment 與一般 assignment。 |
| 35–40 | 找 `categories` 並只用 `literal_eval`；動態 expression 無法 parse 就保留空。 |
| 41–47 | 走訪 `commands` AST calls，只抽 literal `command=` 或 `template=`。 |
| 48–53 | 取 docstring 第一段並對 commands 去重。 |
| 54 | 回傳 test-name metadata map。 |
| 58–65 | `_dict_fields` 只接受 AST dict 與 constant keys，轉成容易 lookup 的 map。 |
| 68–74 | `_literal` 安全 literal-eval；任何動態 expression 都 fallback default，不執行程式。 |

### 11.2 Official fixture filter

| 行 | 說明 |
| --- | --- |
| 77–80 | 遞迴找 unit test `test_*.py` 並 parse AST。 |
| 81–88 | 只找 top-level annotated variable `DATA`，且 value 必須是 dict。 |
| 89–95 | 每個 entry key 必須是二元素 tuple `(VerifyClass, constant case name)`。 |
| 96–98 | case name 必須包含 `failure`；success/skipped 不進 dataset。 |
| 99–101 | 從 entry 找 `expected.messages`，使用 safe literal parser。 |
| 102–103 | messages 必須是非空 list。 |
| 104–112 | 保存 test name、case、literal inputs、string messages、source path。 |
| 113 | 回傳所有可靜態追查的 failure fixtures。 |

### 11.3 Official fixture -> canonical

| 行 | 說明 |
| --- | --- |
| 116–122 | 讀 fixture/test metadata；缺 category fallback system，缺 command fallback test name；映射 domain。 |
| 123 | 所有 messages 都符合 expected/actual regex 才 `exact=True`。 |
| 124–129 | exact -> state mismatch；其他 -> guidance，cause wording也不同。 |
| 130 | ID 由 source path + test + case hash，保證 fixture case 唯一。 |
| 131–140 | reason 明確區分 explicit mismatch 與 unknown RCA。 |
| 141–144 | 每個官方 EOS command 都成為 identify step，failure messages 是 expected evidence。 |
| 145–152 | 只有 exact mismatch 產 generic corrective action；guidance resolution 留空。 |
| 153–159 | verification 永遠是重新跑同一 ANTA test，期望 failure messages 消失。 |
| 159–179 | 組 canonical identity/platform/domain/problem/evidence/diagnosis。Evidence output 是 inputs + oracle messages。 |
| 180 | group hash 用 test + inputs + messages。 |
| 181–186 | metadata 只保留 test、fixture case 與 source path。 |
| 187–190 | schema gate。 |

### 11.4 Optional real ANTA export

| 行 | 說明 |
| --- | --- |
| 193–197 | expected 等於 observed 不是 failed check，直接 reject。 |
| 198–200 | 至少需要一個 curated diagnosis。 |
| 201–202 | 只有 direct check + 單 diagnosis 才 state mismatch，否則 guidance。 |
| 203–214 | 正規化 diagnoses；guidance 強制清空 resolution，避免把可能原因當修法。 |
| 215–219 | 使用 supplied ID 或 stable hash，並映射 category/domain。 |
| 220–241 | 組精簡 canonical row；evidence 只保存 supplied expected/observed。 |
| 242–245 | schema gate。 |

### 11.5 `build_anta`

| 行 | 說明 |
| --- | --- |
| 248–252 | 初始化 lists 與官方 source/fixture paths。 |
| 253–265 | paths 存在時抽 metadata/fixtures、逐筆 parse；錯誤記 case/reason。 |
| 266–267 | snapshot 缺失時寫一筆明確 rejection。 |
| 269–273 | 再讀 optional `anta.jsonl`，逐行 parse，錯誤記 line/reason。 |
| 274 | 合併回傳官方 fixtures 與 optional exports。 |

## 12. `taxonomy.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1 | 模組用途。 |
| 4–5 | NIKA fault type 先轉小寫。 |
| 6–21 | 依關鍵字優先順序映射 BGP、OSPF、DNS、DHCP、ARP、ACL/attack、K8s、P4。第一個命中即 return。 |
| 22–25 | route/gateway/subnet/IP -> routing；link/interface/MTU -> interfaces。 |
| 26 | 無已知關鍵字 fallback system/no protocol。 |
| 29–30 | ANTA categories 正規化為小寫與 underscore。 |
| 31–41 | 定義 protocol categories。 |
| 42–45 | BGP/OSPF/ISIS/BFD -> routing；MLAG/VXLAN/EVPN/STP/VLAN -> layer2。 |
| 46–53 | interfaces/connectivity、security/AAA、services/SNMP/logging、generic routing 的 mapping。 |
| 54 | 未映射 category 原樣當 domain；完全沒有 category 才 system。 |

mapping 有順序：一筆 categories 同時含多個值時，較早命中的 protocol 會決定 domain/protocol。

## 13. `schema.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1–3 | 模組用途與 `Any` type。 |
| 5–10 | 唯一合法的四種 diagnosis types。 |
| 11 | 合法 evidence item types。 |
| 12–17 | 合法 grounding types。`config_diff` 目前為 schema compatibility，active v1 sources 未使用。 |
| 20–21 | validator 回傳 error list，而不是第一個錯就停止。 |
| 22–33 | 必填 top-level fields。 |
| 34–36 | 缺 field 時立即回傳，避免後續 indexing 觸發 KeyError。 |
| 38–44 | 核心 identity/text、protocol 與 grounding validation。 |
| 46–56 | evidence 必須是非空 list；每項要是 object、有合法 type、有非空 output。 |
| 58–67 | diagnoses 必須非空；每項 object 且 type 合法。 |
| 68–73 | cause/reason 非空，identify/verification 必須是非空 list。 |
| 74–78 | resolution 必須是 list；只有 diagnostic guidance 允許空 list。 |
| 80 | 回傳全部 errors；空 list 表示通過。 |

## 14. `render.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1–5 | 模組用途、types、system message 設定。 |
| 8–11 | `_evidence_block` 將一個 evidence render 成 device、command、raw output。缺欄位有保守 fallback。 |
| 14–16 | 所有 evidence 用空行串起，放進 user 的 Problem/Evidence 格式。 |
| 19–24 | `_steps` 支援純文字 step，render 成 bullet。 |
| 25–28 | object step 取 command 與 expected evidence/result，再 render。 |
| 31–40 | assistant 先建立 heading，依 diagnosis type 選不 overclaim 的 label。 |
| 41–49 | 多 diagnosis 編號，依序 render cause、why、investigation。 |
| 50–51 | resolution 非空才出現；guidance 不會印假的 resolution section。 |
| 52–53 | 永遠 render verification/expected result，最後串接 sections。 |
| 56–69 | `to_sft` 輸出扁平的 ID/source/platform/domain/protocol/type，加上 system/user/assistant messages。 |

## 15. `split.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1–7 | 模組用途與 imports。 |
| 10–14 | `assign_splits` 建 `(source, domain) -> group -> rows` 巢狀結構。 |
| 15–21 | 每 row 只依 source/domain stratify；diagnosis type 不參與，以免同 group 被拆開。 |
| 23–27 | 每個 stratum 計算 row total、比例 target、目前已分配 row count。 |
| 28–31 | group 依 seed + stratum + group hash deterministic 排序。 |
| 32–36 | 每個 group 放到目前 `target-current` 缺口最大的 split；tie 用 current/name 穩定決定。 |
| 37–40 | 防止同 group 被不同 stratum 重複指定；加入整個 group 的 row 數。 |
| 41 | 回傳 group-to-split map。 |
| 44–53 | `split_rows` 依 assignment 放 rows，再用 seeded ID hash 排定 split 內順序。 |
| 55–58 | 統計每 source 與 diagnosis type 在各 split 的 row 數。 |
| 59–74 | 建 manifest：seed、ratio、rows、groups、source/type 分布。`leaked_group_ids` 固定空，因 assignment 本身按 group。 |
| 75 | 回傳 canonical splits 與 manifest。 |

## 16. `coverage.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1–6 | imports 與固定 platform 顯示順序。 |
| 9–13 | 直接用 top-level `platform`，Counter 統計 domain×platform、platform×protocol、diagnosis types、sources。 |
| 14–16 | 已知 platform 依固定順序，其餘補在後面；domain alphabetic sort。 |
| 17–33 | 將 Counter 轉成 JSON-friendly nested dictionaries。 |
| 36–40 | 建 Markdown table header。 |
| 41–51 | 每 domain 加一列 counts，附「row count 非 unique label」說明。 |
| 54–55 | review function 內 local import stable hash。 |
| 57–61 | 每 source 按 seeded hash 排序並取前 N 筆；這是 deterministic pseudo-random review。 |
| 62 | 回傳合併 sample。 |

## 17. `feasibility.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1–6 | 模組只做 read-only candidate checks。 |
| 9–15 | `_files` 對多個 glob pattern 收集相對路徑、去重、排序；root 不存在回空。 |
| 18–21 | 搜尋 ITU device outputs、GT candidates 和 FaulT-Bench scenario text files。 |
| 22–34 | ITU 有 output+GT 才標 `schema_join_check_required`；只有 output 就 skip；都無則 not downloaded。無論如何 production disabled。 |
| 35–40 | FaulT-Bench 有 scenario files 才 ready for converter，並明示只補 fault diversity。 |
| 41 | 回傳 feasibility report。 |

這個模組不會自動把 candidate 加入 dataset。

## 18. `build.py` 逐行說明

| 行 | 說明 |
| --- | --- |
| 1–13 | 匯入三個 builders、reports、IO、renderer、splitter。 |
| 16–21 | active production sources 只有 NIKA、NetOpsBench、ANTA；順序固定。 |
| 22–23 | 初始化 merged rows、rejections、source counts。 |
| 25–30 | 逐 source build、立即寫 source canonical、合併 rows/rejections、記 kept/rejected。 |
| 32–34 | merged rows 依 ID 排序，寫 all canonical 與 rejected JSONL。 |
| 36–38 | group-aware split，逐 split render 成 Hugging Face messages JSONL。 |
| 40–47 | 統計 rejection reasons，合併精簡 split/source manifest；`llm_generated` 只在 dataset level 記一次。 |
| 48–53 | 產 coverage JSON/Markdown 與獨立 candidate feasibility JSON。 |
| 54–57 | 每 source deterministic 抽 review rows，render 後寫 review sample。 |
| 58–60 | 寫 manifest、`_SUCCESS` marker，並回傳 manifest。 |
| 63–67 | CLI-like module entry，但沒有 argparse；執行 build 並印三個摘要。 |

## 19. `__init__.py`

| 行 | 說明 |
| --- | --- |
| 1 | package docstring。 |
| 2 | 空白；package initialization 沒有 side effect。import `data_rca` 不會下載或 build。 |

## 20. Catalog 不是程式，但會直接影響 filter

### `fault_catalog.json`

每個 NIKA `fault_type` 定義一次：

```text
cause
reason
evidence_groups       # confirmed gate 的 AND-of-OR tool groups
affected_only         # true 時 confirmation pool 只看 faulty device
identify
resolution
verification
```

修改 `evidence_groups` 會直接改變 confirmed/possible 比例。新增太寬鬆的 tool 會 over-confirm；新增太嚴格
的 group 會讓更多 rows 降為 possible。

### `netopsbench_catalog.json`

每個 NetOpsBench fault type 定義：

```text
cause
domain
protocol
preferred_tools       # relevant observation keep gate + evidence ranking
reason
resolution
verification
```

修改 `preferred_tools` 會直接改變 keep/reject 數。這裡不應加入像 `get_topology` 這種幾乎每個 trace
都有、但不能確認 specific fault 的工具，否則 relevant gate 失去意義。

## 21. Tests 在保護什麼

`tests/test_data_rca.py` 驗證：

1. ANTA 只有 direct single diagnosis export 才成為 state mismatch。
2. NIKA 不讀壞掉的 `submission.json`，並在缺 confirmation tool 時降成 possible。
3. NetOpsBench 只用 tool observation + evaluator GT，trajectory final diagnosis 中的 poison text 不會洩漏。
4. coverage 直接使用 canonical `platform`，能把 Linux/FRR/EOS 分開。
5. 同一 group 永遠不跨 train/val/test。

全 repo test 通過不等於 network semantics 已人工驗證；它證明的是 parser、certainty rule、grounding 與
leakage invariants 沒被程式修改破壞。

## 22. 目前 filter 的已知限制

1. **NIKA 不解析 CLI 語意。** `show ip bgp summary` 出現不等於 output 一定證明 catalog cause。
2. **NetOpsBench 使用字串命中。** device 名稱出現在 arguments/output 就算 location grounded，沒有重新判斷 fault 值。
3. **ANTA 用 failure message regex。** `Expected/Actual` 確認 mismatch，不確認 underlying RCA。
4. **截斷只保留前段。** 關鍵內容在 12,000 字之後可能消失。
5. **最多六筆 evidence。** ranking 可能不是人類認為最有診斷力的六筆。
6. **Catalog 文字是人工規則。** resolution/verification 並非從每個 individual trace 推導。
7. **Group leakage 防護依賴 group key。** key 未捕捉到的 semantic near-duplicate 仍可能跨 split。

因此這份 v1 適合稱為 evidence-grounded supervised troubleshooting corpus，不應描述成每一筆都經過
production engineer semantic validation。`review_sample.jsonl` 與 source/domain coverage review 仍是發佈前必要步驟。
