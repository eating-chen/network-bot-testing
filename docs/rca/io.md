# `data_rca/io.py`

## 角色

提供小型、deterministic 的 JSON/JSONL I/O 和 stable hashing。

## Functions

### `read_json(path)`

- Input：`Path`，UTF-8 JSON file。
- Output：`Any`，`json.loads` 後的 Python object。
- Error：檔案不存在或 JSON 壞掉時直接拋原生 exception。

### `read_jsonl(path)`

- Input：`Path`。
- Output：iterator，每次 yield 一個 `dict[str, Any]`。
- 行為：檔案不存在時是空 iterator；跳過空行；接受 UTF-8 BOM。
- Error：某行不是 JSON object 時，拋含 path 與 line number 的 `ValueError`。

### `write_json(path, value)`

- Input：output `Path` 與 JSON-serializable `value`。
- Output：`None`。
- Side effect：建立 parent directories，寫 pretty/sorted UTF-8 JSON。

### `write_jsonl(path, rows)`

- Input：output `Path` 與 iterable of dictionaries。
- Output：`int`，實際寫入 row 數。
- Side effect：建立 parent directories，每筆一行 sorted-key JSON。

### `stable_hash(*values, length=24)`

- Input：任意 objects，以及輸出長度。
- Output：SHA-256 hex prefix string。
- 用途：ID、group ID、deterministic ordering。相同 inputs 永遠得到相同 output。

## Input / output 長相

```python
read_json(Path("ground_truth.json"))
# -> {"is_anomaly": True, "root_cause_name": ["bgp_asn_misconfig"]}

list(read_jsonl(Path("rows.jsonl")))
# -> [{"id": "row-1"}, {"id": "row-2"}]

write_json(Path("manifest.json"), {"rows": 2140})
# -> None；檔案內容是 formatted JSON

write_jsonl(Path("data.jsonl"), [{"id": "a"}, {"id": "b"}])
# -> 2

stable_hash("bgp", "leaf1")
# -> 24-character hex，例如 "e73c..."
```
