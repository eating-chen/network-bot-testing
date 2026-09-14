# `data_rca/coverage.py`

## 角色

建立 OS/NOS × domain coverage、protocol 統計及 deterministic review sample。

## Functions

### `coverage_report(rows)`

- Input：canonical row list。
- Output：dictionary，包含 platforms、domains、matrix、protocol counts、diagnosis-type counts、source counts。

### `coverage_markdown(report)`

- Input：`coverage_report()` 的 output。
- Output：Markdown table string。

### `review_rows(rows, per_source, seed)`

- Input：rows、每來源筆數、seed。
- Output：抽樣 canonical row list。
- 方法：每個 source 按 `stable_hash(seed, source, id)` 排序後取前 N；不是 nondeterministic random。

## Input / output 長相

```python
rows = [
    {"source": "nika", "platform": "frr",
     "network_domain": "routing", "protocol": "bgp",
     "diagnoses": [{"type": "confirmed_rca"}]},
    {"source": "anta", "platform": "eos",
     "network_domain": "routing", "protocol": "bgp",
     "diagnoses": [{"type": "diagnostic_guidance"}]},
]

coverage_report(rows)
# -> {
#   "platforms": ["frr", "eos"],
#   "domains": ["routing"],
#   "matrix": {"routing": {"frr": 1, "eos": 1}},
#   "protocol_counts": {"frr": {"bgp": 1}, "eos": {"bgp": 1}},
#   "diagnosis_type_counts": {"confirmed_rca": 1, "diagnostic_guidance": 1},
#   "source_counts": {"anta": 1, "nika": 1},
# }
```
