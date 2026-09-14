# `data_rca/split.py`

## 角色與運作方式

做 deterministic、group-aware、source/domain-stratified train/val/test split。分配單位是 group，不是 row。

## Functions

### `assign_splits(rows)`

- Input：canonical row list；每筆至少需要 `source`、`network_domain`、`group_id`。
- Output：`dict[group_id, split_name]`。

流程：

1. 用 `(source, network_domain)` 建 strata。
2. 每個 stratum 裡按 group 收 rows。
3. 用 `stable_hash(seed, stratum, group)` 排 group。
4. 將整個 group 放進目前離 ratio target 最遠的 split。
5. 同 group 若被要求放不同 split，拋 `ValueError`。

Diagnosis type 不參與 strata，避免同一近重複 group 因 confirmed/possible 差異被拆開。

### `split_rows(rows)`

- Input：完整 canonical row list。
- Output：tuple：
  1. `{"train": [...], "val": [...], "test": [...]}`
  2. split manifest dictionary。
- Manifest 包含 rows/groups/source/diagnosis-type counts。
- Split 內 row 順序也由 seed + ID hash 固定。

## Input / output 長相

```python
rows = [
    {"id": "a1", "source": "nika", "network_domain": "routing",
     "group_id": "group-a", "diagnoses": [{"type": "confirmed_rca"}]},
    {"id": "a2", "source": "nika", "network_domain": "routing",
     "group_id": "group-a", "diagnoses": [{"type": "possible_rca"}]},
    {"id": "b1", "source": "anta", "network_domain": "interfaces",
     "group_id": "group-b", "diagnoses": [{"type": "diagnostic_guidance"}]},
]

assign_splits(rows)
# -> {"group-a": "train", "group-b": "val"}  # split 名稱由 seed/全資料決定

splits, manifest = split_rows(rows)
# splits -> {"train": [a1, a2], "val": [b1], "test": []}
# manifest -> {
#   "seed": 42,
#   "rows": {"train": 2, "val": 1, "test": 0},
#   "groups": {"train": 1, "val": 1},
#   "rows_by_source": {...},
#   "rows_by_diagnosis_type": {...},
# }
```

重點是 `a1/a2` 共用 `group-a`，一定一起移動。
