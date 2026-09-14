# `data_rca/feasibility.py`

## 角色

只讀檢查尚未 production-enable 的 ITU Track B 與 FaulT-Bench artifacts。它不建立 training rows。

## Functions

### `_files(root, patterns)`

- Input：root `Path` 與 glob patterns tuple。
- Output：排序、去重後的 relative path list；root 不存在回空 list。

### `source_feasibility()`

- Input：沒有參數；使用 config candidate paths。
- Output：包含 `itu_track_b` 與 `faultbench` 的 status dictionary。
- ITU status：
  - output + GT candidate → `schema_join_check_required`
  - 只有 output → `skip_missing_ground_truth`
  - 都沒有 → `not_downloaded`
- FaulT-Bench：有 `.txt` scenario 才 `ready_for_converter`。
- 兩者 `production_enabled` 都固定 false。

## Output 長相

```json
{
  "itu_track_b": {
    "status": "not_downloaded",
    "device_output_artifacts": [],
    "ground_truth_candidates": [],
    "production_enabled": false
  },
  "faultbench": {
    "status": "not_downloaded",
    "scenario_file_count": 0,
    "production_enabled": false,
    "role": "fault diversity, not NOS diversity"
  }
}
```
