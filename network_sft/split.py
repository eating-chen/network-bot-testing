"""Group-aware, source/category-stratified 80/10/10 split."""

import logging
from collections import Counter, defaultdict
from typing import Any

from network_sft import config
from network_sft.io import jsonl_rows, setup_logging, stable_score, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def assign_groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    strata: defaultdict[tuple[str, str], defaultdict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (row["source"], str(row["metadata"]["category"]))
        strata[key][row["metadata"]["group_id"]].append(row)

    assignments = {}
    for stratum, groups in sorted(strata.items()):
        total = sum(len(items) for items in groups.values())
        target = {split: total * ratio for split, ratio in config.SPLIT_RATIOS.items()}
        current = dict.fromkeys(config.SPLIT_RATIOS, 0)
        ordered = sorted(groups, key=lambda group: stable_score(config.SEED, stratum, group))
        for group in ordered:
            split = max(
                config.SPLIT_RATIOS,
                key=lambda name: (target[name] - current[name], -current[name]),
            )
            assignments[group] = split
            current[split] += len(groups[group])
    return assignments


def run_split() -> dict[str, Any]:
    rows = list(jsonl_rows(config.SELECTED_DIR / "selected.jsonl"))
    assignments = assign_groups(rows)
    splits = {name: [] for name in config.SPLIT_RATIOS}
    for row in rows:
        split = assignments[row["metadata"]["group_id"]]
        row["metadata"]["split"] = split
        splits[split].append(row)
    for split, items in splits.items():
        items.sort(key=lambda row: stable_score(config.SEED, split, row["id"]))
        write_jsonl(config.FINAL_DIR / f"{split}.jsonl", items)

    by_source = Counter((row["source"], row["metadata"]["split"]) for row in rows)
    groups_by_split = Counter(assignments.values())
    manifest = {
        "seed": config.SEED,
        "ratios": config.SPLIT_RATIOS,
        "rows": {split: len(items) for split, items in splits.items()},
        "groups": dict(groups_by_split),
        "rows_by_source": {
            source: {split: by_source[(source, split)] for split in config.SPLIT_RATIOS}
            for source in config.SOURCE_ORDER
        },
        "leaked_group_ids": [],
    }
    seen = {}
    for row in rows:
        group, split = row["metadata"]["group_id"], row["metadata"]["split"]
        if group in seen and seen[group] != split:
            manifest["leaked_group_ids"].append(group)
        seen[group] = split
    if manifest["leaked_group_ids"]:
        raise RuntimeError("group leakage detected")
    write_json(config.FINAL_DIR / "dataset_manifest.json", manifest)
    LOGGER.info("split rows: %s", manifest["rows"])
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_split()
