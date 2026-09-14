"""Deterministic, group-aware split for canonical RCA rows."""

from collections import Counter, defaultdict
from typing import Any

from data_rca import config
from data_rca.io import stable_hash


def assign_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Return group_id -> split while stratifying by source and broad category."""
    strata: defaultdict[tuple[str, str], defaultdict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        category = row["network_domain"]
        # One near-duplicate group can contain both confirmed and possible rows
        # depending on which checks a particular trace collected.  Excluding
        # diagnosis type here keeps that whole group in one split.
        stratum = (row["source"], category)
        strata[stratum][row["group_id"]].append(row)

    assignments: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        total = sum(len(items) for items in groups.values())
        targets = {name: ratio * total for name, ratio in config.SPLIT_RATIOS.items()}
        current = dict.fromkeys(config.SPLIT_RATIOS, 0)
        ordered_groups = sorted(
            groups,
            key=lambda group: stable_hash(config.SEED, stratum, group, length=64),
        )
        for group in ordered_groups:
            split = max(
                config.SPLIT_RATIOS,
                key=lambda name: (targets[name] - current[name], -current[name], name),
            )
            if group in assignments and assignments[group] != split:
                raise ValueError(f"group {group!r} appears in multiple strata")
            assignments[group] = split
            current[split] += len(groups[group])
    return assignments


def split_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    assignments = assign_splits(rows)
    result = {name: [] for name in config.SPLIT_RATIOS}
    for row in rows:
        split = assignments[row["group_id"]]
        result[split].append(row)
    for split, items in result.items():
        items.sort(key=lambda row: stable_hash(config.SEED, split, row["id"], length=64))

    source_counts = Counter((row["source"], assignments[row["group_id"]]) for row in rows)
    task_counts = Counter(
        (row["diagnoses"][0]["type"], assignments[row["group_id"]]) for row in rows
    )
    manifest = {
        "seed": config.SEED,
        "ratios": config.SPLIT_RATIOS,
        "canonical_rows": len(rows),
        "rows": {name: len(items) for name, items in result.items()},
        "groups": dict(Counter(assignments.values())),
        "rows_by_source": {
            source: {split: source_counts[(source, split)] for split in config.SPLIT_RATIOS}
            for source in sorted({row["source"] for row in rows})
        },
        "rows_by_diagnosis_type": {
            task: {split: task_counts[(task, split)] for split in config.SPLIT_RATIOS}
            for task in sorted({row["diagnoses"][0]["type"] for row in rows})
        },
        "leaked_group_ids": [],
    }
    return result, manifest
