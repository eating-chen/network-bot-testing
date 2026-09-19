"""Deterministic split that never separates rows sharing a group_id."""

from collections import Counter, defaultdict

from data_sft_rca import config
from data_sft_rca.io_utils import stable_id


def split_rows(rows: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    strata: dict[tuple[str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        # A FaulT-Bench base case and its wrong-device/wrong-cause variants can
        # have different labels but must remain together, so source is the only
        # safe stratum above group_id.
        strata[(row["source"],)][row["group_id"]].append(row)
    assignments = {}
    for stratum, groups in sorted(strata.items()):
        total = sum(len(items) for items in groups.values())
        targets = {name: ratio * total for name, ratio in config.SPLIT_RATIOS.items()}
        current = dict.fromkeys(config.SPLIT_RATIOS, 0)
        ordered = sorted(
            groups, key=lambda group: stable_id(config.SEED, stratum, group, length=64)
        )
        for group in ordered:
            split = max(
                config.SPLIT_RATIOS,
                key=lambda name: (targets[name] - current[name], -current[name], name),
            )
            if group in assignments and assignments[group] != split:
                raise ValueError(f"group {group} appears in more than one stratum")
            assignments[group] = split
            current[split] += len(groups[group])
    result = {name: [] for name in config.SPLIT_RATIOS}
    for row in rows:
        result[assignments[row["group_id"]]].append(row)
    for split, items in result.items():
        items.sort(key=lambda row: stable_id(config.SEED, split, row["id"], length=64))
    return result, {
        "seed": config.SEED,
        "ratios": config.SPLIT_RATIOS,
        "rows": {name: len(items) for name, items in result.items()},
        "groups": dict(Counter(assignments.values())),
        "leaked_group_ids": [],
    }
