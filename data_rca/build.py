"""Run all available RCA sources, render SFT messages, and create splits."""

from collections import Counter

from data_rca import config
from data_rca.anta import build_anta
from data_rca.coverage import coverage_markdown, coverage_report, review_rows
from data_rca.feasibility import source_feasibility
from data_rca.io import write_json, write_jsonl
from data_rca.netopsbench import build_netopsbench
from data_rca.nika import build_nika
from data_rca.render import to_sft
from data_rca.split import split_rows


def build() -> dict:
    builders = {
        "nika": build_nika,
        "netopsbench": build_netopsbench,
        "anta": build_anta,
    }
    all_rows, all_rejected = [], []
    source_counts = {}

    for source, builder in builders.items():
        rows, rejected = builder()
        write_jsonl(config.CANONICAL_DIR / f"{source}.jsonl", rows)
        all_rows.extend(rows)
        all_rejected.extend(rejected)
        source_counts[source] = {"kept": len(rows), "rejected": len(rejected)}

    all_rows.sort(key=lambda row: row["id"])
    write_jsonl(config.CANONICAL_DIR / "all.jsonl", all_rows)
    write_jsonl(config.REJECTED_PATH, all_rejected)

    canonical_splits, split_manifest = split_rows(all_rows)
    for split, rows in canonical_splits.items():
        write_jsonl(config.FINAL_DIR / f"{split}.jsonl", (to_sft(row) for row in rows))

    rejection_reasons = Counter(row["reason"] for row in all_rejected)
    manifest = {
        **split_manifest,
        "sources": source_counts,
        "rejected_rows": len(all_rejected),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "llm_generated": False,
    }
    coverage = coverage_report(all_rows)
    write_json(config.FINAL_DIR / "coverage_matrix.json", coverage)
    (config.FINAL_DIR / "coverage_matrix.md").write_text(
        coverage_markdown(coverage), encoding="utf-8"
    )
    write_json(config.FINAL_DIR / "source_feasibility.json", source_feasibility())
    write_jsonl(
        config.FINAL_DIR / "review_sample.jsonl",
        (to_sft(row) for row in review_rows(all_rows, config.REVIEW_ROWS_PER_SOURCE, config.SEED)),
    )
    write_json(config.FINAL_DIR / "dataset_manifest.json", manifest)
    (config.FINAL_DIR / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = build()
    print("sources:", result["sources"])
    print("splits:", result["rows"])
    print("rejected:", result["rejected_rows"])
