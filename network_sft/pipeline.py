"""Run SFT data preparation from download through final Parquet files."""

from __future__ import annotations

from network_sft import config
from network_sft.download import run_download
from network_sft.functiongemma import run_functiongemma
from network_sft.general import run_general
from network_sft.io import setup_logging
from network_sft.nika import run_nika
from network_sft.split_merge import run_split_merge
from network_sft.stats import run_stats
from network_sft.validate import run_validate


def run_pipeline() -> None:
    run_download()
    run_general()
    run_functiongemma()
    run_nika()
    run_validate()
    run_split_merge()
    run_validate(
        [config.CANONICAL_DIR / f"{split}.jsonl" for split in ("train", "validation", "test")]
    )
    run_stats()


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
