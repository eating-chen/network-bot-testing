"""Run V1 in the same order shown in the plan."""

from network_sft.curate import run_curate
from network_sft.download import run_download
from network_sft.io import setup_logging
from network_sft.normalize import run_normalize
from network_sft.split import run_split
from network_sft.stats import run_stats


def run_pipeline() -> None:
    run_download()
    run_normalize()
    run_curate()  # quality -> exact/near dedup + group_id -> sampling
    run_split()
    run_stats()


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
