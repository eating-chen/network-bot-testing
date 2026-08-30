"""CCDS-style dataset entry point: run the complete preprocessing DAG."""

from __future__ import annotations

import logging

from network_cpt.clean import run_clean
from network_cpt.download import run_download
from network_cpt.io import setup_logging
from network_cpt.parse import run_parse
from network_cpt.split import run_split

LOGGER = logging.getLogger(__name__)


def make_dataset() -> None:
    LOGGER.info("Network CPT v1 pipeline 開始")
    run_download()
    run_parse()
    run_clean()
    run_split()
    LOGGER.info("Network CPT v1 pipeline 完成")


if __name__ == "__main__":
    setup_logging()
    make_dataset()
