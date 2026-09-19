"""Fetch only the source files used by v2.  Edit config.py; there is no CLI layer."""

import subprocess
from pathlib import Path

import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download

from data_sft_rca import config


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _sparse_clone(url: str, destination: Path, revision: str, patterns: list[str]) -> None:
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                url,
                str(destination),
            ]
        )
    _run(["git", "-C", str(destination), "sparse-checkout", "set", "--no-cone", *patterns])
    _run(["git", "-C", str(destination), "checkout", revision])


def fetch_cloudops() -> None:
    patterns = []
    for category in ("service", "performance", "infrastructure"):
        patterns.extend(
            [
                f"/benchmark/*/{category}/*/metadata.json",
                f"/benchmark/*/{category}/*/tool_cache.json",
                f"/process-label/*/{category}/*/milestone.json",
            ]
        )
    _sparse_clone(
        "https://github.com/LLM4Ops/Cloud-OpsBench.git",
        config.CLOUDOPS_ROOT,
        config.CLOUDOPS_REVISION,
        patterns,
    )


def fetch_faultbench() -> None:
    _sparse_clone(
        "https://github.com/Overlxrd-uwu/FaulT-Bench.git",
        config.FAULTBENCH_ROOT,
        config.FAULTBENCH_REVISION,
        ["/dataset/"],
    )


def fetch_rcaeval() -> None:
    config.RCAEVAL_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = hf_hub_download(
        "phamquiluan/RCAEval",
        "cases.parquet",
        repo_type="dataset",
        revision=config.RCAEVAL_REVISION,
        local_dir=config.RCAEVAL_ROOT,
    )
    rows = parquet.read_table(index_path).to_pylist()
    selected = [row for row in rows if str(row["fault"]).lower() in config.RCAEVAL_FAULTS]
    for number, row in enumerate(selected, 1):
        filename = f"{row['case']}/metrics.parquet"
        hf_hub_download(
            "phamquiluan/RCAEval",
            filename,
            repo_type="dataset",
            revision=config.RCAEVAL_REVISION,
            local_dir=config.RCAEVAL_ROOT,
        )
        if number % 25 == 0 or number == len(selected):
            print(f"RCAEval metrics: {number}/{len(selected)}")


def fetch() -> None:
    fetch_cloudops()
    fetch_faultbench()
    fetch_rcaeval()


if __name__ == "__main__":
    fetch()
