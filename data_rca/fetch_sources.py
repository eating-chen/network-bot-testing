"""Download pinned public snapshots and extract NetOpsBench archives.

This is intentionally a separate explicit step.  ``data_rca.build`` is offline.
"""

import subprocess

from huggingface_hub import snapshot_download

from data_rca import config


def _extract_netopsbench() -> None:
    config.NETOPSBENCH_EXTRACTED.mkdir(parents=True, exist_ok=True)
    for archive in sorted(config.NETOPSBENCH_ROOT.rglob("*.tar.zst")):
        # Every public archive has a unique top-level run/composite directory.
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(config.NETOPSBENCH_EXTRACTED)],
            check=True,
        )


def fetch() -> None:
    snapshot_download(
        repo_id="yyyyyt/netopsbench-trace",
        repo_type="dataset",
        revision=config.NETOPSBENCH_REVISION,
        local_dir=config.NETOPSBENCH_ROOT,
    )
    _extract_netopsbench()
    if not config.ANTA_ROOT.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/aristanetworks/anta.git",
                str(config.ANTA_ROOT),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(config.ANTA_ROOT), "checkout", config.ANTA_REVISION],
            check=True,
        )


if __name__ == "__main__":
    fetch()
    print("Public RCA sources are ready under", config.RAW_DIR)
