"""Download raw SFT sources into data/sft/raw; never touches CPT data."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from network_sft import config
from network_sft.io import setup_logging, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def _download_hf(name: str, repo_id: str, patterns: list[str]) -> dict[str, Any]:
    destination = config.RAW_DIR / name
    if list(destination.rglob("*.parquet")) or list(destination.rglob("*.jsonl")):
        LOGGER.info("沿用既有 Hugging Face snapshot: %s", destination)
        return {"dataset_id": repo_id, "path": str(destination), "reused": True}

    revision = HfApi().dataset_info(repo_id).sha
    LOGGER.info("下載 %s@%s", repo_id, revision)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=destination,
    )
    return {
        "dataset_id": repo_id,
        "revision": revision,
        "path": str(destination),
        "reused": False,
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the upstream published checksum only.
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_nika() -> dict[str, Any]:
    archive = config.RAW_DIR / "nika_traces.zip"
    destination = config.RAW_DIR / "nika"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".zip.part")
        LOGGER.info("下載 NIKA traces")
        request = urllib.request.Request(config.NIKA_URL, headers={"User-Agent": "network-sft/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    if _md5(archive) != config.NIKA_MD5:
        raise RuntimeError(f"NIKA checksum mismatch: {archive}")

    marker = destination / ".extracted"
    if not marker.exists():
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise RuntimeError(f"Unsafe zip member: {member.filename}")
            bundle.extractall(destination)
        marker.write_text("ok\n", encoding="utf-8")
    return {"url": config.NIKA_URL, "md5": config.NIKA_MD5, "path": str(destination)}


def run_download() -> dict[str, Any]:
    manifest = {
        "stage": "sft_download",
        "created_at": utc_now(),
        "general_sft": _download_hf(
            "general_sft",
            config.GENERAL_DATASET_ID,
            [f"data/{config.GENERAL_SOURCE_SPLIT}-*.parquet", "README.md"],
        ),
        "functiongemma_network": _download_hf(
            "functiongemma_network",
            config.FUNCTIONGEMMA_DATASET_ID,
            [f"data/{config.FUNCTIONGEMMA_CONFIG}/*.jsonl", "README.md"],
        ),
        "nika": _download_nika(),
    }
    write_json(config.RAW_DIR / "download_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_download()
