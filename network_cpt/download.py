"""把所有上游原始檔完整下載到 ``data/raw``。"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from network_cpt import config
from network_cpt.io import setup_logging, sha256_file, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def _run(command: list[str], cwd: Path | None = None) -> str:
    """執行一條 shell command，失敗時直接停止 pipeline。"""
    LOGGER.info("執行: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _read_manifest(path: Path) -> dict[str, Any] | None:
    """有 manifest 就沿用，沒有就回傳 None 並繼續下載。"""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def download_huggingface_dataset(name: str, dataset_id: str) -> dict[str, Any]:
    """完整下載一個 Hugging Face dataset repository。"""
    destination = config.RAW_DIR / "huggingface" / name
    manifest_path = destination.parent / f"{name}.manifest.json"

    existing = _read_manifest(manifest_path) if destination.exists() else None
    if existing is not None:
        LOGGER.info("沿用既有 Hugging Face snapshot: %s", destination)
        return existing

    revision = HfApi().dataset_info(dataset_id).sha
    LOGGER.info("下載 Hugging Face dataset: %s@%s", dataset_id, revision)
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
    )

    files = [
        path
        for path in destination.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(destination).parts
    ]
    manifest = {
        "source": name,
        "kind": "huggingface-repository",
        "dataset_id": dataset_id,
        "revision": revision,
        "path": str(destination.relative_to(config.PROJECT_ROOT)),
        "files": len(files),
        "captured_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    return manifest


def download_git_repository(name: str, url: str) -> dict[str, Any]:
    """用 shallow clone 下載 Git repository。"""
    destination = config.RAW_DIR / "repos" / name
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", url, str(destination)])
    elif not (destination / ".git").is_dir():
        raise RuntimeError(f"{destination} 已存在但不是 Git repository")
    else:
        LOGGER.info("沿用既有 Git snapshot: %s", destination)

    commit = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    manifest = {
        "source": name,
        "kind": "git",
        "url": url,
        "commit": commit,
        "path": str(destination.relative_to(config.PROJECT_ROOT)),
        "captured_at": utc_now(),
    }
    write_json(destination.parent / f"{name}.manifest.json", manifest)
    return manifest


def _download_file(url: str, destination: Path) -> None:
    """先寫暫存檔，成功後才改成正式檔名。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "network-cpt-data/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_kea() -> dict[str, Any]:
    """下載固定版本的 Kea Administrator Reference Manual PDF。"""
    destination = config.RAW_DIR / "documents" / "kea" / f"{config.KEA_VERSION}.pdf"
    if not destination.exists():
        LOGGER.info("下載 Kea ARM PDF: %s", config.KEA_PDF_URL)
        _download_file(config.KEA_PDF_URL, destination)
    else:
        LOGGER.info("沿用既有 Kea PDF: %s", destination)

    manifest = {
        "source": "kea",
        "kind": "pdf",
        "version": config.KEA_VERSION,
        "url": config.KEA_PDF_URL,
        "sha256": sha256_file(destination),
        "path": str(destination.relative_to(config.PROJECT_ROOT)),
        "captured_at": utc_now(),
    }
    write_json(destination.with_name("manifest.json"), manifest)
    return manifest


def run_download() -> dict[str, Any]:
    """依照 config 的來源順序完整下載所有 raw data。"""
    config.validate_config()
    snapshots: dict[str, Any] = {}

    for source in config.ENABLED_SOURCES:
        LOGGER.info("處理下載來源: %s", source)
        if source in config.HUGGINGFACE_DATASETS:
            snapshots[source] = download_huggingface_dataset(
                source,
                config.HUGGINGFACE_DATASETS[source],
            )
        elif source == "kea":
            snapshots[source] = download_kea()
        else:
            snapshots[source] = download_git_repository(
                source,
                config.GIT_REPOSITORIES[source],
            )

    manifest = {
        "stage": "download",
        "created_at": utc_now(),
        "effective_config": config.effective_config(),
        "snapshots": snapshots,
    }
    write_json(config.RAW_DIR / "download_manifest.json", manifest)
    LOGGER.info("download 完成: %s", config.RAW_DIR / "download_manifest.json")
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_download()
