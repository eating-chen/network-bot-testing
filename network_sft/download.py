"""Download the six raw sources. Existing snapshots are reused."""

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from network_sft import config
from network_sft.io import setup_logging, source_files, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def _hf_source(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    destination = config.RAW_DIR / name
    if source_files(destination):
        LOGGER.info("reuse raw/%s", name)
        cached = destination / ".cache" / "huggingface" / "trees"
        revisions = sorted(path.stem for path in cached.glob("*.json"))
        return {
            "repo_id": spec["repo_id"],
            "cached_revisions": revisions,
            "path": str(destination),
            "reused": True,
        }
    revision = HfApi().dataset_info(spec["repo_id"]).sha
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        revision=revision,
        allow_patterns=spec["patterns"],
        local_dir=destination,
    )
    data_revision = revision
    if not source_files(destination):
        # Some dataset repos only keep a card on main; the viewer materializes data here.
        data_revision = HfApi().dataset_info(spec["repo_id"], revision="refs/convert/parquet").sha
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=data_revision,
            allow_patterns=["**/*.parquet"],
            local_dir=destination,
        )
    if not source_files(destination):
        raise RuntimeError(f"downloaded no supported data files for {name}")
    return {
        "repo_id": spec["repo_id"],
        "main_revision": revision,
        "data_revision": data_revision,
        "path": str(destination),
        "reused": False,
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - checking the publisher's checksum
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nika() -> dict[str, Any]:
    archive = config.RAW_DIR / "nika_traces.zip"
    destination = config.RAW_DIR / "nika"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".part")
        request = urllib.request.Request(config.NIKA_URL, headers={"User-Agent": "network-sft-v1"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        temporary.replace(archive)
    if _md5(archive) != config.NIKA_MD5:
        raise RuntimeError(f"NIKA checksum mismatch: {archive}")
    marker = destination / ".extracted"
    if not marker.exists():
        destination.mkdir(parents=True, exist_ok=True)
        safe_root = destination.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not (destination / member.filename).resolve().is_relative_to(safe_root):
                    raise RuntimeError(f"unsafe ZIP member: {member.filename}")
            bundle.extractall(destination)
        marker.write_text("ok\n", encoding="utf-8")
    return {"url": config.NIKA_URL, "md5": config.NIKA_MD5, "path": str(destination)}


def run_download() -> dict[str, Any]:
    sources = {name: _hf_source(name, spec) for name, spec in config.HF_SOURCES.items()}
    sources["nika"] = _nika()
    manifest = {"created_at": utc_now(), "sources": sources}
    write_json(config.RAW_DIR / "download_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    setup_logging()
    run_download()
