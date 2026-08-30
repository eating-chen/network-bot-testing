from pathlib import Path
from types import SimpleNamespace

import pytest

from network_cpt import config, download


def test_huggingface_snapshot_downloads_once_and_then_reuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(
        download,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda dataset_id: SimpleNamespace(sha="fixture-revision")
        ),
    )

    calls = []

    def fake_snapshot_download(**kwargs: object) -> None:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"]))
        (destination / "data").mkdir(parents=True)
        (destination / "data" / "raw.parquet").write_bytes(b"fixture")

    monkeypatch.setattr(download, "snapshot_download", fake_snapshot_download)

    first = download.download_huggingface_dataset("fixture", "owner/dataset")
    second = download.download_huggingface_dataset("fixture", "owner/dataset")

    assert first == second
    assert first["revision"] == "fixture-revision"
    assert first["files"] == 1
    assert len(calls) == 1
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "fixture-revision"
    assert (config.RAW_DIR / "huggingface" / "fixture.manifest.json").exists()
