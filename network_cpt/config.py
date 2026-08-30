"""集中管理 v1 pipeline 設定。

第一版刻意不用 argparse。要調整資料量、來源或路徑，直接修改本檔，讓每次執行的
行為都能從版本控制中看出來。每次執行也會把有效設定寫入 manifest。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------- v1 corpus sources ----------
# 不設定 token budget，也不依比例抽樣。啟用的 dataset 會完整下載。
SUPPORTED_SOURCES = (
    "gsma_tcc",
    "tele_data",
    "frrouting",
    "sonic",
    "openvswitch",
    "kea",
)
ENABLED_SOURCES = SUPPORTED_SOURCES

# Hugging Face 來源會用 snapshot_download 完整下載 repository，不做 row filtering。
HUGGINGFACE_DATASETS = {
    "tele_data": "AliMaatouk/Tele-Data",
    "gsma_tcc": "GSMA/Telco-Common-Corpus",
}

# ---------- immutable snapshots ----------
GIT_REPOSITORIES = {
    "frrouting": "https://github.com/FRRouting/frr.git",
    "sonic": "https://github.com/sonic-net/SONiC.git",
    "openvswitch": "https://github.com/openvswitch/ovs.git",
}
# 固定文件版本，不使用 latest，否則同一份設定日後可能得到不同 corpus。
KEA_VERSION = "kea-3.0.4"
KEA_PDF_URL = f"https://kea.readthedocs.io/_/downloads/en/{KEA_VERSION}/pdf/"

# ---------- parsing / cleaning ----------
MIN_TEXT_CHARS = 200
MIN_ALNUM_RATIO = 0.20
MAX_REPLACEMENT_CHAR_RATIO = 0.01

# ---------- document-level split ----------
SPLIT_SEED = 42
TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
TEST_RATIO = 0.05


def effective_config() -> dict[str, object]:
    """回傳這次 pipeline 會寫進 manifest 的主要設定。"""
    return {
        "enabled_sources": ENABLED_SOURCES,
        "kea_version": KEA_VERSION,
        "min_text_chars": MIN_TEXT_CHARS,
        "split_seed": SPLIT_SEED,
        "split_ratios": [TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO],
    }


def validate_config() -> None:
    if abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("split ratios 必須合計為 1.0")
    unknown = set(ENABLED_SOURCES) - set(SUPPORTED_SOURCES)
    if unknown:
        raise ValueError(f"ENABLED_SOURCES 含未知來源: {sorted(unknown)}")
