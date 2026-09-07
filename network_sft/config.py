"""SFT v1 settings. Edit this file so every run remains easy to trace."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_DATA_DIR = PROJECT_ROOT / "data" / "sft"
RAW_DIR = SFT_DATA_DIR / "raw"
INTERMEDIATE_DIR = SFT_DATA_DIR / "intermediate"
CANONICAL_DIR = SFT_DATA_DIR / "canonical"
RENDERED_DIR = SFT_DATA_DIR / "rendered"
REPORTS_DIR = SFT_DATA_DIR / "reports"

GENERAL_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
GENERAL_SOURCE_SPLIT = "train_sft"
GENERAL_SAMPLE_SIZE = 2_000
GENERAL_MAX_CHARS = 32_000
# One upstream prompt_id per line. Useful if a future CPT replay uses UltraChat.
GENERAL_EXCLUDE_IDS = SFT_DATA_DIR / "general_cpt_replay_ids.txt"

FUNCTIONGEMMA_DATASET_ID = "lucasllfsQ/network-agent-functiongemma-en-es"
FUNCTIONGEMMA_CONFIG = "en"

NIKA_URL = "https://zenodo.org/records/17971675/files/NIKA%20Traces.zip?download=1"
NIKA_MD5 = "cc940a7fadd29677d5ef942847a5c45c"
NIKA_SYSTEM_PROMPT = "You are a network troubleshooting assistant."

SPLIT_SEED = 42
TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
TEST_RATIO = 0.05

TOKENIZER_ID = "meta-llama/Llama-3.1-8B-Instruct"


def intermediate_path(source: str) -> Path:
    return INTERMEDIATE_DIR / f"{source}.jsonl"
