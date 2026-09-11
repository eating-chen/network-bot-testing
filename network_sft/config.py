"""Editable V1 settings. There is intentionally no CLI configuration layer."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sft"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "01_normalized"
CURATED_DIR = DATA_DIR / "02_curated"
SELECTED_DIR = DATA_DIR / "03_selected"
FINAL_DIR = DATA_DIR / "network_sft_v1"
REPORTS_DIR = DATA_DIR / "reports"

# snapshot_download keeps the exact Hub revision in download_manifest.json.
HF_SOURCES = {
    "5g_faults": {
        "repo_id": "greenwich157/5G-Faults-Full-v2",
        "patterns": ["**/*.csv", "**/*.parquet", "README.md"],
    },
    "telelogs": {
        "repo_id": "tecnicolaude/Telelogs-CoT",
        "patterns": ["**/*.parquet", "README.md"],
    },
    "ccna": {
        "repo_id": "Rzkoohi/CCNA_small",
        "patterns": ["**/*.parquet", "README.md"],
    },
    # Public, standardized conversion of Team-ACE/ToolACE (11,072 rows).
    "toolace": {
        "repo_id": "minpeter/toolace-parsed",
        "patterns": ["**/*.parquet", "README.md"],
    },
    "when2call": {
        "repo_id": "nvidia/When2Call",
        "patterns": ["train/when2call_train_sft.jsonl", "README.md"],
    },
}

NIKA_URL = "https://zenodo.org/records/17971675/files/NIKA%20Traces.zip?download=1"
NIKA_MD5 = "cc940a7fadd29677d5ef942847a5c45c"
NIKA_SYSTEM = "You are a network troubleshooting assistant."

SOURCE_ORDER = ("5g_faults", "telelogs", "ccna", "nika", "toolace", "when2call")
FULL_SOURCES = ("5g_faults", "telelogs", "nika")
SAMPLE_TARGETS = {"ccna": 3_500, "toolace": 3_000, "when2call": 2_000}

CCNA_QUOTAS = {
    "ip_routing": 700,
    "routing_protocols": 650,
    "vlan_stp_l2": 600,
    "acl_nat_security": 550,
    "network_services": 500,
    "management_cli_misc": 500,
}
TOOLACE_QUOTAS = {
    "single_tool": 750,
    "sequential": 1_050,
    "multi_turn": 750,
    "parallel": 450,
}
WHEN2CALL_QUOTAS = {
    "tool_call": 700,
    "direct_answer": 500,
    "ask_clarification": 500,
    "cannot_solve": 300,
}

SEED = 42
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
NEAR_DUP_JACCARD = 0.86
REVIEW_ROWS = 200

# stats.py records incompatibilities; these are checks, not corpus formatting.
TOKENIZER_IDS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)


def normalized_path(source: str) -> Path:
    return NORMALIZED_DIR / f"{source}.jsonl"


def curated_path(source: str) -> Path:
    return CURATED_DIR / f"{source}.jsonl"
