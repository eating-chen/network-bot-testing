"""Edit these values directly.  The RCA pipeline intentionally has no argparse layer."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETOPSBENCH_REVISION = "f7ef8e5ecde5eba030c9b4cd9ec4cda7703edc42"
ANTA_REVISION = "155732898dee840f8c27ebb2e4ee1eeec3c38bc0"

# Existing NIKA snapshot.  The pipeline never runs the network environment.
NIKA_ROOT = PROJECT_ROOT / "data" / "sft" / "raw" / "nika"

# Public snapshots.  Run ``python -m data_rca.fetch_sources`` once to populate
# these directories.  Building the dataset itself never accesses the network.
RAW_DIR = PROJECT_ROOT / "data" / "rca" / "raw"
NETOPSBENCH_ROOT = RAW_DIR / "netopsbench"
NETOPSBENCH_EXTRACTED = NETOPSBENCH_ROOT / "extracted"
ANTA_ROOT = RAW_DIR / "anta"
ITU_TRACK_B_ROOT = RAW_DIR / "itu_track_b"
FAULTBENCH_ROOT = RAW_DIR / "faultbench"

# Optional hand-inspectable ANTA rows.  Official fixtures are the main EOS
# source; this file is only for additional real ANTA failure exports.
INPUT_DIR = PROJECT_ROOT / "data_rca" / "input"
ANTA_INPUT = INPUT_DIR / "anta.jsonl"

# Generated, inspectable stages.
DATA_DIR = PROJECT_ROOT / "data" / "rca"
CANONICAL_DIR = DATA_DIR / "01_canonical"
FINAL_DIR = DATA_DIR / "network_troubleshooting_sft_v1"
REJECTED_PATH = DATA_DIR / "rejected.jsonl"

NIKA_CATALOG_PATH = PROJECT_ROOT / "data_rca" / "fault_catalog.json"
NETOPSBENCH_CATALOG_PATH = PROJECT_ROOT / "data_rca" / "netopsbench_catalog.json"

SEED = 42
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
MAX_EVIDENCE_ITEMS = 6
MAX_OUTPUT_CHARS = 12_000
REVIEW_ROWS_PER_SOURCE = 8
SYSTEM_MESSAGE = (
    "You are a network troubleshooting assistant. Distinguish root cause analysis, "
    "confirmed state mismatches, and diagnostic guidance. Use only supplied evidence."
)
