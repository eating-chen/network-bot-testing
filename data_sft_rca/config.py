"""Small, editable configuration.  This project intentionally has no argparse."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "data_sft_rca"
RAW_DIR = PACKAGE_DIR / "raw"
OUTPUT_DIR = PACKAGE_DIR / "output"

# Existing source snapshots from the v1 builder.
NIKA_ROOT = ROOT / "data" / "sft" / "raw" / "nika"
NETOPSBENCH_ROOT = ROOT / "data" / "rca" / "raw" / "netopsbench"
ANTA_ROOT = ROOT / "data" / "rca" / "raw" / "anta"

# New v2 sources are kept under this directory.
CLOUDOPS_ROOT = RAW_DIR / "cloudopsbench"
FAULTBENCH_ROOT = RAW_DIR / "faultbench"
RCAEVAL_ROOT = RAW_DIR / "rcaeval"

CANONICAL_DIR = OUTPUT_DIR / "canonical"
VIEWS_DIR = OUTPUT_DIR / "views"
SFT_DIR = OUTPUT_DIR / "qwen_sft"

SEED = 42
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
MAX_EVIDENCE_CHARS = 8_000
ANTA_GUIDANCE_LIMIT = 200
MIN_GRAPH_SUPPORT = 2
FOCUSED_CONFIRMED_LIMIT = 900
PROBABLE_VIEW_LIMIT = 900
NEXT_STEP_VIEW_LIMIT = 730

SYSTEM_MESSAGE = (
    "You are a network troubleshooting assistant. Base conclusions only on the "
    "provided evidence. Distinguish confirmed root causes from hypotheses. Do not "
    "invent device outputs or unsupported causes."
)

CLOUDOPS_FAULTS = {
    "service_selector_mismatch",
    "service_port_mapping_mismatch",
    "service_protocol_mismatch",
    "service_env_var_address_mismatch",
    "gateway_misrouted",
    "service_dns_resolution_failure",
    "pod_network_delay",
    "node_network_delay",
    "node_network_packet_loss",
    "kube_proxy_unavailable",
}

RCAEVAL_FAULTS = {"delay", "loss", "socket"}

CLOUDOPS_REVISION = "d79f1526a4c3315004c09457fbec588eb257dd1c"
FAULTBENCH_REVISION = "ab8c45f808fb0c2a1ae5c029bcfe0194617c36f2"
RCAEVAL_REVISION = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
FROZEN_EVAL_IDS = PACKAGE_DIR / "frozen_eval_ids.txt"
