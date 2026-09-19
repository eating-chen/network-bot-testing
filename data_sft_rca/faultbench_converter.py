"""Convert the 200 FaulT-Bench core scenarios; persona rewrites are excluded."""

import csv
import re
from pathlib import Path

from data_sft_rca import config
from data_sft_rca.io_utils import stable_id

SECTION = re.compile(r"^\[([A-Z-]+)\]\s*$", re.MULTILINE)


def _sections(text: str) -> dict[str, str]:
    matches = list(SECTION.finditer(text))
    result = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match.group(1)] = text[match.end() : end].strip()
    return result


def _field(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.*(?:\n  .*)*)", text, re.MULTILINE)
    return " ".join(match.group(1).split()) if match else ""


def _base_id(scenario: str) -> str:
    return re.sub(r"-(?:misdirected|misdescribed)$", "", scenario)


def _protocol(text: str) -> str:
    lowered = text.lower()
    for value in ("bgp", "ospf", "rip", "eigrp", "dns", "dhcp", "vrrp"):
        if value in lowered:
            return value
    return "ip"


def _healthy_evidence(ground_truth: str) -> str:
    flat = " ".join(ground_truth.split())
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    signals = (
        " healthy",
        " reachable",
        " resolves",
        " established",
        " responds",
        " running",
        " works",
        " succeeds",
        "0% loss",
        " is up",
        " are up",
        " present",
        " intact",
        " forwarding",
        " answering",
        " listening",
        " returns",
    )
    blocked = ("correct diagnosis", "over-diagnos", "negative control", "no fault was")
    chosen = [
        sentence
        for sentence in sentences
        if any(signal in sentence.lower() for signal in signals)
        and not any(word in sentence.lower() for word in blocked)
    ]
    return " ".join(chosen[:5])


def _base_truth(files: list[Path]) -> dict[str, dict[str, str]]:
    result = {}
    for path in files:
        scenario = path.stem
        sections = _sections(path.read_text(encoding="utf-8", errors="replace"))
        truth = sections.get("GROUND-TRUTH", "")
        result[scenario] = {
            "cause": _field(truth, "Root cause") or _field(truth, "Affected component"),
            "affected": _field(truth, "Affected component"),
            "mechanism": _field(truth, "Mechanism"),
            "fix": _field(truth, "Fix") or sections.get("FIX", ""),
        }
    return result


def _load_index(dataset: Path) -> dict[str, dict[str, str]]:
    with (dataset / "dataset_index.csv").open(encoding="utf-8", newline="") as handle:
        return {row["scenario"]: row for row in csv.DictReader(handle)}


def parse_scenario(
    path: Path, index: dict[str, dict[str, str]], bases: dict[str, dict[str, str]]
) -> dict | None:
    scenario = path.stem
    info = index.get(scenario)
    if not info:
        return None
    sections = _sections(path.read_text(encoding="utf-8", errors="replace"))
    problem = sections.get("PROMPT-TO-AGENT", "")
    if not problem:
        return None
    root = _base_id(scenario)
    class_name = info["class"]
    no_fault = class_name == "false-premise"
    evidence_text = (
        _healthy_evidence(sections.get("GROUND-TRUTH", ""))
        if no_fault
        else sections.get("POST-INJECT-CHECK", "")
    )
    if not evidence_text or evidence_text.startswith("# none"):
        evidence_text = _healthy_evidence(sections.get("GROUND-TRUTH", ""))
    if not evidence_text:
        return None
    evidence = [
        {
            "id": "ev1",
            "type": "expected_observed",
            "device": info["topology"],
            "command": "official post-injection verification"
            if not no_fault
            else "official healthy-baseline verification",
            "output": evidence_text,
            "role": "decisive",
        }
    ]
    if no_fault:
        diagnosis = {
            "status": "no_fault",
            "cause_id": "reported_fault_not_reproduced",
            "cause": "The available evidence does not support the reported fault.",
            "evidence_ids": ["ev1"],
            "reason": "The official healthy-baseline checks contradict the ticket premise.",
            "resolution": [],
            "verification": [
                {
                    "command": "Repeat the reported operation from the affected endpoint.",
                    "expected_result": (
                        "The operation succeeds and the reported failure remains unreproduced."
                    ),
                }
            ],
        }
        task_type = "premise_correction"
        fault_family = "reported_fault_not_reproduced"
    else:
        truth = bases.get(root)
        if not truth or not truth["cause"]:
            ground_truth = " ".join(sections.get("GROUND-TRUTH", "").split())
            first_sentence = re.split(r"(?<=[.!?])\s+", ground_truth, maxsplit=1)[0]
            truth = {
                "cause": first_sentence.removeprefix("The real fault is ").rstrip("."),
                "affected": "",
                "mechanism": " ".join(re.split(r"(?<=[.!?])\s+", ground_truth)[:2]),
                "fix": _field(ground_truth, "Fix") or sections.get("FIX", ""),
            }
        if not truth["cause"]:
            return None
        cause_id = index.get(root, info).get("nika_label") or ""
        if not cause_id or cause_id == "none":
            cause_id = re.sub(r"[^a-z0-9]+", "_", info["fault_family"].lower()).strip("_")
        diagnosis = {
            "status": "confirmed",
            "cause_id": cause_id,
            "cause": truth["cause"],
            "evidence_ids": ["ev1"],
            "reason": truth["mechanism"]
            or "The official scenario verification localizes the injected fault.",
            "resolution": [truth["fix"]] if truth["fix"] else [],
            "verification": [
                {
                    "command": "Repeat the scenario post-injection checks after remediation.",
                    "expected_result": (
                        "The failed checks recover while unaffected paths remain healthy."
                    ),
                }
            ],
        }
        if class_name in {"wrong-device", "wrong-cause"}:
            task_type = "premise_correction"
            diagnosis["cause"] = (
                f"The ticket premise is incorrect; the confirmed cause is {diagnosis['cause']}"
            )
        else:
            task_type = "confirmed_rca"
        fault_family = info["fault_family"].replace(" (misdescribed cause)", "")
    protocol = _protocol(" ".join((problem, info["fault_family"])))
    return {
        "id": "faultbench_" + stable_id(scenario),
        "root_case_id": f"faultbench_{root}",
        "group_id": f"faultbench_{root}",
        "source": "faultbench",
        "platform_family": "linux_frr",
        "network_domain": "routing"
        if protocol in {"bgp", "ospf", "rip", "eigrp"}
        else "network_services",
        "protocol": protocol,
        "task_type": task_type,
        "fault_family": fault_family,
        "problem": problem,
        "evidence": evidence,
        "diagnoses": [diagnosis],
        "provenance": {
            "label_source": "scenario_ground_truth",
            "evidence_source": "official_verified_scenario",
            "llm_generated": False,
        },
    }


def build_faultbench() -> list[dict]:
    dataset = config.FAULTBENCH_ROOT / "dataset"
    if not (dataset / "dataset_index.csv").exists():
        return []
    index = _load_index(dataset)
    base_files = sorted(dataset.glob("*_Q/base/*.txt"))
    core_files = base_files + sorted(dataset.glob("*_Q/cex/*.txt"))
    bases = _base_truth(base_files)
    rows = [parse_scenario(path, index, bases) for path in core_files]
    return [row for row in rows if row]


if __name__ == "__main__":
    rows = build_faultbench()
    print(
        {
            "rows": len(rows),
            "no_fault": sum(row["task_type"] == "premise_correction" for row in rows),
        }
    )
