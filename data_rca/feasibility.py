"""Read-only checks for candidate sources that are not yet production inputs."""

from pathlib import Path
from typing import Any

from data_rca import config


def _files(root: Path, patterns: tuple[str, ...]) -> list[str]:
    if not root.exists():
        return []
    found = []
    for pattern in patterns:
        found.extend(str(path.relative_to(root)) for path in root.rglob(pattern))
    return sorted(set(found))


def source_feasibility() -> dict[str, Any]:
    itu_outputs = _files(config.ITU_TRACK_B_ROOT, ("devices_outputs.zip", "*device*output*"))
    itu_truth = _files(config.ITU_TRACK_B_ROOT, ("*ground*truth*", "*answer*.json", "*answer*.csv"))
    faultbench_scenarios = _files(config.FAULTBENCH_ROOT, ("*.txt",))
    return {
        "itu_track_b": {
            "status": (
                "schema_join_check_required"
                if itu_outputs and itu_truth
                else "skip_missing_ground_truth"
                if itu_outputs
                else "not_downloaded"
            ),
            "device_output_artifacts": itu_outputs,
            "ground_truth_candidates": itu_truth,
            "production_enabled": False,
        },
        "faultbench": {
            "status": "ready_for_converter" if faultbench_scenarios else "not_downloaded",
            "scenario_file_count": len(faultbench_scenarios),
            "production_enabled": False,
            "role": "fault diversity, not NOS diversity",
        },
    }
