import json
from pathlib import Path

from data_rca.anta import parse_anta
from data_rca.coverage import coverage_report
from data_rca.netopsbench import parse_netopsbench
from data_rca.nika import parse_case
from data_rca.render import render_user
from data_rca.schema import validate_record
from data_rca.split import split_rows


def _diagnosis(cause: str = "MTU mismatch") -> dict:
    return {
        "cause": cause,
        "reason": "Observed state differs from the intended state.",
        "identify": [{"command": "show interface", "expected_evidence": "A mismatch is shown."}],
        "resolution": ["Correct the setting."],
        "verification": [{"command": "show interface", "expected_result": "Values match."}],
    }


def test_anta_only_confirms_direct_configuration_comparison() -> None:
    base = {
        "id": "case-1",
        "test_name": "VerifyL3MTU",
        "device": "leaf1",
        "command": "show interfaces Ethernet1",
        "expected": 1500,
        "observed": 9000,
        "diagnoses": [_diagnosis()],
    }
    possible = parse_anta(base)
    confirmed = parse_anta({**base, "direct_configuration_check": True})

    assert possible["diagnoses"][0]["type"] == "diagnostic_guidance"
    assert possible["diagnoses"][0]["resolution"] == []
    assert confirmed["diagnoses"][0]["type"] == "confirmed_state_mismatch"
    assert validate_record(confirmed) == []


def test_nika_uses_tool_output_and_ground_truth_but_not_submission(tmp_path: Path) -> None:
    case = tmp_path / "misconfigurations" / "bgp_asn_misconfig" / "123"
    case.mkdir(parents=True)
    (case / "ground_truth.json").write_text(
        json.dumps(
            {
                "is_anomaly": True,
                "faulty_devices": ["r1"],
                "root_cause_name": ["bgp_asn_misconfig"],
            }
        )
    )
    (case / "session_meta.json").write_text(
        json.dumps(
            {
                "scenario_name": "fixture",
                "task_description": (
                    "Network Description: Two BGP routers.\n"
                    "Routers: r1, r2\n\nYour goal is to diagnose."
                ),
            }
        )
    )
    # If parse_case accidentally starts using this file, the test will fail.
    (case / "submission.json").write_text("not json")
    events = [
        {
            "event": "tool_start",
            "tool": {"name": "frr_get_bgp_conf"},
            "input": {"router_name": "r1"},
        },
        {
            "event": "tool_end",
            "output": "content='router bgp 65001' name='frr_get_bgp_conf'",
        },
        {
            "event": "tool_start",
            "tool": {"name": "frr_exec"},
            "input": {"router_name": "r1", "command": "show ip bgp summary"},
        },
        {
            "event": "tool_end",
            "output": "content='Idle' name='frr_exec'",
        },
    ]
    (case / "conversation_diagnosis_agent.log").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    catalog = {
        "bgp_asn_misconfig": {
            "cause": "BGP AS mismatch",
            "reason": "The AS relationship is incorrect.",
            "evidence_groups": [["frr_get_bgp_conf"], ["frr_exec"]],
            "identify": _diagnosis()["identify"],
            "resolution": _diagnosis()["resolution"],
            "verification": _diagnosis()["verification"],
        }
    }

    row = parse_case(case, catalog)

    assert row["diagnoses"][0]["type"] == "confirmed_rca"
    assert {item["command"] for item in row["evidence"]} == {
        "show BGP configuration",
        "show ip bgp summary",
    }
    assert validate_record(row) == []

    # The same ground-truth cause without the state check is retained but
    # downgraded instead of being overstated or dropped.
    (case / "conversation_diagnosis_agent.log").write_text(
        "\n".join(json.dumps(event) for event in events[:2])
    )
    possible = parse_case(case, catalog)
    assert possible["diagnoses"][0]["type"] == "possible_rca"


def test_netopsbench_uses_tool_observation_and_evaluator_truth_only() -> None:
    result = {
        "trace_id": "run:case:trace",
        "run_id": "run",
        "scenario_id": "generated_mtu_mismatch_xs_001",
        "details": {
            "ground_truth": {
                "fault_type": "mtu_mismatch",
                "location": {"device": "spine1", "interface": "Ethernet0"},
            }
        },
    }
    structured = {
        "device": "spine1",
        "interfaces": [{"name": "Ethernet0", "mtu": 1400}],
    }
    trajectory = {
        "trajectory_id": "run:case:trace",
        "extra": {"final_diagnosis": {"reasoning": "MUST NOT BE USED"}},
        "steps": [
            {
                "extra": {"is_initial_context": True},
                "message": json.dumps(
                    {"symptoms": {"observations": {"anomalies_detected": True}}}
                ),
            },
            {
                "extra": {"type": "tool_call", "name": "get_device_interfaces"},
                "tool_calls": [
                    {
                        "arguments": {"device": "spine1"},
                        "function_name": "get_device_interfaces",
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "content": json.dumps(
                                {"artifact": {"structured_content": structured}}
                            )
                        }
                    ]
                },
            },
        ],
    }
    catalog = {
        "mtu_mismatch": {
            "cause": "Interface MTU mismatch",
            "domain": "interfaces",
            "protocol": "ethernet",
            "preferred_tools": ["get_device_interfaces"],
            "reason": "Evaluator ground truth identifies an MTU mismatch.",
            "resolution": ["Correct the MTU."],
            "verification": [
                {"command": "get_device_interfaces", "expected_result": "MTUs match."}
            ],
        }
    }

    row = parse_netopsbench(result, trajectory, catalog)

    assert row["platform"] == "sonic"
    assert row["diagnoses"][0]["type"] == "confirmed_rca"
    assert "1400" in render_user(row)
    assert "MUST NOT BE USED" not in json.dumps(row)
    assert validate_record(row) == []


def test_coverage_uses_platform_detail_for_linux_and_frr() -> None:
    rows = [
        {
            "source": "nika",
            "platform": "frr",
            "network_domain": "routing",
            "protocol": "bgp",
            "diagnoses": [{"type": "confirmed_rca"}],
        },
        {
            "source": "anta",
            "platform": "eos",
            "network_domain": "routing",
            "protocol": "bgp",
            "diagnoses": [{"type": "diagnostic_guidance"}],
        },
    ]

    report = coverage_report(rows)

    assert report["matrix"]["routing"]["frr"] == 1
    assert report["matrix"]["routing"]["eos"] == 1


def test_group_never_crosses_splits() -> None:
    rows = []
    for number in range(20):
        group = f"group-{number // 2}"
        rows.append(
            {
                "id": f"row-{number}",
                "source": "fixture",
                "network_domain": "routing",
                "group_id": group,
                "diagnoses": [{"type": "confirmed_rca", "cause": "fixture"}],
                "metadata": {"category": "fixture"},
            }
        )

    splits, manifest = split_rows(rows)
    locations = {}
    for split, split_rows_list in splits.items():
        for row in split_rows_list:
            locations.setdefault(row["group_id"], set()).add(split)

    assert all(len(values) == 1 for values in locations.values())
    assert manifest["leaked_group_ids"] == []
    assert sum(manifest["rows"].values()) == 20
