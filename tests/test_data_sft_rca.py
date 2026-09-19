from data_sft_rca.build_views import build_views
from data_sft_rca.render_qwen_sft import to_sft
from data_sft_rca.split import split_rows


def _row(number: int = 1) -> dict:
    return {
        "id": f"case-{number}",
        "root_case_id": f"root-{number}",
        "group_id": "shared-group",
        "source": "fixture",
        "platform_family": "linux_frr",
        "network_domain": "routing",
        "protocol": "bgp",
        "task_type": "confirmed_rca",
        "fault_family": "bgp_failure",
        "problem": "The BGP peer is Idle.",
        "evidence": [
            {
                "id": "ev1",
                "type": "cli",
                "device": "r1",
                "command": "show bgp",
                "output": "Idle",
                "role": "symptom",
            },
            {
                "id": "ev2",
                "type": "cli",
                "device": "r1",
                "command": "show route",
                "output": "No peer route",
                "role": "supporting",
            },
            {
                "id": "ev3",
                "type": "config",
                "device": "r1",
                "command": "show config",
                "output": "remote-as 3",
                "role": "decisive",
            },
        ],
        "diagnoses": [
            {
                "status": "confirmed",
                "cause_id": "as_mismatch",
                "cause": "BGP AS mismatch",
                "evidence_ids": ["ev3"],
                "reason": "The configured AS is incorrect.",
                "resolution": ["Correct the AS."],
                "verification": [{"command": "show bgp", "expected_result": "Established"}],
            }
        ],
        "provenance": {
            "label_source": "fixture",
            "evidence_source": "fixture",
            "llm_generated": False,
        },
    }


def test_views_mask_decisive_evidence_and_keep_group() -> None:
    graph = {
        "bgp_failure": [
            {
                "cause_id": "as_mismatch",
                "cause": "BGP AS mismatch",
                "reason": "Compatible.",
                "verification": [],
                "support": 2,
                "platforms": ["linux_frr"],
                "protocols": ["bgp"],
            },
            {
                "cause_id": "peer_unreachable",
                "cause": "Peer unreachable",
                "reason": "Compatible.",
                "verification": [],
                "support": 2,
                "platforms": ["linux_frr"],
                "protocols": ["bgp"],
            },
        ]
    }
    views = build_views([_row()], graph)
    probable = next(row for row in views if row["task_type"] == "probable_cause_analysis")

    assert probable["group_id"] == "shared-group"
    assert "ev3" in probable["masked_evidence_ids"]
    assert "ev3" not in {item["id"] for item in probable["evidence"]}


def test_split_keeps_all_views_of_a_group_together() -> None:
    rows = [_row(1), {**_row(2), "id": "case-2", "root_case_id": "root-2"}]
    splits, _ = split_rows(rows)
    locations = [name for name, items in splits.items() if items]
    assert len(locations) == 1


def test_qwen_renderer_has_only_plain_messages() -> None:
    rendered = to_sft(_row())
    assert [message["role"] for message in rendered["messages"]] == ["system", "user", "assistant"]
    assert "tools" not in rendered
    assert "tool_calls" not in rendered
    assert "<think>" not in rendered["messages"][-1]["content"]
