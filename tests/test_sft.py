import json
from pathlib import Path

from network_sft import config as sft_config
from network_sft.adapters.llama import render_for_llama
from network_sft.functiongemma import convert_row
from network_sft.io import jsonl_rows, write_jsonl
from network_sft.nika import parse_incident
from network_sft.schema import canonical_row, validate_row
from network_sft.split_merge import _nika_assignments

TOOLS = [
    {
        "name": "ping",
        "description": "Ping a device.",
        "parameters": {
            "type": "object",
            "properties": {"host": {"type": "string"}},
            "required": ["host"],
        },
    }
]


def agent_row() -> dict:
    return canonical_row(
        "nika_example",
        "nika",
        "network_agent",
        "Troubleshoot the network.",
        TOOLS,
        [
            {"role": "user", "content": "Check both routers."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"call_id": "call_0", "name": "ping", "arguments": {"host": "r1"}},
                    {"call_id": "call_1", "name": "ping", "arguments": {"host": "r2"}},
                ],
            },
            {"role": "tool", "call_id": "call_0", "name": "ping", "content": "reachable"},
            {"role": "tool", "call_id": "call_1", "name": "ping", "content": "timeout"},
            {"role": "assistant", "content": "r2 is unreachable."},
        ],
    )


def test_sft_directory_is_separate_from_cpt() -> None:
    assert sft_config.SFT_DATA_DIR.name == "sft"
    assert "network_cpt_v1" not in str(sft_config.CANONICAL_DIR)


def test_validator_accepts_parallel_tool_results() -> None:
    assert validate_row(agent_row()) == []


def test_jsonl_round_trip_preserves_nested_objects(tmp_path: Path) -> None:
    output = tmp_path / "sample.jsonl"
    write_jsonl(output, [agent_row()])
    restored = next(jsonl_rows(output))

    assert restored["tools"][0]["parameters"]["properties"]["host"] == {
        "type": "string"
    }
    assert restored["turns"][1]["tool_calls"][0]["arguments"] == {"host": "r1"}
    assert validate_row(restored) == []


def test_llama_adapter_is_the_only_place_that_adds_hf_wrappers() -> None:
    row = agent_row()
    messages, tools = render_for_llama(row)

    assert "type" not in row["tools"][0]
    assert tools[0]["type"] == "function"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "ping"


def test_functiongemma_incomplete_first_turn_is_detected() -> None:
    source = {
        "messages": [
            {"role": "developer", "content": "Use tools."},
            {"role": "user", "content": "Check r1."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ping", "arguments": {"host": "r1"}},
                    }
                ],
            },
        ],
        "tools": [{"type": "function", "function": TOOLS[0]}],
        "module_id": "first_turn_core",
        "split": "train",
    }
    row = convert_row(source, 0)

    assert row["system"] == "Use tools."
    assert row["turns"][0]["role"] == "user"
    assert "assistant response" in " ".join(validate_row(row))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_parse_nika_event_log(tmp_path: Path) -> None:
    incident = tmp_path / "misconfigurations" / "bgp_asn" / "123"
    incident.mkdir(parents=True)
    truth = {
        "is_anomaly": True,
        "faulty_devices": ["r1"],
        "root_cause_name": ["bgp_asn"],
    }
    _write_json(incident / "ground_truth.json", truth)
    _write_json(incident / "submission.json", truth)
    _write_json(
        incident / "session_meta.json",
        {
            "task_description": "Diagnose this BGP network.",
            "scenario_name": "simple_bgp",
            "backend_model": "fixture",
            "scenario_topo_size": "s",
        },
    )
    events = [
        {"event": "llm_end", "generation_info": {"finish_reason": "tool_calls"}},
        {
            "event": "tool_start",
            "tool": {"name": "ping", "description": "Ping a device."},
            "input": "{'host': 'r1'}",
        },
        {
            "event": "tool_end",
            "output": "content='timeout' name='ping' tool_call_id='upstream-id'",
        },
        {
            "event": "llm_end",
            "text": "r1 has a BGP ASN mismatch.",
            "generation_info": {"finish_reason": "stop"},
        },
    ]
    (incident / "conversation_diagnosis_agent.log").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    row = parse_incident(incident)

    assert validate_row(row) == []
    assert row["metadata"]["failure_type"] == "bgp_asn"
    assert row["system"] == sft_config.NIKA_SYSTEM_PROMPT
    assert [turn["role"] for turn in row["turns"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_nika_split_holds_out_complete_failure_types() -> None:
    rows = []
    for failure_type in ("link_down", "bgp_asn", "packet_drop", "vpn", "microburst"):
        for index in range(2):
            rows.append(
                {
                    "id": f"{failure_type}-{index}",
                    "metadata": {"failure_type": failure_type, "scenario": "fixture"},
                }
            )
    assignments = _nika_assignments(rows)
    split_by_type = {}
    for row in rows:
        split_by_type.setdefault(row["metadata"]["failure_type"], set()).add(
            assignments[row["id"]]
        )
    assert all(len(splits) == 1 for splits in split_by_type.values())
    assert {next(iter(splits)) for splits in split_by_type.values()} == {
        "train",
        "validation",
        "test",
    }
