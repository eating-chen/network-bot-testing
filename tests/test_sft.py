import json
from pathlib import Path

import pytest

from network_sft import config
from network_sft.curate import is_ccna_troubleshooting, network_topic, stratified_sample
from network_sft.io import jsonl_rows, write_jsonl
from network_sft.nika import evaluate_submission, parse_incident
from network_sft.normalize import _decision_answer, normalize_toolace
from network_sft.schema import make_row, normalize_tool, validate_row
from network_sft.split import assign_groups
from network_sft.stats import apply_template

PING = normalize_tool(
    {
        "name": "ping",
        "description": "Ping a host.",
        "parameters": {
            "type": "dict",
            "properties": {"host": {"type": "str"}},
            "required": ["host"],
        },
    }
)


def agent_row(identifier: str = "agent-1", group: str = "incident-1") -> dict:
    return make_row(
        identifier,
        "nika",
        "agentic",
        [
            {"role": "user", "content": "Why is r1 unreachable?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "ping", "arguments": {"host": "r1"}},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "ping",
                "tool_call_id": "call_0",
                "content": "timeout",
            },
            {"role": "assistant", "content": "The host is unreachable."},
        ],
        [PING],
        {"group_id": group, "category": "link_down"},
    )


def test_canonical_schema_is_hf_and_trl_conversational_format(tmp_path: Path) -> None:
    row = agent_row()
    assert validate_row(row) == []
    assert PING["function"]["parameters"]["type"] == "object"
    assert PING["function"]["parameters"]["properties"]["host"]["type"] == "string"
    output = tmp_path / "sample.jsonl"
    write_jsonl(output, [row])
    assert next(jsonl_rows(output))["messages"][1]["tool_calls"][0]["id"] == "call_0"


def test_when2call_toolcall_becomes_structured_message() -> None:
    category, message = _decision_answer(
        '<TOOLCALL>[{"name":"weather","arguments":{"city":"Taipei"}}]</TOOLCALL>'
    )
    assert category == "tool_call"
    assert message["tool_calls"][0]["function"]["arguments"] == {"city": "Taipei"}


def test_when2call_text_decision_classes_are_traceable_rules() -> None:
    assert _decision_answer("Could you please specify the city?")[0] == "ask_clarification"
    assert _decision_answer("I cannot access a live database.")[0] == "cannot_solve"
    assert _decision_answer("The answer is 42.")[0] == "direct_answer"


def test_toolace_complexity_is_derived_from_messages() -> None:
    source = {
        "tools": json.dumps([PING]),
        "messages": [
            {"role": "user", "content": "Ping both."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "ping", "arguments": '{"host":"r1"}'}},
                    {"function": {"name": "ping", "arguments": '{"host":"r2"}'}},
                ],
            },
            {"role": "tool", "name": "ping", "content": "ok"},
            {"role": "tool", "name": "ping", "content": "timeout"},
            {"role": "assistant", "content": "r2 failed."},
        ],
    }
    row = normalize_toolace(7, source)
    assert row["metadata"]["category"] == "parallel"
    assert row["metadata"]["num_tool_calls"] == 2
    assert validate_row(row) == []


def test_toolace_terminal_call_is_a_valid_sft_target() -> None:
    row = agent_row()
    row["task_type"] = "tool_calling"
    row["messages"] = row["messages"][:2]
    assert validate_row(row) == []


def test_ccna_filter_and_topic() -> None:
    row = make_row(
        "ccna-1",
        "ccna",
        "diagnostic",
        [
            {"role": "user", "content": "Why is the OSPF neighbor not forming?"},
            {"role": "assistant", "content": "Verify the area and hello timers."},
        ],
        [],
        {"category": "unclassified"},
    )
    assert is_ccna_troubleshooting(row)
    assert network_topic(row["messages"][0]["content"]) == "routing_protocols"


def test_sampling_redistributes_unused_quota() -> None:
    rows = []
    for index, category in enumerate(["a", "a", "a", "b", "b"]):
        row = agent_row(f"row-{index}", f"group-{index}")
        row["source"] = "fixture"
        row["metadata"]["category"] = category
        rows.append(row)
    selected = stratified_sample(rows, 4, {"a": 1, "missing": 2}, "fixture")
    assert len(selected) == 4


def test_group_split_never_leaks() -> None:
    rows = []
    for group in range(20):
        for variant in range(2):
            row = agent_row(f"{group}-{variant}", f"group-{group}")
            rows.append(row)
    assignments = assign_groups(rows)
    assert set(assignments) == {f"group-{group}" for group in range(20)}
    assert set(assignments.values()) == {"train", "val", "test"}


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_nika_only_keeps_correct_complete_trace(tmp_path: Path) -> None:
    incident = tmp_path / "misconfigurations" / "bgp_asn" / "123"
    incident.mkdir(parents=True)
    truth = {"is_anomaly": True, "faulty_devices": ["r1"], "root_cause_name": ["bgp_asn"]}
    _write(incident / "ground_truth.json", truth)
    _write(incident / "submission.json", truth)
    _write(
        incident / "session_meta.json",
        {"task_description": "Diagnose BGP.", "scenario_name": "simple_bgp"},
    )
    events = [
        {"event": "llm_end", "generation_info": {"finish_reason": "tool_calls"}},
        {"event": "tool_start", "tool": {"name": "ping"}, "input": "{'host': 'r1'}"},
        {"event": "tool_end", "output": "content='timeout' name='ping'"},
        {"event": "llm_end", "text": "The BGP ASN is wrong.", "generation_info": {}},
    ]
    (incident / "conversation_diagnosis_agent.log").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    row = parse_incident(incident)
    assert validate_row(row) == []
    assert row["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"host": "r1"}
    assert row["metadata"]["submission_tier"] == "exact_correct"


def test_nika_submission_tiers_keep_exact_gate_traceable() -> None:
    truth = {
        "is_anomaly": True,
        "faulty_devices": ["r1"],
        "root_cause_name": ["link_down"],
    }
    assert evaluate_submission(truth, truth)["tier"] == "exact_correct"
    partial = {**truth, "root_cause_name": ["link_down", "bgp_misconfiguration"]}
    assert evaluate_submission(truth, partial)["tier"] == "partial_correct"
    wrong = {**truth, "root_cause_name": ["bgp_misconfiguration"]}
    assert evaluate_submission(truth, wrong)["tier"] == "wrong"


def test_nika_rejects_ambiguous_parallel_tool_result(tmp_path: Path) -> None:
    incident = tmp_path / "link_failures" / "link_down" / "456"
    incident.mkdir(parents=True)
    truth = {"is_anomaly": True, "faulty_devices": ["r1"], "root_cause_name": ["link_down"]}
    _write(incident / "ground_truth.json", truth)
    _write(incident / "submission.json", truth)
    _write(
        incident / "session_meta.json",
        {"task_description": "Diagnose links.", "scenario_name": "simple_bgp"},
    )
    events = [
        {"event": "llm_end", "generation_info": {"finish_reason": "tool_calls"}},
        {"event": "tool_start", "tool": {"name": "ping"}, "input": {"host": "r1"}},
        {"event": "tool_start", "tool": {"name": "ping"}, "input": {"host": "r2"}},
        {"event": "tool_end", "output": "content='ok' name='ping'"},
        {"event": "tool_end", "output": "content='timeout' name='ping'"},
        {"event": "llm_end", "text": "r1 is down.", "generation_info": {}},
    ]
    (incident / "conversation_diagnosis_agent.log").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ambiguous_tool_result"):
        parse_incident(incident)


class FakeTokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, **kwargs):
        assert kwargs["conversation"][0]["role"] == "user"
        assert kwargs["tools"][0]["type"] == "function"
        return [1, 2, 3]


def test_template_boundary_passes_canonical_messages_and_tools_directly() -> None:
    assert apply_template(FakeTokenizer(), agent_row()) == [1, 2, 3]
    assert config.SPLIT_RATIOS == {"train": 0.8, "val": 0.1, "test": 0.1}


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, **kwargs):
        return {"input_ids": [4, 5, 6, 7]}


def test_template_boundary_handles_transformers_v5_mapping_result() -> None:
    assert apply_template(MappingTokenizer(), agent_row()) == [4, 5, 6, 7]
