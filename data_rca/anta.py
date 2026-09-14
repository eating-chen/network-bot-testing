"""Convert official ANTA failure fixtures into grounded EOS troubleshooting data."""

import ast
import contextlib
import json
import re
from pathlib import Path
from typing import Any

from data_rca import config
from data_rca.io import read_jsonl, stable_hash
from data_rca.schema import validate_record
from data_rca.taxonomy import anta_domain

EXACT_MISMATCH = re.compile(r"\bExpected:\s*.+?\s+Actual:\s*.+", re.IGNORECASE)


def _class_metadata(source_root: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not node.name.startswith("Verify"):
                continue
            categories: list[str] = []
            commands: list[str] = []
            for child in node.body:
                if isinstance(child, ast.AnnAssign):
                    target, value = child.target, child.value
                elif isinstance(child, ast.Assign):
                    target = child.targets[0] if child.targets else None
                    value = child.value
                else:
                    target, value = None, None
                target_name = ""
                if isinstance(target, ast.Name):
                    target_name = target.id
                if target_name == "categories":
                    with contextlib.suppress(ValueError, TypeError):
                        categories = list(ast.literal_eval(value))
                if target_name == "commands" and value is not None:
                    for call in (part for part in ast.walk(value) if isinstance(part, ast.Call)):
                        for keyword in call.keywords:
                            if keyword.arg in {"command", "template"} and isinstance(
                                keyword.value, ast.Constant
                            ):
                                commands.append(str(keyword.value.value))
            doc = ast.get_docstring(node) or ""
            metadata[node.name] = {
                "categories": categories,
                "commands": list(dict.fromkeys(commands)),
                "description": doc.split("\n\n", 1)[0].replace("\n", " ").strip(),
            }
    return metadata


def _dict_fields(node: ast.AST | None) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Dict):
        return {}
    return {
        str(key.value): value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
    }


def _literal(node: ast.AST | None, default: Any) -> Any:
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return default


def _fixture_rows(fixtures_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(fixtures_root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        data_nodes = [
            node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "DATA"
            and isinstance(node.value, ast.Dict)
        ]
        for data in data_nodes:
            for key, value in zip(data.keys, data.values, strict=True):
                if not isinstance(key, ast.Tuple) or len(key.elts) != 2:
                    continue
                test_node, case_node = key.elts
                if not isinstance(test_node, ast.Name) or not isinstance(case_node, ast.Constant):
                    continue
                case = str(case_node.value)
                if "failure" not in case:
                    continue
                fields = _dict_fields(value)
                expected = _dict_fields(fields.get("expected"))
                messages = _literal(expected.get("messages"), [])
                if not isinstance(messages, list) or not messages:
                    continue
                rows.append(
                    {
                        "test_name": test_node.id,
                        "case": case,
                        "inputs": _literal(fields.get("inputs"), {}),
                        "messages": [str(message) for message in messages],
                        "source_path": str(path.relative_to(config.ANTA_ROOT)),
                    }
                )
    return rows


def parse_official_fixture(raw: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    test_name = str(raw["test_name"])
    case = str(raw["case"])
    messages = raw["messages"]
    categories = list(metadata.get("categories") or ["system"])
    commands = list(metadata.get("commands") or [test_name])
    domain, protocol = anta_domain(categories)
    exact = all(EXACT_MISMATCH.search(message) for message in messages)
    diagnosis_type = "confirmed_state_mismatch" if exact else "diagnostic_guidance"
    issue = (
        f"{test_name} confirmed an EOS state mismatch: {'; '.join(messages)}"
        if exact
        else f"{test_name} did not reach its expected EOS operational state."
    )
    identifier = stable_hash(raw["source_path"], test_name, case)
    diagnosis = {
        "type": diagnosis_type,
        "cause": issue,
        "reason": (
            "The official ANTA failure fixture explicitly records different "
            "expected and actual values."
            if exact
            else "The official ANTA fixture records a failed test, but it does not "
            "establish the underlying root cause."
        ),
        "identify": [
            {"command": command, "expected_evidence": "; ".join(messages)}
            for command in commands
        ],
        "resolution": (
            [
                "Correct the EOS configuration or operational state so it matches "
                "the ANTA test input."
            ]
            if exact
            else []
        ),
        "verification": [
            {
                "command": f"Run {test_name} again.",
                "expected_result": "The ANTA test passes without the recorded failure messages.",
            }
        ],
    }
    row = {
        "id": "anta_" + identifier,
        "source": "anta",
        "platform": "eos",
        "network_domain": domain,
        "protocol": protocol,
        "problem": metadata.get("description") or f"The EOS validation {test_name} failed.",
        "evidence": [
            {
                "type": "expected_observed",
                "device": "EOS fixture device",
                "command": ", ".join(commands),
                "output": json.dumps(
                    {"test_inputs": raw.get("inputs", {}), "failure_messages": messages},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        ],
        "diagnoses": [diagnosis],
        "grounding": "official_test_fixture",
        "group_id": "anta_" + stable_hash(test_name, raw.get("inputs", {}), messages),
        "metadata": {
            "test_name": test_name,
            "fixture_case": case,
            "source_path": raw["source_path"],
        },
    }
    errors = validate_record(row)
    if errors:
        raise ValueError("; ".join(errors))
    return row


def parse_anta(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse an optional real ANTA export in the documented compact format."""
    expected, observed = raw.get("expected"), raw.get("observed")
    if expected == observed:
        raise ValueError("ANTA row is not a failed expected/observed check")
    diagnoses = raw.get("diagnoses")
    if not isinstance(diagnoses, list) or not diagnoses:
        raise ValueError("ANTA row needs one or more curated diagnoses")
    exact = bool(raw.get("direct_configuration_check")) and len(diagnoses) == 1
    diagnosis_type = "confirmed_state_mismatch" if exact else "diagnostic_guidance"
    normalized = []
    for diagnosis in diagnoses:
        normalized.append(
            {
                "type": diagnosis_type,
                "cause": diagnosis["cause"],
                "reason": diagnosis["reason"],
                "identify": diagnosis["identify"],
                "resolution": diagnosis["resolution"] if exact else [],
                "verification": diagnosis["verification"],
            }
        )
    identifier = str(
        raw.get("id") or stable_hash(raw.get("test_name"), raw.get("device"), expected, observed)
    )
    categories = [str(raw.get("category", "system"))]
    domain, protocol = anta_domain(categories)
    row = {
        "id": "anta_export_" + identifier,
        "source": "anta",
        "platform": "eos",
        "network_domain": domain,
        "protocol": protocol,
        "problem": str(raw.get("problem") or f"ANTA test {raw.get('test_name')} failed."),
        "evidence": [
            {
                "type": "expected_observed",
                "device": str(raw.get("device", "")),
                "command": str(raw.get("command", raw.get("test_name", "ANTA validation"))),
                "output": json.dumps(
                    {"expected": expected, "observed": observed}, ensure_ascii=False, sort_keys=True
                ),
            }
        ],
        "diagnoses": normalized,
        "grounding": "test_definition",
        "group_id": "anta_export_" + str(raw.get("group_id") or identifier),
        "metadata": {"test_name": raw.get("test_name")},
    }
    errors = validate_record(row)
    if errors:
        raise ValueError("; ".join(errors))
    return row


def build_anta() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_root = config.ANTA_ROOT / "anta" / "tests"
    fixtures_root = config.ANTA_ROOT / "tests" / "units" / "anta_tests"
    if source_root.exists() and fixtures_root.exists():
        metadata = _class_metadata(source_root)
        for raw in _fixture_rows(fixtures_root):
            try:
                rows.append(parse_official_fixture(raw, metadata.get(raw["test_name"], {})))
            except (KeyError, TypeError, ValueError) as error:
                rejected.append(
                    {
                        "source": "anta",
                        "case": f"{raw['test_name']}:{raw['case']}",
                        "reason": str(error),
                    }
                )
    else:
        rejected.append({"source": "anta", "reason": "official_snapshot_not_found"})

    for line_number, raw in enumerate(read_jsonl(config.ANTA_INPUT), 1):
        try:
            rows.append(parse_anta(raw))
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"source": "anta", "line": line_number, "reason": str(error)})
    return rows, rejected
