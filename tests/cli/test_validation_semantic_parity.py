"""Shared semantic evidence preserves each interface's exit/response policy."""

import json
from pathlib import Path

import pytest

import gnn.cli as cli
from gnn.validation import process_semantic_validation, validate_content
from gnn.validation.mcp import validate_gnn_file_mcp

CONTENT = """## ModelName
M
## StateSpaceBlock
StateSpaceBlock {
 Name: s
 Dimensions: 0
}
## Connections
"""


@pytest.mark.parametrize(
    "strict, expected", [(False, cli.EXIT_WARNING), (True, cli.EXIT_ERROR)]
)
def test_cli_exposes_shared_semantics_and_preserves_exit_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], strict: bool, expected: int
) -> None:
    model = tmp_path / "model.md"
    model.write_text(CONTENT)
    args = cli.build_parser().parse_args(
        ["validate", str(model), "--json", *(["--strict"] if strict else [])]
    )
    assert cli._cmd_validate(args) == expected
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["semantic"]["valid"] is False
    assert result["data"]["semantic"]["errors"] == validate_content(CONTENT)["errors"]


def test_mcp_top_level_verdict_includes_semantic_failure(tmp_path: Path) -> None:
    model = tmp_path / "model.md"
    model.write_text(CONTENT)
    result = validate_gnn_file_mcp(str(model))
    assert result["success"] is True  # tool completed; the model is invalid
    assert result["is_valid"] is False
    assert set(result["semantic"]["errors"]) <= set(result["errors"])


def test_raw_sections_and_content_have_identical_semantic_evidence() -> None:
    sections = {
        "ModelName": "M",
        "StateSpaceBlock": "s[2]\no[2]",
        "Connections": "s>missing:label",
    }
    content = "\n\n".join(f"## {name}\n{body}" for name, body in sections.items())
    direct = validate_content(content)
    parsed = process_semantic_validation({"raw_sections": sections})
    assert direct["valid"] is False
    for field in ("valid", "errors", "warnings", "semantic_score"):
        assert parsed[field] == direct[field]


def test_grouped_structured_endpoints_have_same_semantic_verdict() -> None:
    structured = {
        "variables": [{"name": name, "dimensions": [2]} for name in ("s", "o")],
        "connections": [
            {
                "source_variables": ["s"],
                "target_variables": ["o", "missing"],
                "annotation": "edge",
            }
        ],
    }
    direct = validate_content(
        "## StateSpaceBlock\ns[2]\no[2]\n## Connections\ns>(o,missing):edge\n"
    )
    restored = process_semantic_validation(structured)
    assert direct["valid"] is False
    assert restored["errors"] == direct["errors"]


@pytest.mark.parametrize("dimensions", ["-1", "0", "2,-3"])
def test_canonical_nonpositive_dimensions_are_invalid(dimensions: str) -> None:
    result = validate_content(f"## StateSpaceBlock\ns[{dimensions}]\n")
    assert result["valid"] is False
    assert any("dimension" in error.lower() for error in result["errors"])


@pytest.mark.parametrize(
    "state_space", ["# Invalid example s[-1]\ns[2]", "s[2] # index offsets [-1, 0, 1]"]
)
def test_dimension_checks_ignore_comments(state_space: str) -> None:
    assert validate_content(f"## StateSpaceBlock\n{state_space}\n")["valid"] is True
