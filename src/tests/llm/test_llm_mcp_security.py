from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm import mcp as llm_mcp

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_GNN = REPO_ROOT / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"


def test_analyze_gnn_with_llm_rejects_outside_file_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "secret.md"
    secret.write_text("not a repository model", encoding="utf-8")

    def fail_if_called(_path: Path) -> dict[str, Any]:
        raise AssertionError("provider-facing analyzer should not be called")

    monkeypatch.setattr(llm_mcp, "analyze_gnn_file_with_llm", fail_if_called)

    result = llm_mcp.analyze_gnn_with_llm_mcp(str(secret))

    assert result["success"] is False
    assert "repository root" in result["error"]


def test_generate_llm_documentation_rejects_outside_output_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside_output = tmp_path / "doc.json"

    def fail_if_called(_analysis: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("documentation generator should not be called")

    monkeypatch.setattr(llm_mcp, "generate_documentation", fail_if_called)

    result = llm_mcp.generate_llm_documentation_mcp(
        str(SAMPLE_GNN),
        output_path=str(outside_output),
    )

    assert result["success"] is False
    assert "repository root" in result["error"]
    assert not outside_output.exists()
