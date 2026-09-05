"""Deterministic tests for MCP tool functions in llm.mcp (happy paths and
parameter honoring). Path-sandbox security is covered by test_llm_mcp_security.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.defaults import DEFAULT_OLLAMA_MODEL
from llm.mcp import (
    _ANALYSIS_TYPES,
    analyze_gnn_with_llm_mcp,
    get_llm_module_info_mcp,
    get_llm_providers_mcp,
    process_llm_mcp,
    register_tools,
)

pytestmark = pytest.mark.unit

SAMPLE_GNN = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)


class _RecordingMCP:
    def __init__(self) -> None:
        self.registered: list[tuple[str, object, dict, str]] = []

    def register_tool(
        self,
        name: str,
        handler: object,
        schema: dict,
        description: str,
        **kwargs: object,
    ) -> None:
        self.registered.append((name, handler, schema, description))


class TestAnalyzeGnnWithLLMMCP:
    def test_rejects_unknown_analysis_type(self) -> None:
        result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), analysis_type="bogus")
        assert result["success"] is False
        assert "bogus" in result["error"]
        assert "comprehensive" in result["error"]

    def test_provider_override_pins_default_ollama_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_analyze(path: Path, ollama_model: str | None = None) -> dict:
            captured["model"] = ollama_model
            return {"analysis": "ok"}

        monkeypatch.setattr("llm.mcp.analyze_gnn_file_with_llm", fake_analyze)

        result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), provider="ollama")
        assert result["success"] is True
        assert captured["model"] == DEFAULT_OLLAMA_MODEL
        assert result["provider"] == "ollama"
        assert result["analysis_type"] == "comprehensive"

    def test_no_provider_uses_default_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_analyze(path: Path, ollama_model: str | None = None) -> dict:
            captured["model"] = ollama_model
            return {"analysis": "ok"}

        monkeypatch.setattr("llm.mcp.analyze_gnn_file_with_llm", fake_analyze)
        result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), provider=None)
        assert result["success"] is True
        assert captured["model"] is None

    def test_analysis_type_enum_matches_schema(self) -> None:
        assert _ANALYSIS_TYPES == (
            "comprehensive",
            "summary",
            "complexity",
            "connections",
        )


class TestProviderMatrix:
    def test_all_registered_providers_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in (
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "PERPLEXITY_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.setenv(key, "test-key")
        result = get_llm_providers_mcp()
        assert result["success"] is True
        for name in (
            "openai",
            "anthropic",
            "ollama",
            "google",
            "openrouter",
            "perplexity",
        ):
            assert name in result["providers"], name
        for cloud in ("openai", "openrouter", "perplexity", "anthropic"):
            assert result["providers"][cloud]["configured"] is True
        assert result["count"] == len(result["providers"])


class TestModuleInfo:
    def test_lists_all_registered_tools(self) -> None:
        result = get_llm_module_info_mcp()
        assert result["success"] is True
        assert result["tools"] == [
            "process_llm",
            "analyze_gnn_with_llm",
            "generate_llm_documentation",
            "get_llm_providers",
            "get_llm_module_info",
        ]


class TestRegisterTools:
    def test_registers_all_tools_table_driven(self) -> None:
        mcp = _RecordingMCP()
        register_tools(mcp)
        names = [name for name, _, _, _ in mcp.registered]
        assert names == [
            "process_llm",
            "analyze_gnn_with_llm",
            "generate_llm_documentation",
            "get_llm_providers",
            "get_llm_module_info",
        ]
        for _, _, schema, description in mcp.registered:
            assert isinstance(schema, dict)
            assert description


class TestProcessLLMMCP:
    def test_success_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_process_llm(**kwargs: object) -> bool:
            captured.update(kwargs)
            return True

        monkeypatch.setattr("llm.mcp.process_llm", fake_process_llm)
        result = process_llm_mcp(
            str(SAMPLE_GNN.parent.parent), str(SAMPLE_GNN.parent), verbose=True
        )
        assert result["success"] is True
        assert captured["verbose"] is True
