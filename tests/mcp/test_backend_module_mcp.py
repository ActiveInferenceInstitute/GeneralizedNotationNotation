"""Registration contract for the backend-module MCP tools (processing, schema_validator).

The two backend modules register their module-owned tools through the same
discovery path as every other ``mcp.py`` entry point (``gnn.mcp.mcp.MCP.
discover_modules``), so the live registry surface must match the authored
tools. These tests run the real discovery scoped via ``modules_allowlist``
so it stays fast and deterministic; the duplicate-tracking wrapper from
``gnn.mcp.validate_tools`` proves the registration is idempotent under a
forced re-initialize.
"""

from typing import Any

import pytest

from gnn.mcp.mcp import MCP, initialize
from gnn.mcp.validate_tools import _DuplicateRegistrationTracker

pytestmark = pytest.mark.mcp

BACKEND_MODULES = ("processing", "schema_validator")

EXPECTED_TOOLS: dict[str, tuple[str, ...]] = {
    "processing": (
        "processing.check_gnn_file_structure",
        "processing.discover_gnn_files",
        "processing.parse_gnn_file",
    ),
    "schema_validator": (
        "schema_validator.parse_syntax",
        "schema_validator.validate_comprehensive",
    ),
}


def _initialize_scoped() -> MCP:
    """Run full initialization scoped to the two backend modules."""
    mcp, _, _ = initialize(
        halt_on_missing_sdk=False,
        force_proceed_flag=True,
        force_refresh=True,
        modules_allowlist=list(BACKEND_MODULES),
    )
    return mcp


@pytest.fixture(scope="module")
def mcp_scoped() -> MCP:
    """Return an MCP instance initialized with only the backend modules."""
    return _initialize_scoped()


class TestBackendModuleRegistration:
    """Both backend modules register their tools with unique names."""

    def test_both_modules_register_at_least_one_tool(self, mcp_scoped: MCP) -> None:
        for mod_name in BACKEND_MODULES:
            info = mcp_scoped.modules.get(mod_name)
            assert info is not None, f"module {mod_name} not discovered"
            assert info.status == "loaded", (
                f"module {mod_name} status={info.status}: {info.error_message}"
            )
            assert info.tools_count >= 1, (
                f"module {mod_name} registered {info.tools_count} tools"
            )

    def test_all_expected_tools_registered_with_unique_names(
        self, mcp_scoped: MCP
    ) -> None:
        registered = [t for tools in EXPECTED_TOOLS.values() for t in tools]
        assert len(registered) == len(set(registered)), "tool names must be unique"
        for name in registered:
            assert name in mcp_scoped.tools, f"missing tool: {name}"

    def test_tool_module_metadata_matches_directory_name(
        self, mcp_scoped: MCP
    ) -> None:
        """Each tool's module metadata must reflect its owning directory
        (defaulted through the discovery registration context)."""
        for mod_name, tools in EXPECTED_TOOLS.items():
            for name in tools:
                tool = mcp_scoped.tools[name]
                assert tool.module == f"gnn.{mod_name}", (
                    f"{name}: module metadata {tool.module!r} != gnn.{mod_name}"
                )
                assert tool.category == mod_name, (
                    f"{name}: category metadata {tool.category!r} != {mod_name}"
                )

    def test_duplicate_registration_detected_on_double_initialize(self) -> None:
        """A second forced initialize must not produce duplicate registrations.

        The audit-time duplicate tracker wraps ``register_tool`` during
        ``initialize``; re-running discovery from scratch must yield exactly
        the expected tool set with zero duplicate-name events.
        """
        tracker = _DuplicateRegistrationTracker()
        tracker.install()
        try:
            mcp = _initialize_scoped()
        finally:
            tracker.restore()

        expected = {name for tools in EXPECTED_TOOLS.values() for name in tools}
        backend_tools = {
            name
            for name, tool in mcp.tools.items()
            if tool.module in ("gnn.processing", "gnn.schema_validator")
        }
        assert backend_tools == expected
        assert set(tracker.duplicates) == set(), (
            f"duplicate registrations detected: {tracker.duplicates}"
        )


class TestBackendToolExecution:
    """The lightest tool of each module executes via execute_tool."""

    MINIMAL_GNN = (
        "# GNN\n"
        "version: 1.0.0\n"
        "\n"
        "## ModelName\n"
        "SmokeModel\n"
        "\n"
        "## StateSpaceBlock\n"
        "s_f [2,1], type=float\n"
    )

    def test_processing_discover_gnn_files_executes(
        self, mcp_scoped: MCP, tmp_path: Any
    ) -> None:
        (tmp_path / "model.md").write_text(self.MINIMAL_GNN, encoding="utf-8")
        result: dict[str, Any] = mcp_scoped.execute_tool(
            "processing.discover_gnn_files", {"directory": str(tmp_path)}
        )
        assert result["success"] is True
        assert result["count"] == 1
        assert result["files"] == [str(tmp_path / "model.md")]

    def test_schema_validator_validate_comprehensive_executes(
        self, mcp_scoped: MCP, tmp_path: Any
    ) -> None:
        gnn_file = tmp_path / "model.md"
        gnn_file.write_text(self.MINIMAL_GNN, encoding="utf-8")
        result: dict[str, Any] = mcp_scoped.execute_tool(
            "schema_validator.validate_comprehensive",
            {"file_path": str(gnn_file)},
        )
        assert result["success"] is True
        # The tool itself must succeed and surface the pipeline's structural
        # verdict; a skeletal file's validity is asserted only as a bool.
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["errors"], list)
