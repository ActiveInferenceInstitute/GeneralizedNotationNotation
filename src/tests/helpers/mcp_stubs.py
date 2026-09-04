"""In-memory MCP registry stub shared by module MCP-wiring tests.

Many module test directories previously re-declared near-identical
``_CapturingMCP`` / ``StubMCP`` / ``_FakeMCP`` classes. This is the canonical
implementation; the conftest ``test_mcp_tools`` fixture returns an instance.
"""

from __future__ import annotations

from typing import Any, Callable


class MCPTools:
    """Lightweight in-memory MCP registry used by MCP wiring tests.

    Records registered tools/resources and executes tools by name, mirroring
    the small subset of the MCP server surface the wiring tests exercise.
    """

    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, dict[str, Any]] = {}

    def register_tool(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Register a tool; accepts both positional and keyword conventions."""
        function: Callable[..., Any] | None = kwargs.get("function")
        schema = kwargs.get("schema")
        description = kwargs.get("description", "")
        if function is None and args:
            function = args[0]
            if len(args) >= 2 and schema is None:
                schema = args[1]
            if len(args) >= 3 and not description:
                description = args[2]
        self.tools[name] = {
            "function": function,
            "func": function,
            "schema": schema or {},
            "description": description,
        }

    def register_resource(
        self, pattern: str, handler: Any, description: str = ""
    ) -> None:
        """Register a resource handler under a URI pattern."""
        self.resources[pattern] = {"handler": handler, "description": description}

    def execute_tool(self, name: str, **kwargs: Any) -> Any:
        """Invoke a registered tool; unknown names return an error payload."""
        if name not in self.tools:
            return {"error": "tool_not_found", "name": name}
        entry = self.tools[name]
        function = entry.get("function") or entry.get("func")
        assert function is not None, f"tool '{name}' registered without function"
        return function(**kwargs)


__all__ = ["MCPTools"]
