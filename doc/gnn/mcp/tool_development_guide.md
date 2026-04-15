# GNN MCP Tool Development Guide

How to add real, tested MCP tools to any GNN pipeline module.

**Last Updated**: 2026-04-15

## Design Principles

All GNN MCP tools follow four non-negotiable constraints enforced by `test_mcp_audit.py`:

1. **Real named functions** — no lambdas, no `None`, no generic wrappers like `list_functions`
2. **Non-empty descriptions** — every tool must have a docstring or explicit description
3. **Logger call** — `register_tools()` must call `logger.info(f"Registered N tools")` with the real count
4. **Zero placeholders** — the function must do real work, not just return a placeholder

## Module File Structure

Every pipeline module that exposes MCP tools has this layout:

```
src/<module>/
├── __init__.py            ← module exports
├── mcp.py                 ← MCP tool registration (YOU EDIT THIS)
├── processor.py           ← core implementation
└── AGENTS.md              ← module documentation
```

## Writing a New `mcp.py`

Here is the canonical pattern, fully annotated:

```python
"""MCP tools for the <module> module."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(server: Any) -> None:
    """Register <module> MCP tools with the server.

    Registers 3 tools: process_<module>, get_<module>_info, list_<module>_options.
    """
    # 1️⃣  Real named function — NOT a lambda
    def process_<module>(target_dir: str, output_dir: str = "output") -> dict:
        """Run <module> processing pipeline on all GNN files in target_dir."""
        from pathlib import Path
        from .<module>_processor import process_<module>_main
        success = process_<module>_main(
            Path(target_dir), Path(output_dir), logger
        )
        return {"success": success, "output_dir": output_dir}

    def get_<module>_info() -> dict:
        """Return <module> module version and capabilities."""
        return {
            "module": "<module>",
            "version": "2.0.0",
            "capabilities": ["feature_a", "feature_b"],
        }

    def list_<module>_options() -> dict:
        """List configurable options for <module> processing."""
        return {
            "options": {
                "verbose": "bool — enable verbose logging",
                "recursive": "bool — process sub-directories",
            }
        }

    # 2️⃣  Register all tools (each needs a name + description)
    server.register_tool(
        name="process_<module>",
        description="Run <module> processing pipeline on all GNN files in target_dir.",
        func=process_<module>,
    )
    server.register_tool(
        name="get_<module>_info",
        description="Return <module> module version and capabilities.",
        func=get_<module>_info,
    )
    server.register_tool(
        name="list_<module>_options",
        description="List configurable options for <module> processing.",
        func=list_<module>_options,
    )

    # 3️⃣  Required logger.info with exact count
    logger.info("Registered 3 <module> MCP tools")
```

## Checklist Before Submitting

- [ ] All functions are **named** (`def my_tool():`, not `lambda:`)
- [ ] All functions have a **docstring** (the description passed to `register_tool`)
- [ ] The `logger.info` count matches the actual number of `server.register_tool` calls
- [ ] Functions call **real module code** (not `return {}` placeholders)
- [ ] `AGENTS.md` for the module lists the new tools
- [ ] The module's `doc/gnn/modules/NN_<module>.md` has an MCP Tools section

## Running the Audit

```bash
# Full MCP audit (part of src/tests/; suite counts in repository README.md)
uv run pytest src/tests/test_mcp_audit.py -v

# Focus on your new module
uv run pytest src/tests/test_mcp_audit.py -v -k "<module>"

# Generate the tool inventory JSON
uv run python src/mcp/validate_tools.py
# → src/tests/mcp_audit_report.json
```

## What the Audit Tests

| Test Class | What It Checks |
|------------|---------------|
| `TestMCPModuleDiscovery` | 22 modules × 2: module registered + `register_tools` is callable |
| `TestMCPDomainTools` | 131 tools × 2: tool callable + description not empty |
| `TestMCPToolRealness` | No generic placeholders (`list_functions`, `call_function`) |
| `TestMCPLoggingCoverage` | Every `mcp.py` calls `logger.info` in `register_tools` |
| `TestMCPAuditReport` | JSON report generated with correct schema |

If your new tools follow the canonical pattern above, the audit will pass automatically.

## Adding a New Module

If you are adding a **brand-new** pipeline module (e.g., step 25+):

1. Create `src/<module>/mcp.py` following the pattern above
2. Register the module in `src/mcp/mcp_instance.py`'s `_discover_modules()` list
3. Add a row to the `TestMCPModuleDiscovery` fixture in `src/tests/test_mcp_audit.py`
4. Update `doc/gnn/mcp/tool_reference.md` with the new tools
5. Create `doc/gnn/modules/NN_<module>.md` with an MCP Tools section

## See Also

- [Tool Reference](tool_reference.md) — existing tools
- [modules/21_mcp.md](../modules/21_mcp.md) — pipeline step documentation
- [doc/mcp/fastmcp.md](../../../doc/mcp/fastmcp.md) — FastMCP library internals
- [src/mcp/](../../../src/mcp/) — server implementation source
