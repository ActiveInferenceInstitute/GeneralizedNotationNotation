# GNN MCP Tool Development Guide

How to add real, tested MCP tools to any GNN pipeline module.

**Last Updated**: 2026-08-07

## Design Principles

All GNN MCP tools follow six non-negotiable constraints enforced by `test_mcp_audit.py`:

1. **Real named functions** — no lambdas, no `None`, no generic wrappers like `list_functions`
2. **Non-empty descriptions** — every tool must have a docstring or explicit description
3. **Valid JSON schema** — every tool passes a schema that satisfies `MCPTool.validate_schema()`
4. **Non-empty module and category metadata**
5. **Logger call** — `register_tools()` must call `logger.info(...)` naming the real registered count
6. **Real behavior** — the function must do real work, and be callable via `execute_tool` with no required arguments

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
from importlib.metadata import version
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
            "version": version("generalized-notation-notation"),
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

    # 2️⃣  Register each tool. The signature is
    #     register_tool(name, func, schema, description, module=..., category=...)
    #     — the JSON schema is positional third and is NOT optional in practice:
    #     the audit validates it, and module/category must be non-empty.
    server.register_tool(
        "process_<module>",
        process_<module>,
        {
            "type": "object",
            "properties": {
                "target_dir": {
                    "type": "string",
                    "description": "Directory containing GNN files to process",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory to write results to",
                },
            },
            "required": [],
        },
        "Run <module> processing pipeline on all GNN files in target_dir.",
        module="<module>",
        category="processing",
    )
    server.register_tool(
        "get_<module>_info",
        get_<module>_info,
        {"type": "object", "properties": {}, "required": []},
        "Return <module> module version and capabilities.",
        module="<module>",
        category="introspection",
    )
    server.register_tool(
        "list_<module>_options",
        list_<module>_options,
        {"type": "object", "properties": {}, "required": []},
        "List configurable options for <module> processing.",
        module="<module>",
        category="introspection",
    )

    # 3️⃣  Required logger.info naming the real registered count
    logger.info("Registered 3 <module> MCP tools")
```

Every tool must be callable through `execute_tool` **with no arguments**, so keep required-argument lists empty and give parameters sensible defaults.

## Checklist Before Submitting

- [ ] All functions are **named** (`def my_tool():`, not `lambda:`)
- [ ] All functions have a **docstring** (the description passed to `register_tool`)
- [ ] The `logger.info` count matches the actual number of `server.register_tool` calls
- [ ] Functions call **real module code** and return structured results
- [ ] `AGENTS.md` for the module lists the new tools
- [ ] The module's `doc/gnn/modules/NN_<module>.md` has an MCP Tools section

## Running the Audit

```bash
# Full MCP audit (part of src/tests/; suite counts in repository README.md)
uv run --extra dev python -m pytest src/tests/mcp/test_mcp_audit.py -v

# Focus on your new module
uv run --extra dev python -m pytest src/tests/mcp/test_mcp_audit.py -v -k "<module>"

# Generate the tool inventory JSON
uv run python src/mcp/validate_tools.py
```

## What the Audit Tests

| Test Class | What It Checks |
|------------|---------------|
| `TestMCPModuleDiscovery` | Every expected module is registered and its `register_tools` is callable |
| `TestMCPToolRealness` | Named backing functions, non-empty module/category metadata, valid JSON schemas, no generic catch-all tools (`list_functions`, `call_function`) |
| `TestMCPDomainTools` | Each module registers its expected domain tools: callable + description not empty |
| `TestMCPToolExecution` | Tools are callable live via `execute_tool` with no required arguments |
| `TestMCPLoggingCoverage` | Every `mcp.py` calls `logger.info` in `register_tools` |
| `TestMCPAuditReport` | JSON report generated with correct schema |

The expected-module list is `EXPECTED_MODULES` in `src/tests/mcp/test_mcp_audit.py`.

If your new tools follow the canonical pattern above, the audit will pass automatically.

## Adding a New Module

If you are adding a **brand-new** pipeline module (e.g., step 25+):

1. Create `src/<module>/mcp.py` following the pattern above
2. No manual registration step is needed: `MCP.discover_modules()` in
   `src/mcp/mcp.py` dynamically scans the `src/` directory at runtime for
   any subdirectory containing an `mcp.py` file and loads it automatically
   (there is no static module list or `mcp_instance.py` to edit)
3. Add the module name to `EXPECTED_MODULES` in `src/tests/mcp/test_mcp_audit.py`
4. Update `doc/gnn/mcp/tool_reference.md` with the new tools
5. Create `doc/gnn/modules/NN_<module>.md` with an MCP Tools section

## See Also

- [Tool Reference](tool_reference.md) — existing tools
- [modules/21_mcp.md](../modules/21_mcp.md) — pipeline step documentation
- [doc/mcp/fastmcp.md](../../../doc/mcp/fastmcp.md) — FastMCP library internals
- [src/mcp/](../../../src/mcp/) — server implementation source
