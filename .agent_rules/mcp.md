# MCP (Model Context Protocol) Integration

> MCP enables standardized tool registration across all GNN pipeline modules.

## Standard `mcp.py` Structure

```python
"""
MCP Integration for module_name.

Exposes module functionality via Model Context Protocol.
"""
import logging
from typing import Dict, Any, Optional
logger = logging.getLogger(__name__)

def function_name_mcp(mcp_instance_ref=None, **kwargs) -> Dict[str, Any]:
    """
    MCP tool for function description.

    Args:
        mcp_instance_ref: Optional MCP server reference.
        **kwargs: Tool-specific parameters.

    Returns:
        Standardized MCP response dictionary.
    """
    try:
        # Validate required args
        required = ["param1", "param2"]
        for arg in required:
            if arg not in kwargs:
                return {
                    "success": False,
                    "error": f"Missing required argument: {arg}",
                    "error_type": "missing_argument",
                }

        from .processor import function_name
        result = function_name(**kwargs)

        return {
            "success": True,
            "result": result,
            "operation": "function_name",
            "module": "module_name",
        }

    except ImportError as e:
        return {"success": False, "error": str(e), "error_type": "import_error"}
    except ValueError as e:
        return {"success": False, "error": str(e), "error_type": "validation_error"}
    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        return {"success": False, "error": str(e), "error_type": "execution_error"}

def register_tools(mcp=None) -> None:
    """Register all MCP tools with the central server."""
    if mcp is None:
        from mcp.mcp import mcp_instance as mcp

    mcp.register_tool(
        "tool_name",
        function_name_mcp,
        {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Description"},
                "param2": {"type": "boolean", "description": "Description", "default": False},
            },
            "required": ["param1"],
        },
        "Human-readable description of what this tool does",
        module=__package__,
        category="module_name",
    )
    logger.info("module_name: MCP tools registered.")
```

---

## Error Response Format

```python
# Success
{"success": True, "result": {...}, "operation": "...", "module": "...", "execution_time_ms": 123}

# Failure
{
    "success": False,
    "error": "Human-readable message",
    "error_type": "missing_argument|validation_error|execution_error|import_error|timeout_error",
    "details": {"argument": "param1", "expected": "string"}
}
```

---

## Error Type Reference

| `error_type` | Cause |
|-------------|-------|
| `missing_argument` | Required kwarg not provided |
| `validation_error` | Parameter fails validation |
| `execution_error` | Tool logic raised exception |
| `import_error` | Module or dependency not available |
| `not_found` | Requested resource doesn't exist |
| `timeout_error` | Operation exceeded time limit |
| `permission_error` | Access denied |

---

## Tool Categories by Module

| Step | Module | Tools |
|------|--------|-------|
| 3 | gnn | `parse_gnn_content`, `validate_gnn_content`, `process_gnn_directory`, `get_gnn_documentation` |
| 5 | type_checker | `validate_gnn_files`, `validate_single_gnn_file` |
| 7 | export | `process_export`, `export_single_gnn_file`, `list_export_formats`, `validate_export_format` |
| 8 | visualization | `process_visualization`, `get_visualization_options`, `list_visualization_artifacts` |
| 11 | render | `process_render`, `list_render_frameworks`, `render_gnn_to_format` |
| 13 | llm | `process_llm`, `analyze_gnn_with_llm`, `generate_llm_documentation`, `get_llm_providers` |
| 15 | audio | `process_audio`, `check_audio_backends`, `get_audio_generation_options` |
| 21 | mcp | Central registry; verify live inventory with `src/tests/mcp/test_mcp_audit.py` |

---

## MCP Server Operations

```bash
# List all registered tools
python -m src.mcp.cli list

# Call a tool
python -m src.mcp.cli execute parse_gnn_content \
  --params '{"content":"## GNNSection\nActInfPOMDP\n","format_hint":"markdown","enhanced_validation":true}'

# Run MCP step to register all tools
python src/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose
```

---

## Central Registry (`src/mcp/`)

All module `mcp.py` files auto-register with the central registry at startup:
- **Discovery**: Automatic via `src/mcp/processor.py::register_module_tools()`
- **No manual configuration** needed — modules found by directory scan
- **Inventory**: Run `uv run --extra dev python -m pytest src/tests/mcp/test_mcp_audit.py -q` for the current registered tool/module contract

---

**Last Updated**: 2026-06-14 | **Status**: Maintained Standard
