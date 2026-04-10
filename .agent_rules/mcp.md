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

# Registry with full metadata
MCP_TOOLS: Dict[str, Dict[str, Any]] = {
    "tool_name": {
        "function": function_name_mcp,
        "description": "Human-readable description of what this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Description"},
                "param2": {"type": "boolean", "description": "Description", "default": False},
            },
            "required": ["param1"],
        },
    }
}

def register_tools(mcp=None) -> None:
    """Register all MCP tools with the central server."""
    logger.info(f"module_name: {len(MCP_TOOLS)} MCP tools registered.")
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
| 3 | gnn | `gnn_parse`, `gnn_validate`, `gnn_serialize`, `gnn_discover` |
| 5 | type_checker | `validate_gnn_types`, `check_syntax`, `estimate_resources` |
| 7 | export | `export_to_json`, `export_to_xml`, `export_to_yaml`, `export_to_graphml` |
| 8 | visualization | `visualize_graph`, `visualize_matrices`, `export_visualization` |
| 11 | render | `render_pymdp`, `render_rxinfer`, `render_activeinference`, `render_jax`, `render_discopy` |
| 13 | llm | `analyze_gnn`, `explain_model`, `suggest_improvements` |
| 15 | audio | `generate_sapf`, `sonify_model` |
| 21 | mcp | Central registry — 131 tools total across 30 modules |

---

## MCP Server Operations

```bash
# List all registered tools
curl http://localhost:8000/tools

# Call a tool
curl -X POST http://localhost:8000/tools/gnn_parse \
  -H "Content-Type: application/json" \
  -d '{"file_path": "model.md"}'

# Run MCP step to register all tools
python src/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose
```

---

## Central Registry (`src/mcp/`)

All module `mcp.py` files auto-register with the central registry at startup:
- **Discovery**: Automatic via `src/mcp/processor.py::register_module_tools()`
- **No manual configuration** needed — modules found by directory scan
- **131 tools** across 30 modules (as of pipeline v1.3.0)

---

**Last Updated**: March 2026 | **Status**: Production Standard
