# GNN MCP Audit Framework

How `src/tests/test_mcp_audit.py` validates the MCP tool registry.

**Last Updated**: 2026-04-15  
**Source**: [`src/tests/test_mcp_audit.py`](../../../src/tests/test_mcp_audit.py)

## Fixture Design

The module-scoped fixture calls `initialize()` and then **convergence-polls** until the tool count stabilises:

```python
@pytest.fixture(scope="module")
def mcp_initialized():
    from mcp import initialize, mcp_instance
    initialize(halt_on_missing_sdk=False, force_proceed_flag=True)

    # Wait up to 5 s for background registration threads to finish
    prev_count = -1
    for _ in range(25):        # 25 × 0.2 s = 5 s max
        current = len(mcp_instance.tools)
        if current == prev_count:
            break
        prev_count = current
        time.sleep(0.2)

    return mcp_instance
```

Two derived fixtures: `all_tools` (dict snapshot) and `all_modules` (module info dict).

## Test Classes

The audit file contains six test classes (`TestMCPModuleDiscovery`, `TestMCPToolRealness`, `TestMCPDomainTools`, `TestMCPToolExecution`, `TestMCPLoggingCoverage`, `TestMCPAuditReport`). The three primary classes are documented below.

### `TestMCPModuleDiscovery`

22 expected modules × 2 parametrized checks = **44 tests**

```python
EXPECTED_MODULES = [
    "gnn", "utils", "website", "analysis", "render", "export",
    "validation", "ontology", "visualization", "report", "integration",
    "security", "research", "sapf", "audio", "execute", "llm",
    "advanced_visualization", "ml_integration", "intelligent_analysis",
    "gui", "pipeline",
]
```

- `test_expected_module_loaded` — module appears in `all_modules` OR contributes tools
- `test_expected_module_has_tools` — module contributes ≥ 1 registered tool

### `TestMCPToolRealness`

5 aggregate assertions across all registered tools:

| Test | What It Checks |
|------|---------------|
| `test_at_least_50_tools_registered` | `len(all_tools) >= 50` |
| `test_all_tools_have_callable_funcs` | `callable(tool.func)` for every tool |
| `test_no_lambda_tools` | `tool.func.__name__ != "<lambda>"` for every tool |
| `test_all_tools_have_named_functions` | `tool.func.__name__` is non-empty |
| `test_all_tools_have_descriptions` | `tool.description.strip()` is non-empty |

### `TestMCPDomainTools`

76 expected tool names × 1 check = **76 tests**:

```python
DOMAIN_TOOLS = [
    "process_analysis", "get_analysis_results", "compute_complexity_metrics",
    "list_analysis_tools",
    "process_render", "list_render_frameworks", "render_gnn_to_format",
    "process_export", "list_export_formats", "validate_export_format",
    "process_validation", "validate_gnn_file", ...
    # (full list: mcp/tool_reference.md)
]
```

## Running the Audit

```bash
# Full audit (all three classes)
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py -v

# Single class
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py::TestMCPModuleDiscovery -v
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py::TestMCPToolRealness -v
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py::TestMCPDomainTools -v

# JSON report only
PYTHONPATH=src python src/mcp/validate_tools.py
```

## Adding a New Tool to the Audit

1. Add the tool name string to `DOMAIN_TOOLS` in `TestMCPDomainTools`
2. Ensure the module appears in `EXPECTED_MODULES` (or is covered by an existing one)
3. Run the audit — all 3 new assertions should pass

## Why Convergence Polling, Not `time.sleep()`

The MCP server registers modules synchronously but reverts to background threads for modules that time out. A fixed `time.sleep()` can still race if the machine is under load. The polling loop stops as soon as the count stops growing, so it is both faster on idle machines and more robust under load.

## See Also

- [mcp/tool_development_guide.md](../mcp/tool_development_guide.md) — No-placeholder policy, canonical mcp.py pattern
- [mcp/tool_reference.md](../mcp/tool_reference.md) — full DOMAIN_TOOLS list in table form
- [testing/test_patterns.md](test_patterns.md) — general test patterns and fixtures
