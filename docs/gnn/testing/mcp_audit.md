# GNN MCP Audit Framework

How `tests/mcp/test_mcp_audit.py` validates the MCP tool registry.

**Last Updated**: 2026-09-22  
**Source**: [`tests/mcp/test_mcp_audit.py`](../../../tests/mcp/test_mcp_audit.py)

## Fixture Design

The module-scoped fixture calls `initialize()` and then **convergence-polls** until the tool count stabilises:

```python
@pytest.fixture(scope="module")
def mcp_initialized():
    from mcp import initialize, mcp_instance

    initialize(halt_on_missing_sdk=False, force_proceed_flag=True)

    # Wait up to 5 s for background registration threads to finish
    prev_count = -1
    for _ in range(25):  # 25 × 0.2 s = 5 s max
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

32 expected modules × 2 parametrized checks = **64 tests**

```python
EXPECTED_MODULES = (
    "advanced_visualization",
    "analysis",
    "api",
    "audio",
    "cli",
    "doc",
    "execute",
    "export",
    "gnn",
    "integration",
    "intelligent_analysis",
    "llm",
    "lsp",
    "mcp",
    "ml_integration",
    "model_registry",
    "ontology",
    "gui",
    "pipeline",
    "render",
    "report",
    "research",
    "sapf",
    "security",
    "setup",
    "sympy_mcp",
    "template",
    "type_checker",
    "utils",
    "validation",
    "visualization",
    "website",
)
```

- `test_expected_module_loaded` — module appears in `all_modules` OR contributes tools
- `test_expected_module_has_tools` — module contributes ≥ 1 registered tool, *except* the
  two modules in `ZERO_TOOL_MODULES = ("doc", "lsp")`, which are asserted to load and
  contribute exactly 0 tools

### `TestMCPToolRealness`

7 aggregate assertions across all registered tools:

| Test | What It Checks |
|------|---------------|
| `test_at_least_50_tools_registered` | `len(all_tools) >= 50` |
| `test_all_tools_have_callable_funcs` | `callable(tool.func)` for every tool |
| `test_no_lambda_tools` | `tool.func.__name__ != "<lambda>"` for every tool |
| `test_all_tools_have_named_functions` | `tool.func.__name__` is non-empty |
| `test_all_tools_have_descriptions` | `tool.description.strip()` is non-empty |
| `test_all_tools_have_module_and_category_metadata` | every tool carries module + category metadata |
| `test_all_tools_have_valid_json_schemas` | every tool's input schema is valid JSON Schema |

### `TestMCPDomainTools`

74 expected tool names × 2 parametrized checks
(`test_domain_tool_registered` + `test_domain_tool_is_callable`) = **148 tests**:

```python
DOMAIN_TOOLS = [
    "process_analysis",
    "get_analysis_results",
    "compute_complexity_metrics",
    "list_analysis_tools",
    "process_render",
    "list_render_frameworks",
    "render_gnn_to_format",
    "process_export",
    "list_export_formats",
    "validate_export_format",
    "process_validation",
    "validate_gnn_file",
    ...,
    # (full list: mcp/tool_reference.md)
]
```

## Running the Audit

```bash
# Full audit (all three classes)
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py -v

# Single class
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py::TestMCPModuleDiscovery -v
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py::TestMCPToolRealness -v
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py::TestMCPDomainTools -v

# JSON report only
PYTHONPATH=src python src/gnn/mcp/validate_tools.py
```

## Adding a New Tool to the Audit

1. Add the tool name string to `DOMAIN_TOOLS` in `TestMCPDomainTools`
2. Ensure the module appears in `EXPECTED_MODULES` (or is covered by an existing one)
3. Run the audit — all 3 new assertions should pass

## Why Convergence Polling, Not `time.sleep()`

The MCP server registers modules synchronously but reverts to background threads for modules that time out. A fixed `time.sleep()` can still race if the machine is under load. The polling loop stops as soon as the count stops growing, so it is both faster on idle machines and more robust under load.

## Tool-Surface Scope: LSP Visibility

The audit machinery covers the MCP tool surface only. Grepping the parity tests and docs for LSP at HEAD finds `lsp` exclusively as a name: it appears in `EXPECTED_MODULES`, in `ZERO_TOOL_MODULES = ("doc", "lsp")`, and in the committed ledger's `modules_list` (`src/gnn/mcp/audit_report.json`, 35 modules / 156 tools), never as a covered surface. The reason is structural: IDE clients drive the LSP server out-of-band over stdio, not through the MCP server, so `src/gnn/lsp/mcp.py` implements `register_tools` as a documented no-op and the generated quick-reference table [`mcp/tool_reference.md`](../mcp/tool_reference.md) (written by `uv run python src/gnn/mcp/validate_tools.py --markdown`) has no `lsp` row to gain. The actual LSP surface — the pygls server built by `create_server()` in [`src/gnn/lsp/__init__.py`](../../../src/gnn/lsp/__init__.py) (open/save/change diagnostics, hover, completions) — carries its own unit tests under `tests/lsp/`, but no committed capabilities manifest and no parity gate pins that feature set the way `tests/mcp/test_registry_internals.py::TestAuditSurfaceParity` pins the MCP tool count to the ledger. Closing that gap would mean either exposing LSP functionality as MCP tools (so the existing audit ledger covers it) or mirroring the ledger pattern LSP-side: a regenerate-on-change capabilities manifest plus a test comparing it to the live `create_server()` registration.

## See Also

- [mcp/tool_development_guide.md](../mcp/tool_development_guide.md) — Real-tool policy, canonical mcp.py pattern
- [mcp/tool_reference.md](../mcp/tool_reference.md) — full DOMAIN_TOOLS list in table form
- [testing/test_patterns.md](test_patterns.md) — general test patterns and fixtures
