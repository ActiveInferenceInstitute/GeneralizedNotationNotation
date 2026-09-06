# Integration Tests

Pytest coverage for `src/gnn/integration/`.

This folder contains module-focused tests for integration processing, meta-analysis sweep collection, and cross-module coordination.

## Test Files

- `test_integration_functional.py` — functional tests for the integration processor: system-level consistency checks, dependency graph construction, and circular dependency detection.
- `test_integration_mcp.py` — live registration and execution coverage for the integration MCP tools.
- `test_integration_mcp_tools.py` — exercises the integration module MCP tool handlers in `src/gnn/integration/mcp.py` (integration inventory, dependency paths).
- `test_integration_meta_analysis_benchmark_fields.py` — pins the benchmark timing fields `SweepDataCollector` preserves when reading `summaries/execution_summary.json` in `slim_v1` or detail-row form.
- `test_integration_meta_analysis_validation.py` — meta-analysis validator, statistics export, and `run_meta_analysis` wiring.
- `test_integration_overall.py` — module-level aggregate contract for the integration folder.
- `test_integration_parsing_graph.py` — pins the pure extraction primitives in `gnn.integration.parsing`, the system graph builder in `gnn.integration.graph`, and the package-level API.
- `test_integration_processor.py` — empty-directory handling, GNN file processing, and composite pipeline behavior of the integration processor.
- `test_integration_slim_summary_contract.py` — slim execution-summary contract: `SweepDataCollector` keeps harvesting timing data when the Step 12 aggregate uses `slim_v1`.
- `test_meta_analysis_none_states.py` — regression test for the meta-analysis resource-efficiency table with sweep records that carry no sweep parameters.

Run:

```bash
uv run --extra dev python -m pytest tests/integration/ -q
```
