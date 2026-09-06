# Type Checker Tests

Pytest coverage for `src/gnn/type_checker/`.

This folder contains module-focused tests for type checking, dimensions, resource estimation, and reports.

## Test Files

- `test_resource_estimation_contract.py` — regression coverage for real and degenerate type-checker inputs (resource estimation contract).
- `test_type_checker_analysis_outputs.py` — analysis and output utility layers: `analysis_utils` statistics computation and `output_utils` per-file/cross-file report rendering.
- `test_type_checker_content_validation.py` — the additive type-checker surface: section-scoped extraction helpers, the typed validation summary, the pure `validate_content` entry point, and `strict_mode` constructor plumbing.
- `test_type_checker_discovery.py` — discovery contract: the type checker discovers every registered GNN file extension, not just `*.md`.
- `test_type_checker_estimator_cli_mcp.py` — resource estimator, CLI end-to-end path, and MCP integration behavior.
- `test_type_checker_overall.py` — module-level aggregate contract for the type checker folder.

Run:

```bash
uv run --extra dev python -m pytest tests/type_checker/ -q
```
