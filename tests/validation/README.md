# Validation Tests

Pytest coverage for `src/gnn/validation/`.

This folder contains module-focused tests for semantic validation and consistency checks.

## Test Files

- `test_consistency_contract.py` — canonical consistency behavior and best-effort receipts.
- `test_reliability_validation.py` — current-run validation verdicts and replay-safe receipts.
- `test_validation_overall.py` — module-level aggregate contract for the validation folder.
- `test_validation_public_api.py` — public API surface: `get_module_info`, `FEATURES`, `__version__`, `SemanticValidator`/`ConsistencyChecker`/`PerformanceProfiler` edge cases, and `process_validation`.
- `test_workflow_contracts.py` — the step-6 orchestrator contract (template kwargs, stage receipts, accumulation across passes, score averaging), the uniform best-effort error contract, cycle detection, and shared structure helpers.

Run:

```bash
uv run --extra dev python -m pytest tests/validation/ -q
```
