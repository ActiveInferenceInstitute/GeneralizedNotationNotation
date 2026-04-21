# Specification: Testing Documentation

## Scope
Testing strategy, coverage expectations, and quality gates for GNN.
Complements `src/tests/` which holds the actual test suite.

## Coverage Gates
- Minimum coverage: **40%** (gate fails below 31%)
- Zero-mock policy: **no `MagicMock`, no `assert True`** — tests use real
  dependencies or skip-with-guard when deps are unavailable
- Test naming: `src/tests/test_{module}_*.py`
- Baseline at v1.6.0: 2,000+ passing, ≤85 skipped, 0 failures
  (excluding env-blocked `test_uv_environment.py` and optional
  `test_llm_ollama*.py` that require a local Ollama)

## Test Categories
| Category | Files | Purpose |
|----------|-------|---------|
| Unit | `test_<module>_overall.py`, `test_<module>_<area>.py` | Module-level behavior |
| Integration | `test_pipeline_*.py` | Cross-module flows |
| Contract | `test_pymdp_contracts.py`, `test_discrete_models_pymdp.py` | Backend contract tests |
| Regression | `test_<phase_or_fix>.py` | One-off regression guards |

## Running
```bash
uv sync --extra dev
uv run pytest src/tests/ -q --tb=no \
  --ignore=src/tests/test_llm_ollama.py \
  --ignore=src/tests/test_llm_ollama_integration.py
```

## Status
Maintained. Every contract change (e.g., the Phase 1.1 exit-code-2 widening)
must land with a regression test.
