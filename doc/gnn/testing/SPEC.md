# Specification: Testing Documentation

## Scope
Testing strategy, coverage expectations, and quality gates for GNN.
Complements `src/tests/` which holds the actual test suite.

## Coverage Gates
- Coverage threshold enforced via `fail_under = 50` in `pyproject.toml`:
  `[tool.coverage.run] fail_under = 50` — CI will now fail on coverage
  regression below the floor, not just report it.
- Real-implementation policy: tests use real
  dependencies or skip-with-guard when deps are unavailable
- Test naming: `src/tests/test_{module}_*.py`
- Baseline at v1.6.0: 2,000+ passing, ≤85 skipped, 0 failures
  (excluding env-blocked `test_uv_environment.py` and optional
  `test_llm_ollama*.py` that require a local Ollama)

## Reproducibility Requirements
- `PYTHONHASHSEED=0` must be set in the environment for deterministic
  dict iteration across runs (CI enforces this via `ci.yml` env block).
  Without this, JSON output files with unsorted keys may differ between
  pipeline runs even with identical inputs.
- A function-scoped `_auto_seed_rng` autouse fixture (`conftest.py`)
  calls `np.random.seed(0)` before every test, providing a deterministic
  baseline even for unseeded test functions.

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
uv run --extra dev python -m pytest src/tests/ -q --tb=no \
  --ignore=src/tests/llm/test_llm_ollama.py \
  --ignore=src/tests/llm/test_llm_ollama_integration.py
```

## Status
Maintained. Every contract change (e.g., the Phase 1.1 exit-code-2 widening)
must land with a regression test.
