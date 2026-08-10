# Specification: Testing Documentation

## Scope
Testing strategy, coverage expectations, and quality gates for GNN.
Complements `src/tests/` which holds the actual test suite.

## Coverage Gates
- Coverage threshold enforced via `fail_under = 50` in `pyproject.toml`:
  `[tool.coverage.run] fail_under = 50` — CI will now fail on coverage
  regression below the floor, not just report it.
- Real-implementation policy: tests exercise real dependencies. A missing
  surface must fail loudly rather than be skipped or substituted.
- Zero-skip contract: `src/tests/test_zero_skip_contracts.py` scans every
  `test_*.py` under `src/tests/` and fails on any `pytest.skip(`,
  `pytest.importorskip(`, `pytest.xfail(`, `@pytest.mark.skip`,
  `@pytest.mark.skipif`, or `@pytest.mark.xfail` token. The sole exemption is
  `DEFAULT_SKIP_ALLOWLIST` in that same module, which lists the files needing
  software outside the Python environment (the Ollama LLM tests, and the
  Julia-dependent cross-framework and RxInfer visualization-log tests).
  Widening that set is a reviewable edit to the contract itself.
- Test naming: `src/tests/test_{module}_*.py`
- Expected outcome for the default suite is zero failures. Consult the current
  run output for pass counts rather than a number recorded here.

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
