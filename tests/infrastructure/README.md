# Infrastructure Tests

Test-suite infrastructure tests: environment validation, stack validation, coverage meta-tests, and coverage of the custom test runner and its parsing helpers.

## Module Structure

| File | Purpose |
|------|---------|
| `test_environment_python.py` | Validates the Python runtime and interpreter environment used by the test suite |
| `test_environment_dependencies.py` | Validates required and optional test dependencies |
| `test_environment_system.py` | Validates system-level environment (OS, resources) for the suite |
| `test_environment_integration.py` | Validates the integrated test environment end to end |
| `test_environment_overall.py` | Aggregates environment validation into a suite-level check |
| `test_uv_environment.py` | Validates the `uv`-managed environment (`uv sync`, `FEATURES` surface) |
| `test_jax_pymdp_stack_validation.py` | Validates the JAX + PyMDP execution stack availability and behavior |
| `test_coverage_assessment.py` | Assesses suite coverage and reports gaps |
| `test_coverage_gap_infrastructure.py` | Meta-tests for coverage-gap reporting infrastructure |
| `test_coverage_overall.py` | Aggregates coverage validation into a suite-level check |
| `test_runner.py` | Tests the custom `TestRunner` (canonical source, re-exported by `tests.runner`) |
| `test_infrastructure_utils_statistics.py` | Tests pytest output and coverage parsing helpers |

Supporting non-test modules in this directory: `utils.py` (pytest command building, output parsing, dependency checks), `test_config.py` (`TestExecutionConfig` / `TestExecutionResult`), `report_generator.py` (markdown/timeout/error report generation), and `resource_monitor.py` (memory/CPU monitoring during tests).

## Usage

```bash
uv run --extra dev python -m pytest tests/infrastructure -q
```

## See Also

- [Parent: tests/README.md](../README.md)
- [Parent agents: tests/AGENTS.md](../AGENTS.md)
