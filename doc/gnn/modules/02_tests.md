# Step 2: Tests — Test Suite Execution

## Overview

Orchestrates the project test suite using pytest, supporting fast (default) and comprehensive execution modes. Respects environment variable overrides for CI/pipeline integration.

## Usage

```bash
python src/2_tests.py --fast-only --verbose      # Fast tests (default)
python src/2_tests.py --comprehensive --verbose   # All tests
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/2_tests.py` (86 lines) |
| Module | `src/tests/` |
| Module function | `run_tests()` |

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--fast-only` | flag | Run only fast tests (default) |
| `--comprehensive` | flag | Run all tests including slow/integration |

## Environment Variables

| Variable | Effect |
|----------|--------|
| `SKIP_TESTS_IN_PIPELINE` | Skip test execution entirely |
| `FAST_TESTS_TIMEOUT` | Override timeout (default: 600 seconds) |

## Output

- **Directory**: `output/2_tests_output/`
- Test results, coverage reports, and execution logs

## Source

- **Script**: [src/2_tests.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/2_tests.py)
