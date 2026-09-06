# Tests Module Specification

## Overview
Modular test suite for the GNN Processing Pipeline. The live inventory is
owned by pytest collection and `categories.py`; this specification does not
duplicate their changing counts.

## Components

### Core Infrastructure
- `runner.py` - `run_tests()` mode routing; re-exports the canonical `TestRunner` from `infrastructure/`
- `test_runner_modular.py` - `_ModularTestRunner` (category execution) + `create_test_runner` factory
- `test_runner_modes.py` - fast / comprehensive / reliable mode implementations
- `categories.py` - typed `MODULAR_TEST_CATEGORIES` routing table (`TestCategory`) + `missing_category_files()` drift detector
- `infrastructure/` - canonical `TestRunner`, config dataclasses, resource monitor, report generators, output parsers
- `conftest.py` - Pytest configuration, fixtures, and marker registration

### Test Files
Naming convention: `test_{module}_{detail}.py`
- `test_{module}_overall.py` - Comprehensive module tests
- `test_{module}_{feature}.py` - Specific feature tests  
- `test_{module}_integration.py` - Integration tests
- `test_{module}_performance.py` - Performance benchmarks

### Helpers
- `helpers/` - Reusable typed utilities (`load_module_from_path`, `SAMPLE_GNN_CONTENT`, `write_sample_gnn_markdown`, `MCPTools`, `render_gnn_files`)
- `test_data/` - Sample test data and fixtures
- `tests/` - Shared-plumbing regression tests (category contracts, runner unification, helper contracts)

## Test Categories

The live routing table is `categories.py` (`MODULAR_TEST_CATEGORIES`) — 24
categories whose `files` entries resolve relative to `src/tests/` (module
subdirectory paths). `missing_category_files()` must stay empty; the
contract test in `tests/test_categories_contract.py` enforces it. Do not
duplicate counts here; enumerate with:

```bash
uv run --extra dev python -c "from tests.categories import get_all_test_files; print(len(get_all_test_files()))"
```

## Running Tests

```bash
# Fast tests (~2 min)
python src/2_tests.py --fast-only --verbose

# Comprehensive tests (~15 min)
python src/2_tests.py --comprehensive --verbose

# Specific category
uv run --extra dev python -m pytest src/tests/gnn -v

# All tests
uv run --extra dev python -m pytest src/tests/ -v
```

## Key Exports
```python
from tests import run_tests
from tests.runner import TestRunner  # canonical copy: tests.infrastructure
from tests.categories import (
    MODULAR_TEST_CATEGORIES,
    get_all_test_files,
    missing_category_files,
)
from tests.helpers import load_sample_gnn_spec, load_module_from_path
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
