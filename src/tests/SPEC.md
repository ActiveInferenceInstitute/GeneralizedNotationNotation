# Tests Module Specification

## Overview
Modular test suite for the GNN Processing Pipeline with 91 test files organized into 24 categories.

## Components

### Core Infrastructure
- `runner.py` - Test runner and execution engine
- `categories.py` - 24 test categories with all 91 test files
- `conftest.py` - Pytest configuration and fixtures

### Test Files
Naming convention: `test_{module}_{detail}.py`
- `test_{module}_overall.py` - Comprehensive module tests
- `test_{module}_{feature}.py` - Specific feature tests  
- `test_{module}_integration.py` - Integration tests
- `test_{module}_performance.py` - Performance benchmarks

### Helpers
- `helpers/` - Reusable test utilities
- `test_data/` - Sample test data and fixtures

## Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| gnn | 5 | GNN processing and validation |
| render | 4 | Code generation and rendering |
| execute | 9 | Execution and PyMDP simulation |
| analysis | 1 | Analysis and statistics |
| visualization | 8 | Graph and matrix visualization |
| pipeline | 15 | Pipeline orchestration |
| llm | 3 | LLM integration |
| mcp | 5 | Model Context Protocol |
| comprehensive | 11 | API and system tests |

## Running Tests

```bash
# Fast tests (~2 min)
python src/2_tests.py --fast-only --verbose

# Comprehensive tests (~15 min)
python src/2_tests.py --comprehensive --verbose

# Specific category
pytest src/tests/test_gnn*.py -v

# All tests
python -m pytest src/tests/ -v
```

## Key Exports
```python
from tests import run_tests
from tests.categories import MODULAR_TEST_CATEGORIES, get_all_test_files
from tests.helpers import load_sample_gnn_spec
```
