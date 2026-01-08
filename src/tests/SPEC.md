# Tests Module Specification

## Overview
Modular test suite for GNN pipeline.

## Components

### Core
- `runner.py` - Test runner (2089 lines)
- `categories.py` - 24 test categories (250 lines)

### Test Files
- `test_*.py` - Individual test modules

## Test Categories
Defined in `categories.py`:
- Unit tests, Integration tests, Performance tests
- Module-specific test categories

## Key Exports
```python
from tests import run_tests, MODULAR_TEST_CATEGORIES
from tests.categories import MODULAR_TEST_CATEGORIES
```

## Running Tests
```bash
python -m pytest src/tests/ -v
```
