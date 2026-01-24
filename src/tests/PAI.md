# Tests Module - PAI Context

## Quick Reference

**Purpose:** Comprehensive test suite for all pipeline modules and functionality.

**When to use this module:**
- Run full test suite validation
- Test specific module functionality
- Verify pipeline integrity after changes

## Common Operations

```bash
# Run all tests
python -m pytest src/tests/ -v

# Run specific test file
python -m pytest src/tests/test_gnn_parser.py -v

# Run with coverage
python -m pytest src/tests/ --cov=src --cov-report=html

# Run tests via pipeline
python 2_tests.py --verbose
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | All modules | Code under test |
| **Output** | report | Test results, coverage |

## Key Files

- `runner.py` - Test orchestration
- `test_*.py` - Individual test modules
- `conftest.py` - Shared fixtures
- `__init__.py` - Module exports

## Test Categories

| Category | Files | Purpose |
|----------|-------|---------|
| Parser | `test_gnn_parser.py` | GNN parsing |
| Render | `test_render_*.py` | Code generation |
| Execute | `test_execute_*.py` | Simulation execution |
| Analysis | `test_analysis_*.py` | Result analysis |
| Integration | `test_*_integration.py` | End-to-end flows |

## Tips for AI Assistants

1. **Step 2:** Tests run as Step 2 of the pipeline
2. **pytest:** Uses pytest framework with fixtures
3. **Output Location:** `output/2_tests_output/`
4. **Coverage:** Aim for >80% coverage on core modules
5. **Fixtures:** Use `tmp_path` for isolated test directories

---

**Version:** 1.1.3 | **Step:** 2 (Test Execution)
