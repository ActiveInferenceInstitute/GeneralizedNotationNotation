# Testing Framework

> **Run tests**: `uv run pytest src/tests/ -v` or `PYTHONPATH=src pytest src/tests/ -v`

## Import Pattern ⚠️ CRITICAL

```python
# ✅ CORRECT — always sys.path.insert then direct imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn import discover_gnn_files, parse_gnn_file
from render import render_gnn_to_pymdp
from utils import setup_step_logging, EnhancedArgumentParser
from pipeline import get_pipeline_config

# ❌ WRONG — never use src. prefix
from src.gnn import discover_gnn_files
```

---

## Standard Test File Structure

```python
#!/usr/bin/env python3
"""Tests for module_name."""
import pytest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestModuleImports:
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_import_module(self):
        try:
            import module_name
            assert hasattr(module_name, "__version__")
            assert hasattr(module_name, "FEATURES")
        except ImportError:
            pytest.skip("Module not available")

class TestModuleFunctionality:
    @pytest.fixture
    def temp_output(self, tmp_path):
        out = tmp_path / "output"; out.mkdir(); return out

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_process_basic(self, temp_output):
        try:
            from module_name import process_main_function
            import logging
            result = process_main_function(
                target_dir=Path("input/gnn_files"),
                output_dir=temp_output,
                logger=logging.getLogger("test"),
            )
            assert result is True
        except ImportError:
            pytest.skip("Module not available")

class TestModuleMCP:
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_mcp_tools_registered(self):
        try:
            from module_name.mcp import MCP_TOOLS
            assert len(MCP_TOOLS) > 0
        except ImportError:
            pytest.skip("MCP not available")
```

---

## Test Markers

```python
@pytest.mark.unit           # Individual components
@pytest.mark.integration    # Cross-module interactions
@pytest.mark.performance    # Resource/timing tests
@pytest.mark.slow           # Long-running tests
@pytest.mark.fast           # Quick feedback tests
@pytest.mark.safe_to_fail   # Tests safe to skip gracefully
```

**Filter by marker:**
```bash
pytest -m unit                # Unit tests only
pytest -m "not slow"          # Skip slow tests
pytest -m "fast"              # Fast feedback only
```

---

## Safe-to-Fail Pattern

```python
@pytest.mark.safe_to_fail
def test_optional_functionality(self):
    """Test optional feature with graceful skip."""
    try:
        from module_name import optional_function
        result = optional_function()
        assert result is not None
    except ImportError:
        pytest.skip("Optional dep not available")
    except Exception as e:
        pytest.fail(f"Feature available but failed: {e}")
```

---

## Key Testing Rules

1. **No mocks**: Use real implementations; skip when deps are absent
2. **Path setup**: Always `sys.path.insert(0, str(Path(__file__).parent.parent))`
3. **Conftest**: Star-import `from tests.conftest import *` for shared fixtures
4. **Import errors**: Always wrap in `try/except ImportError → pytest.skip()`
5. **Resource cleanup**: Fixtures use `tmp_path` (pytest built-in) for isolation

---

## Running Tests

```bash
# Full suite via pipeline
python src/2_tests.py --verbose

# Individual file
PYTHONPATH=src pytest src/tests/test_gnn_overall.py -v

# Specific class/method
PYTHONPATH=src pytest src/tests/test_gnn_overall.py::TestGNNComprehensive::test_imports -v

# Fast tests only
python src/2_tests.py --fast-only

# Coverage
PYTHONPATH=src pytest --cov=src --cov-report=term-missing src/tests/
```

---

## Coverage Targets

| Scope | Target |
|-------|--------|
| Overall | ≥85% |
| Critical modules (gnn, pipeline, utils) | ≥90% |
| Integration tests | ≥80% |

---

## Test File Naming Convention

```
src/tests/
├── test_MODULENAME_overall.py    # Comprehensive module coverage
├── test_MODULENAME_parsing.py   # Specific area
├── test_MODULENAME_integration.py
├── runner.py                    # Category-based test runner
├── conftest.py                  # Shared fixtures
└── run_fast_tests.py
```

---

**Last Updated**: March 2026 | **Status**: Production Standard
