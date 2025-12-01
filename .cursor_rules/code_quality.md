# Code Quality Standards

> **Environment Note**: Run all quality checks via `uv` to maintain consistent tooling (e.g., `uv run python -m mypy src/`, `uv run black src/`).

## Overview

This document defines the code quality standards for the GNN pipeline. All code contributions must meet these standards to maintain scientific rigor, reliability, and maintainability.

---

## Type Safety Standards

### Complete Type Hints

All public functions must have complete type annotations:

```python
# CORRECT: Complete type hints
def process_gnn_file(
    file_path: Path,
    output_dir: Path,
    options: Dict[str, Any],
    logger: logging.Logger,
    *,
    recursive: bool = False,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """Process a GNN file with full type safety."""
    ...

# INCORRECT: Missing type hints
def process_gnn_file(file_path, output_dir, options):
    ...
```

### Generic Types for Containers

Use proper generic types for containers:

```python
# CORRECT: Explicit generic types
from typing import Dict, List, Optional, Tuple, Any, Set

def get_results() -> Dict[str, List[Tuple[str, int]]]:
    ...

# INCORRECT: Bare container types
def get_results() -> dict:
    ...
```

### Union Types and Optional

Use appropriate union types:

```python
# CORRECT: Clear optional handling
def find_file(name: str) -> Optional[Path]:
    ...

def process(data: Union[str, bytes, Path]) -> bool:
    ...

# Python 3.10+ alternative
def process(data: str | bytes | Path) -> bool:
    ...
```

### Type Checking Validation

Run mypy for type validation:

```bash
# Full type check
uv run python -m mypy src/ --ignore-missing-imports

# Strict mode for new modules
uv run python -m mypy src/new_module/ --strict
```

---

## Documentation Standards

### Comprehensive Docstrings

All public functions, classes, and modules require docstrings:

```python
def render_pymdp_code(
    model: Dict[str, Any],
    output_path: Path,
    *,
    include_imports: bool = True,
    validate: bool = True
) -> Tuple[bool, str]:
    """
    Render a GNN model to PyMDP simulation code.
    
    This function generates executable PyMDP code from a parsed GNN model
    specification. The generated code includes all necessary imports,
    matrix definitions, and simulation logic.
    
    Args:
        model: Parsed GNN model dictionary containing:
            - 'name': Model name (str)
            - 'states': State space definitions (Dict)
            - 'matrices': A, B, C, D matrices (Dict)
        output_path: Path where the generated code will be written.
        include_imports: Whether to include import statements at the top.
            Defaults to True.
        validate: Whether to validate the model before rendering.
            Defaults to True.
    
    Returns:
        A tuple of (success: bool, message: str) where:
            - success: True if rendering completed successfully
            - message: Success message or error description
    
    Raises:
        ValueError: If model is missing required fields.
        IOError: If output_path is not writable.
    
    Example:
        >>> model = {"name": "MyAgent", "states": {...}, "matrices": {...}}
        >>> success, msg = render_pymdp_code(model, Path("output/agent.py"))
        >>> print(f"Render {'succeeded' if success else 'failed'}: {msg}")
    
    Note:
        PyMDP must be installed for the generated code to execute.
        If PyMDP is not available, the code will still be generated
        but will include fallback stubs.
    """
    ...
```

### Module-Level Documentation

Every module must have a module docstring:

```python
"""
GNN Render Module - Code Generation for Simulation Frameworks

This module provides code generation capabilities for translating GNN
model specifications into executable simulation code for various
Active Inference frameworks.

Supported Frameworks:
    - PyMDP: Python-based Active Inference
    - RxInfer.jl: Julia probabilistic programming
    - ActiveInference.jl: Julia Active Inference framework
    - JAX: Pure functional ML approach
    - DisCoPy: Categorical diagrams

Architecture:
    The module follows a thin orchestrator pattern where the main
    script (11_render.py) delegates to framework-specific renderers
    in the src/render/ subdirectories.

Usage:
    # As pipeline step
    python src/11_render.py --target-dir input/gnn_files --output-dir output
    
    # Programmatic usage
    from render import render_gnn_to_pymdp
    success = render_gnn_to_pymdp(model, output_path)

See Also:
    - src/render/pymdp/: PyMDP-specific rendering
    - src/render/jax/: JAX-specific rendering
    - doc/render/: Detailed framework documentation
"""
```

### Class Documentation

Classes require comprehensive docstrings:

```python
class GNNParser:
    """
    Multi-format GNN file parser with semantic validation.
    
    This class provides parsing capabilities for GNN specifications
    across 22+ formats including Markdown, JSON, YAML, XML, and
    various formal specification formats.
    
    Attributes:
        supported_formats: Set of supported file format extensions.
        validation_level: Current validation strictness level.
        last_error: Last parsing error message, if any.
    
    Example:
        >>> parser = GNNParser(validation_level="strict")
        >>> model = parser.parse_file(Path("model.md"))
        >>> if model:
        ...     print(f"Parsed model: {model['name']}")
        ... else:
        ...     print(f"Parse failed: {parser.last_error}")
    """
    
    def __init__(
        self,
        validation_level: str = "standard",
        enable_caching: bool = True
    ) -> None:
        """
        Initialize the GNN parser.
        
        Args:
            validation_level: Validation strictness. One of:
                - "basic": Syntax only
                - "standard": Syntax + structure
                - "strict": Full semantic validation
                - "research": Publication-ready validation
            enable_caching: Whether to cache parsed results.
        """
        ...
```

---

## Testing Standards

### No Mocks Policy

**CRITICAL**: All tests must execute real code paths:

```python
# CORRECT: Real implementation testing
def test_gnn_parsing():
    """Test GNN parsing with real files."""
    parser = GNNParser()
    result = parser.parse_file(Path("tests/fixtures/sample.md"))
    assert result is not None
    assert "name" in result
    assert result["name"] == "SampleModel"

# INCORRECT: Mock-based testing (FORBIDDEN)
def test_gnn_parsing_with_mock():
    """This pattern is NOT allowed."""
    with patch("gnn.parser.GNNParser") as mock_parser:
        mock_parser.parse_file.return_value = {"name": "Fake"}
        # This bypasses real code and provides no value
```

### Skip Pattern for Unavailable Dependencies

When dependencies are unavailable, skip gracefully:

```python
@pytest.mark.safe_to_fail
def test_pymdp_execution():
    """Test PyMDP execution with real framework."""
    try:
        import pymdp
    except ImportError:
        pytest.skip("PyMDP not installed - skipping execution test")
    
    # Real test with actual PyMDP
    agent = pymdp.Agent(...)
    result = agent.step()
    assert result is not None
```

### Coverage Requirements

| Area | Minimum Coverage | Target Coverage |
|------|-----------------|-----------------|
| Core modules (gnn, render) | 90% | 95% |
| Infrastructure (utils, pipeline) | 85% | 90% |
| Integration points | 80% | 85% |
| Error handling paths | 75% | 85% |

Run coverage analysis:

```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=term-missing --cov-report=html

# Check specific module
uv run pytest --cov=src/render --cov-fail-under=90
```

### Test File Naming

Follow the standardized naming convention:

```
src/tests/
├── test_MODULE_overall.py      # Comprehensive module tests
├── test_MODULE_AREA.py         # Specific area tests
├── test_MODULE_integration.py  # Integration tests
└── test_MODULE_performance.py  # Performance tests
```

---

## Linting and Formatting

### Black Formatter

All Python code must be formatted with Black:

```bash
# Format all code
uv run black src/

# Check formatting without changes
uv run black --check src/

# Format specific file
uv run black src/render/jax/jax_renderer.py
```

### Ruff Linter

Use Ruff for fast, comprehensive linting:

```bash
# Run linter
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Check specific rules
uv run ruff check --select E,F,W src/
```

### Import Sorting

Imports must follow isort conventions (handled by Ruff):

```python
# CORRECT: Sorted imports
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from gnn import parse_gnn_file
from utils import setup_step_logging
```

---

## Code Organization Standards

### Module Structure

Every module follows this structure:

```
src/module_name/
├── __init__.py          # Public API exports, version, features
├── processor.py         # Core processing logic
├── mcp.py              # MCP tool registration
├── AGENTS.md           # Agent documentation
├── README.md           # Module documentation
└── subdirectory/       # Specialized components
```

### Public API Exports

The `__init__.py` must clearly export the public API:

```python
"""
Module Name - Brief Description

Extended description of module purpose and capabilities.
"""

from .processor import process_function, ProcessorClass
from .helpers import helper_function

__version__ = "1.0.0"
__author__ = "Active Inference Institute"

FEATURES = {
    "core_processing": True,
    "mcp_integration": True,
    "advanced_features": True,
}

__all__ = [
    "process_function",
    "ProcessorClass",
    "helper_function",
    "FEATURES",
    "__version__",
]
```

### Function Length Guidelines

| Function Type | Maximum Lines | Notes |
|--------------|---------------|-------|
| Simple utilities | 20 | Single responsibility |
| Core logic | 50 | Well-documented |
| Complex algorithms | 100 | Must be well-commented |
| Orchestrators | 150 | Pipeline scripts only |

Functions exceeding these limits should be refactored.

---

## Error Message Standards

### Informative Error Messages

Error messages must be actionable:

```python
# CORRECT: Actionable error message
if not input_path.exists():
    raise FileNotFoundError(
        f"GNN file not found: {input_path}\n"
        f"Searched in: {input_path.parent}\n"
        f"Available files: {list(input_path.parent.glob('*.md'))}\n"
        f"Suggestion: Check the --target-dir argument"
    )

# INCORRECT: Unhelpful error
if not input_path.exists():
    raise FileNotFoundError("File not found")
```

### Logging with Context

Use structured logging with context:

```python
logger.error(
    "Failed to parse GNN file",
    extra={
        "file_path": str(file_path),
        "error_type": type(e).__name__,
        "line_number": getattr(e, "lineno", None),
        "suggestion": "Check GNN syntax at indicated line"
    }
)
```

---

## Performance Standards

### Memory Efficiency

- Peak memory usage: <2GB for standard workloads
- Avoid loading entire files into memory when streaming is possible
- Clean up large objects explicitly when done

### Execution Time

- Full pipeline: <30 minutes
- Individual steps: <5 minutes (except tests)
- Response to user input: <100ms for interactive operations

---

## Review Checklist

Before submitting code, verify:

- [ ] All public functions have complete type hints
- [ ] All public APIs have comprehensive docstrings
- [ ] Tests use real implementations (no mocks)
- [ ] Code is formatted with Black
- [ ] Linting passes with Ruff
- [ ] Test coverage meets requirements
- [ ] Error messages are actionable
- [ ] Performance is within standards
- [ ] Documentation is updated

---

**Last Updated**: December 2025  
**Status**: Production Standard

