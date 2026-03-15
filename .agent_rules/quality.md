# Code Quality Standards

> **Principle**: Real implementations only. No mocks, no placeholders, no stubs.

## Type Safety

All public functions must have complete type hints:

```python
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

def process_gnn_files(
    target_dir: Path,
    output_dir: Path,
    *,
    formats: List[str] = None,
    validate: bool = True,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process GNN files and return results.

    Args:
        target_dir: Directory containing GNN files.
        output_dir: Directory for output artifacts.
        formats: List of output formats (default: all supported).
        validate: Whether to validate files before processing.
        max_files: Maximum number of files to process.

    Returns:
        Dictionary with keys: 'processed', 'failed', 'output_files', 'metrics'.

    Raises:
        FileNotFoundError: If target_dir does not exist.
        ValueError: If an unsupported format is requested.

    Example:
        >>> result = process_gnn_files(Path("input/"), Path("output/"))
        >>> print(result['processed'])
        42
    """
```

---

## Docstring Requirements

Every public function, class, and module must have:
- **Summary line**: One sentence, imperative mood
- **Args**: All parameters with types and descriptions
- **Returns**: What's returned and its structure
- **Raises**: Exceptions that may be raised
- **Example**: At least one working example

---

## Linting and Formatting

```bash
# Format (Black-compatible, line length 100)
uv run ruff format src/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/ --strict
```

**Standards:**
- Line length: 100 characters
- String quotes: double quotes
- Import order: stdlib → third-party → local (ruff enforces)
- No unused imports, variables, or arguments

---

## Implementation Quality Rules

### No Mocks ⚠️
```python
# ❌ WRONG
from unittest.mock import MagicMock, patch
mock_logger = MagicMock()

# ✅ CORRECT
import logging
logger = logging.getLogger("test")
```

### Real Data Testing
```python
# ❌ WRONG — hard-coded fake data
model = {"name": "fake", "states": 2}

# ✅ CORRECT — use actual GNN fixture files
from tests.conftest import sample_gnn_files
model = parse_gnn_file(sample_gnn_files["basic"])
```

### Error Messages — Actionable
```python
# ❌ WRONG
raise ValueError("Invalid input")

# ✅ CORRECT
raise ValueError(
    f"Invalid format '{format_name}'. "
    f"Supported formats: {', '.join(SUPPORTED_FORMATS)}. "
    f"See gnn_standards.md for format specifications."
)
```

---

## Module Completeness Checklist

Before submitting a new module, verify:

- [ ] `__init__.py` with `__version__`, `FEATURES`, `get_module_info()`
- [ ] `processor.py` with standard function signature
- [ ] `mcp.py` with `MCP_TOOLS` dict and `register_tools()`
- [ ] `AGENTS.md` documenting capabilities
- [ ] `README.md` with usage examples
- [ ] Tests in `src/tests/test_MODULENAME_overall.py`
- [ ] All public functions have complete type hints
- [ ] All public functions have docstrings with examples
- [ ] No unused imports
- [ ] No `print()` statements (use `logger.info()` etc.)
- [ ] No hard-coded paths (use `Path` and parametrize)

---

## Logging Standards

```python
import logging
logger = logging.getLogger(__name__)

# Levels:
logger.debug("Detailed trace info for troubleshooting")
logger.info("Normal operation milestones: files found, steps completed")
logger.warning("Recoverable issue: optional dep missing, fallback used")
logger.error("Non-fatal failure: file failed to process, step degraded")
logger.critical("Fatal error: pipeline cannot continue")

# Always include context:
logger.info(f"Processed {n_files} files in {duration:.2f}s")
logger.error(f"Failed to parse {file_path}: {e} — check file format")
```

---

## Performance Standards Per Step

| Step | Target | Maximum |
|------|--------|---------|
| Template/Registry | <1s | 5s |
| GNN Parse | <1s | 10s |
| Type Check/Validation/Export | <1s | 30s |
| Visualization | <1s | 60s |
| Render/Execute | <60s | 300s |
| Tests | <120s | 1200s |
| LLM | <60s | 360s |
| Full pipeline | <3 min | 30 min |

---

**Last Updated**: March 2026 | **Status**: Production Standard
