# Module Patterns and Architecture

> **Environment**: `uv pip install -e .` | `uv run pytest`

## Standard Module Layout

```
src/module_name/
├── __init__.py          # Public API, version, FEATURES dict
├── processor.py         # Core processing logic
├── mcp.py               # MCP tool registration
├── AGENTS.md            # Agent capabilities documentation
├── README.md            # Module usage documentation
└── specialized/         # Optional subdirectories
    ├── __init__.py
    └── component.py
```

---

## Standard `__init__.py`

```python
"""
Module Name — Brief description.

Pipeline Integration: Invoked by step N (N_module.py).
"""
from pathlib import Path
from typing import Dict, Any

from .processor import process_main_function, ProcessorClass

__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Module description"

FEATURES: Dict[str, bool] = {
    "core_processing": True,
    "mcp_integration": True,
    "optional_feature": False,  # Requires optional dependency
}

def get_module_info() -> Dict[str, Any]:
    """Get module metadata."""
    return {
        "name": "module_name",
        "version": __version__,
        "description": __description__,
        "features": FEATURES,
    }

__all__ = [
    "process_main_function", "ProcessorClass",
    "get_module_info", "FEATURES", "__version__",
]
```

## Lazy Import for Optional Dependencies

```python
_optional_module = None

def _get_optional():
    global _optional_module
    if _optional_module is None:
        try:
            import optional_package
            _optional_module = optional_package
        except ImportError:
            _optional_module = False
    return _optional_module if _optional_module else None
```

---

## Standard Processing Function Signature

```python
def process_module(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    *,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process module functionality for GNN models.

    Args:
        target_dir: Directory containing input files.
        output_dir: Step-specific output directory (pre-created).
        logger: Logger instance for this step.
        recursive: Process files recursively.
        verbose: Enable verbose logging.
        **kwargs: Additional step-specific arguments.

    Returns:
        True if processing succeeded, False otherwise.
    """
    try:
        if not target_dir.exists():
            logger.error(f"Target directory not found: {target_dir}")
            return False
        output_dir.mkdir(parents=True, exist_ok=True)
        files = list(target_dir.rglob("*.md") if recursive else target_dir.glob("*.md"))
        for file in files:
            process_file(file, output_dir, **kwargs)
        logger.info(f"Processed {len(files)} files")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False
```

---

## Standard MCP Integration (`mcp.py`)

```python
"""MCP integration for module_name."""
import logging
from typing import Dict, Any
logger = logging.getLogger(__name__)

def module_action_mcp(mcp_instance_ref=None, **kwargs) -> Dict[str, Any]:
    """MCP tool for module action."""
    try:
        from .processor import module_action
        required = ["input_path"]
        for arg in required:
            if arg not in kwargs:
                return {"success": False, "error": f"Missing: {arg}", "error_type": "missing_argument"}
        result = module_action(**kwargs)
        return {"success": True, "result": result, "module": "module_name"}
    except ImportError as e:
        return {"success": False, "error": str(e), "error_type": "import_error"}
    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        return {"success": False, "error": str(e), "error_type": "execution_error"}

MCP_TOOLS: Dict[str, Dict[str, Any]] = {
    "module_action": {
        "function": module_action_mcp,
        "description": "Perform module action",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Path to input file"},
                "verbose": {"type": "boolean", "default": False}
            },
            "required": ["input_path"]
        }
    }
}

def register_tools(mcp=None) -> None:
    """Register MCP tools with server."""
    logger.info("module_name MCP tools registered.")
```

---

## Cross-Module Communication

```python
@dataclass
class ProcessingResult:
    """Standardized result from module processing."""
    success: bool
    module: str
    operation: str
    output_files: List[Path]
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {**self.__dict__, "output_files": [str(p) for p in self.output_files]}
```

---

## AGENTS.md Template (Required for Every Module)

```markdown
# Module Name Agent

**Version**: 1.0.0 | **Status**: Production Ready | **Pipeline Step**: N

## Overview
[Brief description]

## Capabilities
- [Capability 1]

## API Reference
### `process_function(target_dir, output_dir, logger, **kwargs) -> bool`

### MCP Tools
| Tool | Description | Parameters |
|------|-------------|------------|
| tool_name | What it does | param1, param2 |

## Examples
```python
from module_name import process_function
result = process_function(...)
```
```

---

**Last Updated**: March 2026 | **Status**: Production Standard
