# Module Patterns and Architecture

> **Environment Note**: Module development should use `uv` for dependency management and testing (`uv pip install -e .`, `uv run pytest`).

## Overview

This document defines the architectural patterns for GNN pipeline modules. All modules follow consistent patterns to ensure maintainability, testability, and proper integration.

---

## Module Directory Structure

### Standard Module Layout

Every module follows this structure:

```
src/module_name/
├── __init__.py              # Public API, version, feature flags
├── processor.py             # Core processing logic
├── mcp.py                   # MCP tool registration
├── AGENTS.md               # Agent capabilities documentation
├── README.md               # Module usage documentation
└── specialized/            # Optional subdirectories
    ├── __init__.py
    └── component.py
```

### Example: Render Module Structure

```
src/render/
├── __init__.py             # Exports render functions
├── processor.py            # POMDP-aware render processing
├── pomdp_processor.py      # POMDP extraction and rendering
├── mcp.py                  # MCP tool registration
├── AGENTS.md               # Render agent documentation
├── README.md               # Render module documentation
├── pymdp/                  # PyMDP-specific rendering
│   ├── __init__.py
│   └── pymdp_renderer.py
├── rxinfer/                # RxInfer.jl rendering
│   ├── __init__.py
│   └── rxinfer_renderer.py
├── activeinference_jl/     # ActiveInference.jl rendering
│   ├── __init__.py
│   └── activeinference_renderer.py
├── jax/                    # JAX rendering (pure, no Flax)
│   ├── __init__.py
│   └── jax_renderer.py
└── discopy/                # DisCoPy categorical diagrams
    ├── __init__.py
    └── discopy_renderer.py
```

---

## Module Initialization Pattern

### Standard `__init__.py` Structure

```python
"""
Module Name - Brief Description

Extended description of the module's purpose, capabilities,
and integration points within the GNN pipeline.

Key Features:
    - Feature 1: Description
    - Feature 2: Description
    - Feature 3: Description

Usage:
    from module_name import process_function
    result = process_function(input_data, options)

Pipeline Integration:
    This module is invoked by step N (N_module.py) in the pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional

# Import core functionality
from .processor import (
    process_main_function,
    ProcessorClass,
    validate_input,
)

# Import specialized components
from .specialized import specialized_function

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Module description for registry"

# Feature availability flags
FEATURES: Dict[str, bool] = {
    "core_processing": True,
    "mcp_integration": True,
    "advanced_features": True,
    "optional_feature": False,  # Requires optional dependency
}

def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive module information.
    
    Returns:
        Dictionary containing module metadata, version,
        features, and capabilities.
    """
    return {
        "name": "module_name",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "features": FEATURES,
        "supported_formats": ["format1", "format2"],
    }

# Public API exports
__all__ = [
    # Core functions
    "process_main_function",
    "ProcessorClass",
    "validate_input",
    # Specialized
    "specialized_function",
    # Metadata
    "get_module_info",
    "FEATURES",
    "__version__",
]
```

### Lazy Import Pattern for Optional Dependencies

```python
"""Module with optional dependencies."""

from typing import TYPE_CHECKING

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from optional_package import OptionalClass

# Runtime lazy import
_optional_module = None

def _get_optional_module():
    """Lazy load optional module with graceful fallback."""
    global _optional_module
    if _optional_module is None:
        try:
            import optional_package
            _optional_module = optional_package
        except ImportError:
            _optional_module = False
    return _optional_module if _optional_module else None

def function_with_optional_dep():
    """Function that uses optional dependency."""
    opt = _get_optional_module()
    if opt is None:
        # Fallback implementation
        return fallback_implementation()
    # Use optional module
    return opt.actual_implementation()
```

---

## Thin Orchestrator Pattern

### Pipeline Script Structure

Numbered scripts (e.g., `11_render.py`) must be thin orchestrators:

```python
#!/usr/bin/env python3
"""
Step 11: Code Rendering (Thin Orchestrator)

This step orchestrates code generation for simulation frameworks.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates
    core functionality to src/render/. It handles argument parsing,
    logging setup, and calls the actual processing functions.

Pipeline Flow:
    main.py → 11_render.py (this script) → render/ (implementation)

How to run:
    python src/11_render.py --target-dir input/gnn_files --output-dir output
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from render import process_render
except ImportError:
    def process_render(target_dir, output_dir, logger, **kwargs):
        """Fallback when module unavailable."""
        logger.warning("Render module not available - using fallback")
        return True

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "11_render.py",
    process_render,
    "Code generation for simulation frameworks",
    additional_arguments={
        "frameworks": {
            "type": str,
            "help": "Frameworks to render (comma-separated or 'all')",
            "default": "all"
        }
    }
)

def main() -> int:
    """Main entry point."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
```

### What Belongs in the Module vs Script

| Component | In Module (src/module/) | In Script (N_module.py) |
|-----------|------------------------|-------------------------|
| Core processing logic | ✅ | ❌ |
| Algorithm implementations | ✅ | ❌ |
| Helper functions | ✅ | ❌ |
| Argument parsing | ❌ | ✅ |
| Logging setup | ❌ | ✅ |
| Output directory management | ❌ | ✅ |
| Exit code handling | ❌ | ✅ |
| Performance tracking | ❌ | ✅ |

---

## Module Function Signature Pattern

### Standard Processing Function

All module processing functions must follow this signature:

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
        target_dir: Directory containing input files to process.
        output_dir: Step-specific output directory (already created).
        logger: Logger instance for this step.
        recursive: Whether to process files recursively.
        verbose: Whether to enable verbose logging.
        **kwargs: Additional step-specific arguments.
    
    Returns:
        True if processing succeeded, False otherwise.
    """
    try:
        # Validate inputs
        if not target_dir.exists():
            logger.error(f"Target directory does not exist: {target_dir}")
            return False
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main processing logic
        files = discover_files(target_dir, recursive=recursive)
        
        for file in files:
            result = process_file(file, output_dir, **kwargs)
            if not result:
                logger.warning(f"Failed to process: {file}")
        
        logger.info(f"Processed {len(files)} files")
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False
```

---

## MCP Integration Pattern

### Standard MCP Module Structure

```python
"""
Module MCP Integration

Exposes module functionality through Model Context Protocol.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def module_function_mcp(mcp_instance_ref=None, **kwargs) -> Dict[str, Any]:
    """
    MCP tool for module function.
    
    Args:
        mcp_instance_ref: Optional MCP instance reference.
        **kwargs: Function-specific arguments.
    
    Returns:
        Standardized MCP response dictionary.
    """
    try:
        from .processor import module_function
        
        # Validate required arguments
        required = ["input_path"]
        for arg in required:
            if arg not in kwargs:
                return {
                    "success": False,
                    "error": f"Missing required argument: {arg}",
                    "error_type": "missing_argument"
                }
        
        # Call core function
        result = module_function(**kwargs)
        
        return {
            "success": True,
            "result": result,
            "operation": "module_function",
            "module": "module_name"
        }
        
    except ImportError as e:
        return {
            "success": False,
            "error": f"Module import failed: {e}",
            "error_type": "import_error"
        }
    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "execution_error"
        }

# MCP Tool Registry
MCP_TOOLS: Dict[str, Dict[str, Any]] = {
    "module_function": {
        "function": module_function_mcp,
        "description": "Description of what this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to input file"
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose output",
                    "default": False
                }
            },
            "required": ["input_path"]
        }
    }
}

def register_tools(mcp=None) -> None:
    """
    Register MCP tools with the server.
    
    Args:
        mcp: Optional MCP server instance.
    """
    logger.info("module_name module MCP tools registered.")
```

---

## Dependency Injection Pattern

### Configuration Injection

```python
class Processor:
    """Processor with dependency injection."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        cache: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize processor with injected dependencies.
        
        Args:
            config: Configuration dictionary. Uses defaults if None.
            logger: Logger instance. Creates new one if None.
            cache: Result cache. Creates new one if None.
        """
        self.config = config or self._default_config()
        self.logger = logger or logging.getLogger(__name__)
        self.cache = cache if cache is not None else {}
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "timeout": 30,
            "max_retries": 3,
            "validate": True,
        }
    
    def process(self, input_data: Any) -> Any:
        """Process with injected dependencies."""
        if self.config.get("validate"):
            self._validate(input_data)
        return self._do_process(input_data)
```

### Factory Pattern for Framework Selection

```python
class RendererFactory:
    """Factory for creating framework-specific renderers."""
    
    _renderers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, framework: str, renderer_class: type) -> None:
        """Register a renderer for a framework."""
        cls._renderers[framework] = renderer_class
    
    @classmethod
    def create(cls, framework: str, **kwargs) -> "BaseRenderer":
        """
        Create a renderer for the specified framework.
        
        Args:
            framework: Framework name (pymdp, jax, rxinfer, etc.)
            **kwargs: Renderer configuration.
        
        Returns:
            Configured renderer instance.
        
        Raises:
            ValueError: If framework is not supported.
        """
        if framework not in cls._renderers:
            available = ", ".join(cls._renderers.keys())
            raise ValueError(
                f"Unknown framework: {framework}. "
                f"Available: {available}"
            )
        return cls._renderers[framework](**kwargs)

# Register renderers
RendererFactory.register("pymdp", PyMDPRenderer)
RendererFactory.register("jax", JAXRenderer)
RendererFactory.register("rxinfer", RxInferRenderer)
```

---

## Integration Patterns

### Cross-Module Communication

Modules communicate through standardized data structures:

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
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "module": self.module,
            "operation": self.operation,
            "output_files": [str(p) for p in self.output_files],
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
        }
```

### Pipeline Data Flow

```
Step 3 (GNN) → output/3_gnn_output/
    ├── gnn_processing_results.json  # Processing metadata
    └── model_name/
        └── model_name_parsed.json   # Parsed model data
            ↓
Step 11 (Render) → reads parsed.json → output/11_render_output/
    └── model_name/
        ├── pymdp/model_pymdp.py
        ├── jax/model_jax.py
        └── rxinfer/model_rxinfer.jl
            ↓
Step 12 (Execute) → runs generated scripts → output/12_execute_output/
    └── execution_results/
        ├── execution_report.md
        └── results.json
```

---

## Testing Pattern for Modules

### Module Test Structure

```python
"""Tests for module_name module."""

import pytest
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModuleImports:
    """Test module can be imported correctly."""
    
    def test_import_module(self):
        """Test basic import."""
        import module_name
        assert hasattr(module_name, "__version__")
        assert hasattr(module_name, "FEATURES")
    
    def test_import_functions(self):
        """Test function imports."""
        from module_name import process_main_function
        assert callable(process_main_function)


class TestModuleFunctionality:
    """Test core module functionality."""
    
    @pytest.fixture
    def temp_output(self, tmp_path):
        """Create temporary output directory."""
        output = tmp_path / "output"
        output.mkdir()
        return output
    
    def test_process_basic(self, temp_output):
        """Test basic processing."""
        from module_name import process_main_function
        import logging
        
        logger = logging.getLogger("test")
        result = process_main_function(
            target_dir=Path("input/gnn_files"),
            output_dir=temp_output,
            logger=logger,
        )
        
        assert result is True


class TestModuleMCP:
    """Test MCP integration."""
    
    @pytest.mark.safe_to_fail
    def test_mcp_tools_registered(self):
        """Test MCP tools are registered."""
        try:
            from module_name.mcp import MCP_TOOLS
            assert len(MCP_TOOLS) > 0
        except ImportError:
            pytest.skip("MCP not available")
```

---

## Documentation Requirements

### AGENTS.md Template

Every module must have an AGENTS.md file:

```markdown
# Module Name Agent

**Version**: 1.0.0  
**Status**: Production Ready  
**Pipeline Step**: N

## Overview

Brief description of what this agent/module does.

## Capabilities

- Capability 1: Description
- Capability 2: Description
- Capability 3: Description

## API Reference

### Main Functions

#### `process_function(target_dir, output_dir, logger, **kwargs)`

Description of the function.

**Parameters:**
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory
- `logger` (Logger): Logger instance

**Returns:**
- bool: Success status

### MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| tool_name | What it does | param1, param2 |

## Configuration

Environment variables and configuration options.

## Examples

```python
from module_name import process_function
result = process_function(...)
```

## Troubleshooting

Common issues and solutions.
```

---

**Last Updated**: December 2025  
**Status**: Production Standard

