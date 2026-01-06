# Implementation Patterns and Infrastructure Usage

> **Environment Note**: Follow the `uv` workflow for all implementation and tooling activities. Use `uv pip install`, `uv run <script>`, and `uv test` to ensure consistent dependency and environment configuration before resorting to bare `python3` invocations.

## Pipeline Script Implementation Pattern

### Modern Standardized Pattern (PREFERRED)

**All new pipeline scripts should use `create_standardized_pipeline_script`** from `utils.pipeline_template`. This is the current standard pattern used across all 24 pipeline steps (0-23).

```python
#!/usr/bin/env python3
"""
Step N: [Step Name] (Thin Orchestrator)

This step orchestrates [step description] for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/[module_name]/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the module.

Pipeline Flow:
    main.py → N_[module].py (this script) → [module_name]/ (modular implementation)

How to run:
  python src/N_[module].py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - [Module-specific outputs] in the specified output directory
  - Comprehensive reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from [module_name] import [process_function]
except ImportError:
    def [process_function](target_dir, output_dir, logger, **kwargs):
        """Fallback processing when module unavailable."""
        logger.warning("[Module] module not available - using fallback")
        return True

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "N_[module].py",  # Step name (e.g., "11_render.py")
    [process_function],  # Module function to call
    "[Brief step description]",  # Description for help text
    additional_arguments={  # Optional: step-specific arguments
        "custom_option": {
            "type": bool,
            "help": "Custom option description",
            "default": False
        },
        "custom_path": {
            "type": Path,
            "help": "Path to custom resource",
            "flag": "--custom-path"  # Optional custom flag name
        }
    }
)

def main() -> int:
    """Main entry point for the pipeline step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
```

**Key Benefits of Standardized Pattern:**
- **Consistent Argument Parsing**: Automatic handling of `--target-dir`, `--output-dir`, `--verbose`, `--recursive`
- **Automatic Output Directory Management**: Uses `get_output_dir_for_script()` to create step-specific output directories
- **Graceful Degradation**: Fallback argument parser if enhanced parser unavailable
- **Standardized Logging**: Automatic setup of step logging with correlation IDs
- **Error Handling**: Consistent error handling and exit codes
- **Path Normalization**: Automatic conversion of string paths to `Path` objects

### Older Implementation Pattern (For Reference Only)

The following pattern is shown for reference but should NOT be used for new scripts. All existing scripts are being migrated to the standardized pattern above.

### Standardized Main Function Pattern

```python
def main(parsed_args) -> int:
    """
    Main function following standardized pipeline pattern.
    
    Returns:
        0=success, 1=critical error, 2=success with warnings
    """
    
    # Initialize logging with correlation context
    logger = setup_step_logging("[step_name]", verbose=getattr(parsed_args, 'verbose', False))
    log_step_start(logger, "[Step description]")
    
    # Validate requirements
    if not DEPENDENCIES_AVAILABLE:
        log_step_error(logger, "Required dependencies not available")
        return 1
    
    # Get configuration and paths
    config = get_pipeline_config()
    step_config = config.get_step_config("[script_name].py")
    
    input_dir = Path(getattr(parsed_args, 'target_dir', 'input/gnn_files'))
    output_dir = Path(getattr(parsed_args, 'output_dir', 'output'))
    step_output_dir = get_output_dir_for_script("[script_name].py", output_dir)
    step_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract arguments with defaults
    recursive = getattr(parsed_args, 'recursive', True)
    verbose = getattr(parsed_args, 'verbose', False)
    
    # Process with performance tracking
    try:
        with performance_tracker.track_operation("[operation_name]"):
            # Main processing logic here
            success = process_main_logic(input_dir, step_output_dir, {
                'recursive': recursive,
                'verbose': verbose
                # Add step-specific options
            })
        
        if success:
            log_step_success(logger, "[Success message]")
            return 0
        else:
            log_step_warning(logger, "[Warning message]")
            return 2
            
    except Exception as e:
        log_step_error(logger, f"[Error context]: {e}")
        return 1
```

## Argument Parsing Pattern

### Enhanced Argument Parser Usage

```python
def create_argument_parser():
    """Create argument parser with step-specific configuration."""
    
    if UTILS_AVAILABLE:
        from utils import EnhancedArgumentParser
        parser = EnhancedArgumentParser.create_parser(
            description="[Step Description]",
            step_name="[step_name]"
        )
        # Enhanced parser automatically includes common arguments
        
        # Add step-specific arguments
        parser.add_argument('--custom-option', 
                          help="Step-specific option", 
                          default=False, action='store_true')
        
        return parser
    else:
        # Fallback parser for graceful degradation
        parser = argparse.ArgumentParser(description="[Step Description]")
        parser.add_argument('--target-dir', type=str, default='input/gnn_files')
        parser.add_argument('--output-dir', type=str, default='output')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--recursive', action='store_true', default=True)
        # Add step-specific arguments
        return parser

def parse_arguments():
    """Parse arguments with enhanced error handling."""
    
    if UTILS_AVAILABLE:
        from utils import EnhancedArgumentParser
        return EnhancedArgumentParser.parse_step_arguments("[step_name]")
    else:
        parser = create_argument_parser()
        return parser.parse_args()
```

## Module Structure Pattern

### Core Module Organization

```
src/[module_name]/
├── __init__.py              # Module initialization with version and features
├── [module_core].py         # Main functionality implementation
├── mcp.py                   # MCP integration (if applicable)
├── [specialized_files].py   # Additional specialized components
└── [subdirectories]/        # Organized sub-components
```

### Module __init__.py Pattern

**Standard Module Structure:**

```python
#!/usr/bin/env python3
"""
[Module Name] - [Description]

[Module documentation]
"""

from .main_module import [primary_functions]

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "[Module description]"

# Feature availability flags
FEATURES = {
    'core_functionality': True,
    'mcp_integration': True,
    'advanced_features': True,
    # Add module-specific features
}

# Main API exports
__all__ = [
    '[primary_functions]',
    'FEATURES',
    '__version__'
]
```

**Package-Level __init__.py Pattern (src/__init__.py):**

The top-level `src/__init__.py` provides package discovery and metadata:

```python
"""
GNN Pipeline Core Module

This module provides the core functionality for the GNN processing pipeline.
"""

from pathlib import Path
from typing import List

# Package-level metadata expected by tests
__version__ = "1.0.0"
FEATURES = {
    "pipeline_orchestration": True,
    "mcp_integration": True,
}

def _discover_top_level_modules() -> List[str]:
    """Discover all top-level subpackages under the src package.
    
    Returns a sorted list of directory names that contain an __init__.py file
    and do not start with an underscore.
    """
    base_dir = Path(__file__).parent
    module_names: List[str] = []
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith('_'):
            continue
        if (entry / '__init__.py').exists():
            module_names.append(name)
    return sorted(module_names)

def get_module_info() -> dict[str, object]:
    """Get information about the core GNN package.
    
    Returns a dictionary with high-level metadata about the overall package.
    """
    return {
        "name": "GNN Pipeline Core",
        "version": __version__,
        "description": "Core functionality for GNN processing pipeline",
        "modules": _discover_top_level_modules(),
        "features": [
            "Pipeline orchestration",
            "GNN processing",
            "Analysis and statistics",
            # ... additional features
        ],
    }

# Lazy imports for optional dependencies with graceful fallback
import importlib

try:
    # Attempt to import optional modules with fallback placeholders
    optional_module = importlib.import_module('src.optional_module')
except Exception:
    class _OptionalModulePlaceholder:
        __version__ = "1.0.0"
        FEATURES = {'feature': True}
        
        @staticmethod
        def get_module_info() -> dict:
            return {
                'version': _OptionalModulePlaceholder.__version__,
                'description': 'Optional module compatibility shim',
                'features': _OptionalModulePlaceholder.FEATURES,
            }
    
    optional_module = _OptionalModulePlaceholder()

__all__ = [
    'get_module_info',
    'optional_module',
    '__version__',
    'FEATURES',
]
```

**Key Patterns:**
- **Module Discovery**: Automatic discovery of subpackages via `_discover_top_level_modules()`
- **Metadata Exposure**: `get_module_info()` provides standardized module information
- **Lazy Imports**: Optional dependencies imported with graceful fallback to placeholder objects
- **Feature Flags**: `FEATURES` dictionary for capability detection
- **Version Management**: Centralized version tracking

## MCP Integration Pattern

### Standard MCP Module Structure

```python
"""
[Module] MCP Integration

Exposes [module] functionality through Model Context Protocol.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def [module_function]_mcp(mcp_instance_ref, **kwargs) -> Dict[str, Any]:
    """
    MCP tool for [module function].
    
    Args:
        mcp_instance_ref: MCP instance reference
        **kwargs: Function-specific arguments
        
    Returns:
        Standardized MCP response dictionary
    """
    try:
        from .[module_core] import [module_function]
        
        # Extract and validate arguments
        required_args = ['arg1', 'arg2']
        for arg in required_args:
            if arg not in kwargs:
                return {
                    "success": False,
                    "error": f"Missing required argument: {arg}",
                    "error_type": "missing_argument"
                }
        
        # Call core function
        result = [module_function](**kwargs)
        
        return {
            "success": True,
            "result": result,
            "operation": "[module_function]",
            "module": "[module_name]"
        }
        
    except Exception as e:
        logger.error(f"MCP [module_function] error: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "execution_error"
        }

# MCP tool registry
MCP_TOOLS = {
    "[module_function]": {
        "function": [module_function]_mcp,
        "description": "[Function description]",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "Argument description"},
                "arg2": {"type": "boolean", "description": "Argument description"}
            },
            "required": ["arg1"]
        }
    }
}
```

## Error Handling and Logging Patterns

### Structured Error Handling

```python
def robust_function_pattern(input_data, options: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Template for robust function implementation.
    
    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        # Validate inputs
        if not input_data:
            return False, "Input data is required"
            
        # Main processing with nested error handling
        with performance_tracker.track_operation("function_operation"):
            result = process_data(input_data, options)
            
        # Validate results
        if not validate_result(result):
            return False, "Result validation failed"
            
        return True, None
        
    except ValueError as e:
        logger.warning(f"Validation error in function: {e}")
        return False, f"Validation error: {e}"
        
    except Exception as e:
        logger.error(f"Unexpected error in function: {e}")
        return False, f"Processing error: {e}"
```

### Centralized Logging Pattern

```python
# Use structured logging with correlation IDs
logger = setup_step_logging("module_name", verbose=True)

# Step-level logging
log_step_start(logger, "Starting operation", operation_id="op_001")
log_step_success(logger, "Operation completed", operation_id="op_001", duration=1.23)
log_step_warning(logger, "Non-critical issue", operation_id="op_001", issue_type="validation")
log_step_error(logger, "Critical failure", operation_id="op_001", error_type="dependency")

# Performance tracking integration
with performance_tracker.track_operation("expensive_operation"):
    result = expensive_computation()
```

## Configuration Management Pattern

### Configuration Access and Override

```python
def get_step_configuration(step_name: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Get configuration with CLI override support."""
    
    # Load base configuration
    config = get_pipeline_config()
    step_config = config.get_step_config(f"{step_name}.py")
    
    # Create merged configuration with CLI overrides
    merged_config = {
        'target_dir': getattr(cli_args, 'target_dir', config.target_dir),
        'output_dir': getattr(cli_args, 'output_dir', config.output_dir),
        'verbose': getattr(cli_args, 'verbose', config.verbose),
        'recursive': getattr(cli_args, 'recursive', config.recursive),
    }
    
    # Add step-specific configuration
    if step_config:
        merged_config.update({
            'timeout': step_config.timeout,
            'required': step_config.required,
            'dependencies': step_config.dependencies
        })
    
    return merged_config
```

## Testing and Validation Patterns

### Comprehensive Testing Structure

```python
import pytest
from pathlib import Path
import tempfile
import shutil

class TestModulePattern:
    """Template for module testing."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample test data."""
        return {
            'input_files': ['sample1.md', 'sample2.json'],
            'expected_outputs': ['result1.json', 'result2.xml']
        }
    
    def test_core_functionality(self, temp_workspace, sample_data):
        """Test core module functionality."""
        # Setup test environment
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Execute function under test
        from module_name import core_function
        result = core_function(input_dir, output_dir, {'verbose': True})
        
        # Validate results
        assert result is True
        assert (output_dir / "expected_output.json").exists()
    
    def test_error_handling(self, temp_workspace):
        """Test error handling and recovery."""
        # Test with invalid inputs
        result = core_function(None, None, {})
        assert result is False
        
    def test_performance_benchmark(self, temp_workspace, sample_data):
        """Test performance characteristics."""
        with performance_tracker.track_operation("benchmark_test"):
            result = core_function(input_dir, output_dir, sample_data)
        
        # Validate performance metrics
        assert performance_tracker.get_last_duration() < 10.0  # seconds
```

## Quality Assurance Patterns

### Documentation and Communication Standards
- **Direct Updates**: Update existing documentation files directly rather than creating separate report files
- **Functional Focus**: Make smart functional improvements to code and documentation inline
- **Real Demonstrations**: Show functionality through working code and actual outputs
- **Concrete Examples**: Use specific file sizes, execution times, and measurable results
- **Understated Approach**: Avoid promotional language, focus on concrete functionality

### Input Validation Pattern

```python
def validate_inputs(input_dir: Path, output_dir: Path, options: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Comprehensive input validation pattern.
    
    Returns:
        (is_valid: bool, error_messages: List[str])
    """
    errors = []
    
    # Path validation
    if not input_dir.exists():
        errors.append(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        errors.append(f"Input path is not a directory: {input_dir}")
        
    # Output directory validation
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory: {e}")
    
    # Options validation
    required_options = ['format', 'validate']
    for option in required_options:
        if option not in options:
            errors.append(f"Missing required option: {option}")
    
    return len(errors) == 0, errors
```

### Performance Monitoring Pattern

```python
def performance_monitored_function(data, options):
    """Function with comprehensive performance monitoring."""
    
    operation_id = f"func_{int(time.time())}"
    
    with performance_tracker.track_operation(operation_id) as tracker:
        # Track memory usage
        initial_memory = tracker.get_memory_usage()
        
        # Main processing
        result = process_data(data)
        
        # Track peak memory
        peak_memory = tracker.get_peak_memory()
        
        # Log performance metrics
        logger.info(f"Operation {operation_id} completed", extra={
            'duration': tracker.duration,
            'memory_delta': peak_memory - initial_memory,
            'data_size': len(data),
            'operation_type': 'data_processing'
        })
    
    return result
```

## Enhanced Visual Logging Pattern

### Visual Logging Integration

All pipeline steps should integrate with the enhanced visual logging system for better user experience and accessibility:

```python
from utils.visual_logging import (
    create_visual_logger,
    VisualConfig,
    format_step_header,
    format_status_message
)

# Setup enhanced visual logging
visual_config = VisualConfig(
    enable_colors=True,
    enable_progress_bars=True,
    enable_emoji=True,
    enable_animation=True,
    show_timestamps=args.verbose,
    show_correlation_ids=True,
    compact_mode=False
)

visual_logger = create_visual_logger("step_name", visual_config)

# Use visual indicators
visual_logger.print_header(
    "🎨 Step Title",
    f"Processing description | Output: {output_dir}"
)

visual_logger.print_progress(current, total, "Processing items")

visual_logger.print_success("Operation completed successfully")
visual_logger.print_warning("Non-critical issue detected")
visual_logger.print_error("Critical error occurred")
```

**Visual Features:**
- **Progress Indicators**: Real-time progress bars and completion indicators
- **Status Icons**: Standardized emoji icons (✅, ⚠️, ❌, 🔄) for clear status communication
- **Color Coding**: Consistent color schemes (green=success, yellow=warning, red=error, blue=info)
- **Screen Reader Support**: Emoji-free alternatives and structured text for accessibility
- **Correlation IDs**: Unique tracking IDs for debugging and monitoring across pipeline steps
- **Performance Metrics**: Display timing, memory usage, and resource consumption clearly
- **Structured Summaries**: Formatted tables and panels for key metrics and completion status

## Module Function Signature Pattern

### Standard Module Function Interface

All module functions called by pipeline scripts must follow this signature:

```python
def process_module_function(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process [module] functionality for GNN models.
    
    Args:
        target_dir: Directory containing input files to process
        output_dir: Step-specific output directory (already normalized by orchestrator)
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional step-specific arguments
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Validate inputs
        if not target_dir.exists():
            log_step_error(logger, f"Target directory does not exist: {target_dir}")
            return False
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main processing logic here
        # ...
        
        return True
        
    except Exception as e:
        log_step_error(logger, f"Processing failed: {e}")
        return False
```

**Critical Requirements:**
- **First Parameter**: `target_dir: Path` - Input directory
- **Second Parameter**: `output_dir: Path` - Output directory (already normalized by orchestrator)
- **Third Parameter**: `logger: logging.Logger` - Logger instance
- **Standard Parameters**: `recursive: bool = False`, `verbose: bool = False`
- **Flexible Parameters**: `**kwargs` for step-specific options
- **Return Type**: `bool` - True for success, False for failure

## Output Directory Management Pattern

### Standardized Output Directory Structure

All pipeline steps use `get_output_dir_for_script()` to create step-specific output directories:

```python
from pipeline.config import get_output_dir_for_script

# In pipeline script (handled automatically by create_standardized_pipeline_script):
step_output_dir = get_output_dir_for_script("N_module", output_dir)
# Creates: output/N_module_output/

# Output directory structure:
# output/
# ├── 0_template_output/
# ├── 1_setup_output/
# ├── 2_tests_output/
# ├── 3_gnn_output/
# ├── ... (all 24 steps)
# └── pipeline_execution_summary.json
```

**Directory Naming Convention:**
- Format: `{step_number}_{module_name}_output`
- Examples: `3_gnn_output`, `11_render_output`, `22_gui_output`
- All outputs organized under base `output/` directory
- Each step has isolated output directory

## Optional Dependency Handling Pattern

### Detection and Fallback

```python
# Standard pattern for optional dependencies
try:
    import pymdp
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False
    logger.info(
        "PyMDP not available - this is normal if not installed. "
        "Install with: pip install pymdp. "
        "Continuing with fallback mode."
    )

def process_with_optional_dep(model: Dict[str, Any]) -> bool:
    """Process with optional dependency fallback."""
    if PYMDP_AVAILABLE:
        return _process_with_pymdp(model)
    else:
        return _process_fallback(model)
```

### Feature Flags Pattern

```python
# In module __init__.py
FEATURES = {
    "pymdp_simulation": _check_pymdp(),
    "jax_rendering": _check_jax(),
    "julia_execution": _check_julia(),
}

def get_available_features() -> Dict[str, bool]:
    """Get dictionary of available features."""
    return FEATURES.copy()
```

**Key Principles:**
- Optional dependencies should never cause import failures
- Always provide graceful fallback with informative messages
- Log availability status for debugging
- See: [optional_dependencies.md](optional_dependencies.md) for details

## Safe-to-Fail Pattern

### Steps 8, 9, 12 Implementation

These steps implement comprehensive safe-to-fail patterns:

```python
def safe_to_fail_step(target_dir: Path, output_dir: Path, logger) -> bool:
    """
    Safe-to-fail step - NEVER returns False or non-zero exit.
    """
    try:
        # Level 1: Full processing
        result = full_processing(target_dir, output_dir)
        if result:
            return True
    except Exception as e:
        logger.warning(f"Full processing failed: {e}")
    
    try:
        # Level 2: Reduced processing
        result = reduced_processing(target_dir, output_dir)
        if result:
            return True
    except Exception as e:
        logger.warning(f"Reduced processing failed: {e}")
    
    try:
        # Level 3: Fallback report (always succeeds)
        generate_fallback_report(output_dir)
        return True  # ALWAYS return True
    except Exception:
        pass
    
    return True  # Even on complete failure, return True

def main() -> int:
    """Main function - always returns 0."""
    try:
        success = safe_to_fail_step(...)
    except Exception as e:
        logger.error(f"Step failed: {e}")
    
    return 0  # ALWAYS return 0 - never stop pipeline
```

**Key Principles:**
- Never return exit code 1 (critical error)
- Implement multiple fallback levels
- Generate output even on failure
- See: [error_handling.md](error_handling.md) for details

---

## Related Documentation

- **[module_patterns.md](module_patterns.md)**: Advanced module architecture
- **[error_handling.md](error_handling.md)**: Error strategies and recovery
- **[optional_dependencies.md](optional_dependencies.md)**: Optional package handling
- **[code_quality.md](code_quality.md)**: Quality standards

---

**Last Updated**: December 2025  
**Status**: Production Standard