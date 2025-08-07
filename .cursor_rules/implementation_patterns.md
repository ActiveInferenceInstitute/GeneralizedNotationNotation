# Implementation Patterns and Infrastructure Usage

## Pipeline Script Implementation Pattern

### Standard Import Structure
Every pipeline script should follow this exact import pattern:

```python
#!/usr/bin/env python3
"""
[Script Description]
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Standard infrastructure imports with fallback
try:
    from utils import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_warning,
        log_step_error,
        performance_tracker,
        UTILS_AVAILABLE
    )
    
    from pipeline import (
        get_output_dir_for_script,
        get_pipeline_config
    )
    
except ImportError as e:
    # Fallback implementations for graceful degradation
    # [Minimal compatibility functions here]
    UTILS_AVAILABLE = False

# Module-specific imports with dependency validation
try:
    from [module_name] import [specific_functions]
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    log_step_warning(logger, f"Failed to import dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False
```

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

These patterns ensure consistency, reliability, and maintainability across the entire GNN pipeline system. 