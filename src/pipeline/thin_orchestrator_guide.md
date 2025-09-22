# Thin Orchestrator Pattern Implementation Guide

## Overview

The GNN pipeline follows the **Thin Orchestrator Pattern** where numbered scripts (e.g., `2_tests.py`, `5_type_checker.py`) act as minimal orchestrators that delegate core functionality to their corresponding modules.

## ✅ Correct Pattern - Thin Orchestrator

### Example: `3_gnn.py` (30 lines - PERFECT)

```python
#!/usr/bin/env python3
"""
Step 3: GNN File Discovery and Parsing (Thin Orchestrator)

Delegates discovery, parsing, and multi-format serialization to
`gnn/multi_format_processor.py` using the standardized pipeline wrapper.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from gnn.multi_format_processor import process_gnn_multi_format

run_script = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_multi_format,
    "GNN discovery, parsing, and multi-format serialization",
)

def main() -> int:
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
```

**Key characteristics:**
- ✅ **Only 30 lines** - minimal orchestration
- ✅ **Single responsibility** - just delegates to module
- ✅ **No implementation code** - all logic in `gnn/multi_format_processor.py`
- ✅ **Clear delegation** - imports and calls `process_gnn_multi_format()`
- ✅ **Standardized wrapper** - uses `create_standardized_pipeline_script()`

### Example: `6_validation.py` (55 lines - GOOD)

```python
#!/usr/bin/env python3
"""
Step 6: Validation Processing (Thin Orchestrator)

This step performs validation and quality assurance on GNN models,
including semantic validation, performance profiling, and consistency checking.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from validation import process_validation
except ImportError:
    def process_validation(target_dir, output_dir, logger, **kwargs):
        """Fallback validation when module unavailable."""
        logger.warning("Validation module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "6_validation.py",
    process_validation,
    "Validation processing for GNN models",
    additional_arguments={
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "profile": {"type": bool, "help": "Enable performance profiling"}
    }
)

def main() -> int:
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
```

**Key characteristics:**
- ✅ **55 lines** - still reasonably thin
- ✅ **Proper fallback handling** - graceful degradation when module unavailable
- ✅ **Clean delegation** - directly calls `process_validation()`
- ✅ **Additional arguments** - properly configured for step-specific options

## ❌ Incorrect Pattern - Fat Orchestrator

### Example: `2_tests.py` (877 lines - VIOLATION)

**Problems:**
- ❌ **877 lines** - contains substantial implementation code
- ❌ **process_tests_standardized()** function defined directly in script
- ❌ **execute_test_suite()** function defined directly in script
- ❌ **validate_test_syntax()** function defined directly in script
- ❌ **generate_test_report()** function defined directly in script

**Should be refactored to:**
1. Move all implementation to `src/tests/runner.py` or `src/tests/processor.py`
2. Keep only thin orchestration in `2_tests.py`
3. Delegate to module functions like `tests.process_test_suite()`

### Example: `9_advanced_viz.py` (769 lines - VIOLATION)

**Problems:**
- ❌ **769 lines** - extensive implementation code
- ❌ **process_advanced_viz_standardized()** function with 400+ lines
- ❌ **SafeAdvancedVisualizationManager** class defined in script
- ❌ **AdvancedVisualizationResults** dataclass defined in script
- ❌ **generate_fallback_html_visualization()** method defined in script

**Should be refactored to:**
1. Move classes to `src/advanced_visualization/core.py`
2. Move implementation to `src/advanced_visualization/processor.py`
3. Keep only orchestration in `9_advanced_viz.py`
4. Delegate to module like `advanced_visualization.process_advanced_viz()`

## 📋 Implementation Checklist

### For Each Numbered Script

1. **✅ Script should be thin** (ideally <100 lines, maximum <200 lines)
2. **✅ No implementation functions** - only orchestration logic
3. **✅ Proper module delegation** - imports and calls module functions
4. **✅ Standard pipeline wrapper** - uses `create_standardized_pipeline_script()`
5. **✅ Fallback handling** - graceful degradation when modules unavailable
6. **✅ Additional arguments** - properly configured for step-specific options

### Module Structure Pattern

```
src/
├── N_step.py                    # Thin orchestrator
├── step_module/                 # Module directory
│   ├── __init__.py             # Exports module functions
│   ├── processor.py            # Core implementation
│   ├── utils.py                # Helper functions
│   └── mcp.py                  # MCP integration
└── tests/
    └── test_step_integration.py # Integration tests
```

## 🔧 Refactoring Strategy

### Step 1: Identify Fat Orchestrators
- `2_tests.py` - 877 lines (CRITICAL)
- `9_advanced_viz.py` - 769 lines (CRITICAL)
- `5_type_checker.py` - 383 lines (HIGH)
- `4_model_registry.py` - 277 lines (HIGH)
- `12_execute.py` - 175 lines (MEDIUM)
- `10_ontology.py` - 145 lines (MEDIUM)
- `11_render.py` - 136 lines (MEDIUM)

### Step 2: Create Module Structure
For each violating script, create:
- `src/[module_name]/__init__.py`
- `src/[module_name]/processor.py`
- `src/[module_name]/mcp.py` (if applicable)

### Step 3: Move Implementation Code
Move all implementation functions from script to appropriate module files.

### Step 4: Refactor Script to Thin Orchestrator
Replace implementation code with simple delegation to module functions.

### Step 5: Test Integration
Ensure refactored scripts work with existing pipeline.

## 📖 Pattern Examples by Complexity

### Simple Delegation Pattern
```python
from module_name import process_function

run_script = create_standardized_pipeline_script(
    "N_step.py",
    process_function,
    "Step description"
)
```

### Fallback Pattern
```python
try:
    from module_name import process_function
except ImportError:
    def process_function(target_dir, output_dir, logger, **kwargs):
        logger.warning("Module unavailable - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "N_step.py",
    process_function,
    "Step description"
)
```

### Parameterized Pattern
```python
from module_name import process_function

run_script = create_standardized_pipeline_script(
    "N_step.py",
    process_function,
    "Step description",
    additional_arguments={
        "param1": {"type": str, "help": "Parameter description"},
        "param2": {"type": bool, "default": True, "help": "Flag description"}
    }
)
```

This guide demonstrates the thin orchestrator pattern through concrete, working examples rather than abstract descriptions.

