# Thin Orchestrator Pattern - GNN Pipeline Architecture

## Overview

The GNN Processing Pipeline follows a **thin orchestrator pattern** to ensure maintainability, modularity, and comprehensive testability. This pattern separates pipeline orchestration from core domain logic.

## 🏗️ Architectural Pattern

### Core Components

1. **Numbered Scripts** (e.g., `11_render.py`, `10_ontology.py`): Thin orchestrators that handle:
   - Pipeline orchestration
   - Argument parsing and validation
   - Logging setup and management
   - Output directory management
   - Result aggregation and reporting
   - Error handling and recovery

2. **Module `__init__.py`**: Imports and exposes functions from modular files within the module folder

3. **Modular Files** (e.g., `src/render/renderer.py`, `src/ontology/processor.py`): Contain the actual implementation of core methods

4. **Tests**: All methods are tested in `src/tests/` with comprehensive test coverage

## 📁 File Organization

```
src/
├── 11_render.py                    # Thin orchestrator - imports from render/
├── render/
│   ├── __init__.py                 # Imports from renderer.py, pymdp/, etc.
│   ├── renderer.py                 # Core rendering functions
│   ├── pymdp/                      # PyMDP-specific rendering
│   │   ├── __init__.py
│   │   ├── pymdp_renderer.py
│   │   └── pymdp_converter.py
│   ├── rxinfer/                    # RxInfer.jl-specific rendering
│   │   ├── __init__.py
│   │   └── rxinfer_renderer.py
│   └── discopy/                    # DisCoPy-specific rendering
│       ├── __init__.py
│       └── discopy_renderer.py
├── 10_ontology.py                  # Thin orchestrator - imports from ontology/
├── ontology/
│   ├── __init__.py                 # Imports from processor.py
│   └── processor.py                # Core ontology processing functions
└── tests/
    ├── test_render_integration.py  # Tests for render module
    └── test_ontology_integration.py # Tests for ontology module
```

## ✅ Correct Pattern Examples

### Thin Orchestrator Script (`11_render.py`)

```python
#!/usr/bin/env python3
"""
Render GNN specifications to various target languages.
"""

from pathlib import Path
import logging

# Import core functionality from module
from render import (
    generate_pymdp_code,
    generate_rxinfer_code,
    generate_activeinference_jl_code,
    generate_discopy_code
)

def main():
    """Main rendering orchestration."""
    # Setup logging, argument parsing, output directories
    # ...
    
    # Delegate core functionality to module
    pymdp_code = generate_pymdp_code(model_data)
    rxinfer_code = generate_rxinfer_code(model_data)
    # ...
    
    # Handle results and reporting
    # ...
```

### Module `__init__.py` (`src/render/__init__.py`)

```python
"""
Render module for GNN Processing Pipeline.
"""

# Import core functions from modular files
from .renderer import (
    generate_pymdp_code,
    generate_rxinfer_code,
    generate_activeinference_jl_code,
    generate_discopy_code
)

__all__ = [
    'generate_pymdp_code',
    'generate_rxinfer_code',
    'generate_activeinference_jl_code',
    'generate_discopy_code'
]
```

### Modular File (`src/render/renderer.py`)

```python
#!/usr/bin/env python3
"""
Renderer Module - Core rendering functionality.
"""

def generate_pymdp_code(model_data: Dict) -> str:
    """Generate PyMDP simulation code."""
    # Actual implementation here
    # ...

def generate_rxinfer_code(model_data: Dict) -> str:
    """Generate RxInfer.jl simulation code."""
    # Actual implementation here
    # ...
```

## ❌ Incorrect Pattern Examples

### Wrong: Defining Core Logic in Numbered Script

```python
# WRONG: 11_render.py should NOT contain this
def generate_pymdp_code(model_data: Dict) -> str:
    """Generate PyMDP simulation code."""
    # This should be in src/render/renderer.py
    # ...
```

### Wrong: Long Method Definitions in Numbered Scripts

```python
# WRONG: Any method >20 lines should be in modular files
def process_complex_rendering_logic():
    """This is too long for a numbered script."""
    # 50+ lines of implementation
    # Should be moved to src/render/renderer.py
```

## 🧪 Testing Pattern

### Test File (`src/tests/test_render_integration.py`)

```python
#!/usr/bin/env python3
"""
Test render integration functionality.
"""

import pytest
from render import (
    generate_pymdp_code,
    generate_rxinfer_code
)

class TestRenderIntegration:
    """Test render integration functionality."""
    
    def test_generate_pymdp_code(self):
        """Test PyMDP code generation."""
        model_data = {"model_name": "test_model"}
        code = generate_pymdp_code(model_data)
        assert "import pymdp" in code
        assert "test_model" in code
    
    def test_generate_rxinfer_code(self):
        """Test RxInfer.jl code generation."""
        model_data = {"model_name": "test_model"}
        code = generate_rxinfer_code(model_data)
        assert "using RxInfer" in code
        assert "test_model" in code
```

## 🎯 Benefits of This Pattern

1. **Maintainability**: Core logic is separated from orchestration
2. **Testability**: Each component can be tested independently
3. **Modularity**: Easy to add new functionality without modifying numbered scripts
4. **Reusability**: Core functions can be imported and used elsewhere
5. **Clarity**: Clear separation of concerns between pipeline flow and domain logic

## 🔧 Implementation Guidelines

### For New Pipeline Steps

1. **Create the numbered script** (e.g., `12_new_step.py`): Handle orchestration only
2. **Create the module directory** (e.g., `src/new_step/`): Organize core functionality
3. **Create modular files** (e.g., `src/new_step/processor.py`): Implement core logic
4. **Update module `__init__.py`**: Import and expose functions
5. **Create tests** (e.g., `src/tests/test_new_step_integration.py`): Comprehensive testing

### For Existing Steps

1. **Move core logic** from numbered scripts to modular files
2. **Update numbered scripts** to import from modules
3. **Update module `__init__.py`** to expose functions
4. **Create/update tests** for comprehensive coverage

## 📋 Checklist for Compliance

- [ ] Numbered scripts contain only orchestration logic
- [ ] Core methods are defined in modular files within module folders
- [ ] Module `__init__.py` imports and exposes functions from modular files
- [ ] All methods are tested in `src/tests/`
- [ ] No method definitions >20 lines in numbered scripts
- [ ] Clear separation between pipeline flow and domain logic
- [ ] Comprehensive error handling and logging
- [ ] Proper argument parsing and validation

## 🔍 Validation Commands

```bash
# Check for long methods in numbered scripts
find src/ -name "*.py" -maxdepth 1 | grep -E "[0-9]+_.*\.py" | xargs -I {} python -c "
import ast
with open('{}', 'r') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
        print('{}: {} has {} lines'.format('{}', node.name, len(node.body)))
"

# Check module imports
python -c "
import sys
sys.path.insert(0, 'src')
for i in range(23):
    try:
        script = f'{i:02d}_*.py'
        import glob
        files = glob.glob(f'src/{script}')
        if files:
            print(f'Checking {files[0]}')
            with open(files[0], 'r') as f:
                content = f.read()
                if 'def ' in content and len(content.split('def ')) > 5:
                    print(f'  WARNING: {files[0]} may contain too many function definitions')
    except Exception as e:
        print(f'Error checking step {i}: {e}')
"
```
