#!/usr/bin/env python3
"""
DisCoPy/JAX component bootstrap, availability probing, and setup reporting for GNN Step 11.

Extracted from ``render.discopy.translator``.
"""

import datetime
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
)

logger = logging.getLogger(__name__)


# Global variables for DisCoPy components, JAX, and jax.numpy.
# Availability flags below define whether these names contain imported objects.
Dim: Any = None
Box: Any = None
Diagram: Any = None
Id: Any = None
Swap: Any = None
Cup: Any = None
Cap: Any = None
Spider: Any = None
Functor: Any = None
Matrix: Any = None
Ty: Any = None
Word: Any = None
jax: Any = None
jnp: Any = None
discopy_backend: Any = None  # For discopy.matrix.backend

# Availability flags
TENSOR_COMPONENTS_AVAILABLE = False
TY_AVAILABLE = False
JAX_CORE_AVAILABLE = False  # Specific to JAX itself
DISCOPY_MATRIX_MODULE_AVAILABLE = False  # Specific to discopy.matrix module
JAX_AVAILABLE = False  # Overall flag for JAX-backed DisCoPy readiness


# Error reporting and setup guidance
class DisCoPySetupError(Exception):
    """Raised when DisCoPy or JAX components are not available."""


def generate_setup_report() -> str:
    """Generate comprehensive setup instructions for DisCoPy and JAX."""
    return f"""
# DisCoPy and JAX Setup Required

## Current Status
- DisCoPy Tensor Components: {"✓ Available" if TENSOR_COMPONENTS_AVAILABLE else "✗ Not Available"}
- DisCoPy Ty/Word Components: {"✓ Available" if TY_AVAILABLE else "✗ Not Available"}
- JAX Core: {"✓ Available" if JAX_CORE_AVAILABLE else "✗ Not Available"}
- DisCoPy Matrix Backend: {"✓ Available" if DISCOPY_MATRIX_MODULE_AVAILABLE else "✗ Not Available"}

## Required Installation

### 1. Install DisCoPy
```bash
uv pip install discopy
```

### 2. Install JAX (for matrix operations)
```bash
# For CPU only
uv pip install jax jaxlib

# For GPU support (CUDA)
uv pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 3. Verify Installation
```python
# Test DisCoPy
import discopy
from discopy.tensor import Dim, Box, Diagram
from discopy.matrix import Matrix

# Test JAX
import jax
import jax.numpy as jnp
print(f"JAX version: {{jax.__version__}}")
```

## Features Available After Setup

### Basic DisCoPy Operations
- GNN to DisCoPy diagram conversion
- Tensor dimension mapping
- Connection parsing and diagram construction

### JAX-Enhanced Features
- Matrix diagram generation with JAX arrays
- Tensor initialization with random values
- Diagram evaluation and computation
- GPU acceleration support

## Troubleshooting

### Common Issues
1. **ImportError: No module named 'discopy'**
   - Solution: `uv pip install discopy`

2. **ImportError: No module named 'jax'**
   - Solution: `uv pip install jax jaxlib`

3. **JAX backend not available**
   - Solution: Ensure JAX is properly installed and accessible

### Verification Commands
```python
# Test complete setup
from discopy.tensor import Dim, Box, Diagram
from discopy.matrix import Matrix
import jax.numpy as jnp

# Create a simple diagram
d = Dim(2)
box = Box('test', d, d)
diagram = box >> box
print("DisCoPy setup successful!")

# Test JAX integration
arr = jnp.array([[1, 0], [0, 1]])
print(f"JAX array: {{arr}}")
```

## Alternative: Basic Mode
If you only need basic GNN parsing without DisCoPy/JAX:
- Use the GNN parser directly: `from gnn import parse_gnn_file`
- Export to other formats: JSON, XML, GraphML
- Use visualization tools that don't require DisCoPy
"""


def create_discopy_error_report(
    gnn_file_path: Path, error_type: str = "unavailable"
) -> Dict[str, Any]:
    """
    Create a comprehensive error report when DisCoPy functionality is not available.

    Args:
        gnn_file_path: Path to the GNN file that was being processed
        error_type: Type of error ("unavailable", "import_failed", "initialization_failed")

    Returns:
        Dictionary containing error report and setup instructions
    """
    report: dict[str, Any] = {
        "success": False,
        "error_type": error_type,
        "gnn_file": str(gnn_file_path),
        "timestamp": datetime.datetime.now().isoformat(),
        "setup_required": True,
        "availability_status": {
            "tensor_components": TENSOR_COMPONENTS_AVAILABLE,
            "ty_components": TY_AVAILABLE,
            "jax_core": JAX_CORE_AVAILABLE,
            "discopy_matrix": DISCOPY_MATRIX_MODULE_AVAILABLE,
            "overall_jax": JAX_AVAILABLE,
        },
        "setup_instructions": generate_setup_report(),
        "alternative_suggestions": [
            "Use GNN parser directly: from gnn import parse_gnn_file",
            "Export to other formats: JSON, XML, GraphML",
            "Use visualization tools that don't require DisCoPy",
            "Process GNN files with other pipeline steps (1-6, 8-9, 11-13)",
        ],
    }

    return report


def check_discopy_availability() -> Dict[str, bool]:
    """Check availability of all DisCoPy and JAX components."""
    availability: dict[str, Any] = {
        "tensor_components": False,
        "ty_components": False,
        "jax_core": False,
        "discopy_matrix": False,
        "overall_jax": False,
    }

    # Check tensor components
    try:
        from discopy.matrix import Matrix as Matrix_actual
        from discopy.tensor import Box as Box_actual
        from discopy.tensor import Cap as Cap_actual
        from discopy.tensor import Cup as Cup_actual
        from discopy.tensor import Diagram as Diagram_actual
        from discopy.tensor import Dim as Dim_actual
        from discopy.tensor import Functor as Functor_actual
        from discopy.tensor import Id as Id_actual
        from discopy.tensor import Spider as Spider_actual
        from discopy.tensor import Swap as Swap_actual

        _ = (
            Matrix_actual,
            Cap_actual,
            Cup_actual,
            Diagram_actual,
            Functor_actual,
            Id_actual,
            Spider_actual,
            Swap_actual,
        )

        # Test basic functionality
        test_dim = Dim_actual(2)
        Box_actual("test", test_dim, test_dim)

        availability["tensor_components"] = True
        logger.debug("DisCoPy tensor components available and functional")
    except ImportError as e:
        logger.warning(f"DisCoPy tensor components not available: {e}")
    except Exception as e:
        logger.warning(f"DisCoPy tensor components available but not functional: {e}")

    # Check Ty/Word components
    try:
        from discopy.grammar.pregroup import Word as Word_actual
        from discopy.monoidal import Ty as Ty_actual

        # Test basic functionality
        test_ty = Ty_actual("test")
        Word_actual("test", test_ty, test_ty)

        availability["ty_components"] = True
        logger.debug("DisCoPy Ty/Word components available and functional")
    except ImportError as e:
        logger.warning(f"DisCoPy Ty/Word components not available: {e}")
    except Exception as e:
        logger.warning(f"DisCoPy Ty/Word components available but not functional: {e}")

    # Check JAX core
    try:
        import jax as jax_actual
        import jax.numpy as jnp_actual

        _ = jax_actual

        # Test basic functionality
        test_array = jnp_actual.array([1, 2, 3])
        jnp_actual.sum(test_array)

        availability["jax_core"] = True
        logger.debug("JAX core available and functional")
    except ImportError as e:
        logger.warning(f"JAX core not available: {e}")
    except Exception as e:
        logger.warning(f"JAX core available but not functional: {e}")

    # Check DisCoPy matrix backend
    if availability["jax_core"]:
        try:
            from discopy.matrix import backend as backend_actual

            # Test backend context
            with backend_actual("jax"):
                logger.debug("DisCoPy matrix backend accepted the JAX backend context")

            availability["discopy_matrix"] = True
            logger.debug("DisCoPy matrix backend available and functional")
        except ImportError as e:
            logger.warning(f"DisCoPy matrix backend not available: {e}")
        except Exception as e:
            logger.warning(f"DisCoPy matrix backend available but not functional: {e}")

    # Overall JAX availability
    availability["overall_jax"] = (
        availability["tensor_components"]
        and availability["jax_core"]
        and availability["discopy_matrix"]
    )

    return availability


def initialize_discopy_components() -> bool:
    """Initialize DisCoPy and JAX components with proper error handling."""
    global Dim, Box, Diagram, Id, Swap, Cup, Cap, Spider, Functor, Matrix
    global Ty, Word, jax, jnp, discopy_backend
    global TENSOR_COMPONENTS_AVAILABLE, TY_AVAILABLE, JAX_CORE_AVAILABLE
    global DISCOPY_MATRIX_MODULE_AVAILABLE, JAX_AVAILABLE

    # Check availability first
    availability = check_discopy_availability()

    # Initialize tensor components
    if availability["tensor_components"]:
        try:
            from discopy.matrix import Matrix as Matrix_actual
            from discopy.tensor import Box as Box_actual
            from discopy.tensor import Cap as Cap_actual
            from discopy.tensor import Cup as Cup_actual
            from discopy.tensor import Diagram as Diagram_actual
            from discopy.tensor import Dim as Dim_actual
            from discopy.tensor import Functor as Functor_actual
            from discopy.tensor import Id as Id_actual
            from discopy.tensor import Spider as Spider_actual
            from discopy.tensor import Swap as Swap_actual

            Dim = Dim_actual
            Box = Box_actual
            Diagram = Diagram_actual
            Id = Id_actual
            Swap = Swap_actual
            Cup = Cup_actual
            Cap = Cap_actual
            Spider = Spider_actual
            Functor = Functor_actual
            Matrix = Matrix_actual

            TENSOR_COMPONENTS_AVAILABLE = True
            logger.info("DisCoPy tensor components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DisCoPy tensor components: {e}")
            TENSOR_COMPONENTS_AVAILABLE = False

    # Initialize Ty/Word components
    if availability["ty_components"]:
        try:
            from discopy.grammar.pregroup import Word as Word_actual
            from discopy.monoidal import Ty as Ty_actual

            Ty = Ty_actual
            Word = Word_actual

            TY_AVAILABLE = True
            logger.info("DisCoPy Ty/Word components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DisCoPy Ty/Word components: {e}")
            TY_AVAILABLE = False

    # Initialize JAX components
    if availability["jax_core"]:
        try:
            import jax as jax_actual
            import jax.numpy as jnp_actual

            jax = jax_actual
            jnp = jnp_actual

            JAX_CORE_AVAILABLE = True
            logger.info("JAX components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize JAX components: {e}")
            JAX_CORE_AVAILABLE = False

    # Initialize DisCoPy matrix backend
    if availability["discopy_matrix"]:
        try:
            from discopy.matrix import backend as backend_actual

            discopy_backend = backend_actual

            DISCOPY_MATRIX_MODULE_AVAILABLE = True
            logger.info("DisCoPy matrix backend initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DisCoPy matrix backend: {e}")
            DISCOPY_MATRIX_MODULE_AVAILABLE = False

    # Set overall JAX availability
    JAX_AVAILABLE = (
        TENSOR_COMPONENTS_AVAILABLE
        and JAX_CORE_AVAILABLE
        and DISCOPY_MATRIX_MODULE_AVAILABLE
    )

    if JAX_AVAILABLE:
        logger.info("All DisCoPy and JAX components are available and ready for use")
    else:
        logger.warning("Some DisCoPy or JAX components are not available")
        logger.info("Run generate_setup_report() for installation instructions")

    return JAX_AVAILABLE


# Initialize components on module import (lazy: only once)
_discopy_initialized = False
if not _discopy_initialized:
    initialize_discopy_components()
    _discopy_initialized = True
