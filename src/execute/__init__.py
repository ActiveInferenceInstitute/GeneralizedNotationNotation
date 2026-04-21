"""
execute module for GNN Processing Pipeline.

Provides simulation execution across PyMDP, RxInfer.jl, ActiveInference.jl,
JAX, DisCoPy, PyTorch, and NumPyro backends, plus the dependency/environment
validators that gate them.
"""

__version__ = "1.6.0"
FEATURES = {
    "pymdp_execution": True,
    "rxinfer_execution": True,
    "activeinference_jl_execution": True,
    "discopy_execution": True,
    "jax_execution": True,
    "pytorch_execution": True,
    "numpyro_execution": True,
    "validation": True,
    "mcp_integration": True,
}

from pathlib import Path
from typing import Any, Dict, List, Literal

# Constrained type for supported execution framework names.
FrameworkName = Literal[
    'pymdp',
    'rxinfer',
    'jax',
    'discopy',
    'activeinference_jl',
    'pytorch',
    'numpyro',
]

# All execute submodules are in-tree — their import must succeed or tests
# catch it. Any ImportError here is a real bug, not a "missing optional dep"
# situation, and should fail loudly.
from .executor import (
    GNNExecutor,
    execute_gnn_model,
    execute_script_safely,
    run_simulation,
)
from .pymdp import (
    PyMDPSimulation,
    execute_pymdp_simulation,
    execute_pymdp_simulation_from_gnn,
    get_pymdp_health_status,
    validate_pymdp_environment,
)
from .validator import (
    check_dependencies,
    check_file_permissions,
    check_network_connectivity,
    check_python_environment,
    check_system_resources,
    log_validation_results,
    validate_execution_environment,
)
from .processor import process_execute

# ``execute_simulation_from_gnn`` is the canonical name exported from the
# processor (if defined there) or the executor — checked in that order.
try:
    from .processor import execute_simulation_from_gnn
except ImportError:
    execute_simulation_from_gnn = execute_gnn_model

__all__ = [
    '__version__',
    'FEATURES',
    'FrameworkName',

    # Core classes
    'GNNExecutor',
    'PyMDPSimulation',

    # Execution functions
    'process_execute',
    'execute_simulation_from_gnn',
    'execute_gnn_model',
    'run_simulation',
    'execute_script_safely',

    # PyMDP
    'execute_pymdp_simulation_from_gnn',
    'execute_pymdp_simulation',
    'validate_pymdp_environment',
    'get_pymdp_health_status',

    # Validation
    'validate_execution_environment',
    'log_validation_results',
    'check_python_environment',
    'check_system_resources',
    'check_dependencies',
    'check_file_permissions',
    'check_network_connectivity',
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "execute",
        "version": __version__,
        "description": "Multi-framework execution of rendered simulation scripts",
        "features": FEATURES,
    }
