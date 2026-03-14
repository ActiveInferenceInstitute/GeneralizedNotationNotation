"""
execute module for GNN Processing Pipeline.

This module provides execute capabilities with comprehensive safety patterns,
validation, monitoring, and error recovery systems.
"""

__version__ = "1.1.3"
FEATURES = {
    "pymdp_execution": True,
    "rxinfer_execution": True,
    "activeinference_jl_execution": True,
    "discopy_execution": True,
    "jax_execution": True,
    "pytorch_execution": True,
    "numpyro_execution": True,
    "validation": True,
    "error_recovery": True,
    "mcp_integration": True
}

from pathlib import Path
from typing import Dict, Any, List, Literal
import logging

# Constrained type for supported execution framework names.
# Use this annotation on any parameter that accepts a framework identifier.
FrameworkName = Literal[
    'pymdp',
    'rxinfer',
    'jax',
    'discopy',
    'activeinference_jl',
    'pytorch',
    'numpyro',
]

# Import execute functionality
try:
    from .executor import ExecutionEngine, GNNExecutor, execute_gnn_model, run_simulation
    from .pymdp import (
        execute_pymdp_simulation_from_gnn,
        execute_pymdp_simulation,
        PyMDPSimulation,
        validate_pymdp_environment,
        get_pymdp_health_status
    )
    from .validator import (
        validate_execution_environment,
        log_validation_results,
        check_python_environment,
        check_system_resources,
        check_dependencies,
        check_file_permissions,
        check_network_connectivity
    )
    # Backwards-compatibility aliases — PyMdpExecutor is deprecated, use PyMDPSimulation
    execute_simulation_from_gnn = execute_gnn_model
    PyMdpExecutor = PyMDPSimulation  # deprecated alias
    VALIDATION_AVAILABLE = True
    ERROR_RECOVERY_AVAILABLE = True
except ImportError as e:
    import logging as _log
    _log.getLogger(__name__).warning(f"Execute module import failed: {e} - using recovery")
    # Import recovery functions
    from .recovery import (
        ExecutionEngine,
        PyMdpExecutor,
        execute_simulation_from_gnn,
        execute_pymdp_simulation_from_gnn_impl as execute_pymdp_simulation_from_gnn,
        validate_execution_environment_impl as validate_execution_environment,
        validate_execution_environment as validate_execution_environment_fallback,
        get_execution_health_status,
        log_validation_results,
        check_python_environment,
        check_system_dependencies,
        check_python_packages,
        check_file_system,
        analyze_pipeline_error,
        generate_error_recovery_report,
        get_quick_error_suggestions,
        ErrorRecoverySystem
    )
    # Stubs for missing items
    execute_gnn_model = execute_simulation_from_gnn
    GNNExecutor = ExecutionEngine
    PyMDPSimulation = PyMdpExecutor  # deprecated alias; PyMdpExecutor is canonical in recovery path
    execute_pymdp_simulation = execute_pymdp_simulation_from_gnn
    validate_pymdp_environment = lambda: {"valid": False, "issues": ["PyMDP validation not available"]}
    get_pymdp_health_status = lambda: {"status": "unknown"}
    check_system_resources = lambda: []
    check_dependencies = lambda: []
    check_file_permissions = lambda: []
    check_network_connectivity = lambda: {"status": "unknown"}
    run_simulation = lambda cfg: {"success": False, "error": "run_simulation not available"}
    VALIDATION_AVAILABLE = False
    ERROR_RECOVERY_AVAILABLE = False

# Import processor functions
from .processor import (
    process_execute,
)

# Provide execute_script_safely export for tests (validate_execution_environment already imported above)
try:
    from .executor import execute_script_safely
except Exception:
    def execute_script_safely(*args, **kwargs):
        return {"success": False, "error": "execute_script_safely not available"}

# Ensure execute_simulation_from_gnn is exported from top-level module
try:
    # If the processor defines a wrapper, import it; otherwise prefer executor convenience function
    from .processor import execute_simulation_from_gnn as _proc_execute_fn
    execute_simulation_from_gnn = _proc_execute_fn
except Exception:
    try:
        from .executor import execute_gnn_model as execute_simulation_from_gnn
    except Exception:
        execute_simulation_from_gnn = lambda *a, **k: {"success": False, "error": "execute_simulation_from_gnn not available"}

# Add to __all__ for proper exports
__all__ = [
    '__version__',
    'FEATURES',
    'VALIDATION_AVAILABLE',
    'ERROR_RECOVERY_AVAILABLE',
    'FrameworkName',

    # Core classes
    'ExecutionEngine',
    'GNNExecutor',
    'PyMdpExecutor',
    'PyMDPSimulation',

    # Execution functions
    'process_execute',
    'execute_simulation_from_gnn',
    'run_simulation',
    'execute_script_safely',

    # PyMDP functions
    'execute_pymdp_simulation_from_gnn',
    'execute_pymdp_simulation',
    'validate_pymdp_environment',
    'get_pymdp_health_status',

    # Validation functions
    'validate_execution_environment',
    'log_validation_results',
    'check_python_environment',
    'check_system_resources',
    'check_dependencies',
    'check_file_permissions',
    'check_network_connectivity',
]
