"""
execute module for GNN Processing Pipeline.

This module provides execute capabilities with comprehensive safety patterns,
validation, monitoring, and error recovery systems.
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
    "error_recovery": True,
    "mcp_integration": True
}

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal

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
    from .executor import (
        ExecutionEngine,
        GNNExecutor,
        execute_gnn_model,
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
        ErrorRecoverySystem,
        ExecutionEngine,
        PyMdpExecutor,
        analyze_pipeline_error,
        check_file_system,
        check_python_environment,
        check_python_packages,
        check_system_dependencies,
        execute_pymdp_simulation_from_gnn,
        execute_simulation_from_gnn,
        generate_error_recovery_report,
        get_execution_health_status,
        get_quick_error_suggestions,
        log_validation_results,
        validate_execution_environment,
    )
    # Phase 2.1: structured stubs. The pre-Phase-2.1 stubs returned bare
    # ``{}`` / ``[]`` / ``{"valid": False, ...}``, making it impossible for
    # callers to distinguish "validator not available" from "validator ran
    # and found no issues". The shape below gives every stub the same
    # envelope so callers can branch on ``result.get("available")`` rather
    # than empty-collection heuristics.
    execute_gnn_model = execute_simulation_from_gnn
    GNNExecutor = ExecutionEngine
    PyMDPSimulation = PyMdpExecutor  # deprecated alias; PyMdpExecutor is canonical in recovery path
    execute_pymdp_simulation = execute_pymdp_simulation_from_gnn

    def _unavailable(name: str, reason: str = "dependency missing") -> dict:
        """Structured 'not available' envelope used by every stub below.

        Shape:
            available: False (constant — signals "stub engaged")
            reason:    human-readable cause
            name:      which capability was requested
            data:      None (so callers expecting data can detect absence)
        """
        return {"available": False, "reason": reason, "name": name, "data": None}

    def validate_pymdp_environment():
        return {**_unavailable("validate_pymdp_environment", "PyMDP module not importable"),
                "valid": False, "issues": ["PyMDP validation not available"]}

    def get_pymdp_health_status():
        return {**_unavailable("get_pymdp_health_status"), "status": "unknown"}

    def check_system_resources():
        return _unavailable("check_system_resources", "resource probe unavailable")

    def check_dependencies():
        return _unavailable("check_dependencies", "dependency probe unavailable")

    def check_file_permissions():
        return _unavailable("check_file_permissions", "permission probe unavailable")

    def check_network_connectivity():
        return {**_unavailable("check_network_connectivity"), "status": "unknown"}

    def run_simulation(cfg):
        return {"success": False, "available": False,
                "reason": "run_simulation not available (recovery path engaged)",
                "error_type": "NotAvailable", "error": "run_simulation not available"}
    VALIDATION_AVAILABLE = False
    ERROR_RECOVERY_AVAILABLE = False

# Import processor functions
from .processor import (
    process_execute,
)

# Public helper for safe subprocess execution of rendered scripts.
from .executor import execute_script_safely

# Ensure execute_simulation_from_gnn is exported from top-level module
try:
    # If the processor defines a wrapper, import it; otherwise prefer executor convenience function
    from .processor import execute_simulation_from_gnn as _proc_execute_fn
    execute_simulation_from_gnn = _proc_execute_fn
except Exception:
    try:
        from .executor import execute_gnn_model as execute_simulation_from_gnn
    except Exception:
        def execute_simulation_from_gnn(*a, **k):
            # Phase 2.1: structured not-available envelope. Callers branching
            # on error_type="NotAvailable" can distinguish this from real
            # runtime failures, which carry error_type set to the exception
            # class name.
            return {
                "success": False,
                "available": False,
                "reason": "no executor implementations importable",
                "error_type": "NotAvailable",
                "error": "execute_simulation_from_gnn not available",
            }

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
    'PyMDPSimulation',
    # Note: PyMdpExecutor kept as private backward-compat alias, not in __all__

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


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "execute",
        "version": __version__,
        "description": "Multi-framework execution of rendered simulation scripts",
        "features": FEATURES,
    }
