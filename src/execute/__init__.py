"""
execute module for GNN Processing Pipeline.

This module provides execute capabilities with comprehensive safety patterns,
validation, monitoring, and error recovery systems.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

# Import logging helpers with fallback to keep tests import-safe
try:
    from utils.pipeline_template import (
        log_step_start,
        log_step_success,
        log_step_error,
        log_step_warning
    )
except Exception:
    def log_step_start(logger, msg): logger.info(f"🚀 {msg}")
    def log_step_success(logger, msg): logger.info(f"✅ {msg}")
    def log_step_error(logger, msg): logger.error(f"❌ {msg}")
    def log_step_warning(logger, msg): logger.warning(f"⚠️ {msg}")

# Import execute functionality 
try:
    from .executor import ExecutionEngine, execute_simulation_from_gnn
    from .pymdp import execute_pymdp_simulation_from_gnn_impl, PyMdpExecutor
    from .validator import validate_execution_environment_impl
    VALIDATION_AVAILABLE = True
    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    # Import fallback functions
    from .fallback import (
        ExecutionEngine,
        PyMdpExecutor,
        execute_simulation_from_gnn,
        execute_pymdp_simulation_from_gnn_impl,
        validate_execution_environment_impl,
        validate_execution_environment,
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
    VALIDATION_AVAILABLE = False
    ERROR_RECOVERY_AVAILABLE = False

# Import processor functions
from .processor import (
    process_execute,
)

# Provide execute_script_safely and validate_execution_environment exports for tests
try:
    from .executor import execute_script_safely, validate_execution_environment
except Exception:
    def execute_script_safely(*args, **kwargs):
        return {"success": False, "error": "execute_script_safely not available"}

    def validate_execution_environment(*args, **kwargs):
        return {"valid": False, "issues": ["validate_execution_environment not available"]}

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
    # Core classes
    'ExecutionEngine', 
    'PyMdpExecutor',
    
    # Processor functions
    'process_execute',
    'execute_simulation_from_gnn',
    
    # Validation functions
    'validate_execution_environment',
    'get_execution_health_status',
    'log_validation_results',
    'check_python_environment',
    'check_system_dependencies',
    'check_python_packages',
    'check_file_system',
    
    # Error recovery functions
    'analyze_pipeline_error',
    'generate_error_recovery_report',
    'get_quick_error_suggestions',
    'ErrorRecoverySystem'
]
