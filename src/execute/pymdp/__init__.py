#!/usr/bin/env python3
"""
PyMDP Execution Module

This module provides PyMDP simulation execution capabilities for the GNN pipeline.
It includes utilities for running PyMDP simulations configured from GNN specifications
with enhanced safety patterns and comprehensive error handling.
"""

import logging

from .context import create_enhanced_pymdp_context
from .executor import execute_pymdp_simulation, execute_pymdp_simulation_from_gnn
from .package_detector import (
    attempt_pymdp_auto_install,
    detect_pymdp_installation,
    get_pymdp_installation_instructions,
    is_correct_pymdp_package,
    validate_pymdp_for_execution,
)
from .pymdp_simulation import PyMDPSimulation, create_demo_pymdp_simulation
from .pymdp_utils import (
    clean_trace_for_serialization,
    convert_numpy_for_json,
    create_output_directory_with_timestamp,
    format_duration,
    generate_simulation_summary,
    safe_json_dump,
    safe_pickle_dump,
    save_simulation_results,
)
from .simulation import run_pymdp_simulation
from .validator import get_pymdp_health_status, validate_pymdp_environment

__all__ = [
    # Core classes
    "PyMDPSimulation",
    "create_demo_pymdp_simulation",
    # Execution functions
    "run_pymdp_simulation",
    "execute_pymdp_simulation_from_gnn",
    "execute_pymdp_simulation",
    # Validation functions
    "validate_pymdp_environment",
    "get_pymdp_health_status",
    # Package detection functions
    "detect_pymdp_installation",
    "is_correct_pymdp_package",
    "get_pymdp_installation_instructions",
    "attempt_pymdp_auto_install",
    "validate_pymdp_for_execution",
    # Context functions
    "create_enhanced_pymdp_context",
    # Utility functions
    "convert_numpy_for_json",
    "safe_json_dump",
    "safe_pickle_dump",
    "clean_trace_for_serialization",
    "save_simulation_results",
    "generate_simulation_summary",
    "create_output_directory_with_timestamp",
    "format_duration",
]

logger = logging.getLogger(__name__)
