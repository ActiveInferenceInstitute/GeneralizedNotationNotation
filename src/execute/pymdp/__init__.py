#!/usr/bin/env python3
"""
PyMDP Execution Module

This module provides PyMDP simulation execution capabilities for the GNN pipeline.
It includes utilities for running PyMDP simulations configured from GNN specifications
with enhanced safety patterns and comprehensive error handling.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pymdp_simulation import PyMDPSimulation
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

# Soft import: analysis (step 16) is downstream of execute (step 12); wrap to avoid
# hard coupling across pipeline stages.
try:
    from analysis.pymdp.visualizer import (
        PyMDPVisualizer,
        create_visualizer,
        save_all_visualizations,
    )
except ImportError:
    PyMDPVisualizer = None  # type: ignore[assignment,misc]

    def create_visualizer(*args, **kwargs):  # type: ignore[misc]
        return None

    def save_all_visualizations(*args, **kwargs):  # type: ignore[misc]
        return []

# Import execution functions
# Import context functions
from .context import create_enhanced_pymdp_context
from .executor import execute_pymdp_simulation, execute_pymdp_simulation_from_gnn

# Import package detection functions
from .package_detector import (
    attempt_pymdp_auto_install,
    detect_pymdp_installation,
    get_pymdp_installation_instructions,
    is_correct_pymdp_package,
    validate_pymdp_for_execution,
)

# Import validation functions
from .validator import get_pymdp_health_status, validate_pymdp_environment

__all__ = [
    # Core classes
    'PyMDPSimulation',
    'PyMDPVisualizer',

    # Execution functions
    'execute_pymdp_simulation_from_gnn',
    'execute_pymdp_simulation',

    # Validation functions
    'validate_pymdp_environment',
    'get_pymdp_health_status',

    # Package detection functions
    'detect_pymdp_installation',
    'is_correct_pymdp_package',
    'get_pymdp_installation_instructions',
    'attempt_pymdp_auto_install',
    'validate_pymdp_for_execution',

    # Context functions
    'create_enhanced_pymdp_context',

    # Utility functions
    'convert_numpy_for_json',
    'safe_json_dump',
    'safe_pickle_dump',
    'clean_trace_for_serialization',
    'save_simulation_results',
    'generate_simulation_summary',
    'create_output_directory_with_timestamp',
    'format_duration',
    'create_visualizer',
    'save_all_visualizations'
]

logger = logging.getLogger(__name__)
