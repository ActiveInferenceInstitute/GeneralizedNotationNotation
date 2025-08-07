#!/usr/bin/env python3
"""
PyMDP Execution Module

This module provides PyMDP simulation execution capabilities for the GNN pipeline.
It includes utilities for running PyMDP simulations configured from GNN specifications
with enhanced safety patterns and comprehensive error handling.
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .pymdp_simulation import PyMDPSimulation
from .pymdp_utils import (
    convert_numpy_for_json,
    safe_json_dump,
    safe_pickle_dump,
    clean_trace_for_serialization,
    save_simulation_results,
    generate_simulation_summary,
    create_output_directory_with_timestamp,
    format_duration
)
from .pymdp_visualizer import PyMDPVisualizer, create_visualizer, save_all_visualizations

# Import execution functions
from .executor import (
    execute_pymdp_simulation_from_gnn,
    execute_pymdp_simulation,
    execute_pymdp_scripts
)

# Import validation functions
from .validator import (
    validate_pymdp_environment,
    get_pymdp_health_status
)

# Import context functions
from .context import (
    create_enhanced_pymdp_context
)

__all__ = [
    # Core classes
    'PyMDPSimulation',
    'PyMDPVisualizer',
    
    # Execution functions
    'execute_pymdp_simulation_from_gnn',
    'execute_pymdp_simulation',
    'execute_pymdp_scripts',
    
    # Validation functions
    'validate_pymdp_environment',
    'get_pymdp_health_status',
    
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