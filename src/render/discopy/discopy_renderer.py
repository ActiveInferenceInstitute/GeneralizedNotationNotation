"""
DisCoPy Renderer Module for GNN Specifications

This module provides rendering capabilities for GNN specifications to DisCoPy
categorical diagrams and JAX-evaluatable matrix diagrams.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Import DisCoPy functionality
try:
    from ...discopy_translator_module.translator import (
        gnn_file_to_discopy_diagram,
        gnn_file_to_discopy_matrix_diagram,
        JAX_FULLY_OPERATIONAL,
        MATPLOTLIB_AVAILABLE
    )
    from ...discopy_translator_module.visualize_jax_output import plot_tensor_output
    DISCOPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DisCoPy translator module not available: {e}")
    gnn_file_to_discopy_diagram = None
    gnn_file_to_discopy_matrix_diagram = None
    plot_tensor_output = None
    JAX_FULLY_OPERATIONAL = False
    MATPLOTLIB_AVAILABLE = False
    DISCOPY_AVAILABLE = False

# Assume translator is available
from .translator import gnn_spec_to_discopy_code, gnn_spec_to_discopy_jax_code

def render_gnn_to_discopy(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Generate a Python script that creates and draws a DisCoPy categorical diagram from GNN spec.
    """
    options = options or {}
    model_name = gnn_spec.get("name", "gnn_model")
    
    output_path = output_path.with_suffix('.py')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        code = gnn_spec_to_discopy_code(gnn_spec)  # Function to generate code string from spec
        with open(output_path, 'w') as f:
            f.write(code)
        success_msg = f"Successfully generated DisCoPy script to {output_path}"
        return True, success_msg, [str(output_path)]
    except Exception as e:
        error_msg = f"Failed to generate DisCoPy script: {e}"
        return False, error_msg, []

def render_gnn_to_discopy_jax(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Generate a Python script that creates and evaluates a DisCoPy matrix diagram with JAX.
    """
    options = options or {}
    model_name = gnn_spec.get("name", "gnn_model")
    
    output_path = output_path.with_suffix('.py')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        code = gnn_spec_to_discopy_jax_code(gnn_spec, options.get('jax_seed', 0))
        with open(output_path, 'w') as f:
            f.write(code)
        success_msg = f"Successfully generated DisCoPy JAX script to {output_path}"
        return True, success_msg, [str(output_path)]
    except Exception as e:
        error_msg = f"Failed to generate DisCoPy JAX script: {e}"
        return False, error_msg, []

def render_gnn_to_discopy_combined(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Generate a combined Python script that does both diagram creation and JAX evaluation.
    """
    options = options or {}
    model_name = gnn_spec.get("name", "gnn_model")
    
    output_path = output_dir / f"{model_name}_discopy_combined.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        code_diag = gnn_spec_to_discopy_code(gnn_spec)
        code_jax = gnn_spec_to_discopy_jax_code(gnn_spec, options.get('jax_seed', 0))
        combined_code = f"# Combined DisCoPy Script\n\n{code_diag}\n\n# JAX Evaluation Part\n{code_jax}"
        with open(output_path, 'w') as f:
            f.write(combined_code)
        success_msg = f"Successfully generated combined DisCoPy script to {output_path}"
        return True, success_msg, [str(output_path)]
    except Exception as e:
        error_msg = f"Failed to generate combined DisCoPy script: {e}"
        return False, error_msg, []

# Add helper functions if not in translator:
def gnn_spec_to_discopy_code(gnn_spec: Dict) -> str:
    # Logic to generate code string from spec
    # Extract variables, connections, etc.
    # Generate Ty, Box, Diagram code
    return """import discopy\n# Code to define and draw diagram"""

def gnn_spec_to_discopy_jax_code(gnn_spec: Dict, seed: int) -> str:
    # Similar, but with JAX evaluation
    return """import jax\n# Code to define matrix diagram and evaluate""" 