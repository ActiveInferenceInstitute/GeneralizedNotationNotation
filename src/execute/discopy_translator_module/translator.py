#!/usr/bin/env python3
"""
DisCoPy Translator for GNN Processing Pipeline

This module provides translation from GNN specifications to DisCoPy categorical diagrams
with optional JAX evaluation capabilities. Provides graceful degradation when 
dependencies are not available.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import discopy
    from discopy import Diagram, Word, Cap, Cup
    from discopy.quantum import Ket, Bra, X, Z, H
    DISCOPY_AVAILABLE = True
    logger.debug("DisCoPy library available")
except ImportError as e:
    logger.debug(f"DisCoPy not available: {e}")
    DISCOPY_AVAILABLE = False
    # Stub classes for graceful degradation
    Diagram = Word = Cap = Cup = None
    Ket = Bra = X = Z = H = None

try:
    import jax
    import jax.numpy as jnp
    JAX_FULLY_OPERATIONAL = True
    logger.debug("JAX library available")
except ImportError as e:
    logger.debug(f"JAX not available: {e}")
    JAX_FULLY_OPERATIONAL = False
    jax = jnp = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    logger.debug("Matplotlib library available")
except ImportError as e:
    logger.debug(f"Matplotlib not available: {e}")
    MATPLOTLIB_AVAILABLE = False
    matplotlib = plt = None


def gnn_file_to_discopy_diagram(
    gnn_data: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[bool, str, Optional[Any]]:
    """
    Convert GNN specification to DisCoPy diagram.
    
    Args:
        gnn_data: Parsed GNN data dictionary
        output_path: Optional path to save diagram visualization
        
    Returns:
        Tuple of (success, message, diagram_object)
    """
    if not DISCOPY_AVAILABLE:
        logger.warning("DisCoPy not available - cannot create diagrams")
        return False, "DisCoPy library not installed", None
    
    try:
        # Extract variables and connections from GNN data
        variables = gnn_data.get('Variables', {})
        edges = gnn_data.get('Edges', [])
        
        if not variables and not edges:
            return False, "No variables or edges found in GNN data", None
        
        logger.info(f"Creating DisCoPy diagram from {len(variables)} variables and {len(edges)} edges")
        
        # Create a simple categorical diagram representation
        # This is a basic implementation - can be enhanced based on specific DisCoPy patterns needed
        
        # For now, create a simple string diagram representation
        diagram_parts = []
        
        # Add variables as objects
        for var_name, var_info in variables.items():
            dimensions = var_info.get('dimensions', [])
            if dimensions:
                dim_str = 'x'.join(map(str, dimensions))
                diagram_parts.append(f"{var_name}[{dim_str}]")
            else:
                diagram_parts.append(var_name)
        
        # Add edges as morphisms
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            if source and target:
                diagram_parts.append(f"{source} -> {target}")
        
        # Create a basic diagram description
        diagram_description = " ; ".join(diagram_parts)
        
        # If output path specified, save a text representation
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(f"DisCoPy Diagram Representation:\n")
                f.write(f"Variables: {len(variables)}\n")
                f.write(f"Edges: {len(edges)}\n\n")
                f.write(f"Diagram: {diagram_description}\n")
        
        logger.info("Successfully created DisCoPy diagram representation")
        return True, f"Diagram created with {len(variables)} variables and {len(edges)} edges", diagram_description
        
    except Exception as e:
        logger.error(f"Error creating DisCoPy diagram: {e}")
        return False, f"Error creating diagram: {str(e)}", None


def gnn_file_to_discopy_matrix_diagram(
    gnn_data: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[bool, str, Optional[Any]]:
    """
    Convert GNN specification to DisCoPy matrix diagram representation.
    
    Args:
        gnn_data: Parsed GNN data dictionary  
        output_path: Optional path to save matrix diagram visualization
        
    Returns:
        Tuple of (success, message, matrix_diagram_object)
    """
    if not DISCOPY_AVAILABLE:
        logger.warning("DisCoPy not available - cannot create matrix diagrams")
        return False, "DisCoPy library not installed", None
    
    try:
        # Extract matrix-like structures from GNN data
        variables = gnn_data.get('Variables', {})
        
        if not variables:
            return False, "No variables found for matrix diagram", None
        
        # Create matrix representation from variable dimensions
        matrix_info = {}
        for var_name, var_info in variables.items():
            dimensions = var_info.get('dimensions', [])
            if len(dimensions) >= 2:  # Matrix-like structure
                matrix_info[var_name] = {
                    'rows': dimensions[0],
                    'cols': dimensions[1] if len(dimensions) > 1 else dimensions[0],
                    'total_elements': dimensions[0] * (dimensions[1] if len(dimensions) > 1 else dimensions[0])
                }
        
        if not matrix_info:
            return False, "No matrix-like variables found", None
        
        logger.info(f"Creating matrix diagram from {len(matrix_info)} matrix variables")
        
        # Create matrix diagram representation
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write("DisCoPy Matrix Diagram Representation:\n\n")
                for var_name, info in matrix_info.items():
                    f.write(f"{var_name}: {info['rows']}x{info['cols']} matrix "
                           f"({info['total_elements']} elements)\n")
        
        logger.info("Successfully created DisCoPy matrix diagram representation")
        return True, f"Matrix diagram created with {len(matrix_info)} matrices", matrix_info
        
    except Exception as e:
        logger.error(f"Error creating DisCoPy matrix diagram: {e}")
        return False, f"Error creating matrix diagram: {str(e)}", None


def evaluate_diagram_with_jax(
    diagram: Any, 
    input_data: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, Optional[Any]]:
    """
    Evaluate DisCoPy diagram using JAX backend.
    
    Args:
        diagram: DisCoPy diagram object
        input_data: Optional input data for evaluation
        
    Returns:
        Tuple of (success, message, evaluation_result)
    """
    if not JAX_FULLY_OPERATIONAL:
        logger.warning("JAX not available - cannot evaluate diagram")
        return False, "JAX library not installed", None
    
    if not DISCOPY_AVAILABLE:
        logger.warning("DisCoPy not available - cannot evaluate diagram")  
        return False, "DisCoPy library not installed", None
    
    try:
        # Real JAX evaluation
        logger.info("Evaluating diagram with JAX backend")
        
        if hasattr(diagram, 'eval'):
            # Evaluate using DisCoPy's JAX backend
            # We assume the diagram is already constructed with JAX-compatible types or free tensors
            result_tensor = diagram.eval()
            
            # Convert to standard python types for JSON serialization
            if hasattr(result_tensor, 'tolist'):
                result_data = result_tensor.tolist()
            else:
                result_data = str(result_tensor)
                
            evaluation_result = {
                'status': 'evaluated',
                'backend': 'jax',
                'diagram_type': str(type(diagram)),
                'timestamp': datetime.now().isoformat(),
                'result': result_data,
                'shape': str(result_tensor.shape) if hasattr(result_tensor, 'shape') else 'scalar'
            }
        else:
            raise ValueError(f"Diagram object {type(diagram)} does not support .eval()")
        
        logger.info("Successfully evaluated diagram with JAX")
        return True, "Diagram evaluated successfully", evaluation_result
        
    except Exception as e:
        logger.error(f"Error evaluating diagram with JAX: {e}")
        return False, f"Error evaluating diagram: {str(e)}", None
