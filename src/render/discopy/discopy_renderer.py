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


def render_gnn_to_discopy(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a DisCoPy categorical diagram.
    
    Args:
        gnn_spec: The GNN specification as a Python dictionary
        output_path: Path where the diagram image will be saved
        options: Rendering options (verbose, etc.)
        
    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    options = options or {}
    verbose = options.get("verbose", False)
    
    if not DISCOPY_AVAILABLE:
        error_msg = "DisCoPy translator module not available. Cannot render DisCoPy diagram."
        logger.error(error_msg)
        return False, error_msg, []
    
    if not gnn_file_to_discopy_diagram:
        error_msg = "DisCoPy diagram translator not available."
        logger.error(error_msg)
        return False, error_msg, []
    
    try:
        # Create a temporary GNN file since the translator expects a file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            import json
            json.dump(gnn_spec, temp_file, indent=2)
            temp_file_path = Path(temp_file.name)
        
        try:
            logger.info(f"Generating DisCoPy diagram for model: {gnn_spec.get('name', 'unknown')}")
            diagram = gnn_file_to_discopy_diagram(temp_file_path, verbose=verbose)
            
            if diagram is None:
                warning_msg = "No DisCoPy diagram could be generated from the GNN specification"
                logger.warning(warning_msg)
                return False, warning_msg, []
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save diagram visualization
            logger.info(f"Saving DisCoPy diagram to: {output_path}")
            diagram.draw(path=str(output_path), show_types=True, figsize=(10, 6))
            
            success_msg = f"Successfully rendered DisCoPy diagram to {output_path.name}"
            logger.info(success_msg)
            return True, success_msg, [str(output_path.resolve())]
            
        finally:
            # Clean up temporary file
            temp_file_path.unlink(missing_ok=True)
            
    except ImportError as e:
        if "matplotlib" in str(e).lower():
            error_msg = f"Matplotlib not available for diagram visualization: {e}"
            logger.error(error_msg)
            return False, error_msg, []
        else:
            error_msg = f"Missing import for DisCoPy rendering: {e}"
            logger.error(error_msg)
            return False, error_msg, []
    except Exception as e:
        error_msg = f"Failed to render DisCoPy diagram: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, []


def render_gnn_to_discopy_jax(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a DisCoPy matrix diagram and evaluate it using JAX.
    
    Args:
        gnn_spec: The GNN specification as a Python dictionary
        output_path: Path where the evaluation results will be saved
        options: Rendering options (verbose, jax_seed, etc.)
        
    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    options = options or {}
    verbose = options.get("verbose", False)
    jax_seed = options.get("jax_seed", 0)
    
    if not DISCOPY_AVAILABLE:
        error_msg = "DisCoPy translator module not available. Cannot render DisCoPy JAX diagram."
        logger.error(error_msg)
        return False, error_msg, []
    
    if not JAX_FULLY_OPERATIONAL:
        error_msg = "JAX is not fully operational. Cannot perform JAX evaluation."
        logger.error(error_msg)
        return False, error_msg, []
    
    if not gnn_file_to_discopy_matrix_diagram:
        error_msg = "DisCoPy matrix diagram translator not available."
        logger.error(error_msg)
        return False, error_msg, []
    
    try:
        # Create a temporary GNN file since the translator expects a file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            import json
            json.dump(gnn_spec, temp_file, indent=2)
            temp_file_path = Path(temp_file.name)
        
        try:
            logger.info(f"Generating DisCoPy matrix diagram for JAX evaluation: {gnn_spec.get('name', 'unknown')}")
            matrix_diagram = gnn_file_to_discopy_matrix_diagram(temp_file_path, verbose=verbose, jax_seed=jax_seed)
            
            if matrix_diagram is None:
                warning_msg = "No DisCoPy matrix diagram could be generated from the GNN specification"
                logger.warning(warning_msg)
                return False, warning_msg, []
            
            # Set JAX random seed if provided
            if jax_seed is not None:
                try:
                    import jax
                    key = jax.random.PRNGKey(jax_seed)
                    logger.debug(f"Set JAX PRNG seed to {jax_seed}")
                except ImportError:
                    logger.warning("JAX not available for seed setting")
            
            # Evaluate the matrix diagram
            logger.info("Evaluating DisCoPy matrix diagram using JAX backend")
            evaluation_result = matrix_diagram.eval()
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization if matplotlib is available
            artifacts = []
            if plot_tensor_output and MATPLOTLIB_AVAILABLE:
                logger.info(f"Saving JAX evaluation visualization to: {output_path}")
                plot_tensor_output(evaluation_result, str(output_path))
                artifacts.append(str(output_path.resolve()))
            
            success_msg = f"Successfully evaluated DisCoPy matrix diagram with JAX"
            if artifacts:
                success_msg += f" and saved visualization to {output_path.name}"
            
            logger.info(success_msg)
            return True, success_msg, artifacts
            
        finally:
            # Clean up temporary file
            temp_file_path.unlink(missing_ok=True)
            
    except Exception as e:
        error_msg = f"Failed to render and evaluate DisCoPy matrix diagram: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, []


def render_gnn_to_discopy_combined(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to both DisCoPy diagram and JAX evaluation.
    
    Args:
        gnn_spec: The GNN specification as a Python dictionary
        output_dir: Directory where outputs will be saved
        options: Rendering options (verbose, jax_seed, etc.)
        
    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    options = options or {}
    model_name = gnn_spec.get("name", "gnn_model")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_artifacts = []
    all_messages = []
    overall_success = True
    
    # Render DisCoPy diagram
    diagram_path = output_dir / f"{model_name}_diagram.png"
    success_diagram, msg_diagram, artifacts_diagram = render_gnn_to_discopy(
        gnn_spec, diagram_path, options
    )
    
    if success_diagram:
        all_artifacts.extend(artifacts_diagram)
        all_messages.append(f"DisCoPy diagram: {msg_diagram}")
    else:
        overall_success = False
        all_messages.append(f"DisCoPy diagram failed: {msg_diagram}")
    
    # Render JAX evaluation
    jax_path = output_dir / f"{model_name}_jax_evaluation.png"
    success_jax, msg_jax, artifacts_jax = render_gnn_to_discopy_jax(
        gnn_spec, jax_path, options
    )
    
    if success_jax:
        all_artifacts.extend(artifacts_jax)
        all_messages.append(f"JAX evaluation: {msg_jax}")
    else:
        all_messages.append(f"JAX evaluation failed: {msg_jax}")
        # Don't fail overall if just JAX fails, since DisCoPy might still work
    
    combined_message = "; ".join(all_messages)
    
    if overall_success or all_artifacts:
        return True, combined_message, all_artifacts
    else:
        return False, combined_message, [] 