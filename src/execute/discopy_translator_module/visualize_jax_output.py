#!/usr/bin/env python3
"""
JAX Output Visualization for DisCoPy Diagrams

This module provides visualization capabilities for JAX evaluation results 
of DisCoPy diagrams, with graceful degradation when dependencies are unavailable.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    logger.debug("Matplotlib available for JAX output visualization")
except ImportError as e:
    logger.debug(f"Matplotlib not available: {e}")
    MATPLOTLIB_AVAILABLE = False
    matplotlib = plt = None

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    logger.debug("JAX available for output processing")
except ImportError as e:
    logger.debug(f"JAX not available: {e}")
    JAX_AVAILABLE = False
    jax = jnp = None


def plot_tensor_output(
    tensor_data: Union[np.ndarray, Any], 
    output_path: Union[str, Path],
    title: str = "JAX Tensor Output",
    **kwargs
) -> Tuple[bool, str]:
    """
    Plot JAX tensor output with appropriate visualization based on tensor shape.
    
    Args:
        tensor_data: JAX tensor or numpy array to visualize
        output_path: Path to save the visualization
        title: Title for the plot
        **kwargs: Additional plotting parameters
        
    Returns:
        Tuple of (success, message)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - cannot plot tensor output")
        return False, "Matplotlib library not installed"
    
    try:
        # Convert JAX array to numpy if needed
        if JAX_AVAILABLE and hasattr(tensor_data, 'block_until_ready'):
            # JAX array
            numpy_data = np.array(tensor_data)
        elif isinstance(tensor_data, np.ndarray):
            numpy_data = tensor_data
        else:
            # Try to convert to numpy
            numpy_data = np.array(tensor_data)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine plot type based on tensor shape
        shape = numpy_data.shape
        logger.info(f"Plotting tensor with shape {shape}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(shape) == 1:
            # 1D tensor - line plot
            ax.plot(numpy_data)
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title(f"{title} (1D: {shape[0]} elements)")
            
        elif len(shape) == 2:
            # 2D tensor - heatmap
            im = ax.imshow(numpy_data, cmap='viridis', aspect='auto')
            ax.set_xlabel('Column Index') 
            ax.set_ylabel('Row Index')
            ax.set_title(f"{title} (2D: {shape[0]}x{shape[1]})")
            plt.colorbar(im, ax=ax)
            
        elif len(shape) >= 3:
            # 3D+ tensor - show first 2D slice
            slice_data = numpy_data[0] if shape[0] > 0 else numpy_data.reshape(shape[1], shape[2])
            im = ax.imshow(slice_data, cmap='viridis', aspect='auto')
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index') 
            ax.set_title(f"{title} (3D+ slice: {slice_data.shape}, original: {shape})")
            plt.colorbar(im, ax=ax)
            
        else:
            # Scalar - text display
            ax.text(0.5, 0.5, f"Scalar value: {numpy_data}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{title} (Scalar)")
        
        plt.tight_layout()
        
        # Save with error handling
        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved tensor visualization to {output_file}")
        except Exception as save_error:
            logger.warning(f"Error saving with standard settings: {save_error}")
            # Fallback save
            plt.savefig(output_file, dpi=100)
            logger.info(f"Saved with fallback settings to {output_file}")
        
        plt.close()
        
        return True, f"Successfully plotted tensor with shape {shape}"
        
    except Exception as e:
        logger.error(f"Error plotting tensor output: {e}")
        return False, f"Error plotting tensor: {str(e)}"


def plot_multiple_tensor_outputs(
    tensor_dict: Dict[str, Union[np.ndarray, Any]],
    output_dir: Union[str, Path],
    prefix: str = "tensor_output"
) -> Tuple[bool, str, List[str]]:
    """
    Plot multiple tensor outputs to separate files.
    
    Args:
        tensor_dict: Dictionary of tensor_name -> tensor_data
        output_dir: Directory to save visualizations
        prefix: Prefix for output filenames
        
    Returns:
        Tuple of (success, message, list_of_saved_files)
    """
    if not MATPLOTLIB_AVAILABLE:
        return False, "Matplotlib not available", []
    
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    failed_plots = []
    
    for tensor_name, tensor_data in tensor_dict.items():
        try:
            # Sanitize filename
            safe_name = "".join(c for c in tensor_name if c.isalnum() or c in ('_', '-'))
            output_path = output_directory / f"{prefix}_{safe_name}.png"
            
            success, message = plot_tensor_output(
                tensor_data, 
                output_path, 
                title=f"JAX Output: {tensor_name}"
            )
            
            if success:
                saved_files.append(str(output_path))
                logger.info(f"Successfully plotted {tensor_name}")
            else:
                failed_plots.append(tensor_name)
                logger.error(f"Failed to plot {tensor_name}: {message}")
                
        except Exception as e:
            failed_plots.append(tensor_name)
            logger.error(f"Error processing tensor {tensor_name}: {e}")
    
    success_count = len(saved_files)
    total_count = len(tensor_dict)
    
    if success_count == total_count:
        return True, f"Successfully plotted all {total_count} tensors", saved_files
    elif success_count > 0:
        return True, f"Plotted {success_count}/{total_count} tensors (failed: {failed_plots})", saved_files
    else:
        return False, f"Failed to plot any tensors: {failed_plots}", []


def create_summary_visualization(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path]
) -> Tuple[bool, str]:
    """
    Create a summary visualization of JAX evaluation results.
    
    Args:
        evaluation_results: Dictionary containing evaluation results and metadata
        output_path: Path to save the summary visualization
        
    Returns:
        Tuple of (success, message)
    """
    if not MATPLOTLIB_AVAILABLE:
        return False, "Matplotlib not available for summary visualization"
    
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create text summary
        summary_text = []
        summary_text.append("JAX DisCoPy Evaluation Summary")
        summary_text.append("=" * 40)
        
        if 'status' in evaluation_results:
            summary_text.append(f"Status: {evaluation_results['status']}")
        
        if 'backend' in evaluation_results:
            summary_text.append(f"Backend: {evaluation_results['backend']}")
        
        if 'diagram_type' in evaluation_results:
            summary_text.append(f"Diagram Type: {evaluation_results['diagram_type']}")
        
        if 'timestamp' in evaluation_results:
            summary_text.append(f"Evaluation Time: {evaluation_results['timestamp']}")
        
        # Add tensor information if available
        if 'tensors' in evaluation_results:
            tensors = evaluation_results['tensors']
            summary_text.append(f"\nTensor Results: {len(tensors)} tensors")
            for i, (name, info) in enumerate(list(tensors.items())[:5]):  # Show first 5
                if isinstance(info, dict) and 'shape' in info:
                    summary_text.append(f"  {i+1}. {name}: shape {info['shape']}")
                else:
                    summary_text.append(f"  {i+1}. {name}: {type(info).__name__}")
            
            if len(tensors) > 5:
                summary_text.append(f"  ... and {len(tensors) - 5} more")
        
        # Display as text
        ax.text(0.05, 0.95, '\n'.join(summary_text), 
               ha='left', va='top', transform=ax.transAxes, 
               fontfamily='monospace', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("DisCoPy JAX Evaluation Summary", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created evaluation summary visualization at {output_file}")
        return True, "Summary visualization created successfully"
        
    except Exception as e:
        logger.error(f"Error creating summary visualization: {e}")
        return False, f"Error creating summary: {str(e)}"
