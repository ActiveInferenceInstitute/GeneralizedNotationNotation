#!/usr/bin/env python3
"""
Render processor module for GNN code generation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime

def process_render(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process GNN files with code generation for various simulation environments.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    try:
        # Create results directory
        results_dir = output_dir / "render_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "pymdp_generated": [],
            "rxinfer_generated": [],
            "activeinference_jl_generated": [],
            "discopy_generated": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)
            
            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Generate code for different environments
                    pymdp_code = generate_pymdp_code({"model_name": gnn_file.stem})
                    if pymdp_code:
                        results["pymdp_generated"].append({
                            "file": str(gnn_file),
                            "code": pymdp_code
                        })
                    
                    rxinfer_code = generate_rxinfer_code({"model_name": gnn_file.stem})
                    if rxinfer_code:
                        results["rxinfer_generated"].append({
                            "file": str(gnn_file),
                            "code": rxinfer_code
                        })
                    
                    activeinference_jl_code = generate_activeinference_jl_code({"model_name": gnn_file.stem})
                    if activeinference_jl_code:
                        results["activeinference_jl_generated"].append({
                            "file": str(gnn_file),
                            "code": activeinference_jl_code
                        })
                    
                    discopy_code = generate_discopy_code({"model_name": gnn_file.stem})
                    if discopy_code:
                        results["discopy_generated"].append({
                            "file": str(gnn_file),
                            "code": discopy_code
                        })
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
        
        # Save results
        results_file = results_dir / "render_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        return results["success"]
        
    except Exception as e:
        return False

def render_gnn_spec(
    gnn_spec: Dict[str, Any],
    target: str,
    output_directory: Union[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a target language.
    
    Args:
        gnn_spec: GNN specification dictionary
        target: Target language/environment
        output_directory: Output directory for generated code
        options: Additional options
        
    Returns:
        Tuple of (success, message, warnings)
    """
    try:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if target.lower() == "pymdp":
            code = generate_pymdp_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_pymdp.py"
        elif target.lower() == "rxinfer":
            code = generate_rxinfer_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_rxinfer.jl"
        elif target.lower() == "activeinference_jl":
            code = generate_activeinference_jl_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_activeinference.jl"
        elif target.lower() == "discopy":
            code = generate_discopy_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_discopy.py"
        else:
            return False, f"Unsupported target: {target}", []
        
        if code:
            with open(output_file, 'w') as f:
                f.write(code)
            return True, f"Successfully generated {target} code", []
        else:
            return False, f"Failed to generate {target} code", []
            
    except Exception as e:
        return False, f"Error rendering {target}: {e}", []

def get_module_info() -> Dict[str, Any]:
    """Get information about the render module."""
    return {
        "name": "Render Module",
        "version": "1.0.0",
        "description": "Code generation for GNN specifications",
        "supported_targets": ["pymdp", "rxinfer", "activeinference_jl", "discopy"],
        "available_targets": ["pymdp", "rxinfer", "activeinference_jl", "discopy"],
        "features": [
            "PyMDP code generation",
            "RxInfer.jl code generation", 
            "ActiveInference.jl code generation",
            "DisCoPy categorical diagram generation"
        ],
        "supported_formats": ["python", "julia", "python_script"]
    }

def get_available_renderers() -> Dict[str, Dict[str, Any]]:
    """Get information about available renderers."""
    return {
        "pymdp": {
            "name": "PyMDP",
            "description": "Python Markov Decision Process library",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["POMDP", "MDP", "Belief State Updates"],
            "function": "render_gnn_to_pymdp",
            "output_format": "python"
        },
        "rxinfer": {
            "name": "RxInfer.jl",
            "description": "Julia reactive message passing inference engine",
            "language": "Julia",
            "file_extension": ".jl",
            "supported_features": ["Message Passing", "Probabilistic Programming"],
            "function": "render_gnn_to_rxinfer",
            "output_format": "julia"
        },
        "activeinference_jl": {
            "name": "ActiveInference.jl",
            "description": "Julia Active Inference library",
            "language": "Julia", 
            "file_extension": ".jl",
            "supported_features": ["Free Energy Minimization", "Active Inference"],
            "function": "render_gnn_to_activeinference_jl",
            "output_format": "julia"
        },
        "discopy": {
            "name": "DisCoPy",
            "description": "Python library for computing with string diagrams",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["Categorical Diagrams", "String Diagrams"],
            "function": "render_gnn_to_discopy",
            "output_format": "python"
        }
    }
