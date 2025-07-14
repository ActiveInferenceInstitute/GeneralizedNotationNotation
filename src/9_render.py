#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 9: Render

This script renders GNN specifications to target simulation environments.

Usage:
    python 9_render.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
from typing import Dict
import argparse

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

from gnn.parsers.markdown_parser import MarkdownGNNParser 

# Initialize logger for this step
logger = setup_step_logging("9_render", verbose=False)

# Import rendering functionality
try:
    from render.render import render_gnn_spec
    from render.pymdp.pymdp_renderer import render_gnn_to_pymdp
    from render.rxinfer import render_gnn_to_rxinfer_toml
    logger.debug("Successfully imported rendering modules")
    RENDER_AVAILABLE = True
except ImportError as e:
    log_step_error(logger, f"Could not import rendering modules: {e}")
    render_gnn_spec = None
    render_gnn_to_pymdp = None
    render_gnn_to_rxinfer_toml = None
    RENDER_AVAILABLE = False

def render_gnn_files(target_dir: Path, output_dir: Path, recursive: bool = False):
    """Render GNN files to simulation environments."""
    log_step_start(logger, "Rendering GNN files to simulation environments")
    
    # Use centralized output directory configuration
    render_output_dir = get_output_dir_for_script("9_render.py", output_dir)
    render_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir} using pattern '{pattern}'")
        return False

    logger.info(f"Found {len(gnn_files)} GNN files to render")
    
    successful_renders = 0
    failed_renders = 0
    
    # Define rendering targets with appropriate options
    render_targets = [
        ("pymdp", "pymdp"),
        ("rxinfer_toml", "rxinfer"),
        ("discopy_combined", "discopy"),
        ("activeinference_combined", "activeinference_jl"),
        ("jax_pomdp", "jax"),
        ("jax", "jax")
    ]
    
    parser = MarkdownGNNParser()
    
    try:
        with performance_tracker.track_operation("render_all_gnn_files"):
            for gnn_file in gnn_files:
                try:
                    # Parse the GNN file
                    parse_result = parser.parse_file(str(gnn_file))
                    if not parse_result.success:
                        log_step_warning(logger, f"Parsing failed for {gnn_file.name}: {parse_result.errors}")
                        failed_renders += len(render_targets)
                        continue
                    
                    model = parse_result.model
                    
                    # Convert model to dictionary for renderers
                    gnn_spec = {
                        "name": model.model_name,
                        "annotation": model.annotation,
                        "variables": [vars(v) for v in model.variables],
                        "connections": [vars(c) for c in model.connections],
                        "parameters": [vars(p) for p in model.parameters],
                        "equations": model.equations,
                        "time": vars(model.time_specification),
                        "ontology": [vars(m) for m in model.ontology_mappings],
                        "model_parameters": model.extensions.get('model_parameters', {}),
                        "source_file": str(gnn_file)
                    }
                    
                    # Extract InitialParameterization as a dictionary for matrix access
                    initial_params = {}
                    param_lines = [p for p in model.parameters if hasattr(p, 'name') and hasattr(p, 'value')]
                    matrix_keys = ["A", "B", "C", "D", "E"]
                    i = 0
                    while i < len(param_lines):
                        param = param_lines[i]
                        name = param.name.strip()
                        value = param.value
                        if name in matrix_keys and isinstance(value, str) and (value.strip().startswith("{") or value.strip().startswith("(")):
                            block = value.strip()
                            open_brace = block[0]
                            close_brace = '}' if open_brace == '{' else ')'
                            block_lines = []
                            # Always include the first line if it starts with '('
                            if block.startswith('('):
                                block_lines.append(block)
                            i += 1
                            while i < len(param_lines):
                                next_value = param_lines[i].value
                                if isinstance(next_value, str):
                                    next_value_str = next_value.strip()
                                    if next_value_str.startswith('('):
                                        block_lines.append(next_value_str)
                                    if next_value_str.endswith(close_brace):
                                        break
                                i += 1
                            # Wrap with braces/parens
                            full_block = open_brace + '\n' + '\n'.join(block_lines) + '\n' + close_brace
                            print(f"DEBUG: InitialParameterization {name} value =\n{full_block}")
                            initial_params[name] = full_block
                            i += 1
                        elif name in matrix_keys and name not in initial_params:
                            initial_params[name] = value
                            i += 1
                        else:
                            i += 1
                    gnn_spec["InitialParameterization"] = initial_params
                    
                    for target_format, output_subdir in render_targets:
                        try:
                            sub_output_dir = render_output_dir / output_subdir
                            sub_output_dir.mkdir(exist_ok=True)
                            
                            # Use model name for filename
                            base_name = model.model_name.lower().replace(" ", "_")
                            if target_format.endswith("_jl") or target_format == "activeinference_combined":
                                ext = ".jl"
                            elif target_format == "rxinfer_toml":
                                ext = ".toml"
                            else:
                                ext = ".py"
                            
                            output_file = sub_output_dir / f"{base_name}_{target_format}{ext}"
                            
                            with performance_tracker.track_operation(f"render_{target_format}_{gnn_file.name}"):
                                success, message, artifacts = render_gnn_spec(
                                    gnn_spec, 
                                    target_format,
                                    output_file.parent,  # Pass directory
                                    {"output_filename": base_name}  # Pass base name
                                )
                                
                            if success:
                                logger.info(f"{target_format} render successful for {gnn_file.name}: {message}")
                                successful_renders += 1
                            else:
                                logger.warning(f"{target_format} render failed for {gnn_file.name}: {message}")
                                failed_renders += 1
                                
                        except Exception as e:
                            log_step_warning(logger, f"{target_format} rendering failed for {gnn_file.name}: {e}")
                            failed_renders += 1
                        
                except Exception as e:
                    log_step_error(logger, f"Failed to process {gnn_file.name}: {e}")
                    failed_renders += len(render_targets)
        
        # Log results summary
        total_attempts = successful_renders + failed_renders
        success_rate = successful_renders / total_attempts * 100 if total_attempts > 0 else 0.0
        log_step_success(logger, f"Rendering completed. Success rate: {success_rate:.1f}% ({successful_renders}/{total_attempts})")
        return failed_renders == 0
        
    except Exception as e:
        log_step_error(logger, f"Rendering failed: {e}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for rendering operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("9_render.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Code generation for simulation environments')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get input and output directories
    input_dir = getattr(parsed_args, 'target_dir', None)
    if input_dir is None:
        input_dir = Path("input/gnn_files")
    elif isinstance(input_dir, str):
        input_dir = Path(input_dir)
        
    output_dir = Path(getattr(parsed_args, 'output_dir', 'output'))
    recursive = getattr(parsed_args, 'recursive', True)
    
    # Validate input directory
    if input_dir is None or not input_dir.exists():
        log_step_error(logger, f"Input directory does not exist: {input_dir}")
        return 1
    
    # Render GNN files
    success = render_gnn_files(
        target_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive
    )
    
    if success:
        log_step_success(logger, "Rendering completed successfully")
        return 0
    else:
        log_step_error(logger, "Rendering failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("9_render")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Code generation for simulation environments")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 