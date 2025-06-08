"""
Pipeline step for rendering GNN specifications.

This script calls the main rendering logic defined in the src/render/render.py module.
It handles rendering to various formats including PyMDP and RxInfer.jl.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    UTILS_AVAILABLE
)

# Initialize logger for this step
logger = setup_step_logging("9_render", verbose=False)

# Ensure src directory is in path to allow sibling imports
current_script_path = Path(__file__).resolve()
src_dir = current_script_path.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import rendering functions from the render package
try:
    from render import render_gnn_to_pymdp, render_gnn_to_rxinfer_toml
except ImportError as e:
    log_step_error(logger, f"Failed to import render functions: {e}")
    render_gnn_to_pymdp = None
    render_gnn_to_rxinfer_toml = None

def main(args: argparse.Namespace) -> int:
    """Main function for the GNN rendering pipeline step."""
    
    # Update logger verbosity based on args
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    log_step_start(logger, "Starting Step 9: GNN Rendering")
    
    # Validate render functions are available
    if not render_gnn_to_pymdp or not render_gnn_to_rxinfer_toml:
        log_step_error(logger, "Render functions not available. Cannot proceed.")
        return 1
    
    base_output_dir = Path(args.output_dir)
    gnn_export_dir = base_output_dir / "gnn_exports"
    render_output_dir = base_output_dir / "gnn_rendered_simulators"
    
    # Create render output directory
    if not validate_output_directory(base_output_dir, "gnn_rendered_simulators"):
        log_step_error(logger, "Failed to create render output directory")
        return 1
    
    if not gnn_export_dir.is_dir():
        log_step_warning(logger, f"GNN export directory not found: {gnn_export_dir}. Skipping render step.")
        return 0

    # Search for GNN JSON files
    gnn_files = list(gnn_export_dir.rglob("*.json"))

    if not gnn_files:
        log_step_warning(logger, f"No GNN JSON files found in {gnn_export_dir}. Nothing to render.")
        return 0

    logger.info(f"Found {len(gnn_files)} GNN JSON files to render")
    
    # Track rendering results
    render_summary = {
        "total_files": len(gnn_files),
        "pymdp_successes": 0,
        "rxinfer_successes": 0,
        "failures": [],
        "output_directory": str(render_output_dir)
    }
    
    overall_success = True
    
    for gnn_file_path in gnn_files:
        logger.info(f"Processing {gnn_file_path.name}")
        try:
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                gnn_spec: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            error_msg = f"Failed to read GNN spec from {gnn_file_path}: {e}"
            log_step_error(logger, error_msg)
            render_summary["failures"].append({"file": str(gnn_file_path), "error": error_msg})
            overall_success = False
            continue

        model_name_base = gnn_file_path.stem.replace(".gnn", "")

        # Render to PyMDP
        pymdp_output_dir = render_output_dir / "pymdp"
        pymdp_output_dir.mkdir(parents=True, exist_ok=True)
        pymdp_output_path = pymdp_output_dir / f"{model_name_base}_pymdp.py"
        
        try:
            success, msg, _ = render_gnn_to_pymdp(gnn_spec, pymdp_output_path)
            if success:
                logger.info(f"Successfully rendered to PyMDP: {pymdp_output_path}")
                render_summary["pymdp_successes"] += 1
            else:
                error_msg = f"PyMDP render failed for {gnn_file_path.name}: {msg}"
                log_step_error(logger, error_msg)
                render_summary["failures"].append({"file": str(gnn_file_path), "target": "PyMDP", "error": error_msg})
                overall_success = False
        except Exception as e:
            error_msg = f"PyMDP render exception for {gnn_file_path.name}: {e}"
            log_step_error(logger, error_msg)
            render_summary["failures"].append({"file": str(gnn_file_path), "target": "PyMDP", "error": error_msg})
            overall_success = False

        # Render to RxInfer TOML
        rxinfer_output_dir = render_output_dir / "rxinfer_toml"
        rxinfer_output_dir.mkdir(parents=True, exist_ok=True)
        rxinfer_output_path = rxinfer_output_dir / f"{model_name_base}_config.toml"
        
        try:
            success, msg, _ = render_gnn_to_rxinfer_toml(gnn_spec, rxinfer_output_path)
            if success:
                logger.info(f"Successfully rendered to RxInfer TOML: {rxinfer_output_path}")
                render_summary["rxinfer_successes"] += 1
            else:
                error_msg = f"RxInfer render failed for {gnn_file_path.name}: {msg}"
                log_step_error(logger, error_msg)
                render_summary["failures"].append({"file": str(gnn_file_path), "target": "RxInfer", "error": error_msg})
                overall_success = False
        except Exception as e:
            error_msg = f"RxInfer render exception for {gnn_file_path.name}: {e}"
            log_step_error(logger, error_msg)
            render_summary["failures"].append({"file": str(gnn_file_path), "target": "RxInfer", "error": error_msg})
            overall_success = False

    # Save render summary
    summary_file = render_output_dir / "render_summary.json"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(render_summary, f, indent=2)
        logger.info(f"Render summary saved to: {summary_file}")
    except Exception as e:
        log_step_warning(logger, f"Failed to save render summary: {e}")

    if overall_success:
        log_step_success(logger, f"GNN rendering completed successfully - PyMDP: {render_summary['pymdp_successes']}, RxInfer: {render_summary['rxinfer_successes']}")
    else:
        log_step_warning(logger, f"GNN rendering completed with {len(render_summary['failures'])} failures")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render GNN specifications from JSON to target formats")
    
    # Define defaults for standalone execution
    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent
    default_output_dir = project_root / "output"
    
    parser.add_argument("--output-dir", type=Path, default=default_output_dir,
                       help="Main pipeline output directory")
    parser.add_argument("--recursive", action="store_true", 
                       help="Search for GNN JSON files recursively")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()

    # Update logger for standalone execution
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    sys.exit(main(args)) 