"""
Pipeline step for rendering GNN specifications.

This script calls the main rendering logic defined in the src/render/render.py module.
It handles rendering to various formats including PyMDP and RxInfer.jl.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Ensure src directory is in path to allow sibling imports
current_script_path = Path(__file__).resolve()
src_dir = current_script_path.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import rendering functions directly
from render.pymdp_renderer import render_gnn_to_pymdp
from render.rxinfer import render_gnn_to_rxinfer_toml

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace) -> int:
    """
    Main function for the GNN rendering pipeline step.
    """
    base_output_dir = Path(args.output_dir)
    gnn_export_dir = base_output_dir / "gnn_exports"
    render_output_dir = base_output_dir / "gnn_rendered_simulators"
    
    if not gnn_export_dir.is_dir():
        logger.warning(f"GNN export directory not found: {gnn_export_dir}. Skipping render step.")
        return 0

    gnn_files = list(gnn_export_dir.rglob("*.json")) if args.recursive else list(gnn_export_dir.glob("*.json"))

    if not gnn_files:
        logger.warning(f"No GNN JSON files found in {gnn_export_dir}. Nothing to render.")
        return 0

    logger.info(f"Found {len(gnn_files)} GNN JSON files to render.")
    
    overall_success = True
    for gnn_file_path in gnn_files:
        logger.info(f"Processing {gnn_file_path.name}")
        try:
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                gnn_spec: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read or parse GNN spec from {gnn_file_path}: {e}")
            overall_success = False
            continue

        model_name_base = gnn_file_path.stem.replace(".gnn", "")

        # Render to PyMDP
        pymdp_output_dir = render_output_dir / "pymdp"
        pymdp_output_dir.mkdir(parents=True, exist_ok=True)
        pymdp_output_path = pymdp_output_dir / f"{model_name_base}_pymdp.py"
        
        success, msg, _ = render_gnn_to_pymdp(gnn_spec, pymdp_output_path)
        if success:
            logger.info(f"Successfully rendered to PyMDP: {pymdp_output_path}")
        else:
            logger.error(f"Failed to render to PyMDP for {gnn_file_path.name}: {msg}")
            overall_success = False

        # Render to RxInfer TOML
        rxinfer_output_dir = render_output_dir / "rxinfer_toml"
        rxinfer_output_dir.mkdir(parents=True, exist_ok=True)
        rxinfer_output_path = rxinfer_output_dir / f"{model_name_base}_config.toml"
        
        success, msg, _ = render_gnn_to_rxinfer_toml(gnn_spec, rxinfer_output_path)
        if success:
            logger.info(f"Successfully rendered to RxInfer TOML: {rxinfer_output_path}")
        else:
            logger.error(f"Failed to render to RxInfer TOML for {gnn_file_path.name}: {msg}")
            overall_success = False

    return 0 if overall_success else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render GNN specifications from JSON to target formats.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Main pipeline output directory.")
    parser.add_argument("--recursive", action="store_true", help="Search for GNN JSON files recursively.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sys.exit(main(args)) 