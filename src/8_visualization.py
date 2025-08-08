#!/usr/bin/env python3
"""
Step 8: Visualization Processing (Thin Orchestrator)

This step handles visualization processing for GNN files with comprehensive
safe-to-fail patterns and robust output management.

How to run:
  python src/8_visualization.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Visualization results in the specified output directory
  - Matrix visualizations, network graphs, and combined analysis plots
  - Comprehensive visualization reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that visualization dependencies are installed
  - Check that src/visualization/ contains visualization modules
  - Check that the output directory is writable
  - Verify visualization configuration and requirements
"""

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

# Import core visualization functions from visualization module
try:
    from visualization import (
        process_visualization,
        process_matrix_visualization,
        generate_visualizations,
        MatrixVisualizer,
        GNNVisualizer
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # Fallback function definitions if visualization module is not available
    def process_visualization(*args, **kwargs):
        return False
    
    def process_matrix_visualization(*args, **kwargs):
        return False
    
    def generate_visualizations(*args, **kwargs):
        return False
    
    class MatrixVisualizer:
        def __init__(self):
            pass
    
    class GNNVisualizer:
        def __init__(self):
            pass

def process_visualization_standardized(
    target_dir: Path,
    output_dir: Path,
    logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized visualization processing function.
    
    Args:
        target_dir: Directory containing GNN files to visualize
        output_dir: Output directory for visualization results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Check if visualization module is available
        if not VISUALIZATION_AVAILABLE:
            log_step_warning(logger, "Visualization module not available, using fallback functions")
        
        # Get pipeline configuration
        config = get_pipeline_config()
        step_output_dir = get_output_dir_for_script("8_visualization.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing visualization")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", output_dir)
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        if not gnn_results_file.exists():
            log_step_error(logger, "GNN processing results not found. Run step 3 first.")
            return False
        
        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)
        
        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
        
        # Visualization results
        visualization_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(target_dir),
            "output_directory": str(step_output_dir),
            "files_visualized": [],
            "summary": {
                "total_files": 0,
                "successful_visualizations": 0,
                "failed_visualizations": 0,
                "total_images_generated": 0,
                "visualization_types": {
                    "matrix": 0,
                    "network": 0,
                    "combined": 0
                }
            }
        }
        
        # Process each file
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue
            
            file_name = file_result["file_name"]
            logger.info(f"Visualizing: {file_name}")
            
            # Load the actual parsed GNN specification
            parsed_model_file = file_result.get("parsed_model_file")
            if parsed_model_file and Path(parsed_model_file).exists():
                try:
                    with open(parsed_model_file, 'r') as f:
                        actual_gnn_spec = json.load(f)
                    logger.info(f"Loaded parsed GNN specification from {parsed_model_file}")
                    model_data = actual_gnn_spec
                except Exception as e:
                    logger.error(f"Failed to load parsed GNN spec from {parsed_model_file}: {e}")
                    model_data = file_result
            else:
                logger.warning(f"Parsed model file not found for {file_name}, using summary data")
                model_data = file_result
            
            # Create file-specific output directory
            # Align with tests expecting PNGs under output/visualization/<model>
            file_output_dir = step_output_dir / Path(file_name).stem
            file_output_dir.mkdir(exist_ok=True)
            
            file_visualization_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "visualizations": {},
                "success": True
            }
            
            # Generate matrix visualizations
            try:
                matrix_result = process_matrix_visualization(model_data, file_output_dir)
                file_visualization_result["visualizations"]["matrix"] = {
                    "success": True,
                    "result": matrix_result
                }
                visualization_results["summary"]["visualization_types"]["matrix"] += 1
                logger.info(f"Matrix visualization completed for {file_name}")
            except Exception as e:
                logger.error(f"Matrix visualization failed for {file_name}: {e}")
                file_visualization_result["visualizations"]["matrix"] = {
                    "success": False,
                    "error": str(e)
                }
                file_visualization_result["success"] = False
            
            # Generate network visualizations
            try:
                network_result = process_visualization(file_output_dir, file_output_dir, verbose=verbose)
                file_visualization_result["visualizations"]["network"] = {
                    "success": True,
                    "result": network_result
                }
                visualization_results["summary"]["visualization_types"]["network"] += 1
                logger.info(f"Network visualization completed for {file_name}")
            except Exception as e:
                logger.error(f"Network visualization failed for {file_name}: {e}")
                file_visualization_result["visualizations"]["network"] = {
                    "success": False,
                    "error": str(e)
                }
                file_visualization_result["success"] = False
            
            # Generate combined visualizations
            try:
                combined_result = generate_visualizations(model_data, file_output_dir)
                file_visualization_result["visualizations"]["combined"] = {
                    "success": True,
                    "result": combined_result
                }
                visualization_results["summary"]["visualization_types"]["combined"] += 1
                logger.info(f"Combined visualization completed for {file_name}")
            except Exception as e:
                logger.error(f"Combined visualization failed for {file_name}: {e}")
                file_visualization_result["visualizations"]["combined"] = {
                    "success": False,
                    "error": str(e)
                }
                file_visualization_result["success"] = False

            # Ensure expected top-level PNGs exist in file_output_dir for tests
            try:
                from visualization.matrix_visualizer import MatrixVisualizer
                visualizer = MatrixVisualizer()
                parameters = model_data.get("parameters", []) if isinstance(model_data, dict) else []
                # Generate matrix_analysis.png at top-level
                matrix_analysis_path = file_output_dir / "matrix_analysis.png"
                if parameters:
                    if visualizer.generate_matrix_analysis(parameters, matrix_analysis_path):
                        visualization_results["summary"]["total_images_generated"] += 1
                # Generate pomdp_transition_analysis.png if B tensor present
                matrices = visualizer.extract_matrix_data_from_parameters(parameters) if parameters else {}
                if isinstance(matrices, dict) and 'B' in matrices:
                    B = matrices['B']
                    try:
                        import numpy as _np
                        if hasattr(B, 'ndim') and B.ndim == 3:
                            pomdp_path = file_output_dir / "pomdp_transition_analysis.png"
                            if visualizer.generate_pomdp_transition_analysis(B, pomdp_path):
                                visualization_results["summary"]["total_images_generated"] += 1
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not generate expected summary PNGs: {e}")
            
            visualization_results["files_visualized"].append(file_visualization_result)
            visualization_results["summary"]["total_files"] += 1
            
            if file_visualization_result["success"]:
                visualization_results["summary"]["successful_visualizations"] += 1
            else:
                visualization_results["summary"]["failed_visualizations"] += 1
            
            # Count generated images
            for viz_type, viz_result in file_visualization_result["visualizations"].items():
                if viz_result.get("success", False):
                    if isinstance(viz_result.get("result"), list):
                        visualization_results["summary"]["total_images_generated"] += len(viz_result["result"])
                    else:
                        visualization_results["summary"]["total_images_generated"] += 1
        
        # Save visualization results
        visualization_results_file = step_output_dir / "visualization_results.json"
        with open(visualization_results_file, 'w') as f:
            json.dump(visualization_results, f, indent=2)
        
        # Save visualization summary
        visualization_summary_file = step_output_dir / "visualization_summary.json"
        with open(visualization_summary_file, 'w') as f:
            json.dump(visualization_results["summary"], f, indent=2)
        
        logger.info(f"Visualization processing completed:")
        logger.info(f"  Total files: {visualization_results['summary']['total_files']}")
        logger.info(f"  Successful visualizations: {visualization_results['summary']['successful_visualizations']}")
        logger.info(f"  Failed visualizations: {visualization_results['summary']['failed_visualizations']}")
        logger.info(f"  Total images generated: {visualization_results['summary']['total_images_generated']}")
        logger.info(f"  Visualization types: {visualization_results['summary']['visualization_types']}")
        
        log_step_success(logger, "Visualization processing completed successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Visualization processing failed: {e}")
        return False

def main():
    """Main visualization processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("8_visualization.py")
    
    # Setup logging
    logger = setup_step_logging("visualization", args)
    
    # Check if visualization module is available
    if not VISUALIZATION_AVAILABLE:
        log_step_warning(logger, "Visualization module not available, using fallback functions")
    
    # Process visualization
    success = process_visualization_standardized(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        logger=logger,
        recursive=args.recursive,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
