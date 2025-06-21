#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 6: Visualization

This script generates visualizations of GNN models and their components.

Usage:
    python 6_visualization.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
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

# Initialize logger for this step
logger = setup_step_logging("6_visualization", verbose=False)

# Import visualization functionality
try:
    from visualization import GNNVisualizer
    from visualization.matrix_visualizer import MatrixVisualizer
    from visualization.ontology_visualizer import OntologyVisualizer
    logger.debug("Successfully imported visualization modules")
    visualizer = GNNVisualizer  # Create alias for backwards compatibility
except ImportError as e:
    log_step_error(logger, f"Could not import visualization modules: {e}")
    GNNVisualizer = None
    visualizer = None
    MatrixVisualizer = None
    OntologyVisualizer = None

def generate_visualizations(target_dir: Path, output_dir: Path, recursive: bool = False):
    """Generate visualizations for GNN models."""
    log_step_start(logger, f"Generating visualizations for GNN files in: {target_dir}")
    
    if not visualizer:
        log_step_error(logger, "Visualization modules not available")
        return False
    
    # Use centralized output directory configuration
    viz_output_dir = get_output_dir_for_script("6_visualization.py", output_dir)
    
    try:
        # Create GNN visualizer instance
        gnn_visualizer = GNNVisualizer(output_dir=str(viz_output_dir))
        
        # Initialize results dictionary
        results = {'success': False, 'files_processed': 0}
        
        # Use performance tracking for visualization generation
        with performance_tracker.track_operation("generate_all_visualizations"):
            # Find GNN files
            if recursive:
                gnn_files = list(target_dir.rglob("*.md"))
            else:
                gnn_files = list(target_dir.glob("*.md"))
            
            log_step_success(logger, f"Found {len(gnn_files)} GNN files to visualize")
            
            # Process each file
            processed_count = 0
            for gnn_file in gnn_files:
                try:
                    output_path = gnn_visualizer.visualize_file(str(gnn_file))
                    log_step_success(logger, f"Generated visualization for {gnn_file.name}: {output_path}")
                    processed_count += 1
                except Exception as e:
                    log_step_warning(logger, f"Failed to visualize {gnn_file.name}: {e}")
            
            results['files_processed'] = processed_count
            results['success'] = processed_count > 0
        
        # Generate matrix visualizations if available
        if MatrixVisualizer:
            try:
                with performance_tracker.track_operation("generate_matrix_visualizations"):
                    matrix_viz = MatrixVisualizer()
                    matrix_results = matrix_viz.visualize_directory(
                        target_dir=target_dir,
                        output_dir=viz_output_dir / "matrices"
                    )
                    results.update(matrix_results)
                log_step_success(logger, "Matrix visualizations completed")
            except Exception as e:
                log_step_warning(logger, f"Matrix visualization failed: {e}")
        
        # Generate ontology visualizations if available
        if OntologyVisualizer:
            try:
                with performance_tracker.track_operation("generate_ontology_visualizations"):
                    ontology_viz = OntologyVisualizer()
                    ontology_results = ontology_viz.visualize_directory(
                        target_dir=target_dir,
                        output_dir=viz_output_dir / "ontology"
                    )
                    results.update(ontology_results)
                log_step_success(logger, "Ontology visualizations completed")
            except Exception as e:
                log_step_warning(logger, f"Ontology visualization failed: {e}")
        
        # Log results summary
        if results.get('success', False):
            log_step_success(logger, f"Visualization generation completed successfully. Files processed: {results.get('files_processed', 0)}")
        else:
            log_step_warning(logger, f"Visualization generation completed with issues. Files processed: {results.get('files_processed', 0)}")
        
        return results.get('success', False)
        
    except Exception as e:
        log_step_error(logger, f"Visualization generation failed: {e}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for visualization generation."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("6_visualization.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Graph visualization generation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Generate visualizations
    success = generate_visualizations(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        recursive=getattr(parsed_args, 'recursive', False)
    )
    
    if success:
        log_step_success(logger, "Visualization generation completed successfully")
        return 0
    else:
        log_step_error(logger, "Visualization generation failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("6_visualization")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Graph visualization generation")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 