#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 9: Advanced Visualization

This step provides advanced visualization capabilities for GNN models,
including 3D visualizations, interactive dashboards, and exploration tools.

Usage:
    python 9_advanced_viz.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
import json
import datetime
from typing import Dict, Any, List, Optional, Union

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
    get_output_dir_for_script,
    get_pipeline_config
)

from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("9_advanced_viz", verbose=False)

# Import step-specific modules
try:
    from advanced_visualization.visualizer import AdvancedVisualizer
    from advanced_visualization.dashboard import DashboardGenerator
    
    DEPENDENCIES_AVAILABLE = True
    logger.debug("Successfully imported advanced visualization dependencies")
    
except ImportError as e:
    log_step_warning(logger, f"Failed to import advanced visualization dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

def process_advanced_viz_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized advanced visualization processing function.
    
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
        # Start performance tracking
        with performance_tracker.track_operation("advanced_viz_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Get configuration
            config = get_pipeline_config()
            step_config = config.get_step_config("9_advanced_viz.py")
            
            # Set up output directory
            step_output_dir = get_output_dir_for_script("9_advanced_viz.py", output_dir)
            step_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate step requirements
            if not DEPENDENCIES_AVAILABLE:
                log_step_warning(logger, "Advanced visualization dependencies are not available, functionality will be limited")
            
            # Log processing parameters
            logger.info(f"Processing GNN files from: {target_dir}")
            logger.info(f"Output directory: {step_output_dir}")
            logger.info(f"Recursive processing: {recursive}")
            
            # Extract additional parameters from kwargs
            viz_type = kwargs.get('viz_type', 'all')
            interactive = kwargs.get('interactive', True)
            export_formats = kwargs.get('export_formats', ['html', 'json'])
            
            logger.info(f"Visualization type: {viz_type}")
            logger.info(f"Interactive: {interactive}")
            logger.info(f"Export formats: {export_formats}")
            
            # Validate input directory
            if not target_dir.exists():
                log_step_error(logger, f"Input directory does not exist: {target_dir}")
                return False
            
            # Find GNN files
            pattern = "**/*.md" if recursive else "*.md"
            gnn_files = list(target_dir.glob(pattern))
            
            if not gnn_files:
                log_step_warning(logger, f"No GNN files found in {target_dir}")
                return True  # Not an error, just no files to process
            
            logger.info(f"Found {len(gnn_files)} GNN files to visualize")
            
            # Process files
            successful_files = 0
            failed_files = 0
            
            # Initialize visualizers
            visualizer = AdvancedVisualizer() if DEPENDENCIES_AVAILABLE else None
            dashboard_gen = DashboardGenerator() if DEPENDENCIES_AVAILABLE else None
            
            # Process each file
            visualization_results = []
            for gnn_file in gnn_files:
                try:
                    # Read file content
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_result = {
                        "file_path": str(gnn_file),
                        "file_name": gnn_file.name,
                        "visualization_type": viz_type,
                        "interactive": interactive,
                        "export_formats": export_formats,
                        "generated_files": [],
                        "status": "unknown"
                    }
                    
                    # Generate visualizations
                    if visualizer:
                        # Use actual visualizer implementation
                        viz_files = visualizer.generate_visualizations(
                            content, gnn_file.stem, step_output_dir, viz_type, interactive, export_formats
                        )
                        file_result["generated_files"] = viz_files
                    else:
                        # Fallback implementation using basic HTML generation
                        viz_files = generate_advanced_viz_fallback(content, gnn_file, step_output_dir, viz_type, export_formats)
                        file_result["generated_files"] = viz_files
                    
                    # Generate dashboard if requested
                    if dashboard_gen and interactive:
                        dashboard_file = dashboard_gen.generate_dashboard(
                            content, gnn_file.stem, step_output_dir
                        )
                        if dashboard_file:
                            file_result["generated_files"].append(str(dashboard_file))
                    
                    # Determine status
                    if file_result["generated_files"]:
                        file_result["status"] = "success"
                        successful_files += 1
                        logger.info(f"Generated {len(file_result['generated_files'])} visualizations for {gnn_file.name}")
                    else:
                        file_result["status"] = "failed"
                        failed_files += 1
                        logger.warning(f"No visualizations generated for {gnn_file.name}")
                    
                    visualization_results.append(file_result)
                    
                except Exception as e:
                    log_step_error(logger, f"Unexpected error visualizing {gnn_file}: {e}")
                    failed_files += 1
            
            # Generate summary report
            summary_file = step_output_dir / "advanced_visualization_summary.json"
            summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step_name": "advanced_visualization",
                "input_directory": str(target_dir),
                "output_directory": str(step_output_dir),
                "total_files": len(gnn_files),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "visualization_type": viz_type,
                "interactive": interactive,
                "export_formats": export_formats,
                "performance_metrics": performance_tracker.get_summary()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Generate detailed report
            detailed_file = step_output_dir / "advanced_visualization_detailed.json"
            with open(detailed_file, 'w') as f:
                json.dump(visualization_results, f, indent=2)
            
            logger.info(f"Summary report saved: {summary_file}")
            logger.info(f"Detailed report saved: {detailed_file}")
            
            # Determine success
            if failed_files == 0:
                log_step_success(logger, f"Successfully generated advanced visualizations for {successful_files} GNN models")
                return True
            elif successful_files > 0:
                log_step_warning(logger, f"Partially successful: {failed_files} files failed visualization")
                return True  # Still consider successful for pipeline continuation
            else:
                log_step_error(logger, "All files failed visualization")
                return False
            
    except Exception as e:
        log_step_error(logger, f"Advanced visualization processing failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def generate_advanced_viz_fallback(content: str, gnn_file: Path, output_dir: Path, viz_type: str, export_formats: List[str]) -> List[str]:
    """
    Fallback implementation for advanced visualization when visualizer module is not available.
    
    Args:
        content: GNN file content
        gnn_file: Path to GNN file
        output_dir: Output directory
        viz_type: Type of visualization
        export_formats: List of export formats
        
    Returns:
        List of generated file paths
    """
    generated_files = []
    
    try:
        # Create file-specific output directory
        file_output_dir = output_dir / gnn_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate simple HTML visualization
        if 'html' in export_formats:
            html_content = generate_html_visualization(content, gnn_file.stem)
            html_file = file_output_dir / f"{gnn_file.stem}_advanced_viz.html"
            with open(html_file, 'w') as f:
                f.write(html_content)
            generated_files.append(str(html_file))
        
        # Generate JSON data for interactive visualization
        json_data = extract_visualization_data(content)
        json_file = file_output_dir / f"{gnn_file.stem}_viz_data.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        generated_files.append(str(json_file))
        
        logger.debug(f"Generated fallback visualizations for {gnn_file.stem}")
        
    except Exception as e:
        logger.error(f"Failed to generate fallback visualizations for {gnn_file}: {e}")
    
    return generated_files

def generate_html_visualization(content: str, model_name: str) -> str:
    """
    Generate a simple HTML visualization for the GNN model (fallback).
    
    Args:
        content: GNN file content
        model_name: Name of the model
        
    Returns:
        HTML content
    """
    # Simple fallback HTML visualization
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Visualization - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: rgba(255, 255, 255, 0.95); padding: 30px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .content {{ text-align: center; padding: 50px; }}
        .message {{ font-size: 1.2em; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Advanced GNN Visualization</h1>
            <h2>Model: {model_name}</h2>
        </div>
        <div class="content">
            <div class="message">
                <p>Advanced visualization features are not available in fallback mode.</p>
                <p>Please ensure the advanced_visualization module is properly installed.</p>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    return html

def extract_visualization_data(content: str) -> Dict[str, Any]:
    """
    Extract basic visualization data from GNN content (fallback).
    
    Args:
        content: GNN file content
        
    Returns:
        Dictionary with basic visualization data
    """
    # Simple fallback data extraction
    return {
        "blocks": [],
        "connections": [],
        "total_blocks": 0,
        "total_connections": 0,
        "message": "Advanced visualization data extraction not available in fallback mode"
    }

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "9_advanced_viz.py",
    process_advanced_viz_standardized,
    "Advanced visualization and exploration",
    additional_arguments={
        "viz_type": {
            "type": str,
            "choices": ["all", "3d", "interactive", "dashboard"],
            "default": "all",
            "help": "Type of visualization to generate"
        },
        "interactive": {"type": bool, "default": True, "help": "Generate interactive visualizations"},
        "export_formats": {"type": str, "nargs": "+", "default": ["html", "json"], "help": "Export formats"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 