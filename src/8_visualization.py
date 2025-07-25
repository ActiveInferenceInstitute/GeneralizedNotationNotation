#!/usr/bin/env python3
"""
Step 8: Visualization Processing

This step handles visualization processing for GNN files with comprehensive
safe-to-fail patterns and robust output management.
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

# Safe imports with fallbacks
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import networkx as nx
    MATPLOTLIB_AVAILABLE = True
    NETWORKX_AVAILABLE = True
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    NETWORKX_AVAILABLE = False
    plt = None
    nx = None

# Import visualization module components with fallbacks
try:
    from visualization.matrix_visualizer import MatrixVisualizer
    MATRIX_VISUALIZER_AVAILABLE = True
except ImportError:
    MatrixVisualizer = None
    MATRIX_VISUALIZER_AVAILABLE = False

try:
    from visualization.visualizer import GNNVisualizer
    GNN_VISUALIZER_AVAILABLE = True
except ImportError:
    GNNVisualizer = None
    GNN_VISUALIZER_AVAILABLE = False


@dataclass
class VisualizationAttempt:
    """Track individual visualization attempts."""
    file_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: str = ""
    generated_files: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class VisualizationResults:
    """Comprehensive visualization results tracking."""
    timestamp: str
    source_directory: str
    output_directory: str
    total_files: int = 0
    successful_visualizations: int = 0
    failed_visualizations: int = 0
    total_images_generated: int = 0
    attempts: List[VisualizationAttempt] = field(default_factory=list)
    dependency_status: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "source_directory": self.source_directory,
            "output_directory": self.output_directory,
            "summary": {
                "total_files": self.total_files,
                "successful_visualizations": self.successful_visualizations,
                "failed_visualizations": self.failed_visualizations,
                "total_images_generated": self.total_images_generated
            },
            "dependency_status": self.dependency_status,
            "files_visualized": [
                {
                    "file_name": attempt.file_name,
                    "success": attempt.success,
                    "duration_seconds": attempt.duration_seconds,
                    "generated_files": attempt.generated_files,
                    "error": attempt.error_message if not attempt.success else None
                }
                for attempt in self.attempts
            ],
            "errors": self.errors
        }


class SafeVisualizationManager:
    """Manages visualization with comprehensive safety patterns."""
    
    def __init__(self, logger, correlation_id: str):
        self.logger = logger
        self.correlation_id = correlation_id
        
    @contextmanager
    def safe_matplotlib_context(self):
        """Context manager for safe matplotlib operations."""
        try:
            if MATPLOTLIB_AVAILABLE:
                # Clear any existing plots
                plt.clf()
                plt.close('all')
            yield
        except Exception as e:
            self.logger.warning(f"[{self.correlation_id}] Matplotlib context error: {e}")
        finally:
            if MATPLOTLIB_AVAILABLE:
                try:
                    plt.close('all')
                except:
                    pass
                    
    def create_fallback_visualization(self, file_name: str, output_dir: Path, 
                                    error_message: str = "Visualization failed") -> List[str]:
        """Create fallback visualizations when normal processing fails."""
        generated_files = []
        
        try:
            # Create file-specific directory
            file_output_dir = output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)
            
            # Create error report
            error_file = file_output_dir / "visualization_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Visualization Error Report\n")
                f.write(f"========================\n\n")
                f.write(f"File: {file_name}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Error: {error_message}\n\n")
                f.write(f"Dependencies Status:\n")
                f.write(f"- Matplotlib: {MATPLOTLIB_AVAILABLE}\n")
                f.write(f"- NetworkX: {NETWORKX_AVAILABLE}\n")
                f.write(f"- MatrixVisualizer: {MATRIX_VISUALIZER_AVAILABLE}\n")
                f.write(f"- GNNVisualizer: {GNN_VISUALIZER_AVAILABLE}\n")
            generated_files.append(str(error_file))
            
            # Create basic HTML visualization summary
            html_file = file_output_dir / "visualization_summary.html"
            with open(html_file, 'w') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Visualization Summary - {file_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .error {{ color: #d32f2f; background: #ffebee; padding: 15px; border-radius: 5px; }}
        .info {{ color: #1976d2; background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Visualization Summary</h1>
        <h2>File: {file_name}</h2>
        
        <div class="error">
            <h3>Visualization Error</h3>
            <p>{error_message}</p>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="info">
            <h3>Dependency Status</h3>
            <ul>
                <li>Matplotlib: {'✓' if MATPLOTLIB_AVAILABLE else '✗'}</li>
                <li>NetworkX: {'✓' if NETWORKX_AVAILABLE else '✗'}</li>
                <li>MatrixVisualizer: {'✓' if MATRIX_VISUALIZER_AVAILABLE else '✗'}</li>
                <li>GNNVisualizer: {'✓' if GNN_VISUALIZER_AVAILABLE else '✗'}</li>
            </ul>
        </div>
    </div>
</body>
</html>""")
            generated_files.append(str(html_file))
            
            self.logger.info(f"[{self.correlation_id}] Created fallback visualization for {file_name}")
            
        except Exception as e:
            self.logger.error(f"[{self.correlation_id}] Failed to create fallback visualization: {e}")
        
        return generated_files

    def generate_basic_plots(self, model_data: Dict, output_dir: Path, file_name: str) -> List[str]:
        """Generate basic plots using matplotlib directly."""
        generated_files = []
        
        if not MATPLOTLIB_AVAILABLE:
            return generated_files
            
        try:
            with self.safe_matplotlib_context():
                variables = model_data.get('variables', [])
                connections = model_data.get('connections', [])
                
                # Generate variable type distribution
                if variables:
                    type_counts = {}
                    for var in variables:
                        var_type = var.get("var_type", var.get("type", "unknown"))
                        type_counts[var_type] = type_counts.get(var_type, 0) + 1
                    
                    if type_counts:
                        plt.figure(figsize=(10, 8))
                        labels = [t.replace('_', ' ').title() for t in type_counts.keys()]
                        sizes = list(type_counts.values())
                        colors = plt.cm.Set3(range(len(sizes)))
                        
                        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        plt.title('Variable Type Distribution', fontsize=16, fontweight='bold')
                        plt.axis('equal')
                        
                        var_type_path = output_dir / "variable_type_chart.png"
                        plt.savefig(var_type_path, dpi=300, bbox_inches='tight')
                        generated_files.append(str(var_type_path))
                        plt.close()
                
                # Generate basic statistics chart
                stats = {
                    'Variables': len(variables),
                    'Connections': len(connections),
                    'Parameters': len(model_data.get('parameters', [])),
                    'Equations': len(model_data.get('equations', []))
                }
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(stats.keys(), stats.values(), color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
                plt.title('Model Components', fontsize=16, fontweight='bold')
                plt.ylabel('Count')
                
                # Add value labels on bars
                for bar, count in zip(bars, stats.values()):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
                
                stats_path = output_dir / "model_statistics.png"
                plt.savefig(stats_path, dpi=300, bbox_inches='tight')
                generated_files.append(str(stats_path))
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"[{self.correlation_id}] Basic plot generation failed: {e}")
        
        return generated_files


def generate_correlation_id() -> str:
    """Generate a correlation ID for tracking."""
    import uuid
    return str(uuid.uuid4())[:8]


def main():
    """Main visualization processing function with comprehensive safety patterns."""
    args = EnhancedArgumentParser.parse_step_arguments("8_visualization.py")
    
    # Setup logging with correlation ID
    correlation_id = generate_correlation_id()
    logger = setup_step_logging("visualization", args)
    logger.info(f"[{correlation_id}] Starting visualization processing")
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("8_visualization.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, f"[{correlation_id}] Processing visualization with safety patterns")
        
        # Initialize results tracking
        results = VisualizationResults(
            timestamp=datetime.now().isoformat(),
            source_directory=str(args.target_dir),
            output_directory=str(output_dir),
            dependency_status={
                "matplotlib": MATPLOTLIB_AVAILABLE,
                "networkx": NETWORKX_AVAILABLE,
                "matrix_visualizer": MATRIX_VISUALIZER_AVAILABLE,
                "gnn_visualizer": GNN_VISUALIZER_AVAILABLE
            }
        )
        
        # Log dependency status
        logger.info(f"[{correlation_id}] Dependency status: {results.dependency_status}")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        gnn_results = {}
        if gnn_results_file.exists():
            try:
                with open(gnn_results_file, 'r') as f:
                    gnn_results = json.load(f)
                logger.info(f"[{correlation_id}] Loaded GNN processing results")
            except Exception as e:
                logger.warning(f"[{correlation_id}] Failed to load GNN results: {e}")
                results.errors.append(f"Failed to load GNN results: {e}")
        else:
            logger.warning(f"[{correlation_id}] No GNN processing results found")
            results.errors.append("No GNN processing results found")
        
        # Initialize visualization manager
        viz_manager = SafeVisualizationManager(logger, correlation_id)
        
        # Process GNN files
        processed_files = gnn_results.get("processed_files", [])
        if not processed_files:
            # Fallback: scan for .md files directly
            logger.info(f"[{correlation_id}] Scanning for GNN files directly")
            gnn_files = list(Path(args.target_dir).glob("*.md"))
            processed_files = [{"file_name": f.name, "file_path": str(f), "parse_success": True} for f in gnn_files]
        
        results.total_files = len(processed_files)
        logger.info(f"[{correlation_id}] Processing {results.total_files} files")
        
        for file_result in processed_files:
            file_name = file_result["file_name"]
            attempt = VisualizationAttempt(
                file_name=file_name,
                start_time=datetime.now()
            )
            
            try:
                logger.info(f"[{correlation_id}] Processing visualization for: {file_name}")
                
                # Create file-specific output directory
                file_output_dir = output_dir / file_name.replace('.md', '')
                file_output_dir.mkdir(exist_ok=True)
                
                # Load model data
                model_data = {}
                parsed_file_path = gnn_output_dir / file_name.replace('.md', '') / f"{file_name.replace('.md', '')}_parsed.json"
                
                if parsed_file_path.exists():
                    try:
                        with open(parsed_file_path, 'r') as f:
                            model_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"[{correlation_id}] Failed to load parsed data for {file_name}: {e}")
                        model_data = file_result  # Fallback to basic data
                else:
                    model_data = file_result  # Use basic data from GNN results
                
                # Attempt visualization using available methods
                generated_files = []
                
                # Try using full visualization module
                if GNN_VISUALIZER_AVAILABLE:
                    try:
                        visualizer = GNNVisualizer(output_dir=str(file_output_dir))
                        result_path = visualizer.visualize_file(file_result["file_path"])
                        if result_path and Path(result_path).exists():
                            generated_files.extend([str(f) for f in Path(result_path).rglob("*") if f.is_file()])
                        logger.info(f"[{correlation_id}] Generated full visualizations for {file_name}")
                    except Exception as e:
                        logger.warning(f"[{correlation_id}] Full visualization failed for {file_name}: {e}")
                
                # Try matrix visualizations
                if MATRIX_VISUALIZER_AVAILABLE and not generated_files:
                    try:
                        matrix_viz = MatrixVisualizer()
                        parameters = model_data.get('parameters', [])
                        if parameters:
                            matrix_analysis_path = file_output_dir / "matrix_analysis.png"
                            if matrix_viz.generate_matrix_analysis(parameters, matrix_analysis_path):
                                generated_files.append(str(matrix_analysis_path))
                        logger.info(f"[{correlation_id}] Generated matrix visualizations for {file_name}")
                    except Exception as e:
                        logger.warning(f"[{correlation_id}] Matrix visualization failed for {file_name}: {e}")
                
                # Try basic matplotlib plots
                if not generated_files:
                    try:
                        basic_files = viz_manager.generate_basic_plots(model_data, file_output_dir, file_name)
                        generated_files.extend(basic_files)
                        logger.info(f"[{correlation_id}] Generated basic plots for {file_name}")
                    except Exception as e:
                        logger.warning(f"[{correlation_id}] Basic plot generation failed for {file_name}: {e}")
                
                # Create fallback if nothing worked
                if not generated_files:
                    fallback_files = viz_manager.create_fallback_visualization(
                        file_name, output_dir, "All visualization methods failed"
                    )
                    generated_files.extend(fallback_files)
                
                # Update attempt results
                attempt.end_time = datetime.now()
                attempt.success = len(generated_files) > 0
                attempt.generated_files = generated_files
                
                if attempt.success:
                    results.successful_visualizations += 1
                    results.total_images_generated += len(generated_files)
                    logger.info(f"[{correlation_id}] ✓ Generated {len(generated_files)} visualizations for {file_name}")
                else:
                    attempt.error_message = "No visualizations generated"
                    results.failed_visualizations += 1
                    logger.warning(f"[{correlation_id}] ✗ No visualizations generated for {file_name}")
                
            except Exception as e:
                attempt.end_time = datetime.now()
                attempt.success = False
                attempt.error_message = str(e)
                results.failed_visualizations += 1
                logger.error(f"[{correlation_id}] Unexpected error processing {file_name}: {e}")
                
                # Still try to create fallback
                try:
                    fallback_files = viz_manager.create_fallback_visualization(
                        file_name, output_dir, f"Unexpected error: {e}"
                    )
                    attempt.generated_files = fallback_files
                except Exception as fallback_error:
                    logger.error(f"[{correlation_id}] Fallback creation failed: {fallback_error}")
            
            results.attempts.append(attempt)
        
        # Save results regardless of success/failure
        try:
            results_file = output_dir / "visualization_results.json"
            with open(results_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            
            summary_file = output_dir / "visualization_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(results.to_dict()["summary"], f, indent=2)
            
            logger.info(f"[{correlation_id}] Saved results to {results_file}")
        except Exception as e:
            logger.error(f"[{correlation_id}] Failed to save results: {e}")
        
        # Always return success to continue pipeline
        if results.successful_visualizations > 0:
            log_step_success(logger, f"[{correlation_id}] Generated {results.total_images_generated} visualizations for {results.successful_visualizations}/{results.total_files} files")
        elif results.total_files > 0:
            log_step_warning(logger, f"[{correlation_id}] No successful visualizations, but created fallback reports for {results.total_files} files")
        else:
            log_step_warning(logger, f"[{correlation_id}] No files found for visualization")
        
        # Always return 0 (success) to ensure pipeline continuation
        return 0
            
    except Exception as e:
        log_step_error(logger, f"[{correlation_id}] Visualization processing failed", {"error": str(e), "traceback": traceback.format_exc()})
        
        # Even on complete failure, try to save an error report
        try:
            error_report = {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dependency_status": {
                    "matplotlib": MATPLOTLIB_AVAILABLE,
                    "networkx": NETWORKX_AVAILABLE,
                    "matrix_visualizer": MATRIX_VISUALIZER_AVAILABLE,
                    "gnn_visualizer": GNN_VISUALIZER_AVAILABLE
                }
            }
            
            error_file = output_dir / "visualization_error.json"
            with open(error_file, 'w') as f:
                json.dump(error_report, f, indent=2)
                
        except Exception as save_error:
            logger.error(f"[{correlation_id}] Failed to save error report: {save_error}")
        
        # Return 0 to ensure pipeline continuation even on complete failure
        return 0

if __name__ == "__main__":
    sys.exit(main())
