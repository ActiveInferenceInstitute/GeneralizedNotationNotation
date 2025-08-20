#!/usr/bin/env python3
"""
Step 9: Advanced Visualization (Thin Orchestrator)

This step provides advanced visualization capabilities for GNN models with
comprehensive safe-to-fail patterns and robust output management.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/advanced_visualization/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the advanced_visualization module.

Pipeline Flow:
    main.py → 9_advanced_viz.py (this script) → advanced_visualization/ (modular implementation)
"""

import sys
import logging
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker
)

from pipeline import (
    get_output_dir_for_script,
    get_pipeline_config
)

from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("9_advanced_viz", verbose=False)

# Safe imports with fallback handling
try:
    from advanced_visualization.visualizer import AdvancedVisualizer
    ADVANCED_VISUALIZER_AVAILABLE = True
except ImportError as e:
    log_step_warning(logger, f"AdvancedVisualizer not available: {e}")
    ADVANCED_VISUALIZER_AVAILABLE = False
    AdvancedVisualizer = None

try:
    from advanced_visualization.dashboard import DashboardGenerator
    DASHBOARD_GENERATOR_AVAILABLE = True
except ImportError as e:
    log_step_warning(logger, f"DashboardGenerator not available: {e}")
    DASHBOARD_GENERATOR_AVAILABLE = False
    DashboardGenerator = None

try:
    from advanced_visualization.data_extractor import extract_visualization_data
    DATA_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    log_step_warning(logger, f"DataExtractor not available: {e}")
    DATA_EXTRACTOR_AVAILABLE = False
    extract_visualization_data = None


@dataclass
class AdvancedVisualizationAttempt:
    """Track individual advanced visualization attempts."""
    file_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: str = ""
    generated_files: List[str] = field(default_factory=list)
    fallback_used: bool = False
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class AdvancedVisualizationResults:
    """Comprehensive advanced visualization results."""
    timestamp: str
    correlation_id: str
    source_directory: str
    output_directory: str
    viz_type: str
    interactive: bool
    export_formats: List[str]
    total_files: int = 0
    successful_visualizations: int = 0
    failed_visualizations: int = 0
    fallback_visualizations: int = 0
    total_files_generated: int = 0
    attempts: List[AdvancedVisualizationAttempt] = field(default_factory=list)
    dependency_status: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "source_directory": self.source_directory,
            "output_directory": self.output_directory,
            "configuration": {
                "visualization_type": self.viz_type,
                "interactive": self.interactive,
                "export_formats": self.export_formats
            },
            "summary": {
                "total_files": self.total_files,
                "successful_visualizations": self.successful_visualizations,
                "failed_visualizations": self.failed_visualizations,
                "fallback_visualizations": self.fallback_visualizations,
                "total_files_generated": self.total_files_generated,
                "success_rate": f"{(self.successful_visualizations / max(1, self.total_files)) * 100:.1f}%"
            },
            "dependency_status": self.dependency_status,
            "processing_details": [
                {
                    "file_name": attempt.file_name,
                    "success": attempt.success,
                    "fallback_used": attempt.fallback_used,
                    "duration_seconds": attempt.duration_seconds,
                    "generated_files": attempt.generated_files,
                    "error": attempt.error_message if not attempt.success else None
                }
                for attempt in self.attempts
            ],
            "errors": self.errors
        }


class SafeAdvancedVisualizationManager:
    """Manages advanced visualization with comprehensive safety patterns."""
    
    def __init__(self, logger, correlation_id: str):
        self.logger = logger
        self.correlation_id = correlation_id
        
    @contextmanager
    def safe_processing_context(self, file_name: str):
        """Context manager for safe processing operations."""
        try:
            self.logger.debug(f"[{self.correlation_id}] Starting processing context for {file_name}")
            yield
        except Exception as e:
            self.logger.error(f"[{self.correlation_id}] Processing context error for {file_name}: {e}")
            raise
        finally:
            self.logger.debug(f"[{self.correlation_id}] Completed processing context for {file_name}")
    
    def create_robust_fallback_visualization(self, content: str, file_name: str, 
                                           output_dir: Path, viz_type: str, 
                                           export_formats: List[str],
                                           error_message: str = "Advanced visualization failed") -> List[str]:
        """Create comprehensive fallback visualizations."""
        generated_files = []
        
        try:
            # Create file-specific output directory
            file_output_dir = output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate fallback HTML if requested
            if 'html' in export_formats:
                html_content = self.generate_fallback_html_visualization(content, file_name, error_message)
                html_file = file_output_dir / f"{file_name.replace('.md', '')}_advanced_viz.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                generated_files.append(str(html_file))
            
            # Generate fallback JSON data if requested
            if 'json' in export_formats:
                json_data = self.extract_fallback_visualization_data(content, error_message)
                json_file = file_output_dir / f"{file_name.replace('.md', '')}_viz_data.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                generated_files.append(str(json_file))
            
            # Create comprehensive error report
            error_file = file_output_dir / "advanced_viz_error_report.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Advanced Visualization Error Report\n")
                f.write(f"===================================\n\n")
                f.write(f"File: {file_name}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Visualization Type: {viz_type}\n")
                f.write(f"Export Formats: {export_formats}\n")
                f.write(f"Error: {error_message}\n\n")
                f.write(f"Dependency Status:\n")
                f.write(f"- AdvancedVisualizer: {ADVANCED_VISUALIZER_AVAILABLE}\n")
                f.write(f"- DashboardGenerator: {DASHBOARD_GENERATOR_AVAILABLE}\n")
                f.write(f"- DataExtractor: {DATA_EXTRACTOR_AVAILABLE}\n\n")
                f.write(f"Generated Files ({len(generated_files)}):\n")
                for i, gf in enumerate(generated_files, 1):
                    f.write(f"{i}. {gf}\n")
            generated_files.append(str(error_file))
            
            # Create basic analysis summary
            summary_file = file_output_dir / "content_analysis.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Content Analysis Summary\n")
                f.write(f"=======================\n\n")
                f.write(f"File: {file_name}\n")
                f.write(f"Content Length: {len(content)} characters\n")
                f.write(f"Lines: {len(content.splitlines())}\n")
                f.write(f"Contains GNN structure: {'#' in content and 'Variables' in content}\n")
                f.write(f"Processing Mode: Fallback\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            generated_files.append(str(summary_file))
            
            self.logger.info(f"[{self.correlation_id}] Created {len(generated_files)} fallback files for {file_name}")
            
        except Exception as e:
            self.logger.error(f"[{self.correlation_id}] Failed to create fallback visualization for {file_name}: {e}")
        
        return generated_files
    
    def generate_fallback_html_visualization(self, content: str, model_name: str, error_message: str) -> str:
        """Generate comprehensive fallback HTML visualization."""
        # Extract basic statistics from content
        lines = content.splitlines()
        word_count = len(content.split())
        section_count = len([line for line in lines if line.startswith('#')])
        variable_mentions = len([line for line in lines if 'Variables' in line])
        connection_mentions = len([line for line in lines if any(keyword in line for keyword in ['connection', 'edge', 'link'])])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Visualization - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .header h2 {{
            color: #7f8c8d;
            margin: 10px 0 0 0;
            font-weight: 300;
        }}
        .content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .panel {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }}
        .panel h3 {{
            color: #2c3e50;
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
        .error-panel {{
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 5px solid #f44336;
        }}
        .stats-panel {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 5px solid #2196f3;
        }}
        .status-panel {{
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border-left: 5px solid #9c27b0;
        }}
        .info-panel {{
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            border-left: 5px solid #4caf50;
        }}
        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }}
        .stat-item:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            font-weight: 600;
            color: #2c3e50;
        }}
        .stat-value {{
            color: #27ae60;
            font-weight: bold;
        }}
        .error-message {{
            background: rgba(244, 67, 54, 0.1);
            padding: 15px;
            border-radius: 8px;
            color: #d32f2f;
            font-weight: 500;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-available {{ background-color: #4caf50; }}
        .status-unavailable {{ background-color: #f44336; }}
        .full-width {{
            grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Advanced GNN Visualization</h1>
            <h2>Model: {model_name}</h2>
        </div>
        
        <div class="content">
            <div class="panel error-panel">
                <h3>⚠️ Processing Status</h3>
                <div class="error-message">
                    {error_message}
                </div>
                <p>This visualization was generated using fallback mode due to the error above. While advanced interactive features are not available, basic analysis and content summary are provided below.</p>
            </div>
            
            <div class="panel stats-panel">
                <h3>📊 Content Statistics</h3>
                <div class="stat-item">
                    <span class="stat-label">Total Lines:</span>
                    <span class="stat-value">{len(lines)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Word Count:</span>
                    <span class="stat-value">{word_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Sections:</span>
                    <span class="stat-value">{section_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Variable References:</span>
                    <span class="stat-value">{variable_mentions}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Connection References:</span>
                    <span class="stat-value">{connection_mentions}</span>
                </div>
            </div>
            
            <div class="panel status-panel">
                <h3>🔧 Dependency Status</h3>
                <div class="stat-item">
                    <span class="stat-label">
                        <span class="status-indicator {'status-available' if ADVANCED_VISUALIZER_AVAILABLE else 'status-unavailable'}"></span>
                        AdvancedVisualizer:
                    </span>
                    <span class="stat-value">{'Available' if ADVANCED_VISUALIZER_AVAILABLE else 'Unavailable'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">
                        <span class="status-indicator {'status-available' if DASHBOARD_GENERATOR_AVAILABLE else 'status-unavailable'}"></span>
                        Dashboard Generator:
                    </span>
                    <span class="stat-value">{'Available' if DASHBOARD_GENERATOR_AVAILABLE else 'Unavailable'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">
                        <span class="status-indicator {'status-available' if DATA_EXTRACTOR_AVAILABLE else 'status-unavailable'}"></span>
                        Data Extractor:
                    </span>
                    <span class="stat-value">{'Available' if DATA_EXTRACTOR_AVAILABLE else 'Unavailable'}</span>
                </div>
            </div>
            
            <div class="panel info-panel">
                <h3>💡 Recovery Information</h3>
                <p><strong>To enable full advanced visualization:</strong></p>
                <ul>
                    <li>Install missing dependencies</li>
                    <li>Verify advanced_visualization module setup</li>
                    <li>Check file permissions and paths</li>
                    <li>Review the error message above for specific issues</li>
                </ul>
                <p><strong>Current capabilities:</strong></p>
                <ul>
                    <li>✓ Content analysis and statistics</li>
                    <li>✓ Basic structure detection</li>
                    <li>✓ Fallback HTML visualization</li>
                    <li>✓ Error reporting and diagnostics</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: Fallback | Pipeline: GNN Advanced Visualization</p>
        </div>
    </div>
</body>
</html>"""
        return html
    
    def extract_fallback_visualization_data(self, content: str, error_message: str) -> Dict[str, Any]:
        """Extract basic visualization data from content (fallback mode)."""
        try:
            lines = content.splitlines()
            
            # Basic structure detection
            sections = []
            variables = []
            connections = []
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    current_section = line.lstrip('#').strip()
                    sections.append(current_section)
                elif 'variable' in line.lower() or any(var in line for var in ['A', 'B', 'C', 'D']):
                    variables.append(line)
                elif any(keyword in line.lower() for keyword in ['connection', 'edge', 'link', '->']):
                    connections.append(line)
            
            return {
                "metadata": {
                    "processing_mode": "fallback",
                    "error_message": error_message,
                    "timestamp": datetime.now().isoformat(),
                    "content_length": len(content),
                    "line_count": len(lines)
                },
                "structure": {
                    "sections": sections,
                    "section_count": len(sections)
                },
                "elements": {
                    "variables": variables[:10],  # Limit for JSON size
                    "variable_count": len(variables),
                    "connections": connections[:10],  # Limit for JSON size
                    "connection_count": len(connections)
                },
                "statistics": {
                    "total_lines": len(lines),
                    "word_count": len(content.split()),
                    "character_count": len(content),
                    "has_gnn_structure": any(keyword in content for keyword in ['Variables', 'Connections', 'Parameters'])
                },
                "visualization_status": {
                    "advanced_visualizer_available": ADVANCED_VISUALIZER_AVAILABLE,
                    "dashboard_generator_available": DASHBOARD_GENERATOR_AVAILABLE,
                    "data_extractor_available": DATA_EXTRACTOR_AVAILABLE
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to extract fallback data: {e}",
                "timestamp": datetime.now().isoformat(),
                "processing_mode": "fallback_failed"
            }


def process_advanced_viz_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized advanced visualization processing with comprehensive safety patterns.
    """
    correlation_id = kwargs.get('correlation_id', f"adviz_{int(time.time())}")
    
    try:
        with performance_tracker.track_operation("advanced_viz_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Get configuration
            config = get_pipeline_config()
            step_output_dir = get_output_dir_for_script("9_advanced_viz.py", output_dir)
            step_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract processing parameters
            viz_type = kwargs.get('viz_type', 'all')
            interactive = kwargs.get('interactive', True)
            export_formats = kwargs.get('export_formats', ['html', 'json'])
            
            logger.info(f"[{correlation_id}] Starting advanced visualization processing")
            logger.info(f"[{correlation_id}] Target: {target_dir}, Output: {step_output_dir}")
            logger.info(f"[{correlation_id}] Config: type={viz_type}, interactive={interactive}, formats={export_formats}")
            
            # Initialize results tracking
            results = AdvancedVisualizationResults(
                timestamp=datetime.now().isoformat(),
                correlation_id=correlation_id,
                source_directory=str(target_dir),
                output_directory=str(step_output_dir),
                viz_type=viz_type,
                interactive=interactive,
                export_formats=export_formats,
                dependency_status={
                    "advanced_visualizer": ADVANCED_VISUALIZER_AVAILABLE,
                    "dashboard_generator": DASHBOARD_GENERATOR_AVAILABLE,
                    "data_extractor": DATA_EXTRACTOR_AVAILABLE
                }
            )
            
            # Validate input directory
            if not target_dir.exists():
                log_step_warning(logger, f"[{correlation_id}] Input directory does not exist: {target_dir}")
                return True  # Not an error for pipeline continuation
            
            # Find GNN files
            pattern = "**/*.md" if recursive else "*.md"
            gnn_files = list(target_dir.glob(pattern))
            
            if not gnn_files:
                log_step_warning(logger, f"[{correlation_id}] No GNN files found in {target_dir}")
                # Still save results with empty state
                _save_results(results, step_output_dir, logger, correlation_id)
                return True
            
            results.total_files = len(gnn_files)
            logger.info(f"[{correlation_id}] Found {results.total_files} GNN files to process")
            
            # Initialize processing manager
            viz_manager = SafeAdvancedVisualizationManager(logger, correlation_id)
            
            # Initialize visualizers if available
            visualizer = AdvancedVisualizer() if ADVANCED_VISUALIZER_AVAILABLE else None
            dashboard_gen = DashboardGenerator() if DASHBOARD_GENERATOR_AVAILABLE else None
            
            # Process each file
            for gnn_file in gnn_files:
                attempt = AdvancedVisualizationAttempt(
                    file_name=gnn_file.name,
                    start_time=datetime.now()
                )
                
                try:
                    with viz_manager.safe_processing_context(gnn_file.name):
                        # Read file content
                        try:
                            with open(gnn_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception as e:
                            raise Exception(f"Failed to read file: {e}")
                        
                        logger.info(f"[{correlation_id}] Processing {gnn_file.name}")
                        
                        # Attempt full advanced visualization
                        if visualizer:
                            try:
                                viz_files = visualizer.generate_visualizations(
                                    content, gnn_file.stem, step_output_dir, viz_type, interactive, export_formats
                                )
                                attempt.generated_files.extend(viz_files)
                                logger.info(f"[{correlation_id}] Generated advanced visualizations for {gnn_file.name}")
                            except Exception as e:
                                logger.warning(f"[{correlation_id}] Advanced visualizer failed for {gnn_file.name}: {e}")
                        
                        # Attempt dashboard generation
                        if dashboard_gen and interactive and not attempt.generated_files:
                            try:
                                dashboard_file = dashboard_gen.generate_dashboard(
                                    content, gnn_file.stem, step_output_dir
                                )
                                if dashboard_file:
                                    attempt.generated_files.append(str(dashboard_file))
                                logger.info(f"[{correlation_id}] Generated dashboard for {gnn_file.name}")
                            except Exception as e:
                                logger.warning(f"[{correlation_id}] Dashboard generation failed for {gnn_file.name}: {e}")
                        
                        # Create fallback if no success yet
                        if not attempt.generated_files:
                            fallback_files = viz_manager.create_robust_fallback_visualization(
                                content, gnn_file.name, step_output_dir, viz_type, export_formats,
                                "Advanced visualization modules unavailable"
                            )
                            attempt.generated_files.extend(fallback_files)
                            attempt.fallback_used = True
                            results.fallback_visualizations += 1
                            logger.info(f"[{correlation_id}] Created fallback visualization for {gnn_file.name}")
                        
                        # Update results
                        attempt.end_time = datetime.now()
                        attempt.success = len(attempt.generated_files) > 0
                        
                        if attempt.success:
                            if not attempt.fallback_used:
                                results.successful_visualizations += 1
                            results.total_files_generated += len(attempt.generated_files)
                            logger.info(f"[{correlation_id}] ✓ Generated {len(attempt.generated_files)} files for {gnn_file.name}")
                        else:
                            attempt.error_message = "No files generated despite attempts"
                            results.failed_visualizations += 1
                            logger.warning(f"[{correlation_id}] ✗ No files generated for {gnn_file.name}")
                
                except Exception as e:
                    attempt.end_time = datetime.now()
                    attempt.success = False
                    attempt.error_message = str(e)
                    results.failed_visualizations += 1
                    logger.error(f"[{correlation_id}] Processing failed for {gnn_file.name}: {e}")
                    
                    # Still try to create fallback
                    try:
                        fallback_files = viz_manager.create_robust_fallback_visualization(
                            "", gnn_file.name, step_output_dir, viz_type, export_formats,
                            f"Processing error: {e}"
                        )
                        attempt.generated_files = fallback_files
                        attempt.fallback_used = True
                        results.fallback_visualizations += 1
                    except Exception as fallback_error:
                        logger.error(f"[{correlation_id}] Fallback creation failed: {fallback_error}")
                
                results.attempts.append(attempt)
            
            # Save comprehensive results
            _save_results(results, step_output_dir, logger, correlation_id)
            
            # Determine success (always return True for pipeline continuation)
            total_generated = results.successful_visualizations + results.fallback_visualizations
            if total_generated > 0:
                log_step_success(logger, f"[{correlation_id}] Generated visualizations for {total_generated}/{results.total_files} files ({results.successful_visualizations} full, {results.fallback_visualizations} fallback)")
            else:
                log_step_warning(logger, f"[{correlation_id}] No visualizations generated, but processing completed")
            
            return True  # Always return True for pipeline continuation
            
    except Exception as e:
        log_step_error(logger, f"[{correlation_id}] Advanced visualization processing failed: {e}")
        if verbose:
            logger.error(f"[{correlation_id}] Full traceback: {traceback.format_exc()}")
        
        # Save error state
        try:
            error_results = AdvancedVisualizationResults(
                timestamp=datetime.now().isoformat(),
                correlation_id=correlation_id,
                source_directory=str(target_dir),
                output_directory=str(step_output_dir),
                viz_type=kwargs.get('viz_type', 'all'),
                interactive=kwargs.get('interactive', True),
                export_formats=kwargs.get('export_formats', ['html', 'json']),
                errors=[str(e)]
            )
            _save_results(error_results, step_output_dir, logger, correlation_id)
        except Exception as save_error:
            logger.error(f"[{correlation_id}] Failed to save error results: {save_error}")
        
        return True  # Still return True for pipeline continuation


def _save_results(results: AdvancedVisualizationResults, output_dir: Path, 
                 logger: logging.Logger, correlation_id: str):
    """Save visualization results safely."""
    try:
        # Save detailed results
        detailed_file = output_dir / "advanced_visualization_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / "advanced_visualization_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict()["summary"], f, indent=2)
        
        logger.info(f"[{correlation_id}] Saved results to {detailed_file}")
        
    except Exception as e:
        logger.error(f"[{correlation_id}] Failed to save results: {e}")


def generate_correlation_id() -> str:
    """Generate a correlation ID for tracking."""
    import uuid
    return str(uuid.uuid4())[:8]


# Create standardized pipeline script with safe defaults
run_script = create_standardized_pipeline_script(
    "9_advanced_viz.py",
    lambda target_dir, output_dir, logger, **kwargs: process_advanced_viz_standardized(
        target_dir, output_dir, logger, 
        correlation_id=generate_correlation_id(),
        **kwargs
    ),
    "Advanced visualization and exploration with safe-to-fail patterns",
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
    # Ensure pipeline continuation even on complete failure
    try:
        exit_code = run_script()
        sys.exit(0)  # Always return success for pipeline continuation
    except Exception as e:
        logger.error(f"Critical failure in advanced visualization: {e}")
        sys.exit(0)  # Still return success for pipeline continuation 