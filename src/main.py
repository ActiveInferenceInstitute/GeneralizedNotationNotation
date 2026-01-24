#!/usr/bin/env python3
"""
GNN Processing Pipeline

This script orchestrates the 24-step GNN processing pipeline (steps 0-23).
The pipeline transforms GNN specifications into executable simulations, visualizations,
and advanced representations including audio sonification.

Pipeline Steps:
0. Template initialization (0_template.py)
1. Environment setup (1_setup.py)
2. Test suite execution (2_tests.py)
3. GNN file processing (3_gnn.py)
4. Model registry (4_model_registry.py)
5. Type checking (5_type_checker.py)
6. Validation (6_validation.py)
7. Multi-format export (7_export.py)
8. Visualization (8_visualization.py)
9. Advanced visualization (9_advanced_viz.py)
10. Ontology processing (10_ontology.py)
11. Code rendering (11_render.py)
12. Execution (12_execute.py)
13. LLM processing (13_llm.py)
14. ML integration (14_ml_integration.py)
15. Audio processing (15_audio.py)
16. Analysis (16_analysis.py)
17. Integration (17_integration.py)
18. Security (18_security.py)
19. Research (19_research.py)
20. Website generation (20_website.py)
21. Model Context Protocol processing (21_mcp.py)
22. GUI (Interactive GNN Constructor) (22_gui.py)
23. Report generation (23_report.py)

Usage:
    python src/main.py [options]
    
Examples:
    # Run full pipeline
    python src/main.py --target-dir input/gnn_files --verbose
    
    # Run specific steps only
    python src/main.py --only-steps "0,1,2,3" --verbose
    
    # Skip certain steps
    python src/main.py --skip-steps "15,16" --verbose

For complete usage information, see:
- README.md: Project overview and quick start
- doc/pipeline/README.md: Detailed pipeline documentation
- src/README.md: Pipeline safety and reliability documentation
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Detect project root and ensure we're working from there
SCRIPT_DIR = Path(__file__).parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent     # project root (one level up from src/)

# Change working directory to project root if not already there
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
    print(f"Changed working directory to project root: {PROJECT_ROOT}")

# Add src to path for imports
sys.path.insert(0, str(SCRIPT_DIR))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import ArgumentParser, PipelineArguments
from pipeline.config import get_output_dir_for_script, get_pipeline_config
from utils.resource_manager import get_current_memory_usage
from utils.pipeline_validator import validate_step_prerequisites, validate_pipeline_step_sequence

# Optional logging and progress tracking
try:
    from utils.logging_utils import (
        PipelineProgressTracker,
        VisualLoggingEnhancer,
        PipelineLogger,
        PipelineLogger,
        log_pipeline_summary,
        reset_progress_tracker,
        setup_step_logging,
        rotate_logs
    )
    from utils.visual_logging import (
        VisualLogger,
        VisualConfig,
        print_pipeline_banner,
        print_step_summary,
        print_completion_summary,
        format_step_header,
        format_status_message,
        create_visual_logger
    )
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError as e:
    STRUCTURED_LOGGING_AVAILABLE = False
    # Provide fallback implementations when visual logging is not available
    from dataclasses import dataclass

    @dataclass
    class VisualConfig:
        """Fallback visual config when visual_logging is not available."""
        enable_colors: bool = True
        enable_progress_bars: bool = True
        enable_emoji: bool = True
        enable_animation: bool = True
        max_width: int = 80
        show_timestamps: bool = False
        show_correlation_ids: bool = True
        compact_mode: bool = False

    class VisualLogger:
        """Fallback visual logger when visual_logging is not available."""
        def __init__(self, name, config=None):
            self.name = name
            self.config = config
            self._correlation_id = None
        def set_correlation_id(self, cid): self._correlation_id = cid
        def print_progress(self, current, total, message): print(f"[{current}/{total}] {message}")
        def print_step_header(self, step_num, description, total): print(f"\n=== Step {step_num}/{total}: {description} ===")

    def create_visual_logger(name, config): return VisualLogger(name, config)
    def print_pipeline_banner(title, subtitle): print(f"\n{'='*60}\n{title}\n{subtitle}\n{'='*60}")
    def print_step_summary(step, desc, status, duration, stats): print(f"Step {step}: {desc} - {status} ({duration:.2f}s)")
    def print_completion_summary(success, duration, stats): print(f"\n{'='*60}\nPipeline {'COMPLETED' if success else 'FAILED'} in {duration:.2f}s\n{stats}\n{'='*60}")

def main():
    """Main pipeline orchestration function."""
    # Parse arguments
    parser = ArgumentParser.create_main_parser()
    parsed = parser.parse_args()

    # Convert to PipelineArguments
    kwargs = {}
    for key, value in vars(parsed).items():
        if value is not None:
            kwargs[key] = value

    args = PipelineArguments(**kwargs)

    # Setup visual logging
    visual_config = VisualConfig(
        enable_colors=True,
        enable_progress_bars=True,
        enable_emoji=True,
        enable_animation=True,
        show_timestamps=args.verbose,
        show_correlation_ids=True,
        compact_mode=False
    )

    visual_logger = create_visual_logger("pipeline", visual_config)

    # Setup logging (use structured if available, fallback to standard)
    if STRUCTURED_LOGGING_AVAILABLE:
        # Prepare log directory
        log_dir = args.output_dir / "00_pipeline_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Rotate existing logs
        rotate_logs(log_dir)

        # Enable JSON logging
        PipelineLogger.enable_json_logging(log_dir)

        logger = setup_step_logging("pipeline", args.verbose, enable_structured=True)
        # Reset progress tracker for new pipeline run
        reset_progress_tracker()
    else:
        # Use the logging_utils version directly to avoid duplicate setup
        try:
            from utils.logging_utils import setup_step_logging as _setup_step_logging
            logger = _setup_step_logging("pipeline", args.verbose)
        except ImportError:
            logger = setup_step_logging("pipeline", args.verbose)

    # Set correlation ID for tracking
    import uuid
    correlation_id = str(uuid.uuid4())[:8]
    visual_logger.set_correlation_id(correlation_id)

    # Initialize steps_to_execute outside try block
    steps_to_execute = []

    # Define pipeline steps (outside try block for proper scope)
    pipeline_steps = [
        ("0_template.py", "Template initialization"),
        ("1_setup.py", "Environment setup"),
        ("2_tests.py", "Test suite execution"),
        ("3_gnn.py", "GNN file processing"),
        ("4_model_registry.py", "Model registry"),
        ("5_type_checker.py", "Type checking"),
        ("6_validation.py", "Validation"),
        ("7_export.py", "Multi-format export"),
        ("8_visualization.py", "Visualization"),
        ("9_advanced_viz.py", "Advanced visualization"),
        ("10_ontology.py", "Ontology processing"),
        ("11_render.py", "Code rendering"),
        ("12_execute.py", "Execution"),
        ("13_llm.py", "LLM processing"),
        ("14_ml_integration.py", "ML integration"),
        ("15_audio.py", "Audio processing"),
        ("16_analysis.py", "Analysis"),
        ("17_integration.py", "Integration"),
        ("18_security.py", "Security"),
        ("19_research.py", "Research"),
        ("20_website.py", "Website generation"),
        ("21_mcp.py", "Model Context Protocol processing"),
        ("22_gui.py", "GUI (Interactive GNN Constructor)"),
        ("23_report.py", "Report generation"),
        ("24_intelligent_analysis.py", "Intelligent pipeline analysis")
    ]

    # Handle step filtering with automatic dependency resolution
    steps_to_execute = pipeline_steps  # Initialize with all steps

    # Initialize pipeline execution summary
    pipeline_summary = {
        "start_time": datetime.now().isoformat(),
        "arguments": args.to_dict(),
        "steps": [],
        "end_time": None,
        "overall_status": "RUNNING",
        "total_duration_seconds": None,
        "environment_info": get_environment_info(),
        "performance_summary": {
            "peak_memory_mb": 0.0,
            "total_steps": len(steps_to_execute),
            "failed_steps": 0,
            "critical_failures": 0,
            "successful_steps": 0,
            "warnings": 0
        }
    }

    try:
        # Project identification banner (displayed at very top)
        print_pipeline_banner(
            "Generalized Notation Notation (GNN)",
            "https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation | Active Inference Institute"
        )

        # Pipeline start with visual banner
        print_pipeline_banner(
            "🚀 GNN Processing Pipeline",
            f"Starting execution with {len(steps_to_execute)} steps | Correlation ID: {correlation_id}"
        )

        # Visual progress indicator
        visual_logger.print_progress(0, len(steps_to_execute), "Pipeline initialization")

        # Pipeline start logging
        if STRUCTURED_LOGGING_AVAILABLE:
            PipelineLogger.log_structured(
                logger, logging.INFO,
                "🚀 Starting GNN Processing Pipeline",
                 total_steps=24,
                target_dir=str(args.target_dir),
                output_dir=str(args.output_dir),
                event_type="pipeline_start"
            )
        else:
            log_step_start(logger, "Starting GNN Processing Pipeline")
        
        # Handle step filtering with automatic dependency resolution
        # (pipeline_steps already defined above, steps_to_execute initialized)
        if args.only_steps:
            requested_step_numbers = parse_step_list(args.only_steps)
            
            # Automatic dependency resolution
            step_dependencies = {
                11: [3],     # 11_render.py needs 3_gnn.py
                12: [3, 11], # 12_execute.py needs 3_gnn.py and 11_render.py
                8: [3],      # 8_visualization.py needs 3_gnn.py
                9: [3, 8],   # 9_advanced_viz.py needs 3_gnn.py and 8_visualization.py
                13: [3],     # 13_llm.py needs 3_gnn.py
                23: [8, 13], # 23_report.py needs 8_visualization.py and 13_llm.py
                20: [8],     # 20_website.py needs 8_visualization.py
                5: [3],      # 5_type_checker.py needs 3_gnn.py
                6: [3, 5],   # 6_validation.py needs 3_gnn.py and 5_type_checker.py
                7: [3],      # 7_export.py needs 3_gnn.py
                10: [3],     # 10_ontology.py needs 3_gnn.py
                15: [3],     # 15_audio.py needs 3_gnn.py
                16: [3, 7],  # 16_analysis.py needs 3_gnn.py and 7_export.py
                24: [23],    # 24_intelligent_analysis.py needs 23_report.py (and implicitly the summary it generates/pipeline completion)
            }
            
            # Include dependencies automatically
            resolved_step_numbers = set(requested_step_numbers)
            added_dependencies = []
            for step_num in requested_step_numbers:
                if step_num in step_dependencies:
                    for dep in step_dependencies[step_num]:
                        if dep not in resolved_step_numbers:
                            resolved_step_numbers.add(dep)
                            added_dependencies.append(dep)
            
            if added_dependencies:
                logger.info(f"Auto-including dependency steps: {sorted(added_dependencies)}")
            
            steps_to_execute = [pipeline_steps[i] for i in sorted(resolved_step_numbers) if 0 <= i < len(pipeline_steps)]
            logger.info(f"Executing steps: {[step[0] for step in steps_to_execute]}")
        
        if args.skip_steps:
            skip_numbers = parse_step_list(args.skip_steps)
            steps_to_execute = [step for i, step in enumerate(pipeline_steps) if i not in skip_numbers]
            logger.info(f"Skipping steps: {[pipeline_steps[i][0] for i in skip_numbers if 0 <= i < len(pipeline_steps)]}")

        # Update pipeline summary with actual step count
        pipeline_summary["performance_summary"]["total_steps"] = len(steps_to_execute)

        # Initialize progress tracker if enhanced logging is available
        progress_tracker = None
        if STRUCTURED_LOGGING_AVAILABLE:
            progress_tracker = PipelineProgressTracker(len(steps_to_execute))
            # Set global progress tracker so structured logging functions can use it
            import utils.logging_utils as logging_utils_module
            logging_utils_module._global_progress_tracker = progress_tracker
        
        # Validate step sequence before execution
        sequence_validation = validate_pipeline_step_sequence(steps_to_execute, logger)
        if sequence_validation["warnings"]:
            for warning in sequence_validation["warnings"]:
                logger.warning(f"Pipeline sequence: {warning}")
        if sequence_validation["recommendations"]:
            for rec in sequence_validation["recommendations"]:
                logger.info(f"Recommendation: {rec}")
        
        # Execute each step
        for step_index, (script_name, description) in enumerate(steps_to_execute, 0):
            # Use the actual script name as step identifier for clarity
            actual_step_number = step_index + 1
            step_start_time = time.time()
            step_start_datetime = datetime.now()
            
            # Step start with visual indicators
            visual_logger.print_step_header(actual_step_number, description, len(steps_to_execute))

            # Step start logging with progress tracking
            if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
                progress_header = progress_tracker.start_step(actual_step_number, description)
                logger.info(progress_header)
                # Use structured logging functions that support additional parameters
                from utils.logging_utils import log_step_start as structured_log_step_start
                structured_log_step_start(logger, f"Starting {description}",
                                      step_number=actual_step_number,
                                      total_steps=len(steps_to_execute),
                                      script_name=script_name)
            else:
                logger.info(f"🔄 Executing step {actual_step_number}: {description}")
            
            # Execute the step
            step_result = execute_pipeline_step(script_name, args, logger)
            
            # Calculate step duration
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_end_datetime = datetime.now()
            
            # Update step result with timing information and enhanced metadata
            step_result.update({
                "step_number": actual_step_number,
                "script_name": script_name,
                "description": description,
                "start_time": step_start_datetime.isoformat(),
                "end_time": step_end_datetime.isoformat(),
                "duration_seconds": step_duration,
                "exit_code": step_result.get("exit_code", 0),
                "retry_count": step_result.get("retry_count", 0),
                "prerequisite_check": step_result.get("prerequisite_check", True),
                "dependency_warnings": step_result.get("dependency_warnings", []),
                "recoverable": step_result.get("recoverable", False),
                "memory_usage_mb": step_result.get("memory_usage_mb", 0.0),
                "peak_memory_mb": step_result.get("peak_memory_mb", 0.0),
                "memory_delta_mb": step_result.get("memory_delta_mb", 0.0)
            })
            
            # Check for warnings in both stdout and stderr (precise regex matching)
            combined_output = f"{step_result.get('stdout', '')}\n{step_result.get('stderr', '')}"
            # More precise warning detection - look for actual log levels or warning symbols
            import re
            
            # Known safe warnings that should not trigger SUCCESS_WITH_WARNINGS
            safe_warning_patterns = [
                r"matplotlib.*?backend",  # Matplotlib backend selection messages
                r"using agg backend",  # Headless backend in use
                r"no display",  # No display available (expected in CI/headless)
                r"pymdp.*?not available",  # Optional dependency message
                r"optional.*?dependency",  # Optional dependency notifications
                r"plotly.*?not available",  # Optional plotly dependency
                r"numpy.*?not available",  # Optional numpy dependency
                r"seaborn.*?not available",  # Optional seaborn dependency
                r"bokeh.*?not available",  # Optional bokeh dependency
                r"d2.*?not available",  # Optional D2 CLI dependency
                r"d2 cli.*?not available",  # Optional D2 CLI dependency (explicit)
                r"d2 visualizer.*?not available",  # D2 visualizer module
                r"d2 cli.*?install",  # D2 CLI installation instructions
                r"interactive.*?limited",  # Interactive visualization limitations
                r"numeric.*?limited",  # Numeric visualization limitations
                r"warnings: 0",  # Zero warnings count in validation logs
            ]
            
            # Combine safe patterns into single regex
            safe_patterns = "|".join(f"({p})" for p in safe_warning_patterns)
            safe_warning_pattern = re.compile(safe_patterns, re.IGNORECASE)
            
            # Check for warnings but exclude safe patterns
            warning_pattern = re.compile(r"(WARNING|⚠️|warn)", re.IGNORECASE)
            has_warning = bool(warning_pattern.search(combined_output))
            
            # If warning found, check if it's a "safe" warning
            if has_warning:
                has_warning = not bool(safe_warning_pattern.search(combined_output))

            # Propagate SUCCESS_WITH_WARNINGS status if applicable
            if step_result["status"] == "SUCCESS" and has_warning:
                step_result["status"] = "SUCCESS_WITH_WARNINGS"

            # Add to pipeline summary
            pipeline_summary["steps"].append(step_result)

            # Update performance summary
            if step_result["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS"):
                pipeline_summary["performance_summary"]["successful_steps"] += 1
            elif step_result["status"] == "FAILED":
                pipeline_summary["performance_summary"]["failed_steps"] += 1
                if step_result.get("exit_code", 0) == 1:
                    pipeline_summary["performance_summary"]["critical_failures"] += 1

            # Count warnings
            if has_warning:
                pipeline_summary["performance_summary"]["warnings"] += 1
            
            # Update peak memory usage with better tracking
            step_memory = step_result.get("memory_usage_mb", 0.0)
            step_peak_memory = step_result.get("peak_memory_mb", 0.0)
            current_peak = pipeline_summary["performance_summary"]["peak_memory_mb"]

            # Use the higher of step memory or peak memory
            new_peak = max(step_memory, step_peak_memory, current_peak)
            pipeline_summary["performance_summary"]["peak_memory_mb"] = new_peak
            
            # Update total steps count
            pipeline_summary["performance_summary"]["total_steps"] = len(steps_to_execute)
            
            # Step completion with visual indicators
            status_for_logging = step_result["status"]

            # Visual step completion summary
            step_stats = {
                "Status": status_for_logging,
                "Duration": f"{step_duration:.2f}s",
                "Memory": f"{step_result.get('peak_memory_mb', 0):.1f}MB",
                "Exit Code": step_result.get("exit_code", 0)
            }

            print_step_summary(actual_step_number, description, status_for_logging, step_duration, step_stats)

            # Step completion logging with progress tracking
            if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
                # Use structured logging functions that support additional parameters
                # These functions will handle progress tracking via the global tracker
                if status_for_logging.startswith("SUCCESS"):
                    from utils.logging_utils import log_step_success as structured_log_step_success
                    structured_log_step_success(logger, f"{description} completed",
                                        step_number=actual_step_number,
                                        duration=step_duration,
                                            status=status_for_logging)
                elif "WARNING" in status_for_logging:
                    from utils.logging_utils import log_step_warning as structured_log_step_warning
                    structured_log_step_warning(logger, f"{description} completed with warnings",
                                        step_number=actual_step_number,
                                        duration=step_duration,
                                            status=status_for_logging)
                else:
                    from utils.logging_utils import log_step_error as structured_log_step_error
                    structured_log_step_error(logger, f"{description} failed",
                                        step_number=actual_step_number,
                                        duration=step_duration,
                                            status=status_for_logging)
            else:
                if str(status_for_logging).startswith("SUCCESS"):
                    logger.info(f"✅ Step {actual_step_number} completed successfully in {step_duration:.2f}s")
                elif status_for_logging == "PARTIAL_SUCCESS" or "WARNING" in status_for_logging:
                    logger.warning(f"⚠️ Step {actual_step_number} completed with warnings in {step_duration:.2f}s")
                else:
                    logger.error(f"❌ Step {actual_step_number} failed with status: {status_for_logging}")
        
        # Complete pipeline summary
        end_time_dt = datetime.now()
        pipeline_summary["end_time"] = end_time_dt.isoformat()
        start_time_dt = datetime.fromisoformat(pipeline_summary["start_time"])
        pipeline_summary["total_duration_seconds"] = (end_time_dt - start_time_dt).total_seconds()
        
        # Determine overall status with enhanced logic
        perf_summary = pipeline_summary["performance_summary"]
        if perf_summary["critical_failures"] > 0:
            pipeline_summary["overall_status"] = "FAILED"
        elif perf_summary["failed_steps"] > 0:
            # Check if failures are recoverable vs critical
            total_steps = perf_summary["total_steps"]
            failed_ratio = perf_summary["failed_steps"] / total_steps if total_steps > 0 else 0

            if failed_ratio > 0.5:  # More than half failed
                pipeline_summary["overall_status"] = "FAILED"
            elif failed_ratio > 0.2:  # 20-50% failed
                pipeline_summary["overall_status"] = "PARTIAL_SUCCESS"
            else:  # Less than 20% failed
                pipeline_summary["overall_status"] = "SUCCESS_WITH_WARNINGS"
        else:
            pipeline_summary["overall_status"] = "SUCCESS"
        
        # Save pipeline summary with validation and error handling
        summary_path = args.output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving pipeline summary to: {summary_path}")

        try:
            # Validate summary structure before saving
            validate_pipeline_summary(pipeline_summary, logger)

            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=4, default=str)
            logger.info(f"Pipeline summary saved successfully")

            # Log summary statistics
            steps = pipeline_summary["steps"]
            successful = sum(1 for step in steps if step["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS"))
            failed = sum(1 for step in steps if step["status"] == "FAILED")
            logger.info(f"Summary: {successful}/{len(steps)} steps successful, {failed} failed")

        except Exception as e:
            logger.error(f"Failed to save pipeline summary: {e}")
            # Try to save a minimal summary as fallback
            try:
                minimal_summary = {
                    "start_time": pipeline_summary.get("start_time"),
                    "end_time": datetime.now().isoformat(),
                    "overall_status": "FAILED",
                    "error": str(e),
                    "arguments": pipeline_summary.get("arguments", {}),
                    "steps_count": len(pipeline_summary.get("steps", [])),
                    "performance_summary": pipeline_summary.get("performance_summary", {}),
                    "steps": pipeline_summary.get("steps", [])
                }
                with open(summary_path, 'w') as f:
                    json.dump(minimal_summary, f, indent=4, default=str)
                logger.info("Minimal summary saved as fallback")
            except Exception as fallback_error:
                logger.error(f"Failed to save even minimal summary: {fallback_error}")
        
        # Final completion summary with visual indicators
        total_duration = pipeline_summary['total_duration_seconds']
        perf_summary = pipeline_summary['performance_summary']

        completion_stats = {
            "Total Steps": len(pipeline_summary.get('steps', [])),
            "Successful": perf_summary['successful_steps'],
            "Failed": perf_summary['failed_steps'],
            "Warnings": perf_summary['warnings'],
            "Peak Memory": f"{perf_summary['peak_memory_mb']:.1f}MB",
            "Duration": f"{total_duration:.1f}s"
        }

        success = pipeline_summary['overall_status'] == "SUCCESS"
        print_completion_summary(success, total_duration, completion_stats)

        # Final status logging
        if STRUCTURED_LOGGING_AVAILABLE:
            # Log overall progress summary
            if progress_tracker:
                overall_progress = progress_tracker.get_overall_progress()
                logger.info(overall_progress)

            # Log detailed pipeline summary
            log_pipeline_summary(logger, pipeline_summary)
        else:
            # Log final status with visual indicators
            if success:
                logger.info(f"🎯 Pipeline completed successfully in {total_duration:.2f}s")
            else:
                logger.info(f"⚠️ Pipeline completed with issues in {total_duration:.2f}s")
        
        return 0 if pipeline_summary["overall_status"] == "SUCCESS" else 1
        
    except Exception as e:
        # Update pipeline summary with error
        end_time_dt = datetime.now()
        pipeline_summary["end_time"] = end_time_dt.isoformat()
        pipeline_summary["overall_status"] = "FAILED"
        start_time_dt = datetime.fromisoformat(pipeline_summary["start_time"])
        pipeline_summary["total_duration_seconds"] = (end_time_dt - start_time_dt).total_seconds()
        
        # Save pipeline summary even on error
        summary_path = args.pipeline_summary_file
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=4)
        
        log_step_error(logger, f"Pipeline failed: {str(e)}")
        return 1

def execute_pipeline_step(script_name: str, args: PipelineArguments, logger) -> Dict[str, Any]:
    """Execute a single pipeline step with comprehensive monitoring."""
    import subprocess
    import os
    import shutil
    import time
    
    # Initialize performance tracking
    start_memory = get_current_memory_usage()
    peak_memory = start_memory
    
    step_result = {
        "status": "UNKNOWN",
        "stdout": "",
        "stderr": "",
        "memory_usage_mb": 0.0,
        "peak_memory_mb": 0.0,
        "memory_delta_mb": 0.0,
        "exit_code": -1,
        "retry_count": 0,
        "prerequisite_check": True,
        "dependency_warnings": []
    }
    
    try:
        # Validate step prerequisites
        prereq_result = validate_step_prerequisites(script_name, args, logger)
        step_result["prerequisite_check"] = prereq_result["passed"]
        step_result["dependency_warnings"] = prereq_result.get("warnings", [])
        
        # Log prerequisite warnings if any
        if prereq_result.get("warnings"):
            for warning in prereq_result["warnings"]:
                logger.warning(f"Prerequisite warning for {script_name}: {warning}")
        
        # Get script path
        script_path = Path(__file__).parent / script_name
        
        # Get virtual environment Python path
        project_root = Path(__file__).parent.parent
        venv_python = project_root / ".venv" / "bin" / "python"
        
        # Use virtual environment Python if available, otherwise fall back to system Python
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        # Prepare command using the argument builder
        from utils.argument_utils import build_step_command_args
        
        cmd = build_step_command_args(
            script_name.replace('.py', ''),
            args,
            python_executable,
            script_path
        )
        
        # Log the command being executed (only in verbose mode)
        if args.verbose:
            logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute step with proper working directory (project root)
        project_root = Path(__file__).parent.parent
        # Ensure unbuffered output from children
        import os as _os
        _env = _os.environ.copy()
        _env.setdefault("PYTHONUNBUFFERED", "1")

        # Execute step with timeout
        try:
            # Use appropriate timeout for different step types
            if script_name == "2_tests.py":
                # Check if comprehensive testing is requested
                comprehensive_requested = any("--comprehensive" in str(arg) for arg in sys.argv)
                step_timeout_seconds = 1200 if comprehensive_requested else 900  # 20 min for comprehensive, 15 min for others
            elif script_name == "13_llm.py":
                # LLM processing can take longer due to API calls and analysis
                step_timeout_seconds = 600  # 10 minutes for LLM processing
            elif script_name == "22_gui.py":
                # GUI step needs more time for initialization even in headless mode
                step_timeout_seconds = 600  # 10 minutes for GUI initialization
            elif script_name in ["11_render.py", "12_execute.py"]:
                # Rendering and execution steps may take longer with multiple frameworks
                step_timeout_seconds = 300  # 5 minutes for framework operations
            else:
                step_timeout_seconds = 120  # 2 minutes default for other steps

            # Track memory during execution
            process_start_time = time.time()
            
            # Use streaming execution for real-time feedback
            # This is critical for long-running steps like tests
            from utils.execution_utils import execute_command_streaming
            
            # Execute with streaming
            # For 2_tests.py, we definitely want to see stdout/stderr in real-time
            # For others, we also benefit from real-time feedback
            result = execute_command_streaming(
                cmd,
                cwd=project_root,
                env=_env,
                timeout=step_timeout_seconds,
                print_stdout=True,  # Always print to stdout so user sees progress
                print_stderr=True,  # Always print stderr so user sees errors
                capture_output=True # Capture for logging/summary
            )
            
            # Capture final memory measurements
            end_memory = get_current_memory_usage()
            peak_memory = max(peak_memory, end_memory)
            
            step_result["stdout"] = result.get("stdout", "")
            step_result["stderr"] = result.get("stderr", "")
            step_result["exit_code"] = result.get("exit_code", -1)
            step_result["memory_usage_mb"] = end_memory
            step_result["peak_memory_mb"] = peak_memory
            step_result["memory_delta_mb"] = end_memory - start_memory

            # Log captured output if verbose (it was already printed to console)
            if args.verbose:
                 logger.info("Command completed with exit code: " + str(step_result["exit_code"]))

        except Exception as e:
            step_result["status"] = "TIMEOUT" if "Timeout" in str(e) else "FAILED"
            step_result["exit_code"] = -1
            step_result["stderr"] = str(e)
            return step_result
        
        # Determine status
        if step_result["exit_code"] == 0:
            step_result["status"] = "SUCCESS"
            # Check for any dependency warnings that might affect downstream steps
            if step_result["dependency_warnings"]:
                step_result["status"] = "SUCCESS_WITH_WARNINGS"
        else:
            # Respect the child process exit code to avoid masking failures
            step_result["status"] = "FAILED"
            # Log detailed failure information
            logger.error(f"Step {script_name} failed with exit code {step_result['exit_code']}")
            if step_result["stderr"]:
                logger.error(f"Error output: {step_result['stderr'][:500]}...")  # Limit to first 500 chars

        # Steps determine their own output directories via get_output_dir_for_script
        
        # Add recovery status to result
        if not step_result.get("recoverable"):
            step_result["recoverable"] = False
            
        return step_result
        
    except Exception as e:
        logger.error(f"Exception in execute_pipeline_step for {script_name}: {e}")
        step_result["status"] = "FAILED"
        step_result["exit_code"] = -1
        step_result["stderr"] = str(e)
        return step_result

def parse_step_list(step_str: str) -> List[int]:
    """Parse comma-separated step list into list of integers."""
    try:
        return [int(s.strip()) for s in step_str.split(',') if s.strip()]
    except ValueError:
        return []

def get_environment_info() -> Dict[str, Any]:
    """Get environment information."""
    import platform
    import os
    
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "working_directory": str(Path.cwd()),
        "user": os.getenv("USER", "unknown")
    }
    
    try:
        import psutil
        info["memory_total_gb"] = f"{psutil.virtual_memory().total / 1024**3:.1f}"
    except ImportError:
        info["memory_total_gb"] = "unavailable (psutil not installed)"
    
    try:
        import psutil
        info["disk_free_gb"] = f"{psutil.disk_usage('/').free / 1024**3:.1f}"
    except ImportError:
        info["disk_free_gb"] = "unavailable (psutil not installed)"
    
    return info


def validate_pipeline_summary(summary: dict, logger) -> None:
    """
    Validate pipeline summary structure and data integrity.

    Args:
        summary: Pipeline summary dictionary to validate
        logger: Logger instance for validation messages
    """
    required_fields = [
        "start_time", "arguments", "steps", "end_time",
        "overall_status", "total_duration_seconds",
        "environment_info", "performance_summary"
    ]

    # Check required fields
    for field in required_fields:
        if field not in summary:
            logger.warning(f"Pipeline summary missing required field: {field}")
        elif summary[field] is None and field not in ["end_time"]:
            logger.warning(f"Pipeline summary field '{field}' is None")

    # Validate steps structure
    steps = summary.get("steps", [])
    if not isinstance(steps, list):
        logger.error("Pipeline summary 'steps' should be a list")
        return

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            logger.error(f"Step {i} should be a dictionary")
            continue

        step_required = ["status", "step_number", "script_name", "description"]
        for field in step_required:
            if field not in step:
                logger.warning(f"Step {i} missing required field: {field}")

        # Validate step status values
        if "status" in step:
            valid_statuses = ["SUCCESS", "SUCCESS_WITH_WARNINGS", "PARTIAL_SUCCESS", "FAILED", "TIMEOUT"]
            if step["status"] not in valid_statuses:
                logger.warning(f"Step {i} has invalid status: {step['status']}")

        # Validate numeric fields
        numeric_fields = ["step_number", "exit_code", "retry_count", "duration_seconds",
                         "memory_usage_mb", "peak_memory_mb", "memory_delta_mb"]
        for field in numeric_fields:
            if field in step and not isinstance(step[field], (int, float)):
                logger.warning(f"Step {i} field '{field}' should be numeric, got {type(step[field])}")

    # Validate performance summary
    perf = summary.get("performance_summary", {})
    if not isinstance(perf, dict):
        logger.error("Performance summary should be a dictionary")
        return

    # Validate numeric fields
    numeric_fields = ["peak_memory_mb", "total_steps", "failed_steps", "critical_failures", "successful_steps", "warnings"]
    for field in numeric_fields:
        if field in perf:
            if not isinstance(perf[field], (int, float)):
                logger.warning(f"Performance summary field '{field}' should be numeric, got {type(perf[field])}")

    # Validate overall status
    if "overall_status" in summary:
        valid_statuses = ["SUCCESS", "SUCCESS_WITH_WARNINGS", "PARTIAL_SUCCESS", "FAILED"]
        if summary["overall_status"] not in valid_statuses:
            logger.warning(f"Invalid overall status: {summary['overall_status']}")

    # Validate timing consistency
    if "start_time" in summary and "end_time" in summary and "total_duration_seconds" in summary:
        try:
            start_dt = datetime.fromisoformat(summary["start_time"])
            end_dt = datetime.fromisoformat(summary["end_time"])
            calculated_duration = (end_dt - start_dt).total_seconds()
            reported_duration = summary["total_duration_seconds"]

            # Allow for small timing differences due to processing
            if abs(calculated_duration - reported_duration) > 1.0:
                logger.warning(f"Timing inconsistency: calculated {calculated_duration:.2f}s vs reported {reported_duration:.2f}s")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not validate timing: {e}")


# All pipeline utility functions have been moved to appropriate utils modules:
# - validate_step_prerequisites, validate_pipeline_step_sequence → utils/pipeline_validator.py  
# - get_current_memory_usage → utils/resource_manager.py
# - attempt_step_recovery and recovery functions → utils/error_recovery.py
# - generate_pipeline_health_report → utils/pipeline_monitor.py
# - generate_execution_plan → utils/pipeline_planner.py


if __name__ == "__main__":
    sys.exit(main()) 
