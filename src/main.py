#!/usr/bin/env python3
"""
Main GNN Processing Pipeline

This script orchestrates the complete 24-step GNN processing pipeline (steps 0-23).
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
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

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
from utils.error_recovery import attempt_step_recovery, is_failure_recoverable
from utils.pipeline_monitor import generate_pipeline_health_report
from utils.pipeline_validator import validate_step_prerequisites, validate_pipeline_step_sequence
from utils.pipeline_planner import generate_execution_plan

# Optional logging and progress tracking
try:
    from utils.logging_utils import (
        PipelineProgressTracker,
        VisualLoggingEnhancer,
        PipelineLogger,
        log_pipeline_summary,
        reset_progress_tracker,
        setup_step_logging
    )
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError:
    STRUCTURED_LOGGING_AVAILABLE = False

# UV utilities for environment management
try:
    from utils.uv_utils import (
        check_uv_available,
        ensure_uv_environment,
        get_uv_environment_info,
        run_python_script
    )
    UV_AVAILABLE = True
except ImportError:
    UV_AVAILABLE = False
    logging.warning("UV utilities not available - falling back to standard Python execution")

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
    
    # Setup logging
    if STRUCTURED_LOGGING_AVAILABLE:
        logger = setup_step_logging("pipeline", args.verbose, enable_structured=True)
        # Reset progress tracker for new pipeline run
        reset_progress_tracker()
    else:
        logger = setup_step_logging("pipeline", args.verbose)
    
    # Check UV environment if available
    if UV_AVAILABLE:
        logger.info("Checking UV environment...")
        
        if not check_uv_available():
            logger.warning("UV is not available - using standard Python execution")
        else:
            # Ensure UV environment is ready
            if not ensure_uv_environment(
                dev=True,
                extras=["llm", "visualization", "audio", "gui"],
                verbose=args.verbose
            ):
                logger.warning("UV environment setup failed - continuing with standard Python execution")
            else:
                logger.info("UV environment is ready")
    
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
             "total_steps": 24,
            "failed_steps": 0,
            "critical_failures": 0,
            "successful_steps": 0,
            "warnings": 0
        }
    }
    
    try:
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
        
        # Define pipeline steps
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
            ("23_report.py", "Report generation")
        ]
        
        # Handle step filtering with automatic dependency resolution
        steps_to_execute = pipeline_steps
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
        
        # Initialize progress tracker if enhanced logging is available
        progress_tracker = None
        if STRUCTURED_LOGGING_AVAILABLE:
            progress_tracker = PipelineProgressTracker(len(steps_to_execute))
        
        # Validate step sequence before execution
        sequence_validation = validate_pipeline_step_sequence(steps_to_execute, logger)
        if sequence_validation["warnings"]:
            for warning in sequence_validation["warnings"]:
                logger.warning(f"Pipeline sequence: {warning}")
        if sequence_validation["recommendations"]:
            for rec in sequence_validation["recommendations"]:
                logger.info(f"Recommendation: {rec}")
        
        # Execute each step
        for step_number, (script_name, description) in enumerate(steps_to_execute, 1):
            step_start_time = time.time()
            step_start_datetime = datetime.now()
            
            # Step start logging with progress tracking
            if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
                progress_header = progress_tracker.start_step(step_number, description)
                logger.info(progress_header)
                # Use structured logging functions that support additional parameters
                from utils.logging_utils import log_step_start as structured_log_step_start
                structured_log_step_start(logger, f"Starting {description}",
                                      step_number=step_number,
                                      total_steps=len(steps_to_execute),
                                      script_name=script_name)
            else:
                logger.info(f"Executing step {step_number}: {description}")
            
            # Execute the step
            step_result = execute_pipeline_step(script_name, args, logger)
            
            # Calculate step duration
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_end_datetime = datetime.now()
            
            # Update step result with timing information
            step_result.update({
                "step_number": step_number,
                "script_name": script_name,
                "description": description,
                "start_time": step_start_datetime.isoformat(),
                "end_time": step_end_datetime.isoformat(),
                "duration_seconds": step_duration
            })
            
            # Check for warnings in both stdout and stderr (case-insensitive and symbol-aware)
            combined_output = f"{step_result.get('stdout', '')}\n{step_result.get('stderr', '')}"
            has_warning = ("WARNING" in combined_output) or ("⚠️" in combined_output) or ("warning" in combined_output.lower())

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
            
            # Update peak memory usage
            step_memory = step_result.get("memory_usage_mb", 0.0)
            if step_memory > pipeline_summary["performance_summary"]["peak_memory_mb"]:
                pipeline_summary["performance_summary"]["peak_memory_mb"] = step_memory
            
            # Update total steps count
            pipeline_summary["performance_summary"]["total_steps"] = len(steps_to_execute)
            
            # Step completion logging with progress tracking
            if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
                status_for_logging = step_result["status"]
                completion_summary = progress_tracker.complete_step(step_number, status_for_logging, step_duration)
                logger.info(completion_summary)

                # Use structured logging functions that support additional parameters
                if status_for_logging.startswith("SUCCESS"):
                    from utils.logging_utils import log_step_success as structured_log_step_success
                    structured_log_step_success(logger, f"{description} completed",
                                        step_number=step_number,
                                        duration=step_duration,
                                            status=status_for_logging)
            else:
                status_for_logging = step_result["status"]
                if str(status_for_logging).startswith("SUCCESS"):
                    logger.info(f"✅ Step {step_number} completed with {status_for_logging} in {step_duration:.2f}s")
                elif status_for_logging == "PARTIAL_SUCCESS":
                    logger.warning(f"⚠️ Step {step_number} completed with PARTIAL_SUCCESS in {step_duration:.2f}s")
                else:
                    logger.error(f"❌ Step {step_number} failed with status: {status_for_logging}")
        
        # Complete pipeline summary
        pipeline_summary["end_time"] = datetime.now().isoformat()
        pipeline_summary["total_duration_seconds"] = time.time() - time.mktime(
            datetime.fromisoformat(pipeline_summary["start_time"]).timetuple()
        )
        
        # Determine overall status
        if pipeline_summary["performance_summary"]["critical_failures"] > 0:
            pipeline_summary["overall_status"] = "FAILED"
        elif pipeline_summary["performance_summary"]["failed_steps"] > 0:
            pipeline_summary["overall_status"] = "PARTIAL_SUCCESS"
        else:
            pipeline_summary["overall_status"] = "SUCCESS"
        
        # Save pipeline summary
        summary_path = args.pipeline_summary_file
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving pipeline summary to: {summary_path}")
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=4)
        logger.info(f"Pipeline summary saved successfully")
        
        # Final status logging
        if STRUCTURED_LOGGING_AVAILABLE:
            # Log overall progress summary
            if progress_tracker:
                overall_progress = progress_tracker.get_overall_progress()
                logger.info(overall_progress)

            # Log detailed pipeline summary
            log_pipeline_summary(logger, pipeline_summary)
        else:
            # Log final status
            if pipeline_summary["overall_status"] == "SUCCESS":
                log_step_success(logger, f"Pipeline completed successfully in {pipeline_summary['total_duration_seconds']:.2f}s")
            else:
                log_step_error(logger, f"Pipeline completed with status: {pipeline_summary['overall_status']}")
        
        return 0 if pipeline_summary["overall_status"] == "SUCCESS" else 1
        
    except Exception as e:
        # Update pipeline summary with error
        pipeline_summary["end_time"] = datetime.now().isoformat()
        pipeline_summary["overall_status"] = "FAILED"
        pipeline_summary["total_duration_seconds"] = time.time() - time.mktime(
            datetime.fromisoformat(pipeline_summary["start_time"]).timetuple()
        )
        
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
            # Use appropriate timeout for the tests step based on test mode
            if script_name == "2_tests.py":
                # Check if comprehensive testing is requested
                comprehensive_requested = any("--comprehensive" in str(arg) for arg in sys.argv)
                step_timeout_seconds = 900 if comprehensive_requested else 600  # 15 min for comprehensive, 10 min for others
            else:
                step_timeout_seconds = 60

            # Track memory during execution
            process_start_time = time.time()
            
            result = subprocess.run(
                cmd,
                cwd=project_root,
                env=_env,
                capture_output=True,
                text=True,
                timeout=step_timeout_seconds
            )
            
            # Capture final memory measurements
            end_memory = get_current_memory_usage()
            peak_memory = max(peak_memory, end_memory)
            
            step_result["stdout"] = result.stdout
            step_result["stderr"] = result.stderr
            step_result["exit_code"] = result.returncode
            step_result["memory_usage_mb"] = end_memory
            step_result["peak_memory_mb"] = peak_memory
            step_result["memory_delta_mb"] = end_memory - start_memory

            # Log output if verbose
            if args.verbose:
                if result.stdout:
                    logger.info(f"Step output:\n{result.stdout}")
                if result.stderr:
                    logger.warning(f"Step errors:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            step_result["status"] = "TIMEOUT"
            step_result["exit_code"] = -1
            return step_result
        
        # Determine status with enhanced logic
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

# All pipeline utility functions have been moved to appropriate utils modules:
# - validate_step_prerequisites, validate_pipeline_step_sequence → utils/pipeline_validator.py  
# - get_current_memory_usage → utils/resource_manager.py
# - attempt_step_recovery and recovery functions → utils/error_recovery.py
# - generate_pipeline_health_report → utils/pipeline_monitor.py
# - generate_execution_plan → utils/pipeline_planner.py


if __name__ == "__main__":
    sys.exit(main()) 
