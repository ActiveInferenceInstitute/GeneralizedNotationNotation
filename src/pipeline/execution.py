#!/usr/bin/env python3
"""
Pipeline Execution Utilities

Handles the core execution logic for running pipeline steps,
monitoring performance, and managing subprocess execution with enhanced visual logging.
"""

import os
import sys
import subprocess
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import argparse

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .config import (
    get_pipeline_config,
    get_output_dir_for_script
)

from utils.logging_utils import (
    log_step_start, log_step_success, log_step_warning, log_step_error,
    log_pipeline_summary, reset_progress_tracker, get_progress_summary,
    VisualLoggingEnhancer, performance_tracker, EnhancedPipelineLogger
)

logger = logging.getLogger(__name__)

class StepExecutionResult:
    """Encapsulates the result of executing a pipeline step."""
    
    def __init__(self, step_number: int, script_name: str):
        self.step_number = step_number
        self.script_name = script_name
        self.status = "UNKNOWN"
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.duration_seconds: Optional[float] = None
        self.details = ""
        self.stdout = ""
        self.stderr = ""
        self.memory_usage_mb: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.retry_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_number": self.step_number,
            "script_name": self.script_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "memory_usage_mb": self.memory_usage_mb,
            "exit_code": self.exit_code,
            "retry_count": self.retry_count
        }

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    
    try:
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0

def build_command_args(script_name: str, script_path: Path, args, python_executable: str) -> List[str]:
    """Build command line arguments for a pipeline step."""
    full_args = []
    
    # Scripts that don't accept --target-dir
    no_target_dir_scripts = []  # All scripts now accept --target-dir
    
    # Build common arguments based on script compatibility
    common_args = []
    
    # --target-dir (not supported by all scripts)
    if script_name not in no_target_dir_scripts:
        target_dir = getattr(args, 'target_dir', None)
        if target_dir is not None:
            common_args.append(('--target-dir', target_dir))
    
    # --output-dir (supported by all scripts)
    output_dir = getattr(args, 'output_dir', None)
    if output_dir is not None:
        common_args.append(('--output-dir', output_dir))
    
    # --verbose (supported by all scripts)
    verbose = getattr(args, 'verbose', False)
    if verbose:
        common_args.append(('--verbose', verbose))
    
    # --recursive (supported by most scripts)
    if hasattr(args, 'recursive') and script_name not in ['2_setup.py', '3_tests.py']:
        recursive = getattr(args, 'recursive', True)  # Default to True
        if recursive:
            common_args.append(('--recursive', recursive))
    
    for flag, value in common_args:
        if value is not None:
            if isinstance(value, bool):
                if value:
                    full_args.append(flag)
            else:
                full_args.extend([flag, str(value)])
    
    # Add step-specific arguments based on script name
    if script_name in ['1_gnn.py']:
        # GNN processing step supports recursive (already handled above)
        pass
    
    if 'type_checker' in script_name:
        # Already handled recursive above
        if getattr(args, 'strict', False):
            full_args.append('--strict')
        if getattr(args, 'estimate_resources', True):
            full_args.append('--estimate-resources')
    
    if 'ontology' in script_name:
        ontology_file = getattr(args, 'ontology_terms_file', None)
        if ontology_file:
            full_args.extend(['--ontology-terms-file', str(ontology_file)])
    
    if 'llm' in script_name:
        llm_tasks = getattr(args, 'llm_tasks', None)
        if llm_tasks:
            full_args.extend(['--llm-tasks', str(llm_tasks)])
        llm_timeout = getattr(args, 'llm_timeout', None)
        if llm_timeout:
            full_args.extend(['--llm-timeout', str(llm_timeout)])
    
    if 'setup' in script_name:
        if getattr(args, 'recreate_venv', False):
            full_args.append('--recreate-venv')
        if getattr(args, 'dev', False):
            full_args.append('--dev')

    return [python_executable, str(script_path)] + full_args

def validate_step_dependencies(step_name: str) -> Tuple[bool, List[str]]:
    """
    Validate that a step's dependencies are available before execution.
    Uses subprocess validation to ensure we're checking the correct Python environment.
    
    Returns:
        Tuple of (is_valid, missing_dependencies)
    """
    missing_deps = []
    
    # Define step-specific dependency requirements
    step_dependencies = {
        "9_render.py": ["render", "render.pymdp.pymdp_renderer", "render.rxinfer.gnn_parser"],
        "10_execute.py": ["execute", "execute.pymdp_runner"],
        "11_llm.py": ["llm", "llm.providers"],
        "12_site.py": []  # Site generator dependency handled internally
    }
    
    required_deps = step_dependencies.get(step_name, [])
    
    # Get the Python executable that will be used for pipeline execution
    import sys
    from pathlib import Path
    
    # Try to find virtual environment Python (same logic as main.py)
    current_dir = Path(__file__).resolve().parent.parent
    venv_candidates = [
        current_dir / ".venv" / "bin" / "python",
        current_dir.parent / "venv" / "bin" / "python", 
        current_dir.parent / ".venv" / "bin" / "python",
    ]
    
    python_executable = sys.executable  # Default fallback
    for venv_path in venv_candidates:
        if venv_path.exists():
            python_executable = str(venv_path)
            break
    
    # Use subprocess to check dependencies in the target environment
    import subprocess
    for dep in required_deps:
        try:
            result = subprocess.run(
                [python_executable, "-c", f"import {dep}"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Import failed"
                missing_deps.append(f"{dep}: {error_msg}")
                
        except subprocess.TimeoutExpired:
            missing_deps.append(f"{dep}: Import check timed out")
        except Exception as e:
            missing_deps.append(f"{dep}: Validation error - {e}")
    
    return len(missing_deps) == 0, missing_deps

def execute_pipeline_step(script_name: str, step_number: int, total_steps: int,
                         python_executable: str, target_dir: Path, output_dir: Path,
                         args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    """
    Execute a single pipeline step with enhanced visual feedback and error handling.
    
    Args:
        script_name: Name of the script to execute
        step_number: Current step number (1-based)
        total_steps: Total number of steps in pipeline
        python_executable: Python executable to use
        target_dir: Target directory for processing
        output_dir: Output directory for results
        args: Parsed command-line arguments
        logger: Logger instance for output
        
    Returns:
        Dictionary containing execution results and metadata
    """
    start_time = time.time()
    initial_memory = get_memory_usage_mb()
    
    # Create execution result tracker
    result = StepExecutionResult(step_number, script_name)
    result.start_time = datetime.datetime.now().isoformat()
    
    # Get pipeline configuration for this step
    pipeline_config = get_pipeline_config()
    step_config = pipeline_config.get_step_config(script_name)
    is_required = step_config.required if step_config else True
    step_timeout = step_config.timeout if step_config else None
    
    # Enhanced step start logging with visual progress
    log_step_start(
        logger, 
        f"Starting {step_config.description if step_config else script_name}",
        step_number=step_number,
        total_steps=total_steps,
        script_name=script_name,
        is_required=is_required,
        timeout=step_timeout
    )
    
    try:
        # Validate step dependencies with enhanced feedback
        is_valid, missing_deps = validate_step_dependencies(script_name)
        if not is_valid:
            EnhancedPipelineLogger.log_structured(
                logger, logging.WARNING,
                f"⚠️ Step {step_number}/{total_steps} ({script_name}) has missing dependencies",
                step_number=step_number,
                missing_dependencies=missing_deps,
                event_type="dependency_warning"
            )
            
            for dep in missing_deps:
                logger.warning(f"   - {dep}")
            
            if is_required:
                result.status = "FAILED_DEPENDENCIES"
                result.details = f"Missing required dependencies: {'; '.join(missing_deps)}"
                log_step_error(
                    logger,
                    f"Required step {script_name} cannot proceed due to missing dependencies",
                    step_number=step_number,
                    dependencies=missing_deps
                )
                return result.to_dict()
            else:
                logger.info(f"⏭️ Step {step_number}/{total_steps} ({script_name}) has missing dependencies but is non-critical, continuing with limited functionality")
        
        # Build command with enhanced argument handling
        script_path = Path(__file__).parent.parent / script_name
        command = build_command_args(script_name, script_path, args, python_executable)
        
        # Log command execution with enhanced formatting
        command_display = ' '.join(command)
        if len(command_display) > 100:
            command_display = command_display[:97] + "..."
        
        EnhancedPipelineLogger.log_structured(
            logger, logging.DEBUG,
            f"📋 Executing command: {command_display}",
            step_number=step_number,
            script_name=script_name,
            command=command,
            timeout=step_timeout,
            event_type="command_start"
        )
        
        # Special handling for long-running steps
        if step_timeout and step_timeout > 60:
            logger.info(f"⏳ {step_config.description if step_config else 'Processing'} (timeout: {step_timeout}s). This may take several minutes...")
        
        # Execute subprocess with enhanced monitoring
        with performance_tracker.track_operation(f"step_{step_number}_{script_name}", 
                                                {"step_number": step_number, "script_name": script_name}):
            
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=step_timeout,
                cwd=Path(__file__).parent.parent  # Run from src directory
            )
            
            # Calculate execution metrics
            duration = time.time() - start_time
            final_memory = get_memory_usage_mb()
            memory_delta = final_memory - initial_memory
            
            # Store results
            result.duration_seconds = duration
            result.memory_usage_mb = final_memory
            result.exit_code = process.returncode
            result.stdout = process.stdout
            result.stderr = process.stderr
            result.end_time = datetime.datetime.now().isoformat()
            
            # Enhanced result processing based on exit code
            if process.returncode == 0:
                result.status = "SUCCESS"
                result.details = f"Completed successfully in {VisualLoggingEnhancer.format_duration(duration)}"
                
                log_step_success(
                    logger,
                    f"Step {step_number}/{total_steps} ({script_name}) completed successfully",
                    step_number=step_number,
                    duration=duration,
                    memory_mb=final_memory,
                    memory_delta_mb=memory_delta,
                    exit_code=process.returncode
                )
                
            elif process.returncode == 2:
                # Exit code 2 typically indicates success with warnings
                result.status = "SUCCESS_WITH_WARNINGS"
                result.details = f"Completed with warnings in {VisualLoggingEnhancer.format_duration(duration)}"
                
                log_step_warning(
                    logger,
                    f"Step {step_number}/{total_steps} ({script_name}) completed with warnings",
                    step_number=step_number,
                    duration=duration,
                    memory_mb=final_memory,
                    exit_code=process.returncode
                )
                
            else:
                # Non-zero exit code indicates failure
                result.status = "FAILED" if is_required else "FAILED_NON_CRITICAL"
                error_preview = process.stderr[:200] + "..." if len(process.stderr) > 200 else process.stderr
                result.details = f"Failed with exit code {process.returncode}: {error_preview}"
                
                if is_required:
                    log_step_error(
                        logger,
                        f"Critical step {step_number}/{total_steps} ({script_name}) failed",
                        step_number=step_number,
                        duration=duration,
                        exit_code=process.returncode,
                        error_preview=error_preview
                    )
                else:
                    log_step_warning(
                        logger,
                        f"Step {step_number}/{total_steps} ({script_name}) failed but is non-critical",
                        step_number=step_number,
                        duration=duration,
                        exit_code=process.returncode
                    )
    
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        result.duration_seconds = duration
        result.status = "TIMEOUT"
        result.details = f"Timed out after {step_timeout}s"
        result.end_time = datetime.datetime.now().isoformat()
        
        log_step_error(
            logger,
            f"Step {step_number}/{total_steps} ({script_name}) timed out",
            step_number=step_number,
            duration=duration,
            timeout=step_timeout,
            event_type="timeout_error"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        result.duration_seconds = duration
        result.status = "ERROR"
        result.details = f"Unexpected error: {str(e)}"
        result.end_time = datetime.datetime.now().isoformat()
        
        log_step_error(
            logger,
            f"Step {step_number}/{total_steps} ({script_name}) encountered unexpected error",
            step_number=step_number,
            duration=duration,
            error=str(e),
            error_type=type(e).__name__,
            event_type="execution_error"
        )
    
    # Return comprehensive execution results
    return result.to_dict()

def execute_pipeline_steps(scripts: List[Tuple[int, str]], python_executable: str, 
                          target_dir: Path, output_dir: Path, args: argparse.Namespace,
                          logger: logging.Logger) -> Dict[str, Any]:
    """
    Execute multiple pipeline steps with enhanced progress tracking and visual feedback.
    
    Args:
        scripts: List of (step_number, script_name) tuples
        python_executable: Python executable to use
        target_dir: Target directory for processing
        output_dir: Output directory for results
        args: Parsed command-line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing overall execution results and statistics
    """
    # Reset progress tracker for new pipeline run
    reset_progress_tracker()
    
    # Initialize execution tracking
    total_steps = len(scripts)
    pipeline_start_time = time.time()
    initial_memory = get_memory_usage_mb()
    
    results = []
    success_count = 0
    warning_count = 0
    failure_count = 0
    critical_failure = False
    
    # Enhanced pipeline start logging
    EnhancedPipelineLogger.log_structured(
        logger, logging.INFO,
        f"🚀 Starting pipeline execution: {total_steps} steps",
        total_steps=total_steps,
        target_dir=str(target_dir),
        output_dir=str(output_dir),
        python_executable=python_executable,
        event_type="pipeline_start"
    )
    
    # Execute each step with enhanced feedback
    for step_number, script_name in scripts:
        step_result = execute_pipeline_step(
            script_name, step_number, total_steps,
            python_executable, target_dir, output_dir, args, logger
        )
        
        results.append(step_result)
        
        # Update counters based on step result
        status = step_result.get('status', 'UNKNOWN')
        if status == 'SUCCESS':
            success_count += 1
        elif 'WARNING' in status:
            warning_count += 1
        elif 'FAILED' in status:
            failure_count += 1
            
            # Check if this was a critical failure
            pipeline_config = get_pipeline_config()
            step_config = pipeline_config.get_step_config(script_name)
            if step_config and step_config.required:
                critical_failure = True
                EnhancedPipelineLogger.log_structured(
                    logger, logging.ERROR,
                    f"💥 Critical step failure: {script_name}",
                    step_number=step_number,
                    script_name=script_name,
                    status=status,
                    event_type="critical_failure"
                )
                break  # Stop pipeline on critical failure
    
    # Calculate final metrics
    pipeline_duration = time.time() - pipeline_start_time
    final_memory = get_memory_usage_mb()
    memory_delta = final_memory - initial_memory
    
    # Determine overall pipeline status
    if critical_failure:
        overall_status = "FAILED"
    elif failure_count > 0:
        overall_status = "COMPLETED_WITH_ERRORS"
    elif warning_count > 0:
        overall_status = "COMPLETED_WITH_WARNINGS"
    else:
        overall_status = "SUCCESS"
    
    # Create comprehensive summary
    summary = {
        "status": overall_status,
        "total_steps": total_steps,
        "executed_steps": len(results),
        "success_count": success_count,
        "warning_count": warning_count,
        "failure_count": failure_count,
        "critical_failure": critical_failure,
        "total_duration_seconds": pipeline_duration,
        "memory_usage_mb": final_memory,
        "memory_delta_mb": memory_delta,
        "start_time": datetime.datetime.fromtimestamp(pipeline_start_time).isoformat(),
        "end_time": datetime.datetime.now().isoformat(),
        "steps": results,
        "performance_summary": performance_tracker.get_summary()
    }
    
    # Log enhanced pipeline summary
    log_pipeline_summary(logger, summary)
    
    # Final status message with enhanced formatting
    status_icon = "✅" if overall_status == "SUCCESS" else "⚠️" if "WARNING" in overall_status else "❌"
    status_color = "GREEN" if overall_status == "SUCCESS" else "YELLOW" if "WARNING" in overall_status else "RED"
    
    final_message = f"{status_icon} Pipeline execution {overall_status.lower().replace('_', ' ')}"
    final_message_colored = VisualLoggingEnhancer.colorize(final_message, status_color, True)
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.INFO if overall_status == "SUCCESS" else logging.WARNING if "WARNING" in overall_status else logging.ERROR,
        final_message_colored,
        **summary,
        event_type="pipeline_complete"
    )
    
    return summary 

def run_pipeline(target_dir: str, output_dir: str, steps: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run the GNN pipeline with specified parameters.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory for pipeline outputs
        steps: List of step numbers to run (e.g., ['1', '2', '3'])
    
    Returns:
        Dictionary with pipeline execution results
    """
    try:
        from pathlib import Path
        import sys
        
        # Convert paths to Path objects
        target_path = Path(target_dir)
        output_path = Path(output_dir)
        
        # Ensure directories exist
        target_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find Python executable
        python_executable = sys.executable
        
        # Create mock args object
        class MockArgs:
            def __init__(self):
                self.target_dir = str(target_path)
                self.output_dir = str(output_path)
                self.verbose = True
                self.recursive = True
                self.strict = False
                self.estimate_resources = True
        
        args = MockArgs()
        
        # Get all available scripts
        src_dir = Path(__file__).parent.parent
        scripts = []
        for script_file in src_dir.glob("*.py"):
            if script_file.name.startswith(("1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_", "9_", "10_", "11_", "12_", "13_")):
                step_num = int(script_file.name.split("_")[0])
                scripts.append((step_num, script_file.name))
        
        # Sort by step number
        scripts.sort(key=lambda x: x[0])
        
        # Filter by requested steps
        if steps:
            requested_steps = set(int(s) for s in steps)
            scripts = [(num, name) for num, name in scripts if num in requested_steps]
        
        # Execute pipeline
        results = execute_pipeline_steps(scripts, python_executable, target_path, output_path, args, logger)
        
        return {
            "success": True,
            "target_dir": str(target_path),
            "output_dir": str(output_path),
            "steps_executed": len(scripts),
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_pipeline_status() -> Dict[str, Any]:
    """
    Get the current status of the pipeline.
    
    Returns:
        Dictionary with pipeline status information
    """
    try:
        from .config import get_pipeline_config, STEP_METADATA
        
        config = get_pipeline_config()
        
        status = {
            "pipeline_version": "1.0.0",
            "total_steps": len(STEP_METADATA),
            "available_steps": [],
            "configuration": config
        }
        
        # Add step information
        for step_num, step_info in STEP_METADATA.items():
            status["available_steps"].append({
                "step_number": step_num,
                "name": step_info.get("name", f"Step {step_num}"),
                "description": step_info.get("description", ""),
                "required": step_info.get("required", False),
                "module": step_info.get("module", "")
            })
        
        return status
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def validate_pipeline_config() -> Dict[str, Any]:
    """
    Validate the pipeline configuration.
    
    Returns:
        Dictionary with validation results
    """
    try:
        from .config import get_pipeline_config, STEP_METADATA
        
        config = get_pipeline_config()
        validation_results = {
            "config_valid": True,
            "missing_configs": [],
            "invalid_configs": [],
            "warnings": []
        }
        
        # Check required configuration fields
        required_fields = ["output_base_dir", "log_level"]
        for field in required_fields:
            if field not in config:
                validation_results["missing_configs"].append(field)
                validation_results["config_valid"] = False
        
        # Check step metadata
        if not STEP_METADATA:
            validation_results["warnings"].append("No step metadata found")
        
        # Check output directory
        output_dir = config.get("output_base_dir")
        if output_dir:
            from pathlib import Path
            output_path = Path(output_dir)
            if not output_path.exists():
                validation_results["warnings"].append(f"Output directory does not exist: {output_dir}")
        
        return validation_results
        
    except Exception as e:
        return {
            "config_valid": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_pipeline_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the pipeline.
    
    Returns:
        Dictionary with pipeline information
    """
    info = {
        "pipeline_name": "GNN Processing Pipeline",
        "version": "1.0.0",
        "description": "Comprehensive pipeline for processing GNN models",
        "total_steps": 13,
        "modules": [
            "gnn", "setup", "tests", "type_checker", "export", "visualization",
            "mcp", "ontology", "render", "execute", "llm", "site", "sapf"
        ]
    }
    
    # Add step descriptions
    info["steps"] = {
        "1": "GNN file discovery and parsing",
        "2": "Environment setup and dependency installation",
        "3": "Test suite execution",
        "4": "Type checking and validation",
        "5": "Multi-format export",
        "6": "Visualization generation",
        "7": "MCP tool registration",
        "8": "Ontology processing",
        "9": "Code rendering for simulators",
        "10": "Model execution",
        "11": "LLM-enhanced analysis",
        "12": "Site generation",
        "13": "SAPF audio generation"
    }
    
    return info


def create_pipeline_config(config_path: str, **kwargs) -> bool:
    """
    Create a new pipeline configuration file.
    
    Args:
        config_path: Path where the configuration should be created
        **kwargs: Configuration parameters
    
    Returns:
        True if configuration created successfully
    """
    try:
        from pathlib import Path
        import json
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        default_config = {
            "output_base_dir": "output",
            "log_level": "INFO",
            "max_workers": 4,
            "timeout_seconds": 300,
            "retry_attempts": 3
        }
        
        # Update with provided parameters
        default_config.update(kwargs)
        
        # Write configuration file
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create pipeline config: {e}")
        return False 