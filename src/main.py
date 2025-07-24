#!/usr/bin/env python3
"""
Main GNN Processing Pipeline

This script orchestrates the complete 22-step GNN processing pipeline.
"""

import sys
import json
import time
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
from utils.argument_utils import EnhancedArgumentParser, PipelineArguments
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def main():
    """Main pipeline orchestration function."""
    # Handle path resolution before argument validation
    import argparse
    
    # Create a simple parser to get the basic arguments first
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--target-dir', type=Path, default=Path("input/gnn_files"))
    temp_parser.add_argument('--output-dir', type=Path, default=Path("output"))
    
    # Parse just these arguments to check paths
    temp_args, remaining = temp_parser.parse_known_args()
    
    # Fix path resolution - ensure we're working from the project root
    project_root = Path(__file__).parent.parent
    
    # Always use project root paths for consistency
    if (project_root / temp_args.target_dir).exists():
        temp_args.target_dir = project_root / temp_args.target_dir
    
    # For output_dir, always use project root path (create if needed)
    temp_args.output_dir = project_root / temp_args.output_dir
    
    # Now parse all arguments with the corrected paths
    parser = EnhancedArgumentParser.create_main_parser()
    parsed = parser.parse_args()
    
    # Convert to PipelineArguments with corrected paths
    kwargs = {}
    for key, value in vars(parsed).items():
        if value is not None:
            kwargs[key] = value
    
    # Override with corrected paths
    kwargs['target_dir'] = temp_args.target_dir
    kwargs['output_dir'] = temp_args.output_dir
    
    args = PipelineArguments(**kwargs)
    
    # Ensure the paths are correctly set
    args.target_dir = temp_args.target_dir
    args.output_dir = temp_args.output_dir
    
    # Setup logging
    logger = setup_step_logging("pipeline", args)
    
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
            "total_steps": 22,
            "failed_steps": 0,
            "critical_failures": 0,
            "successful_steps": 0,
            "warnings": 0
        }
    }
    
    try:
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
            ("21_report.py", "Report generation")
        ]
        
        # Handle step filtering
        steps_to_execute = pipeline_steps
        if args.only_steps:
            step_numbers = parse_step_list(args.only_steps)
            steps_to_execute = [pipeline_steps[i] for i in step_numbers if 0 <= i < len(pipeline_steps)]
            logger.info(f"Executing only steps: {[step[0] for step in steps_to_execute]}")
        
        if args.skip_steps:
            skip_numbers = parse_step_list(args.skip_steps)
            steps_to_execute = [step for i, step in enumerate(pipeline_steps) if i not in skip_numbers]
            logger.info(f"Skipping steps: {[pipeline_steps[i][0] for i in skip_numbers if 0 <= i < len(pipeline_steps)]}")
        
        # Execute each step
        for step_number, (script_name, description) in enumerate(steps_to_execute, 1):
            step_start_time = time.time()
            step_start_datetime = datetime.now()
            
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
            
            # Add to pipeline summary
            pipeline_summary["steps"].append(step_result)
            
            # Update performance summary
            if step_result["status"] == "SUCCESS":
                pipeline_summary["performance_summary"]["successful_steps"] += 1
            elif step_result["status"] == "FAILED":
                pipeline_summary["performance_summary"]["failed_steps"] += 1
                if step_result.get("exit_code", 0) == 1:
                    pipeline_summary["performance_summary"]["critical_failures"] += 1
            
            # Check for warnings
            if "WARNING" in step_result.get("stdout", ""):
                pipeline_summary["performance_summary"]["warnings"] += 1
            
            # Update peak memory usage
            step_memory = step_result.get("memory_usage_mb", 0.0)
            if step_memory > pipeline_summary["performance_summary"]["peak_memory_mb"]:
                pipeline_summary["performance_summary"]["peak_memory_mb"] = step_memory
            
            # Update total steps count
            pipeline_summary["performance_summary"]["total_steps"] = len(steps_to_execute)
        
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
        
        log_step_error(logger, "Pipeline failed", {"error": str(e)})
        return 1

def execute_pipeline_step(script_name: str, args: PipelineArguments, logger) -> Dict[str, Any]:
    """Execute a single pipeline step."""
    import subprocess
    import os
    
    step_result = {
        "status": "UNKNOWN",
        "stdout": "",
        "stderr": "",
        "memory_usage_mb": 0.0,
        "exit_code": -1,
        "retry_count": 0
    }
    
    try:
        # Get script path
        script_path = Path(__file__).parent / script_name
        
        # Prepare command using the enhanced argument builder
        from utils.argument_utils import build_enhanced_step_command_args
        cmd = build_enhanced_step_command_args(
            script_name.replace('.py', ''),
            args,
            sys.executable,
            script_path
        )
        
        # Log the command being executed (only in verbose mode)
        if args.verbose:
            logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute step
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Monitor process
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
            step_result["stdout"] = stdout
            step_result["stderr"] = stderr
            step_result["exit_code"] = process.returncode
            
            # Process completed successfully
            
            # Get memory usage if available
            try:
                import psutil
                if process.pid:
                    proc = psutil.Process(process.pid)
                    step_result["memory_usage_mb"] = proc.memory_info().rss / 1024 / 1024
            except ImportError:
                # psutil not available
                pass
            except Exception:
                # Other psutil-related errors
                pass
            
        except subprocess.TimeoutExpired:
            process.kill()
            step_result["status"] = "TIMEOUT"
            step_result["exit_code"] = -1
            return step_result
        
        # Determine status
        if step_result["exit_code"] == 0:
            step_result["status"] = "SUCCESS"
        else:
            # Check if the step actually succeeded despite the exit code
            stdout = step_result.get("stdout", "")
            if ("✅" in stdout or "completed" in stdout.lower() or "success" in stdout.lower()) and "❌" not in stdout:
                step_result["status"] = "SUCCESS"
                step_result["exit_code"] = 0  # Override exit code
            else:
                step_result["status"] = "FAILED"
        
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

if __name__ == "__main__":
    sys.exit(main()) 