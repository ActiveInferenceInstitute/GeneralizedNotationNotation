"""
Pipeline Execution Utilities

Handles the core execution logic for running pipeline steps,
monitoring performance, and managing subprocess execution.
"""

import os
import sys
import subprocess
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .config import (
    get_step_timeout, 
    is_critical_step, 
    get_output_dir_for_script,
    ARG_PROPERTIES,
    SCRIPT_ARG_SUPPORT
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
    supported_args = SCRIPT_ARG_SUPPORT.get(script_name, [])
    
    for arg_key in supported_args:
        if arg_key not in ARG_PROPERTIES:
            continue

        prop = ARG_PROPERTIES[arg_key]
        if not hasattr(args, arg_key):
            continue
        
        value = getattr(args, arg_key)

        # Special handling for output_dir to route to script-specific subdirectories
        if arg_key == 'output_dir':
            output_dir_val = get_output_dir_for_script(script_name, Path(args.output_dir))
            full_args.extend([prop['flag'], str(output_dir_val)])
            continue

        if prop['type'] == 'store_true':
            if value:
                full_args.append(prop['flag'])
        elif prop['type'] == 'bool_optional':
            if value is True:
                full_args.append(prop['flag'])
        elif prop['type'] == 'value':
            if value is not None:
                full_args.extend([prop['flag'], str(value)])

    return [python_executable, str(script_path)] + full_args

def execute_pipeline_step(
    step_info: Dict[str, Union[int, str, Path]], 
    step_index: int, 
    total_steps: int,
    args,
    python_executable: str
) -> StepExecutionResult:
    """Execute a single pipeline step and return detailed results."""
    
    script_num = step_info['num']
    script_name = str(step_info['basename'])
    script_path = step_info['path']
    
    result = StepExecutionResult(script_num, script_name)
    step_timeout = get_step_timeout(script_name, args)
    is_critical = is_critical_step(script_name)
    
    step_header = f"Step {step_index}/{total_steps} ({script_num}: {script_name})"
    logger.info(f"🚀 Starting {step_header}")
    
    # Build command
    cmd = build_command_args(script_name, script_path, args, python_executable)
    logger.debug(f"📋 Executing command: {' '.join(cmd)}")
    
    # Performance monitoring setup
    start_time = time.time()
    result.start_time = datetime.datetime.now().isoformat()
    initial_memory = get_memory_usage_mb()
    
    # Provide user feedback for slow steps
    if script_name == "2_setup.py":
        logger.info(f"⏳ Setting up environment and dependencies (timeout: {step_timeout}s). This may take several minutes...")
        sys.stdout.flush()
    
    try:
        # Execute subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='replace'
        )
        
        try:
            stdout_data, stderr_data = process.communicate(timeout=step_timeout)
        except subprocess.TimeoutExpired as e:
            logger.warning(f"⚠️ {step_header} timed out after {step_timeout} seconds. Terminating process...")
            process.terminate()
            final_stdout, final_stderr = process.communicate()
            
            # Combine output
            stdout_from_timeout = e.stdout.decode(errors='replace') if e.stdout else ""
            stderr_from_timeout = e.stderr.decode(errors='replace') if e.stderr else ""
            stdout_data = stdout_from_timeout + final_stdout
            stderr_data = stderr_from_timeout + final_stderr
            
            raise subprocess.TimeoutExpired(cmd, step_timeout, output=stdout_data, stderr=stderr_data)

        # Log captured output
        if stdout_data:
            log_level = logging.INFO if not args.verbose else logging.DEBUG
            logger.log(log_level, f"--- Output from {script_name} (stdout) ---")
            for line in stdout_data.strip().split('\n'):
                if line.strip():  # Skip empty lines
                    logger.log(log_level, f"    [STDOUT] {line}")
            logger.log(log_level, f"--- End of {script_name} output ---")

        if stderr_data:
            logger.warning(f"--- Output from {script_name} (stderr) ---")
            for line in stderr_data.strip().split('\n'):
                if line.strip():  # Skip empty lines
                    logger.warning(f"    [STDERR] {line}")
            logger.warning(f"--- End of {script_name} output ---")
        
        # Record results
        result.stdout = stdout_data
        result.stderr = stderr_data
        result.exit_code = process.returncode
        
        # Performance metrics
        final_memory = get_memory_usage_mb()
        memory_delta = final_memory - initial_memory
        result.memory_usage_mb = memory_delta
        
        end_time = time.time()
        duration = end_time - start_time
        
        if process.returncode == 0:
            result.status = "SUCCESS"
            logger.info(f"✅ {step_header} - COMPLETED successfully in {duration:.1f} seconds.")
            
            if stderr_data:
                logger.warning(f"   -> Note: {script_name} completed successfully but wrote to stderr (see details above or in logs).")
        else:
            result.status = "FAILED_NONZERO_EXIT"
            result.details = f"Process exited with code {process.returncode}"
            logger.error(f"❌ {step_header} - FAILED with exit code {process.returncode} after {duration:.1f} seconds.")
            
            if is_critical:
                logger.critical(f"🔥 Critical step {script_name} failed with exit code {process.returncode}. Pipeline should halt.")
                result.details += " Critical step failure."
        
    except subprocess.TimeoutExpired as e:
        end_time = time.time()
        duration = end_time - start_time
        result.status = "FAILED_TIMEOUT"
        result.details = f"Process timed out after {duration:.1f} seconds (limit: {step_timeout}s)"
        
        if e.stdout:
            result.stdout = e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors='replace')
        if e.stderr:
            result.stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors='replace')
            
        logger.error(f"❌ {step_header} - FAILED due to timeout after {duration:.1f} seconds.")
        
        if is_critical:
            logger.critical(f"🔥 Critical step {script_name} timed out. Pipeline should halt.")
            result.details += " Critical step timeout."

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        result.status = "ERROR_UNHANDLED_EXCEPTION"
        result.details = f"Unhandled exception after {duration:.1f} seconds: {str(e)}"
        logger.error(f"❌ Unhandled exception in {step_header}: {e}")
        logger.debug("Full exception details:", exc_info=True)
        
        if is_critical:
            logger.critical(f"🔥 Critical step {script_name} failed due to unhandled exception. Pipeline should halt.")
            result.details += " Critical step failure."
    
    finally:
        result.end_time = datetime.datetime.now().isoformat()
        if result.start_time:
            duration_obj = datetime.datetime.fromisoformat(result.end_time) - datetime.datetime.fromisoformat(result.start_time)
            result.duration_seconds = duration_obj.total_seconds()
        
        logger.info("")  # Add spacing after step completion
    
    return result 