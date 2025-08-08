#!/usr/bin/env python3
"""
Step 12: Execute Processing

This step executes rendered simulation code from step 11 with comprehensive
safe-to-fail patterns, detailed error classification, and robust logging.
"""

import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import tempfile
import shutil

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
from execute.validator import validate_execution_environment, log_validation_results


class ExecutionErrorType(Enum):
    """Classification of execution errors for better handling and recovery."""
    DEPENDENCY_MISSING = "dependency_missing"
    SYNTAX_ERROR = "syntax_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TIMEOUT = "timeout"
    PERMISSION_ERROR = "permission_error"
    FILE_NOT_FOUND = "file_not_found"
    RUNTIME_ERROR = "runtime_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ExecutionAttempt:
    """Track individual execution attempts with detailed metadata."""
    attempt_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_type: Optional[ExecutionErrorType] = None
    error_message: str = ""
    resource_usage: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class ExecutionContext:
    """Execution context with safety features."""
    script_path: Path
    script_name: str
    output_dir: Path
    gnn_spec: Optional[Dict[str, Any]] = None
    max_attempts: int = 3
    timeout_seconds: int = 300
    correlation_id: str = ""
    attempts: List[ExecutionAttempt] = field(default_factory=list)
    
    def add_attempt(self, attempt: ExecutionAttempt):
        """Add an execution attempt to the context."""
        self.attempts.append(attempt)
        
    @property
    def total_attempts(self) -> int:
        return len(self.attempts)
        
    @property
    def successful_attempts(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.success)
        
    @property
    def last_attempt(self) -> Optional[ExecutionAttempt]:
        return self.attempts[-1] if self.attempts else None


class ExecutionManager:
    """Manages execution with comprehensive safety patterns."""
    
    def __init__(self, logger):
        self.logger = logger
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retry_attempts": 0
        }
        
    def classify_error(self, error_message: str, exit_code: int) -> ExecutionErrorType:
        """Classify error type based on error message and exit code."""
        error_lower = error_message.lower()
        
        if "modulenotfounderror" in error_lower or "importerror" in error_lower:
            return ExecutionErrorType.DEPENDENCY_MISSING
        elif "syntaxerror" in error_lower or "indentationerror" in error_lower:
            return ExecutionErrorType.SYNTAX_ERROR
        elif "memoryerror" in error_lower or "out of memory" in error_lower:
            return ExecutionErrorType.RESOURCE_EXHAUSTED
        elif "timeout" in error_lower or exit_code == -9:
            return ExecutionErrorType.TIMEOUT
        elif "permissionerror" in error_lower or "permission denied" in error_lower:
            return ExecutionErrorType.PERMISSION_ERROR
        elif "filenotfounderror" in error_lower or "no such file" in error_lower:
            return ExecutionErrorType.FILE_NOT_FOUND
        elif "connectionerror" in error_lower or "network" in error_lower:
            return ExecutionErrorType.NETWORK_ERROR
        elif exit_code != 0:
            return ExecutionErrorType.RUNTIME_ERROR
        else:
            return ExecutionErrorType.UNKNOWN_ERROR
            
    def get_recovery_suggestion(self, error_type: ExecutionErrorType) -> str:
        """Get recovery suggestion based on error type."""
        suggestions = {
            ExecutionErrorType.DEPENDENCY_MISSING: "Install missing dependencies by adding to pyproject and running 'uv sync' or use 'uv pip install'",
            ExecutionErrorType.SYNTAX_ERROR: "Check script syntax and fix syntax errors",
            ExecutionErrorType.RESOURCE_EXHAUSTED: "Reduce batch size or increase available memory",
            ExecutionErrorType.TIMEOUT: "Increase timeout or optimize script performance",
            ExecutionErrorType.PERMISSION_ERROR: "Check file permissions and directory access",
            ExecutionErrorType.FILE_NOT_FOUND: "Verify file paths and ensure all required files exist",
            ExecutionErrorType.RUNTIME_ERROR: "Check script logic and input data validity",
            ExecutionErrorType.NETWORK_ERROR: "Check network connectivity and retry",
            ExecutionErrorType.UNKNOWN_ERROR: "Review error details and contact support if needed"
        }
        return suggestions.get(error_type, "Review error details and try again")
        
    @contextmanager
    def execution_safety_context(self, context: ExecutionContext):
        """Context manager for safe execution with cleanup."""
        temp_dir = None
        try:
            # Create temporary directory for execution
            temp_dir = tempfile.mkdtemp(prefix=f"gnn_exec_{context.correlation_id}_")
            context.temp_dir = Path(temp_dir)
            
            self.logger.info(f"[{context.correlation_id}] Created temp directory: {temp_dir}")
            
            yield context
            
        except Exception as e:
            self.logger.error(f"[{context.correlation_id}] Execution context error: {e}")
            raise
        finally:
            # Cleanup temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"[{context.correlation_id}] Cleaned up temp directory")
                except Exception as e:
                    self.logger.warning(f"[{context.correlation_id}] Failed to cleanup temp directory: {e}")
                    
    def execute_with_retry(self, context: ExecutionContext) -> bool:
        """Execute with retry logic."""
        
        with self.execution_safety_context(context):
            for attempt_num in range(1, context.max_attempts + 1):
                attempt = ExecutionAttempt(
                    attempt_number=attempt_num,
                    start_time=datetime.now()
                )
                
                self.logger.info(f"[{context.correlation_id}] Execution attempt {attempt_num}/{context.max_attempts}")
                
                try:
                    success = self._execute_single_attempt(context, attempt)
                    
                    attempt.end_time = datetime.now()
                    attempt.success = success
                    context.add_attempt(attempt)
                    
                    if success:
                        self.execution_stats["successful_executions"] += 1
                        self.logger.info(f"[{context.correlation_id}] Execution successful on attempt {attempt_num}")
                        return True
                    else:
                        self.execution_stats["retry_attempts"] += 1
                        
                        # Exponential backoff for retry
                        if attempt_num < context.max_attempts:
                            backoff_time = min(2 ** attempt_num, 30)  # Max 30 seconds
                            self.logger.info(f"[{context.correlation_id}] Retrying in {backoff_time} seconds...")
                            time.sleep(backoff_time)
                            
                except Exception as e:
                    attempt.end_time = datetime.now()
                    attempt.success = False
                    attempt.error_message = str(e)
                    attempt.error_type = self.classify_error(str(e), -1)
                    context.add_attempt(attempt)
                    
                    self.logger.error(f"[{context.correlation_id}] Attempt {attempt_num} failed: {e}")
                    
                    if attempt_num < context.max_attempts:
                        backoff_time = min(2 ** attempt_num, 30)
                        time.sleep(backoff_time)
            
            # All attempts failed
            self.execution_stats["failed_executions"] += 1
            self.logger.error(f"[{context.correlation_id}] All {context.max_attempts} attempts failed")
            return False
            
    def _execute_single_attempt(self, context: ExecutionContext, attempt: ExecutionAttempt) -> bool:
        """Execute a single attempt with comprehensive monitoring."""
        import subprocess
        
        try:
            # Log execution details
            self.logger.info(f"[{context.correlation_id}] Executing: {context.script_path}")
            if context.gnn_spec:
                self.logger.info(f"[{context.correlation_id}] Using GNN spec: {context.gnn_spec.get('model_name', 'Unknown')}")
            
            # Execute script
            result = self._run_subprocess(context)
                
            # Analyze results
            if result.returncode == 0:
                self.logger.info(f"[{context.correlation_id}] Execution completed successfully")
                return True
            else:
                error_type = self.classify_error(result.stderr, result.returncode)
                attempt.error_type = error_type
                attempt.error_message = result.stderr[:500]  # Truncate long error messages
                
                suggestion = self.get_recovery_suggestion(error_type)
                self.logger.warning(f"[{context.correlation_id}] Execution failed (exit code {result.returncode})")
                self.logger.warning(f"[{context.correlation_id}] Error type: {error_type.value}")
                self.logger.warning(f"[{context.correlation_id}] Recovery suggestion: {suggestion}")
                
                return False
                
        except subprocess.TimeoutExpired as e:
            attempt.error_type = ExecutionErrorType.TIMEOUT
            attempt.error_message = f"Execution timed out after {context.timeout_seconds} seconds"
            self.logger.error(f"[{context.correlation_id}] Execution timed out")
            return False
            
        except Exception as e:
            attempt.error_type = self.classify_error(str(e), -1)
            attempt.error_message = str(e)
            self.logger.error(f"[{context.correlation_id}] Execution error: {e}")
            return False
            
    def _run_subprocess(self, context: ExecutionContext):
        """Run subprocess with monitoring."""
        import subprocess
        import os
        
        # Prepare environment
        env = os.environ.copy()
        env["GNN_EXECUTION_ID"] = context.correlation_id
        env["GNN_OUTPUT_DIR"] = str(context.output_dir)
        env["GNN_TEMP_DIR"] = str(getattr(context, 'temp_dir', context.output_dir))
        env["GNN_TIMEOUT_SECONDS"] = str(context.timeout_seconds)
        
        command = [sys.executable, str(context.script_path)]
        
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=context.timeout_seconds,
            env=env,
            cwd=context.output_dir
        )


def generate_correlation_id() -> str:
    """Generate a correlation ID for tracking execution."""
    import uuid
    return str(uuid.uuid4())[:8]


def main():
    """Main execute processing function with comprehensive safety patterns."""
    args = EnhancedArgumentParser.parse_step_arguments("12_execute")
    
    # Setup logging with correlation ID
    correlation_id = generate_correlation_id()
    logger = setup_step_logging("execute", args)
    logger.info(f"[{correlation_id}] Starting execute processing")
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("12_execute.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, f"[{correlation_id}] Processing execute with safety patterns")
        
        # Validate execution environment
        validation_results = validate_execution_environment()
        log_validation_results(validation_results, logger)
        
        if validation_results["summary"]["total_errors"] > 0:
            logger.warning(f"[{correlation_id}] Environment validation failed - continuing with degraded functionality")
            # Create degraded execution results but continue pipeline
            degraded_results = {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "source_directory": str(args.target_dir),
                "output_directory": str(output_dir),
                "environment_validation": validation_results,
                "executions": [],
                "summary": {
                    "total_scripts": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "gnn_based_executions": 0,
                    "total_attempts": 0,
                    "environment_degraded": True
                }
            }
            
            # Save degraded results
            results_file = output_dir / "execution_results.json"
            with open(results_file, 'w') as f:
                json.dump(degraded_results, f, indent=2, default=str)
                
            log_step_warning(logger, f"[{correlation_id}] Execution skipped due to missing dependencies, but pipeline continues")
            return 0  # Continue pipeline even with environment issues
        
        # Load GNN processing results to get full specifications
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        gnn_specs = {}
        if gnn_results_file.exists():
            logger.info(f"[{correlation_id}] Loading GNN specifications for execution")
            with open(gnn_results_file, 'r') as f:
                gnn_results = json.load(f)
            
            # Build map of file names to GNN specifications
            for file_result in gnn_results.get("processed_files", []):
                if file_result.get("parse_success"):
                    file_name = file_result["file_name"]
                    gnn_specs[file_name] = file_result
                    logger.info(f"[{correlation_id}] Loaded GNN spec for {file_name}")
        else:
            logger.warning(f"[{correlation_id}] No GNN processing results found - execution will use basic parameters")
        
        # Check for rendered PyMDP scripts from step 11
        render_output_dir = get_output_dir_for_script("11_render.py", Path(args.output_dir))
        pymdp_scripts = list(render_output_dir.glob("**/*_pymdp*.py"))
        
        execution_results = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id,
            "source_directory": str(args.target_dir),
            "output_directory": str(output_dir),
            "environment_validation": validation_results,
            "executions": [],
            "summary": {
                "total_scripts": len(pymdp_scripts),
                "successful_executions": 0,
                "failed_executions": 0,
                "gnn_based_executions": 0,
                "total_attempts": 0
            }
        }
        
        # Initialize execution manager
        executor = ExecutionManager(logger)
        success = False
        
        if pymdp_scripts:
            logger.info(f"[{correlation_id}] Found {len(pymdp_scripts)} PyMDP scripts to execute")
            
            try:
                from execute.pymdp import execute_pymdp_simulation_from_gnn
                pymdp_available = True
                logger.info(f"[{correlation_id}] PyMDP pipeline module available")
            except ImportError as e:
                logger.warning(f"[{correlation_id}] PyMDP pipeline module not available: {e}")
                pymdp_available = False
            
            for script_path in pymdp_scripts:
                script_correlation_id = f"{correlation_id}-{script_path.stem}"
                
                execution_result = {
                    "script_path": str(script_path),
                    "script_name": script_path.name,
                    "correlation_id": script_correlation_id,
                    "execution_success": False,
                    "gnn_based": False,
                    "output_directory": None,
                    "error": None,
                    "attempts": [],
                    "execution_time_seconds": 0,
                    "resource_usage": {}
                }
                
                start_time = time.time()
                
                try:
                    logger.info(f"[{script_correlation_id}] Starting execution of: {script_path.name}")
                    
                    # Find corresponding GNN specification
                    corresponding_gnn = None
                    for gnn_file, gnn_spec in gnn_specs.items():
                        if gnn_file.replace('.md', '') in script_path.name:
                            corresponding_gnn = gnn_spec
                            break
                    
                    # Create execution context
                    exec_output_dir = output_dir / f"pymdp_{script_path.stem}"
                    exec_output_dir.mkdir(exist_ok=True)
                    
                    context = ExecutionContext(
                        script_path=script_path,
                        script_name=script_path.name,
                        output_dir=exec_output_dir,
                        gnn_spec=corresponding_gnn,
                        max_attempts=3,
                        timeout_seconds=300,
                        correlation_id=script_correlation_id
                    )
                    
                    # Execute with safety patterns
                    execution_success = executor.execute_with_retry(context)
                    
                    execution_result["execution_success"] = execution_success
                    execution_result["attempts"] = [
                        {
                            "attempt_number": attempt.attempt_number,
                            "duration_seconds": attempt.duration_seconds,
                            "success": attempt.success,
                            "error_type": attempt.error_type.value if attempt.error_type else None,
                            "error_message": attempt.error_message[:200],  # Truncate for JSON
                            "resource_usage": attempt.resource_usage
                        }
                        for attempt in context.attempts
                    ]
                    execution_result["output_directory"] = str(exec_output_dir)
                    execution_result["gnn_based"] = corresponding_gnn is not None
                    
                    if execution_success:
                        execution_results["summary"]["successful_executions"] += 1
                        if corresponding_gnn:
                            execution_results["summary"]["gnn_based_executions"] += 1
                        logger.info(f"[{script_correlation_id}] ✓ Execution successful")
                    else:
                        execution_results["summary"]["failed_executions"] += 1
                        last_attempt = context.last_attempt
                        if last_attempt:
                            execution_result["error"] = f"{last_attempt.error_type.value}: {last_attempt.error_message[:200]}"
                        logger.error(f"[{script_correlation_id}] ✗ Execution failed after {context.total_attempts} attempts")
                
                except Exception as e:
                    execution_result["error"] = f"Unexpected error: {str(e)[:200]}"
                    execution_results["summary"]["failed_executions"] += 1
                    logger.error(f"[{script_correlation_id}] Unexpected error: {e}")
                    logger.debug(f"[{script_correlation_id}] Traceback: {traceback.format_exc()}")
                
                execution_result["execution_time_seconds"] = time.time() - start_time
                execution_results["executions"].append(execution_result)
                execution_results["summary"]["total_attempts"] += len(execution_result.get("attempts", []))
            
            # Update summary with executor statistics
            execution_results["executor_stats"] = executor.execution_stats
            
            success = execution_results["summary"]["successful_executions"] > 0
            
            # Save execution results
            results_file = output_dir / "execution_results.json"
            with open(results_file, 'w') as f:
                json.dump(execution_results, f, indent=2, default=str)
            
            # Generate execution report
            report_file = output_dir / "execution_report.md"
            with open(report_file, 'w') as f:
                f.write(f"# Execution Report\n\n")
                f.write(f"**Correlation ID:** {correlation_id}\n")
                f.write(f"**Generated:** {execution_results['timestamp']}\n")
                f.write(f"**Total Scripts:** {execution_results['summary']['total_scripts']}\n")
                f.write(f"**Successful:** {execution_results['summary']['successful_executions']}\n")
                f.write(f"**Failed:** {execution_results['summary']['failed_executions']}\n")
                f.write(f"**GNN-based:** {execution_results['summary']['gnn_based_executions']}\n")
                f.write(f"**Total Attempts:** {execution_results['summary']['total_attempts']}\n\n")
                
                f.write("## Environment Status\n\n")
                env_status = validation_results["overall_status"]
                f.write(f"**Status:** {env_status}\n")
                if validation_results["summary"]["total_errors"] > 0:
                    f.write(f"**Errors:** {validation_results['summary']['total_errors']}\n")
                if validation_results["summary"]["total_warnings"] > 0:
                    f.write(f"**Warnings:** {validation_results['summary']['total_warnings']}\n")
                f.write("\n")
                
                f.write("## Execution Results\n\n")
                for exec_result in execution_results["executions"]:
                    status_icon = "✅" if exec_result["execution_success"] else "❌"
                    f.write(f"### {status_icon} {exec_result['script_name']}\n")
                    f.write(f"- **Success:** {exec_result['execution_success']}\n")
                    f.write(f"- **GNN-based:** {exec_result['gnn_based']}\n")
                    f.write(f"- **Execution Time:** {exec_result['execution_time_seconds']:.2f}s\n")
                    f.write(f"- **Attempts:** {len(exec_result.get('attempts', []))}\n")
                    if exec_result.get("error"):
                        f.write(f"- **Error:** {exec_result['error']}\n")
                    f.write("\n")
            
            if success:
                successful = execution_results["summary"]["successful_executions"]
                gnn_based = execution_results["summary"]["gnn_based_executions"]
                total_attempts = execution_results["summary"]["total_attempts"]
                log_step_success(logger, f"[{correlation_id}] Execution completed: {successful} successful ({gnn_based} GNN-based, {total_attempts} total attempts)")
            else:
                log_step_warning(logger, f"[{correlation_id}] No simulations executed successfully, but execution step completed")
        else:
            logger.info(f"[{correlation_id}] No PyMDP scripts found for execution")
            success = True  # Don't fail if no scripts to execute
        
        # Always return 0 to ensure pipeline continuation
        return 0
            
    except Exception as e:
        log_step_error(logger, f"[{correlation_id}] Execute processing failed: {str(e)}")
        
        # Even on complete failure, try to save an error report
        try:
            error_results = {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "summary": {
                    "total_scripts": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "critical_failure": True
                }
            }
            
            error_file = output_dir / "execution_error.json"
            with open(error_file, 'w') as f:
                json.dump(error_results, f, indent=2)
                
            logger.info(f"[{correlation_id}] Saved error report to {error_file}")
                
        except Exception as save_error:
            logger.error(f"[{correlation_id}] Failed to save error report: {save_error}")
        
        # Always return 0 to ensure pipeline continuation even on critical failure
        return 0

if __name__ == "__main__":
    sys.exit(main())
