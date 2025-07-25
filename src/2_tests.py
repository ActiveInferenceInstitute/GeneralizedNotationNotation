#!/usr/bin/env python3
"""
Step 2: Test Suite Execution

This step runs comprehensive tests for the GNN pipeline.
"""

import sys
from pathlib import Path
import subprocess
import json
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def main():
    """Main test execution function."""
    args = EnhancedArgumentParser.parse_step_arguments("2_tests.py")
    
    # Setup logging
    logger = setup_step_logging("tests", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("2_tests.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Running comprehensive test suite")
        
        # Get project root and virtual environment path
        project_root = Path(__file__).parent.parent
        venv_python = project_root / ".venv" / "bin" / "python"
        
        # Use virtual environment Python if available, otherwise fall back to system Python
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        # Build test command - filter out empty strings
        test_command = [
            python_executable, "-m", "pytest",
            "src/tests/",
            "--cov=src/",
            "--cov-report=json:coverage.json"
        ]
        
        # Add verbose flag if requested
        if args.verbose:
            test_command.append("-v")
        
        # Set up environment variables
        test_env = os.environ.copy()
        src_path = str(project_root / "src")
        test_env['PYTHONPATH'] = src_path + os.pathsep + test_env.get('PYTHONPATH', '')
        test_env['GNN_TEST_MODE'] = 'true'
        test_env['PYTHONWARNINGS'] = 'ignore::ResourceWarning'
        
        logger.info(f"Running test command: {' '.join(test_command)}")
        logger.info(f"Working directory: {project_root}")
        logger.info(f"Python executable: {python_executable}")
        
        result = subprocess.run(
            test_command, 
            capture_output=True, 
            text=True, 
            cwd=project_root, 
            env=test_env,
            timeout=600  # 10 minute timeout
        )
        
        # Save results
        results_file = output_dir / "test_results.json"
        test_results = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "command": ' '.join(test_command),
            "working_directory": str(project_root),
            "python_executable": python_executable
        }
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Log output for debugging
        if args.verbose:
            logger.info("Test stdout:")
            for line in result.stdout.splitlines()[-50:]:  # Last 50 lines
                logger.info(f"  {line}")
        
        if result.stderr:
            logger.warning("Test stderr:")
            for line in result.stderr.splitlines()[-20:]:  # Last 20 lines
                logger.warning(f"  {line}")
        
        if result.returncode == 0:
            log_step_success(logger, "All tests passed successfully")
            return 0
        else:
            log_step_error(logger, f"Some tests failed (exit code: {result.returncode})")
            return 1
            
    except subprocess.TimeoutExpired:
        log_step_error(logger, "Test execution timed out after 10 minutes")
        return 1
    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 