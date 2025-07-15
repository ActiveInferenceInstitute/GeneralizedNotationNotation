#!/usr/bin/env python3
"""
Fast Test Runner

This script runs only the fast tests for quick validation of the GNN pipeline.
It's designed to complete in under 30 seconds and provide basic confidence
that the system is working correctly.
"""

import sys
import subprocess
import time
from pathlib import Path

def run_fast_tests():
    """Run only the fast tests."""
    print("Running fast test suite...")
    start_time = time.time()
    
    # Prepare pytest command for fast tests only
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "--quiet",
        "--tb=short",
        "--maxfail=5",
        "--durations=5",
        "--disable-warnings",
        "-m", "fast",
        "src/tests/test_fast_suite.py"
    ]
    
    try:
        # Run pytest with 60 second timeout
        result = subprocess.run(
            pytest_cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Project root
            timeout=60
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"Fast tests completed in {elapsed_time:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("Test output:")
            print(result.stdout)
        
        if result.stderr:
            print("Test errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Fast tests timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Error running fast tests: {e}")
        return False

if __name__ == "__main__":
    success = run_fast_tests()
    sys.exit(0 if success else 1) 