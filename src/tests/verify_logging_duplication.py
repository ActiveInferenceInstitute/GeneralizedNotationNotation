#!/usr/bin/env python3
import logging
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_duplication():
    print("Verifying logging duplication...")
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    try:
        # Import old logging utils
        from utils.logging_utils import setup_step_logging, PipelineLogger
        
        # Simulate main logging setup
        PipelineLogger.setup()
        
        # Setup logging for a step
        logger = setup_step_logging("test_step")
        
        # Log a message
        test_msg = "LOG_VERIFICATION_MESSAGE"
        logger.info(test_msg)
        
        # Get output
        output = stdout_capture.getvalue() + stderr_capture.getvalue()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
    print(f"Captured output:\n{output}")
    
    occurrences = output.count(test_msg)
    print(f"Occurrences of message: {occurrences}")
    
    if occurrences > 1:
        print("DUPLICATION DETECTED! (Status Quo confirmed)")
        return True
    else:
        print("No duplication detected.")
        return False

if __name__ == "__main__":
    verify_duplication()
