"""
Execution Utilities
==================

This module provides utilities for executing external commands with real-time output streaming.
It allows long-running processes (like test suites) to display their progress immediately
rather than buffering all output until completion.
"""

import subprocess
import sys
import os
import threading
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

def execute_command_streaming(
    cmd: List[str],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    print_stdout: bool = True,
    print_stderr: bool = True,
    capture_output: bool = True
) -> Dict[str, Any]:
    """
    Execute a command with real-time output streaming.
    
    Args:
        cmd: Command and arguments list
        cwd: Working directory
        env: Environment variables
        timeout: Timeout in seconds
        print_stdout: Whether to print stdout to sys.stdout in real-time
        print_stderr: Whether to print stderr to sys.stderr in real-time
        capture_output: Whether to return captured stdout/stderr in the result
        
    Returns:
        Dictionary containing:
        - exit_code: Process exit code
        - stdout: Captured stdout (if capture_output=True)
        - stderr: Captured stderr (if capture_output=True)
        - status: 'SUCCESS', 'FAILED', or 'TIMEOUT'
    """
    if cwd:
        cwd = str(cwd)
        
    # Ensure environment is properly set up
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
        
    # Force unbuffered output
    process_env["PYTHONUNBUFFERED"] = "1"
    
    # buffers for captured output
    stdout_captured = []
    stderr_captured = []
    
    # detailed result structure
    result = {
        "exit_code": -1,
        "stdout": "",
        "stderr": "",
        "status": "UNKNOWN"
    }
    
    try:
        # Start process with pipes
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Reader threads
        def read_stream(stream, is_stderr):
            try:
                for line in iter(stream.readline, ''):
                    if not line:
                        break
                        
                    # Print immediately if requested
                    if is_stderr:
                        if print_stderr:
                            sys.stderr.write(line)
                            sys.stderr.flush()
                        if capture_output:
                            stderr_captured.append(line)
                    else:
                        if print_stdout:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                        if capture_output:
                            stdout_captured.append(line)
            except (ValueError, OSError):
                # Handle cases where stream is closed
                pass
            finally:
                stream.close()

        # Start threads
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, False))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, True))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process with timeout
        try:
            exit_code = process.wait(timeout=timeout)
            
            # Wait for threads to finish reading
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            
            result["exit_code"] = exit_code
            result["status"] = "SUCCESS" if exit_code == 0 else "FAILED"
            
        except subprocess.TimeoutExpired:
            process.kill()
            result["status"] = "TIMEOUT"
            result["exit_code"] = -1
            # Try to join threads briefly
            stdout_thread.join(timeout=0.1)
            stderr_thread.join(timeout=0.1)
            
    except Exception as e:
        result["status"] = "FAILED"
        result["stderr"] = str(e)
        if print_stderr:
            sys.stderr.write(f"Execution error: {e}\n")
            
    # Combine captured output
    if capture_output:
        result["stdout"] = "".join(stdout_captured)
        result["stderr"] = "".join(stderr_captured)
        
    return result
