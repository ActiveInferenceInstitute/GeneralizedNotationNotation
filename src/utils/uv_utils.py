"""
UV Utility Functions for GNN Pipeline

This module provides centralized UV operations and utilities for the GNN pipeline.
It ensures consistent UV usage across all pipeline steps and provides fallback
mechanisms for environments where UV is not available.
"""

import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import time

logger = logging.getLogger(__name__)

# UV configuration
UV_TIMEOUT = 300  # 5 minutes timeout for UV operations
UV_RETRY_ATTEMPTS = 3

def check_uv_available() -> bool:
    """
    Check if UV is available and working.
    
    Returns:
        True if UV is available and working, False otherwise
    """
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False

def get_uv_version() -> Optional[str]:
    """
    Get the UV version string.
    
    Returns:
        UV version string if available, None otherwise
    """
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None

def run_uv_command(
    command: List[str],
    cwd: Optional[Path] = None,
    timeout: int = UV_TIMEOUT,
    retry_attempts: int = UV_RETRY_ATTEMPTS,
    verbose: bool = False
) -> Tuple[bool, str, str]:
    """
    Run a UV command with retry logic and error handling.
    
    Args:
        command: UV command as list of strings
        cwd: Working directory for the command
        timeout: Timeout in seconds
        retry_attempts: Number of retry attempts
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    if not check_uv_available():
        return False, "", "UV is not available"
    
    cwd = cwd or Path.cwd()
    
    for attempt in range(retry_attempts):
        try:
            if verbose:
                logger.info(f"Running UV command (attempt {attempt + 1}/{retry_attempts}): {' '.join(command)}")
            
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                if verbose:
                    logger.info("UV command completed successfully")
                return True, result.stdout, result.stderr
            else:
                if verbose:
                    logger.warning(f"UV command failed (attempt {attempt + 1}): {result.stderr}")
                
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return False, result.stdout, result.stderr
                    
        except subprocess.TimeoutExpired:
            if verbose:
                logger.warning(f"UV command timed out (attempt {attempt + 1})")
            if attempt < retry_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return False, "", "Command timed out"
                
        except Exception as e:
            if verbose:
                logger.warning(f"UV command failed with exception (attempt {attempt + 1}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return False, "", str(e)
    
    return False, "", "All retry attempts failed"

def sync_dependencies(
    extras: Optional[List[str]] = None,
    dev: bool = False,
    frozen: bool = True,
    cwd: Optional[Path] = None,
    verbose: bool = False
) -> bool:
    """
    Sync dependencies using UV.
    
    Args:
        extras: List of optional dependency groups to install
        dev: Install development dependencies
        frozen: Use frozen lock file if available
        cwd: Working directory
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    command = ["uv", "sync"]
    
    if dev:
        command.extend(["--extra", "dev"])
    
    if extras:
        for extra in extras:
            command.extend(["--extra", str(extra)])
    
    if frozen:
        command.append("--frozen")
    
    success, stdout, stderr = run_uv_command(command, cwd=cwd, verbose=verbose)
    
    if success:
        if verbose:
            logger.info("Dependencies synced successfully")
        return True
    else:
        logger.error(f"Failed to sync dependencies: {stderr}")
        return False

def install_package(
    package: str,
    cwd: Optional[Path] = None,
    verbose: bool = False
) -> bool:
    """
    Install a package using UV.
    
    Args:
        package: Package name to install
        cwd: Working directory
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    command = ["uv", "add", package]
    
    success, stdout, stderr = run_uv_command(command, cwd=cwd, verbose=verbose)
    
    if success:
        if verbose:
            logger.info(f"Package {package} installed successfully")
        return True
    else:
        logger.error(f"Failed to install package {package}: {stderr}")
        return False

def run_python_script(
    script_path: Path,
    args: Optional[List[str]] = None,
    cwd: Optional[Path] = None,
    verbose: bool = False
) -> Tuple[bool, str, str]:
    """
    Run a Python script using UV.
    
    Args:
        script_path: Path to Python script
        args: Script arguments
        cwd: Working directory
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    command = ["uv", "run", "python", str(script_path)]
    
    if args:
        command.extend(args)
    
    return run_uv_command(command, cwd=cwd, verbose=verbose)

def get_installed_packages(cwd: Optional[Path] = None) -> Dict[str, str]:
    """
    Get list of installed packages using UV.
    
    Args:
        cwd: Working directory
        
    Returns:
        Dictionary of package names and versions
    """
    command = ["uv", "pip", "list", "--format=json"]
    
    success, stdout, stderr = run_uv_command(command, cwd=cwd, verbose=False)
    
    if success:
        try:
            packages = json.loads(stdout)
            return {pkg["name"]: pkg["version"] for pkg in packages}
        except json.JSONDecodeError:
            logger.warning("Failed to parse package list JSON")
            return {}
    else:
        logger.warning(f"Failed to get package list: {stderr}")
        return {}

def check_package_installed(package: str, cwd: Optional[Path] = None) -> bool:
    """
    Check if a package is installed using UV.
    
    Args:
        package: Package name to check
        cwd: Working directory
        
    Returns:
        True if package is installed, False otherwise
    """
    packages = get_installed_packages(cwd)
    return package in packages

def create_uv_environment(
    python_version: str = "3.12",
    cwd: Optional[Path] = None,
    verbose: bool = False
) -> bool:
    """
    Create a UV environment.
    
    Args:
        python_version: Python version to use
        cwd: Working directory
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    command = ["uv", "venv", "--python", python_version]
    
    success, stdout, stderr = run_uv_command(command, cwd=cwd, verbose=verbose)
    
    if success:
        if verbose:
            logger.info("UV environment created successfully")
        return True
    else:
        logger.error(f"Failed to create UV environment: {stderr}")
        return False

def validate_uv_environment(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """
    Validate UV environment and return status information.
    
    Args:
        cwd: Working directory
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "uv_available": False,
        "uv_version": None,
        "environment_exists": False,
        "python_executable": None,
        "packages_installed": 0,
        "core_packages": {},
        "overall_status": False
    }
    
    # Check UV availability
    validation["uv_available"] = check_uv_available()
    if validation["uv_available"]:
        validation["uv_version"] = get_uv_version()
    
    # Check environment
    cwd = cwd or Path.cwd()
    venv_path = cwd / ".venv"
    validation["environment_exists"] = venv_path.exists()
    
    if validation["environment_exists"]:
        # Check Python executable
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            validation["python_executable"] = str(python_exe)
    
    # Check installed packages
    packages = get_installed_packages(cwd)
    validation["packages_installed"] = len(packages)
    
    # Check core packages
    core_packages = ["numpy", "matplotlib", "pytest", "pandas"]
    for pkg in core_packages:
        validation["core_packages"][pkg] = pkg in packages
    
    # Overall status
    validation["overall_status"] = (
        validation["uv_available"] and
        validation["environment_exists"] and
        validation["python_executable"] is not None and
        validation["packages_installed"] > 0
    )
    
    return validation

def get_uv_environment_info(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get comprehensive UV environment information.
    
    Args:
        cwd: Working directory
        
    Returns:
        Dictionary with environment information
    """
    info = {
        "uv_version": get_uv_version(),
        "uv_available": check_uv_available(),
        "validation": validate_uv_environment(cwd),
        "installed_packages": get_installed_packages(cwd),
        "timestamp": time.time()
    }
    
    return info

def ensure_uv_environment(
    python_version: str = "3.12",
    dev: bool = True,
    extras: Optional[List[str]] = None,
    cwd: Optional[Path] = None,
    verbose: bool = False
) -> bool:
    """
    Ensure UV environment is set up and dependencies are installed.
    
    Args:
        python_version: Python version to use
        dev: Install development dependencies
        extras: List of optional dependency groups
        cwd: Working directory
        verbose: Enable verbose logging
        
    Returns:
        True if environment is ready, False otherwise
    """
    cwd = cwd or Path.cwd()
    
    # Check if UV is available
    if not check_uv_available():
        logger.error("UV is not available")
        return False
    
    # Check if environment exists
    venv_path = cwd / ".venv"
    if not venv_path.exists():
        logger.info("Creating UV environment...")
        if not create_uv_environment(python_version, cwd, verbose):
            return False
    
    # Sync dependencies
    logger.info("Syncing dependencies...")
    if not sync_dependencies(extras=extras, dev=dev, cwd=cwd, verbose=verbose):
        return False
    
    # Validate environment
    validation = validate_uv_environment(cwd)
    if not validation["overall_status"]:
        logger.error("Environment validation failed")
        return False
    
    logger.info("UV environment is ready")
    return True
