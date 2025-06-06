"""
Main setup script for the GNN project.

This script handles the creation of a virtual environment and the installation
of project dependencies when called directly or via its perform_full_setup function.
"""

import os
import subprocess
import sys
import platform
import re
import shutil
from pathlib import Path
import logging
import argparse
import tempfile
import time  # Added for progress tracking

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# --- Configuration ---
VENV_DIR = ".venv"  # Name of the virtual environment directory
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_DEV_FILE = "requirements-dev.txt"

# PROJECT_ROOT should be the 'src/' directory.
# Since this script is in 'src/setup/', Path(__file__).parent is 'src/setup'.
# So, Path(__file__).parent.parent is 'src/'.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

VENV_PATH = PROJECT_ROOT / VENV_DIR
REQUIREMENTS_PATH = PROJECT_ROOT / REQUIREMENTS_FILE
REQUIREMENTS_DEV_PATH = PROJECT_ROOT / REQUIREMENTS_DEV_FILE

# Determine the correct Python interpreter path for the virtual environment
if sys.platform == "win32":
    VENV_PYTHON = VENV_PATH / "Scripts" / "python.exe"
    VENV_PIP = VENV_PATH / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_PATH / "bin" / "python"
    VENV_PIP = VENV_PATH / "bin" / "pip"

# Minimum required versions of system packages
MIN_PYTHON_VERSION = (3, 9)  # Python 3.9 or higher
MIN_PIP_VERSION = (21, 0)    # pip 21.0 or higher

# Known dependency conflicts and resolutions
DEPENDENCY_CONFLICTS = {
    "jax": ["tensorflow<2.10"],  # JAX conflicts with older TensorFlow versions
    "discopy": ["category-theory-for-programming<0.3.0"]  # Example of potential conflict
}

# --- Helper Functions ---

def run_command(command: list[str], cwd: Path = PROJECT_ROOT, check: bool = True, verbose: bool = False) -> subprocess.CompletedProcess:
    """
    Runs a shell command and logs its output based on verbosity.

    Args:
        command: The command and its arguments as a list of strings.
        cwd: The current working directory for the command.
        check: If True, raises CalledProcessError if the command returns a non-zero exit code.
        verbose: If True, enables detailed (DEBUG level) logging for this setup process.
        
    Returns:
        The completed process object with stdout and stderr attributes.
    """
    command_str_list = [str(c) for c in command]
    if verbose:
        logger.debug(f"Running command: '{' '.join(command_str_list)}' in {cwd}")
    else:
        logger.debug(f"Running command: '{command_str_list[0]} ...' in {cwd}")
    
    try:
        process = subprocess.run(command_str_list, cwd=cwd, check=check, capture_output=True, text=True, errors='replace')
        if verbose:
            if process.stdout:
                logger.debug(f"Stdout:\n{process.stdout.strip()}")
            if process.stderr:
                logger.debug(f"Stderr:\n{process.stderr.strip()}")
        if not check and process.returncode != 0:
            logger.warning(f"Command returned non-zero exit code: {process.returncode}")
            if process.stdout:
                logger.warning(f"Stdout:\n{process.stdout.strip()}")
            if process.stderr:
                logger.warning(f"Stderr:\n{process.stderr.strip()}")
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: '{' '.join(e.cmd)}'")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout.strip()}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr.strip()}")
        if check:
            raise
        return e
    except FileNotFoundError as e:
        logger.error(f"Error: Command not found - {command_str_list[0]}. Ensure it is installed and in PATH.")
        logger.error(f"Details: {e}")
        if check:
            raise
        raise

def check_system_requirements(verbose: bool = False) -> bool:
    """
    Checks if the system meets the minimum requirements for the GNN project.
    
    Args:
        verbose: If True, enables detailed logging.
        
    Returns:
        True if all requirements are met, False otherwise.
    """
    logger.info("🔍 Checking system requirements...")
    # Check Python version
    python_version = sys.version_info
    if python_version < MIN_PYTHON_VERSION:
        logger.error(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is below the minimum required version {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}")
        return False
    else:
        logger.info(f"Python version check passed: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check pip version
    try:
        logger.debug("Checking pip version...")
        pip_process = run_command([sys.executable, "-m", "pip", "--version"], check=True, verbose=verbose)
        pip_version_match = re.search(r"pip (\d+)\.(\d+)", pip_process.stdout)
        if pip_version_match:
            pip_major = int(pip_version_match.group(1))
            pip_minor = int(pip_version_match.group(2))
            if (pip_major, pip_minor) < MIN_PIP_VERSION:
                logger.error(f"pip version {pip_major}.{pip_minor} is below the minimum required version {MIN_PIP_VERSION[0]}.{MIN_PIP_VERSION[1]}")
                return False
            else:
                logger.info(f"pip version check passed: {pip_major}.{pip_minor}")
        else:
            logger.warning("Could not determine pip version. Continuing anyway.")
    except Exception as e:
        logger.error(f"Error checking pip version: {e}")
        return False
    
    # Check for required system packages on Linux
    if platform.system() == "Linux":
        required_packages = ["build-essential", "python3-dev"]
        for package in required_packages:
            try:
                process = run_command(["dpkg", "-s", package], check=False, verbose=verbose)
                if process.returncode != 0:
                    logger.warning(f"System package {package} may not be installed. This might cause issues during dependency installation.")
            except Exception as e:
                logger.warning(f"Could not check if {package} is installed: {e}")
    
    # Check disk space (at least 2GB free in the project directory)
    try:
        disk_usage = shutil.disk_usage(PROJECT_ROOT)
        free_space_gb = disk_usage.free / (1024 * 1024 * 1024)  # Convert to GB
        if free_space_gb < 2:
            logger.warning(f"Low disk space: {free_space_gb:.2f}GB free. At least 2GB recommended for dependency installation.")
        else:
            logger.info(f"Disk space check passed: {free_space_gb:.2f}GB free")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    
    return True

def validate_dependency_versions(verbose: bool = False) -> dict:
    """
    Validates installed dependency versions against requirements.
    
    Args:
        verbose: If True, enables detailed logging.
        
    Returns:
        Dictionary with validation results.
    """
    if not VENV_PYTHON.exists():
        logger.error(f"Virtual environment Python not found at {VENV_PYTHON}")
        return {"success": False, "error": "Virtual environment not found"}
    
    results = {
        "success": True,
        "mismatches": [],
        "missing": [],
        "validated": []
    }
    
    try:
        # Get installed packages
        logger.info("📋 Validating installed dependencies...")
        process = run_command([str(VENV_PIP), "list", "--format=json"], check=True, verbose=verbose)
        import json
        installed_packages = json.loads(process.stdout)
        installed_dict = {pkg["name"].lower(): pkg["version"] for pkg in installed_packages}
        
        # Read requirements file
        if REQUIREMENTS_PATH.exists():
            with open(REQUIREMENTS_PATH, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            
            # Check each requirement
            for req in requirements:
                if req.startswith('-r '):
                    continue  # Skip included requirements files
                
                # Parse requirement with version specification
                req_parts = re.match(r"([^<>=~]+)(?:[<>=~]+(.+))?", req)
                if req_parts:
                    pkg_name = req_parts.group(1).strip().lower()
                    req_version = req_parts.group(2).strip() if req_parts.group(2) else None
                    
                    if pkg_name in installed_dict:
                        installed_version = installed_dict[pkg_name]
                        results["validated"].append(f"{pkg_name}: {installed_version}")
                        
                        # Simple version check - just log for now
                        if req_version and req_version != installed_version:
                            logger.warning(f"Version mismatch for {pkg_name}: required {req_version}, installed {installed_version}")
                            results["mismatches"].append(f"{pkg_name}: required {req_version}, installed {installed_version}")
                    else:
                        logger.warning(f"Package {pkg_name} not installed")
                        results["missing"].append(pkg_name)
                        results["success"] = False
        else:
            logger.warning(f"Requirements file not found at {REQUIREMENTS_PATH}")
            results["success"] = False
    
    except Exception as e:
        logger.error(f"Error validating dependency versions: {e}")
        results["success"] = False
        results["error"] = str(e)
    
    return results

def resolve_dependency_conflicts(verbose: bool = False) -> bool:
    """
    Checks for and resolves known dependency conflicts.
    
    Args:
        verbose: If True, enables detailed logging.
        
    Returns:
        True if conflicts were resolved or no conflicts exist, False otherwise.
    """
    if not VENV_PIP.exists():
        logger.error(f"Virtual environment pip not found at {VENV_PIP}")
        return False
    
    try:
        # Get installed packages
        logger.info("🔍 Checking for dependency conflicts...")
        process = run_command([str(VENV_PIP), "list"], check=True, verbose=verbose)
        installed_packages = process.stdout.strip().split('\n')[2:]  # Skip header lines
        
        conflicts_found = False
        
        # Check for each known conflict
        for pkg, conflicts_with in DEPENDENCY_CONFLICTS.items():
            pkg_pattern = re.compile(rf"{pkg}\s+(\S+)", re.IGNORECASE)
            
            for line in installed_packages:
                if pkg_pattern.search(line):
                    # Found package that might have conflicts
                    for conflict in conflicts_with:
                        conflict_pattern = re.compile(rf"{conflict.split('<')[0]}\s+(\S+)", re.IGNORECASE)
                        for conflict_line in installed_packages:
                            match = conflict_pattern.search(conflict_line)
                            if match:
                                conflicts_found = True
                                conflicting_version = match.group(1)
                                logger.warning(f"Dependency conflict detected: {pkg} conflicts with {conflict.split('<')[0]} version {conflicting_version}")
                                
                                # Try to resolve by uninstalling conflicting package
                                logger.info(f"Attempting to resolve conflict by uninstalling {conflict.split('<')[0]}")
                                try:
                                    run_command([str(VENV_PIP), "uninstall", "-y", conflict.split('<')[0]], check=True, verbose=verbose)
                                    logger.info(f"Successfully uninstalled {conflict.split('<')[0]}")
                                except Exception as e:
                                    logger.error(f"Failed to uninstall {conflict.split('<')[0]}: {e}")
                                    return False
        
        if not conflicts_found:
            logger.info("No dependency conflicts detected.")
        
        return True
    except Exception as e:
        logger.error(f"Error checking for dependency conflicts: {e}")
        return False

def create_virtual_environment(verbose: bool = False, recreate: bool = False) -> bool:
    """
    Creates a virtual environment if it doesn't already exist, or recreates it if specified.

    Args:
        verbose: If True, enables detailed (DEBUG level) logging for this setup process.
        recreate: If True, deletes and recreates an existing virtual environment.
        
    Returns:
        True if successful, False otherwise.
    """
    if VENV_PATH.exists() and recreate:
        logger.info(f"🔄 Recreating virtual environment in {VENV_PATH}...")
        try:
            shutil.rmtree(VENV_PATH)
            logger.info(f"Removed existing virtual environment at {VENV_PATH}")
        except Exception as e:
            logger.error(f"Failed to remove existing virtual environment: {e}", exc_info=verbose)
            return False
    
    if not VENV_PATH.exists():
        logger.info(f"🔧 Creating virtual environment in {VENV_PATH}...")
        try:
            run_command([sys.executable, "-m", "venv", VENV_DIR], cwd=PROJECT_ROOT, verbose=verbose)
            logger.info(f"✅ Virtual environment created successfully at {VENV_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}", exc_info=verbose)
            return False
    else:
        logger.info(f"✓ Virtual environment already exists at {VENV_PATH}")
        return True

def cleanup_virtual_environment(verbose: bool = False) -> bool:
    """
    Cleans up the virtual environment by removing unnecessary files.
    
    Args:
        verbose: If True, enables detailed logging.
        
    Returns:
        True if successful, False otherwise.
    """
    if not VENV_PATH.exists():
        logger.warning(f"Virtual environment not found at {VENV_PATH}")
        return False
    
    try:
        logger.info("🧹 Cleaning up virtual environment...")
        # Remove __pycache__ directories
        for pycache_dir in VENV_PATH.rglob('__pycache__'):
            if pycache_dir.is_dir():
                shutil.rmtree(pycache_dir)
                if verbose:
                    logger.debug(f"Removed {pycache_dir}")
        
        # Remove .pyc files
        for pyc_file in VENV_PATH.rglob('*.pyc'):
            pyc_file.unlink()
            if verbose:
                logger.debug(f"Removed {pyc_file}")
        
        # Remove pip cache if exists
        pip_cache = VENV_PATH / "pip-cache"
        if pip_cache.exists():
            shutil.rmtree(pip_cache)
            if verbose:
                logger.debug(f"Removed {pip_cache}")
        
        logger.info("✅ Virtual environment cleanup completed")
        return True
    except Exception as e:
        logger.error(f"Error during virtual environment cleanup: {e}")
        return False

def install_dependencies(verbose: bool = False, dev: bool = False) -> bool:
    """
    Installs or updates dependencies from the requirements.txt file
    into the virtual environment.

    Args:
        verbose: If True, enables detailed (DEBUG level) logging for this setup process.
        dev: If True, also installs development dependencies (marked as such in requirements.txt).
        
    Returns:
        True if successful, False otherwise.
    """
    if not REQUIREMENTS_PATH.exists():
        logger.warning(f"{REQUIREMENTS_PATH} not found. Skipping dependency installation from this file.")
        return False

    logger.info(f"📦 Installing/updating dependencies from {REQUIREMENTS_PATH} into {VENV_PATH}...")
    # Force log flush to make sure output is visible in main.py
    sys.stdout.flush()
    for handler in logger.handlers:
        handler.flush()
    
    try:
        if not VENV_PIP.exists():
            logger.error(f"Pip executable not found at {VENV_PIP}. Cannot install dependencies.")
            raise FileNotFoundError(f"Pip not found at {VENV_PIP}")

        # Read all requirements
        with open(REQUIREMENTS_PATH, 'r') as f:
            all_reqs_raw = f.readlines()
        
        # Separate core and dev requirements
        core_reqs = []
        dev_reqs = []
        in_dev_section = False
        
        for line in all_reqs_raw:
            line = line.strip()
            if not line or line.startswith('#'):
                # Check if we're entering a dev section
                if line.startswith('# Development Dependencies'):
                    in_dev_section = True
                continue
            
            # Strip comments from line
            req = line.split('#')[0].strip()
            if not req:
                continue
                
            if in_dev_section:
                dev_reqs.append(req)
            else:
                core_reqs.append(req)

        # Specific packages for staged install
        jax_reqs = [r for r in core_reqs if r.startswith('jax>=') or r.startswith('jaxlib>=')]
        discopy_req = next((r for r in core_reqs if r.startswith('discopy')), None)
        pymdp_req = next((r for r in core_reqs if 'pymdp' in r.lower()), None)
        
        other_reqs = [r for r in core_reqs if not (r.startswith('jax>=') or 
                                                 r.startswith('jaxlib>=') or 
                                                 (discopy_req and r == discopy_req) or
                                                 (pymdp_req and r == pymdp_req))]

        total_packages = len(jax_reqs) + (1 if discopy_req else 0) + (1 if pymdp_req else 0) + len(other_reqs) + len(dev_reqs)
        logger.info(f"Total packages to install: {total_packages}")
        # Force log flush
        sys.stdout.flush()
        
        # Log before major stages for better progress tracking
        logger.info(f"Beginning 4-stage dependency installation process...")
        sys.stdout.flush()  # Force flush to ensure output is visible

        # Stage 1: Install JAX and JAXLIB first
        if jax_reqs:
            logger.info(f"⏳ Stage 1/4: Installing JAX prerequisites: {', '.join(jax_reqs)}...")
            sys.stdout.flush()  # Force flush
            start_time = time.time()
            logger.info("This may take several minutes as JAX packages are large...")
            # Force flush again before long-running operation
            sys.stdout.flush()
            
            # Progress tracking for long operations
            jax_proc = subprocess.Popen(
                [str(VENV_PIP), "install", "--no-cache-dir"] + jax_reqs,
                cwd=PROJECT_ROOT, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor and provide updates during JAX installation
            while jax_proc.poll() is None:
                # Log progress every 20 seconds during JAX installation
                time.sleep(20)
                current_time = time.time()
                elapsed = current_time - start_time
                logger.info(f"⏳ Still installing JAX packages... (elapsed: {elapsed:.1f}s)")
                sys.stdout.flush()
            
            returncode = jax_proc.wait()
            stdout, stderr = jax_proc.communicate()
            
            if verbose and stdout:
                logger.debug(f"JAX installation stdout:\n{stdout[:500]}...")
            if returncode != 0:
                logger.error("Failed to install JAX prerequisites.")
                if stderr:
                    logger.error(f"Error details: {stderr[:500]}...")
                return False
            
            duration = time.time() - start_time
            logger.info(f"✅ JAX prerequisites installed/updated with --no-cache-dir (took {duration:.1f} seconds).")
            sys.stdout.flush()  # Force flush
        else:
            logger.info("Stage 1/4: No specific JAX prerequisites found in requirements.txt (expected jax>=... and jaxlib>=...).")

        # Stage 1.5: Uninstall any existing DisCoPy to ensure a clean slate for matrix extras
        if discopy_req: # Check if discopy is in requirements to attempt uninstall
            logger.info(f"🔄 Stage 1.5/4: Attempting to uninstall any existing 'discopy' before reinstalling...")
            sys.stdout.flush()  # Force flush
            try:
                run_command([str(VENV_PIP), "uninstall", "-y", "discopy"], cwd=PROJECT_ROOT, verbose=verbose, check=False) # check=False as it might not be installed
                logger.info("'discopy' (if present) uninstalled.")
            except Exception as e_uninstall:
                logger.warning(f"Could not uninstall 'discopy' (it might not have been installed): {e_uninstall}")
        
        # Stage 2: Explicitly install/reinstall discopy
        if discopy_req:
            logger.info(f"⏳ Stage 2/4: Force reinstalling and upgrading {discopy_req} into {VENV_PATH}...")
            sys.stdout.flush()  # Force flush
            start_time = time.time()
            logger.info("This may take a minute or two to resolve dependencies...")
            
            # Progress tracking for discopy installation
            discopy_proc = subprocess.Popen(
                [str(VENV_PIP), "install", "--no-cache-dir", "--force-reinstall", "--upgrade", discopy_req],
                cwd=PROJECT_ROOT, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor and provide updates during discopy installation
            while discopy_proc.poll() is None:
                # Log progress every 15 seconds
                time.sleep(15)
                current_time = time.time()
                elapsed = current_time - start_time
                logger.info(f"⏳ Still installing DisCoPy... (elapsed: {elapsed:.1f}s)")
                sys.stdout.flush()
            
            returncode = discopy_proc.wait()
            stdout, stderr = discopy_proc.communicate()
            
            if returncode != 0:
                logger.error(f"Failed to install {discopy_req}.")
                if stderr:
                    logger.error(f"Error details: {stderr[:500]}...")
                return False
                
            duration = time.time() - start_time
            logger.info(f"✅ {discopy_req} force reinstalled and upgraded successfully with --no-cache-dir (took {duration:.1f} seconds).")
            sys.stdout.flush()  # Force flush
        else:
            logger.warning("Stage 2/4: discopy not found in requirements.txt. Skipping explicit reinstall/upgrade.")
            
        # Stage 2.5: Install PyMDP separately to handle version issues
        if pymdp_req:
            logger.info(f"⏳ Stage 2.5/4: Installing PyMDP ({pymdp_req}) with special handling...")
            sys.stdout.flush()  # Force flush
            start_time = time.time()
            logger.info("Attempting several PyMDP installation approaches...")
            try:
                # First try the exact requirement
                logger.info(f"Trying installation of exact requirement: {pymdp_req}")
                sys.stdout.flush()  # Force flush
                
                result = run_command([str(VENV_PIP), "install", "--no-cache-dir", pymdp_req], 
                                     cwd=PROJECT_ROOT, verbose=verbose, check=False)
                
                if result.returncode != 0:
                    logger.warning(f"Failed to install {pymdp_req} exactly as specified.")
                    sys.stdout.flush()  # Force flush
                    
                    # Try alternate approach - get available versions
                    logger.info("Checking available versions of inferactively-pymdp...")
                    versions_result = run_command([str(VENV_PIP), "index", "versions", "inferactively-pymdp"], 
                                                 cwd=PROJECT_ROOT, verbose=verbose, check=False)
                    
                    if versions_result.returncode == 0 and versions_result.stdout:
                        logger.info(f"Available versions: {versions_result.stdout.strip()}")
                        # Try to install the latest available version
                        logger.info("Attempting to install the latest available version...")
                        sys.stdout.flush()  # Force flush
                        
                        result = run_command([str(VENV_PIP), "install", "--no-cache-dir", "inferactively-pymdp"], 
                                           cwd=PROJECT_ROOT, verbose=verbose, check=False)
                        
                        if result.returncode == 0:
                            logger.info("Successfully installed latest available version of inferactively-pymdp")
                        else:
                            logger.warning("Failed to install latest available version of inferactively-pymdp")
                    else:
                        logger.warning("Could not retrieve available versions of inferactively-pymdp")
                        
                    # Also try to install just 'pymdp' as fallback
                    logger.info("Attempting to install 'pymdp' as a fallback...")
                    sys.stdout.flush()  # Force flush
                    
                    fallback_result = run_command([str(VENV_PIP), "install", "--no-cache-dir", "pymdp"], 
                                                cwd=PROJECT_ROOT, verbose=verbose, check=False)
                    
                    if fallback_result.returncode == 0:
                        logger.info("Successfully installed 'pymdp' as fallback")
                    else:
                        logger.warning("Failed to install 'pymdp' fallback. PyMDP functionality may be limited.")
                else:
                    logger.info(f"Successfully installed {pymdp_req}")
                
                duration = time.time() - start_time
                logger.info(f"✅ PyMDP installation attempts completed (took {duration:.1f} seconds)")
                sys.stdout.flush()  # Force flush
            except Exception as e:
                logger.warning(f"Error during PyMDP installation: {e}")
                logger.warning("Continuing setup process despite PyMDP installation issues.")
        else:
            logger.warning("No PyMDP requirement found in requirements.txt.")

        # Stage 3: Install all other dependencies from requirements.txt
        if other_reqs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_other_reqs_file = Path(temp_file.name)
                for req in other_reqs:
                    temp_file.write(req + '\n')
            
            logger.info(f"⏳ Stage 3/4: Installing remaining core dependencies from a temporary list ({len(other_reqs)} packages)...")
            sys.stdout.flush()  # Force flush
            start_time = time.time()
            logger.info(f"Installing packages: {', '.join(other_reqs[:5])}{'...' if len(other_reqs) > 5 else ''}")
            sys.stdout.flush()  # Force flush
            
            # For larger batches, use progress monitoring
            if len(other_reqs) > 5:
                # Progress tracking for package installation
                other_proc = subprocess.Popen(
                    [str(VENV_PIP), "install", "--no-cache-dir", "-r", str(temp_other_reqs_file)],
                    cwd=PROJECT_ROOT, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Monitor and provide updates
                while other_proc.poll() is None:
                    # Log progress every 15 seconds
                    time.sleep(15)
                    current_time = time.time()
                    elapsed = current_time - start_time
                    logger.info(f"⏳ Still installing core packages... (elapsed: {elapsed:.1f}s)")
                    sys.stdout.flush()
                
                returncode = other_proc.wait()
                stdout, stderr = other_proc.communicate()
                
                if returncode != 0:
                    logger.error("Failed to install remaining core dependencies.")
                    if stderr:
                        logger.error(f"Error details: {stderr[:500]}...")
                    temp_other_reqs_file.unlink()
                    return False
            else:
                # For smaller batches, use simple command
                result = run_command([str(VENV_PIP), "install", "--no-cache-dir", "-r", str(temp_other_reqs_file)], cwd=PROJECT_ROOT, verbose=verbose)
                if result.returncode != 0:
                    logger.error("Failed to install remaining core dependencies.")
                    temp_other_reqs_file.unlink()
                    return False
                
            duration = time.time() - start_time
            logger.info(f"✅ Remaining core dependencies installed/updated with --no-cache-dir (took {duration:.1f} seconds).")
            sys.stdout.flush()  # Force flush
            temp_other_reqs_file.unlink() # Clean up temporary file
        else:
            logger.info("Stage 3/4: No other core dependencies to install.")
        
        # Stage 4: Install development dependencies if requested
        if dev and dev_reqs:
            logger.info(f"⏳ Stage 4/4: Installing {len(dev_reqs)} development dependencies...")
            sys.stdout.flush()  # Force flush
            start_time = time.time()
            logger.info(f"Installing dev packages: {', '.join(dev_reqs[:5])}{'...' if len(dev_reqs) > 5 else ''}")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_dev_reqs_file = Path(temp_file.name)
                for req in dev_reqs:
                    temp_file.write(req + '\n')
            
            # For larger batches, use progress monitoring
            if len(dev_reqs) > 5:
                # Progress tracking for dev package installation
                dev_proc = subprocess.Popen(
                    [str(VENV_PIP), "install", "--no-cache-dir", "-r", str(temp_dev_reqs_file)],
                    cwd=PROJECT_ROOT, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Monitor and provide updates
                while dev_proc.poll() is None:
                    # Log progress every 15 seconds
                    time.sleep(15)
                    current_time = time.time()
                    elapsed = current_time - start_time
                    logger.info(f"⏳ Still installing dev packages... (elapsed: {elapsed:.1f}s)")
                    sys.stdout.flush()
                
                returncode = dev_proc.wait()
                stdout, stderr = dev_proc.communicate()
                
                if returncode != 0:
                    logger.warning("Some development dependencies failed to install, but continuing...")
                    if stderr:
                        logger.warning(f"Error details: {stderr[:500]}...")
                else:
                    duration = time.time() - start_time
                    logger.info(f"✅ Development dependencies installed successfully (took {duration:.1f} seconds).")
                    sys.stdout.flush()  # Force flush
            else:
                # For smaller batches, use simple command            
                result = run_command([str(VENV_PIP), "install", "--no-cache-dir", "-r", str(temp_dev_reqs_file)], 
                                   cwd=PROJECT_ROOT, verbose=verbose)
                
                if result.returncode != 0:
                    logger.warning("Some development dependencies failed to install, but continuing...")
                else:
                    duration = time.time() - start_time
                    logger.info(f"✅ Development dependencies installed successfully (took {duration:.1f} seconds).")
                    sys.stdout.flush()  # Force flush
                
            temp_dev_reqs_file.unlink() # Clean up temporary file
        elif dev:
            logger.warning("Stage 4/4: No development dependencies found in requirements.txt.")
        else:
            logger.info("Stage 4/4: Skipping development dependencies (--dev not specified).")

        logger.info("🎉 All dependency installation stages completed.")
        sys.stdout.flush()  # Force flush
        
        # Verify and resolve conflicts
        logger.info("🔍 Final step: Verifying and resolving dependency conflicts...")
        sys.stdout.flush()  # Force flush
        if not resolve_dependency_conflicts(verbose):
            logger.warning("Dependency conflict resolution completed with warnings.")
        
        # Validate installed versions
        logger.info("🔍 Final step: Validating installed dependency versions...")
        sys.stdout.flush()  # Force flush
        validation_results = validate_dependency_versions(verbose)
        if not validation_results["success"]:
            logger.warning("Dependency validation completed with issues:")
            if "missing" in validation_results and validation_results["missing"]:
                logger.warning(f"Missing packages: {', '.join(validation_results['missing'])}")
            if "mismatches" in validation_results and validation_results["mismatches"]:
                logger.warning(f"Version mismatches: {', '.join(validation_results['mismatches'])}")
        else:
            logger.info("Dependency validation successful.")
        
        return True

    except subprocess.CalledProcessError as e:
        logger.error("Failed to install dependencies from requirements.txt.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during dependency installation from requirements.txt: {e}", exc_info=verbose)
        return False

# --- Callable Main Function ---
def perform_full_setup(verbose: bool = False, recreate_venv: bool = False, dev: bool = False):
    """
    Performs the full setup: creates virtual environment and installs dependencies.
    This function is intended to be called by other scripts.

    Args:
        verbose (bool): If True, enables detailed (DEBUG level) logging for this setup process.
        recreate_venv (bool): If True, recreates the virtual environment even if it already exists.
        dev (bool): If True, also installs development dependencies.
        
    Returns:
        int: 0 if successful, 1 if failed
    """
    
    # Configure logger for this module based on verbosity passed from caller
    log_level_to_set = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level_to_set)
    
    # Ensure we have a console handler with a clear format
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.propagate = False

    start_time = time.time()
    logger.info("🚀 Starting environment setup...")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"⚙️ Configuration: verbose={verbose}, recreate_venv={recreate_venv}, dev={dev}")
    sys.stdout.flush()
    
    try:
        # Phase 1: System Requirements
        logger.info("\n📋 Phase 1/4: Checking system requirements...")
        sys.stdout.flush()
        if not check_system_requirements(verbose):
            logger.error("❌ System requirements check failed")
            return 1
        logger.info("✅ System requirements check passed")
        
        # Phase 2: Virtual Environment
        logger.info("\n📋 Phase 2/4: Setting up virtual environment...")
        sys.stdout.flush()
        venv_start = time.time()
        if not create_virtual_environment(verbose, recreate_venv):
            logger.error("❌ Failed to create virtual environment")
            return 1
        venv_duration = time.time() - venv_start
        logger.info(f"✅ Virtual environment setup completed in {venv_duration:.1f}s")
        
        # Phase 3: Dependencies
        logger.info("\n📋 Phase 3/4: Installing dependencies...")
        logger.info("⏳ This may take several minutes...")
        sys.stdout.flush()
        deps_start = time.time()
        
        def progress_callback():
            current_time = time.time()
            elapsed = current_time - deps_start
            logger.info(f"⏳ Still installing dependencies... (elapsed: {elapsed:.1f}s)")
            sys.stdout.flush()
            return True
        
        deps_result = _install_dependencies_with_progress(verbose, dev, progress_callback)
        if not deps_result:
            logger.error("❌ Failed to install dependencies")
            return 1
            
        deps_duration = time.time() - deps_start
        logger.info(f"✅ Dependencies installed in {deps_duration:.1f}s")
        
        # Phase 4: Cleanup
        logger.info("\n📋 Phase 4/4: Finalizing setup...")
        sys.stdout.flush()
        cleanup_start = time.time()
        if not cleanup_virtual_environment(verbose):
            logger.warning("⚠️ Virtual environment cleanup had issues, but proceeding")
        cleanup_duration = time.time() - cleanup_start
        
        total_duration = time.time() - start_time
        logger.info("\n🎉 Setup completed successfully!")
        logger.info(f"⏱️ Total time: {total_duration:.1f}s")
        logger.info("\nTo activate the virtual environment:")
        if sys.platform == "win32":
            logger.info(f"  .\\{VENV_DIR}\\Scripts\\activate")
        else:
            logger.info(f"  source {VENV_DIR}/bin/activate")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}", exc_info=verbose)
        return 1

def _install_dependencies_with_progress(verbose: bool = False, dev: bool = False, progress_callback=None) -> bool:
    """
    Installs dependencies with progress reporting.
    
    Args:
        verbose (bool): If True, enables detailed logging
        dev (bool): If True, also installs development dependencies
        progress_callback: Optional callback function for progress updates
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        # First, upgrade pip itself
        logger.info("📦 Upgrading pip...")
        sys.stdout.flush()
        run_command([str(VENV_PIP), "install", "--upgrade", "pip"], verbose=verbose)
        
        # Install base requirements
        logger.info("📦 Installing base requirements...")
        sys.stdout.flush()
        if REQUIREMENTS_PATH.exists():
            run_command([str(VENV_PIP), "install", "-r", str(REQUIREMENTS_PATH)], verbose=verbose)
        else:
            logger.error(f"❌ Requirements file not found: {REQUIREMENTS_PATH}")
            return False
            
        # Install dev requirements if requested
        if dev and REQUIREMENTS_DEV_PATH.exists():
            logger.info("📦 Installing development requirements...")
            sys.stdout.flush()
            run_command([str(VENV_PIP), "install", "-r", str(REQUIREMENTS_DEV_PATH)], verbose=verbose)
        elif dev:
            logger.warning("⚠️ Development requirements file not found: {REQUIREMENTS_DEV_PATH}")
            
        # Special handling for PyMDP
        logger.info("📦 Checking PyMDP installation...")
        sys.stdout.flush()
        try:
            import pymdp
            logger.info("✅ PyMDP is already installed")
        except ImportError:
            logger.info("📦 Installing PyMDP...")
            sys.stdout.flush()
            run_command([str(VENV_PIP), "install", "inferactively-pymdp"], verbose=verbose)
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during dependency installation: {e}")
        return False

# --- Main Execution (for direct script running) ---

if __name__ == "__main__":
    # Basic argument parsing for direct execution
    parser = argparse.ArgumentParser(description="Direct execution of GNN project setup script (venv and dependencies).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG level) logging.")
    parser.add_argument("--recreate-venv", action="store_true", help="Recreate virtual environment even if it already exists.")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies.")
    cli_args = parser.parse_args()

    # Setup basic logging for direct run if not already configured by perform_full_setup's internal check
    # This initial basicConfig is for messages before perform_full_setup potentially reconfigures its own handler.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG if cli_args.verbose else logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )

    logger.info("Running src/setup/setup.py directly...")
    exit_code = perform_full_setup(verbose=cli_args.verbose, recreate_venv=cli_args.recreate_venv, dev=cli_args.dev)
    if exit_code == 0:
        logger.info("Direct execution of setup.py completed.")
    else:
        logger.error("Direct execution of setup.py failed.")
    sys.exit(exit_code) 