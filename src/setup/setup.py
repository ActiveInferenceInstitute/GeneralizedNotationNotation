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
import time
import json

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
    sys.stdout.flush()
    
    # Check Python version
    python_version = sys.version_info
    if python_version < MIN_PYTHON_VERSION:
        logger.error(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is below the minimum required version {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}")
        return False
    else:
        logger.info(f"✅ Python version check passed: {python_version.major}.{python_version.minor}.{python_version.micro}")
        sys.stdout.flush()
    
    # Check pip availability
    try:
        logger.debug("Checking pip availability...")
        pip_process = run_command([sys.executable, "-m", "pip", "--version"], check=True, verbose=verbose)
        logger.info(f"✅ pip is available: {pip_process.stdout.strip()}")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"❌ Error checking pip: {e}")
        return False
    
    # Check for venv module
    try:
        logger.debug("Checking venv module availability...")
        run_command([sys.executable, "-c", "import venv"], check=True, verbose=verbose)
        logger.info("✅ venv module is available")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"❌ Error: venv module not available: {e}")
        return False
    
    # Check disk space (at least 1GB free in the project directory)
    try:
        disk_usage = shutil.disk_usage(PROJECT_ROOT)
        free_space_gb = disk_usage.free / (1024 * 1024 * 1024)  # Convert to GB
        if free_space_gb < 1:
            logger.warning(f"⚠️ Low disk space: {free_space_gb:.2f}GB free. At least 1GB recommended for dependency installation.")
        else:
            logger.info(f"✅ Disk space check passed: {free_space_gb:.2f}GB free")
        sys.stdout.flush()
    except Exception as e:
        logger.warning(f"⚠️ Could not check disk space: {e}")
    
    return True

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
        sys.stdout.flush()
        try:
            shutil.rmtree(VENV_PATH)
            logger.info(f"Removed existing virtual environment at {VENV_PATH}")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"❌ Failed to remove existing virtual environment: {e}")
            return False
    
    if not VENV_PATH.exists():
        logger.info(f"🔧 Creating virtual environment in {VENV_PATH}...")
        sys.stdout.flush()
        try:
            start_time = time.time()
            run_command([sys.executable, "-m", "venv", VENV_DIR], cwd=PROJECT_ROOT, verbose=verbose)
            duration = time.time() - start_time
            logger.info(f"✅ Virtual environment created successfully at {VENV_PATH} (took {duration:.1f}s)")
            sys.stdout.flush()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    else:
        logger.info(f"✓ Using existing virtual environment at {VENV_PATH}")
        sys.stdout.flush()
        return True

def install_dependencies(verbose: bool = False, dev: bool = False) -> bool:
    """
    Installs dependencies from requirements.txt into the virtual environment.
    Uses a streaming approach to show progress during installation.

    Args:
        verbose: If True, enables detailed logging.
        dev: If True, also installs development dependencies.
        
    Returns:
        True if successful, False otherwise.
    """
    if not VENV_PIP.exists():
        logger.error(f"❌ pip not found in virtual environment at {VENV_PIP}")
        return False

    if not REQUIREMENTS_PATH.exists():
        logger.error(f"❌ Requirements file not found at {REQUIREMENTS_PATH}")
        return False

    logger.info(f"📦 Installing dependencies from {REQUIREMENTS_PATH}...")
    sys.stdout.flush()
    
    try:
        # First upgrade pip itself
        logger.info("📦 Upgrading pip in virtual environment...")
        sys.stdout.flush()
        upgrade_pip_cmd = [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"]
        process = subprocess.Popen(
            upgrade_pip_cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logger.debug(line.strip())
                if "Successfully installed" in line:
                    logger.info("✅ pip upgraded successfully")
                    sys.stdout.flush()
        
        # Install main dependencies with progress reporting
        logger.info("📦 Installing main dependencies (this may take several minutes)...")
        sys.stdout.flush()
        
        install_cmd = [str(VENV_PIP), "install", "-r", str(REQUIREMENTS_PATH)]
        start_time = time.time()
        
        process = subprocess.Popen(
            install_cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Variables to track progress
        last_progress_time = start_time
        progress_interval = 15  # seconds
        completed = False
        installing_package = None
        
        # Stream output and provide periodic progress updates
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            # Process the line for progress information
            if line:
                line = line.strip()
                logger.debug(line)
                
                # Extract package being installed
                if "Collecting" in line:
                    package = line.split("Collecting ")[1].split()[0]
                    installing_package = package
                    logger.info(f"📦 Collecting {package}...")
                    sys.stdout.flush()
                elif "Installing" in line and ".whl" in line:
                    logger.info(f"📦 Installing wheel for {installing_package or 'package'}...")
                    sys.stdout.flush()
                elif "Successfully installed" in line:
                    packages = line.replace("Successfully installed", "").strip()
                    logger.info(f"✅ Successfully installed: {packages}")
                    sys.stdout.flush()
                    completed = True
            
            # Periodic progress updates for long-running processes
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                elapsed = current_time - start_time
                logger.info(f"⏳ Still installing dependencies... (elapsed: {elapsed:.1f}s)")
                sys.stdout.flush()
                last_progress_time = current_time
        
        # Check final status
        if process.returncode != 0:
            logger.error(f"❌ Failed to install dependencies (exit code: {process.returncode})")
            return False
        
        if not completed:
            logger.info("✅ Dependencies installation completed")
        
        # Install dev dependencies if requested
        if dev and REQUIREMENTS_DEV_PATH.exists():
            logger.info(f"📦 Installing development dependencies from {REQUIREMENTS_DEV_PATH}...")
            sys.stdout.flush()
            
            dev_install_cmd = [str(VENV_PIP), "install", "-r", str(REQUIREMENTS_DEV_PATH)]
            dev_process = subprocess.run(
                dev_install_cmd, 
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False
            )
            
            if dev_process.returncode == 0:
                logger.info("✅ Development dependencies installed successfully")
                sys.stdout.flush()
            else:
                logger.warning(f"⚠️ Some development dependencies failed to install")
                if verbose:
                    logger.warning(f"Details: {dev_process.stderr}")
                sys.stdout.flush()
        
        duration = time.time() - start_time
        logger.info(f"✅ All dependencies installed successfully (took {duration:.1f}s)")
        sys.stdout.flush()
        
        # Report installed package versions
        get_installed_package_versions(verbose)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during dependency installation: {e}")
        sys.stdout.flush()
        return False

def get_installed_package_versions(verbose: bool = False) -> dict:
    """
    Get a list of all installed packages and their versions in the virtual environment.
    
    Args:
        verbose: If True, logs the full package list.
        
    Returns:
        A dictionary of package names and their versions.
    """
    if not VENV_PIP.exists():
        logger.warning("⚠️ Cannot list packages: pip not found in virtual environment")
        return {}
    
    logger.info("📋 Getting list of installed packages...")
    sys.stdout.flush()
    
    try:
        # Get list of installed packages
        list_cmd = [str(VENV_PIP), "list", "--format=json"]
        result = subprocess.run(
            list_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.warning(f"⚠️ Failed to get package list (exit code: {result.returncode})")
            if verbose:
                logger.warning(f"Error: {result.stderr.strip()}")
            return {}
        
        # Parse JSON output
        try:
            packages = json.loads(result.stdout)
            package_dict = {pkg["name"]: pkg["version"] for pkg in packages}
            
            # Count and log summary
            package_count = len(package_dict)
            logger.info(f"📦 Found {package_count} installed packages in the virtual environment")
            
            # Log all packages if verbose
            if verbose:
                logger.info("📋 Installed packages:")
                for name, version in sorted(package_dict.items()):
                    logger.info(f"  - {name}: {version}")
            else:
                # Log just a few key packages even in non-verbose mode
                key_packages = ["pip", "pytest", "numpy", "matplotlib", "scipy"]
                logger.info("📋 Key installed packages:")
                for pkg in key_packages:
                    if pkg in package_dict:
                        logger.info(f"  - {pkg}: {package_dict[pkg]}")
            
            # Save package list to a file in the virtual environment directory
            package_list_file = VENV_PATH / "installed_packages.json"
            with open(package_list_file, 'w') as f:
                json.dump(package_dict, f, indent=2, sort_keys=True)
            logger.info(f"📄 Full package list saved to: {package_list_file}")
            
            return package_dict
            
        except json.JSONDecodeError:
            logger.warning("⚠️ Failed to parse package list JSON")
            if verbose:
                logger.warning(f"Output: {result.stdout}")
            return {}
            
    except Exception as e:
        logger.error(f"❌ Error while getting package versions: {e}")
        return {}

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
        logger.info("\n📋 Phase 1/3: Checking system requirements...")
        sys.stdout.flush()
        if not check_system_requirements(verbose):
            logger.error("❌ System requirements check failed")
            sys.stdout.flush()
            return 1
        logger.info("✅ System requirements check passed")
        sys.stdout.flush()
        
        # Phase 2: Virtual Environment
        logger.info("\n📋 Phase 2/3: Setting up virtual environment...")
        sys.stdout.flush()
        venv_start = time.time()
        if not create_virtual_environment(verbose, recreate_venv):
            logger.error("❌ Failed to create virtual environment")
            sys.stdout.flush()
            return 1
        venv_duration = time.time() - venv_start
        logger.info(f"✅ Virtual environment setup completed in {venv_duration:.1f}s")
        sys.stdout.flush()
        
        # Phase 3: Dependencies
        logger.info("\n📋 Phase 3/3: Installing dependencies...")
        logger.info("⏳ This may take several minutes...")
        sys.stdout.flush()
        deps_start = time.time()
        
        if not install_dependencies(verbose, dev):
            logger.error("❌ Failed to install dependencies")
            sys.stdout.flush()
            return 1
            
        deps_duration = time.time() - deps_start
        logger.info(f"✅ Dependencies installed in {deps_duration:.1f}s")
        sys.stdout.flush()
        
        total_duration = time.time() - start_time
        logger.info("\n🎉 Setup completed successfully!")
        logger.info(f"⏱️ Total time: {total_duration:.1f}s")
        logger.info("\nTo activate the virtual environment:")
        if sys.platform == "win32":
            logger.info(f"  .\\{VENV_DIR}\\Scripts\\activate")
        else:
            logger.info(f"  source {VENV_DIR}/bin/activate")
        sys.stdout.flush()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.stdout.flush()
        return 1

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