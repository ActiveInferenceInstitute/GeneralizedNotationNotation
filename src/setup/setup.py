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
from typing import Dict, Any

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# --- Configuration ---
VENV_DIR = ".venv"  # Name of the virtual environment directory
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_DEV_FILE = "requirements-dev.txt"

# PROJECT_ROOT should be the repository root directory.
# Since this script is in 'src/setup/', Path(__file__).parent is 'src/setup'.
# So, Path(__file__).parent.parent.parent is the repo root.
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

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

def install_jax_and_test(verbose: bool = False) -> bool:
    """
    Ensure JAX, Optax, and Flax are installed and working. Detect hardware and install the correct JAX version if needed.
    After install, run a self-test: import JAX, print device info, check Optax/Flax, log results, and raise errors if not successful.
    """
    import importlib.util
    import platform
    import sys
    import subprocess
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        import jax
        import optax
        import flax
        logger.info(f"JAX version: {jax.__version__}")
        logger.info(f"Optax version: {optax.__version__}")
        logger.info(f"Flax version: {flax.__version__}")
        
        # Check available devices
        devices = jax.devices()
        logger.info(f"Available JAX devices: {[str(d) for d in devices]}")
        
        # Test basic JAX functionality
        x = jax.numpy.array([1.0, 2.0, 3.0])
        y = jax.numpy.sin(x)
        logger.info(f"JAX basic test passed: {y}")
        
        # Test JIT compilation
        @jax.jit
        def test_jit(x):
            return jax.numpy.sum(jax.numpy.sin(x))
        
        result = test_jit(jax.numpy.array([1.0, 2.0, 3.0]))
        logger.info(f"JAX JIT test passed: {result}")
        
        # Test vmap
        def test_vmap(x):
            return jax.numpy.sin(x)
        
        vmapped_fn = jax.vmap(test_vmap)
        vmap_result = vmapped_fn(jax.numpy.array([[1.0, 2.0], [3.0, 4.0]]))
        logger.info(f"JAX vmap test passed: {vmap_result}")
        
        # Test Optax
        optimizer = optax.adam(0.01)
        params = {"w": jax.numpy.ones((2, 2))}
        opt_state = optimizer.init(params)
        logger.info("Optax test passed")
        
        # Test Flax
        class SimpleModel(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x):
                return flax.linen.Dense(1)(x)
        
        model = SimpleModel()
        variables = model.init(jax.random.PRNGKey(0), jax.numpy.ones((1, 2)))
        output = model.apply(variables, jax.numpy.ones((1, 2)))
        logger.info(f"Flax test passed: {output.shape}")
        
        # Test POMDP-like operations
        def test_pomdp_ops():
            # Belief update simulation
            belief = jax.numpy.array([0.5, 0.5])
            transition = jax.numpy.array([[0.8, 0.2], [0.2, 0.8]])
            observation = jax.numpy.array([0.9, 0.1])
            
            # Belief prediction
            belief_pred = transition @ belief
            
            # Belief update
            numerator = observation * belief_pred
            denominator = jax.numpy.sum(numerator)
            updated_belief = numerator / denominator
            
            return updated_belief
        
        pomdp_result = test_pomdp_ops()
        logger.info(f"POMDP operations test passed: {pomdp_result}")
        
        logger.info("JAX, Optax, and Flax are working correctly with POMDP capabilities")
        return True
        
    except ImportError as e:
        logger.warning(f"JAX, Optax, or Flax not installed: {e}")
        
        # Try to install JAX
        try:
            logger.info("Attempting to install JAX...")
            
            # Detect hardware
            if platform.system() == "Darwin":  # macOS
                install_cmd = ["pip", "install", "--upgrade", "jax[cpu]"]
            else:
                # Check for CUDA
                try:
                    subprocess.run(["nvidia-smi"], capture_output=True, check=True)
                    logger.info("CUDA detected, installing JAX with CUDA support")
                    install_cmd = ["pip", "install", "--upgrade", "jax[cuda12]", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"]
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.info("No CUDA detected, installing CPU-only JAX")
                    install_cmd = ["pip", "install", "--upgrade", "jax[cpu]"]
            
            if verbose:
                logger.info(f"Running: {' '.join(install_cmd)}")
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("JAX installed successfully")
                
                # Install Optax and Flax
                subprocess.run(["pip", "install", "--upgrade", "optax", "flax"], check=True)
                logger.info("Optax and Flax installed successfully")
                
                # Test again
                return install_jax_and_test(verbose)
            else:
                logger.error(f"Failed to install JAX: {result.stderr}")
                return False
                
        except Exception as install_error:
            logger.error(f"Failed to install JAX: {install_error}")
            return False
    
    except Exception as e:
        logger.error(f"JAX test failed: {e}")
        return False

# --- Callable Main Function ---
def perform_full_setup(verbose: bool = False, recreate_venv: bool = False, dev: bool = False, skip_jax_test: bool = False):
    """
    Performs the full setup: creates virtual environment and installs dependencies.
    This function is intended to be called by other scripts.

    Args:
        verbose (bool): If True, enables detailed (DEBUG level) logging for this setup process.
        recreate_venv (bool): If True, recreates the virtual environment even if it already exists.
        dev (bool): If True, also installs development dependencies.
        skip_jax_test (bool): If True, skips JAX/Optax/Flax installation testing (faster setup).
        
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
    logger.info(f"⚙️ Configuration: verbose={verbose}, recreate_venv={recreate_venv}, dev={dev}, skip_jax_test={skip_jax_test}")
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
        
        # After dependency install, ensure JAX/Optax/Flax are present and working
        if not skip_jax_test:
            if not install_jax_and_test(verbose=verbose):
                logger.warning("JAX/Optax/Flax installation or self-test failed, but continuing setup.")
        else:
            logger.warning("JAX/Optax/Flax testing was skipped. JAX functionality may not be available.")
        
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

def setup_environment(verbose: bool = False, recreate: bool = False, dev: bool = False) -> bool:
    """
    Set up the complete GNN environment.
    
    Args:
        verbose: Enable verbose logging
        recreate: Recreate virtual environment if it exists
        dev: Install development dependencies
    
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Check system requirements
        if not check_system_requirements(verbose):
            return False
        
        # Create virtual environment
        if not create_virtual_environment(verbose, recreate):
            return False
        
        # Install dependencies
        if not install_dependencies(verbose, dev):
            return False
        
        # Test JAX installation
        if not install_jax_and_test(verbose):
            logger.warning("JAX installation test failed, but continuing...")
        
        logger.info("✅ GNN environment setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False


def validate_setup() -> Dict[str, Any]:
    """
    Validate the current setup and return status information.
    
    Returns:
        Dictionary with setup validation results
    """
    validation_results = {
        "system_requirements": False,
        "virtual_environment": False,
        "dependencies": False,
        "jax_installation": False,
        "overall_status": False
    }
    
    try:
        # Check system requirements
        validation_results["system_requirements"] = check_system_requirements()
        
        # Check virtual environment
        if VENV_PATH.exists():
            validation_results["virtual_environment"] = True
        
        # Check dependencies
        try:
            versions = get_installed_package_versions()
            if versions:
                validation_results["dependencies"] = True
        except Exception:
            pass
        
        # Check JAX
        try:
            import jax
            validation_results["jax_installation"] = True
        except ImportError:
            pass
        
        # Overall status
        validation_results["overall_status"] = all([
            validation_results["system_requirements"],
            validation_results["virtual_environment"],
            validation_results["dependencies"]
        ])
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
    
    return validation_results


def get_setup_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the current setup.
    
    Returns:
        Dictionary with setup information
    """
    info = {
        "project_root": str(PROJECT_ROOT),
        "virtual_environment_path": str(VENV_PATH),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "setup_status": validate_setup()
    }
    
    # Add package versions if available
    try:
        info["installed_packages"] = get_installed_package_versions()
    except Exception:
        info["installed_packages"] = {}
    
    return info


def cleanup_setup() -> bool:
    """
    Clean up the setup (remove virtual environment).
    
    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        if VENV_PATH.exists():
            shutil.rmtree(VENV_PATH)
            logger.info(f"Removed virtual environment at {VENV_PATH}")
            return True
        else:
            logger.info("No virtual environment to clean up")
            return True
    except Exception as e:
        logger.error(f"Failed to clean up setup: {e}")
        return False


def setup_gnn_project(project_path: str, verbose: bool = False) -> bool:
    """
    Set up a new GNN project at the specified path.
    
    Args:
        project_path: Path where the project should be set up
        verbose: Enable verbose logging
    
    Returns:
        True if setup successful, False otherwise
    """
    try:
        project_path = Path(project_path)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create basic project structure
        (project_path / "input" / "gnn_files").mkdir(parents=True, exist_ok=True)
        (project_path / "output").mkdir(parents=True, exist_ok=True)
        (project_path / "src").mkdir(parents=True, exist_ok=True)
        
        # Create basic configuration files
        config_file = project_path / "config.yaml"
        if not config_file.exists():
            with open(config_file, 'w') as f:
                f.write("# GNN Project Configuration\n")
                f.write("project_name: gnn_project\n")
                f.write("version: 1.0.0\n")
        
        logger.info(f"GNN project structure created at {project_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up GNN project: {e}")
        return False 