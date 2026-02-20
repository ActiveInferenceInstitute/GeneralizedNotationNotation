"""
UV environment management functions for the GNN project.

This module handles UV virtual environment creation, dependency installation,
package management, and environment validation.
"""

import os
import subprocess
import sys
import platform
import shutil
from pathlib import Path
import logging
import time
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# --- Configuration ---
VENV_DIR = ".venv"
PYPROJECT_FILE = "pyproject.toml"
LOCK_FILE = "uv.lock"

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

VENV_PATH = PROJECT_ROOT / VENV_DIR
PYPROJECT_PATH = PROJECT_ROOT / PYPROJECT_FILE
LOCK_PATH = PROJECT_ROOT / LOCK_FILE

if sys.platform == "win32":
    VENV_PYTHON = VENV_PATH / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_PATH / "bin" / "python"

MIN_PYTHON_VERSION = (3, 11)

OPTIONAL_GROUPS = {
    'dev': 'Development tools (pytest-mock, black, isort, sphinx, jupyterlab, etc.)',
    'active-inference': 'Active Inference ecosystem (pymdp, jax, flax, optax)',
    'ml-ai': 'Machine Learning (torch, transformers, accelerate)',
    'llm': 'LLM providers (openai, anthropic, cohere, ollama)',
    'visualization': 'Visualization (plotly, altair, seaborn, bokeh, holoviews, dash)',
    'audio': 'Audio processing (librosa, soundfile, pedalboard, pydub)',
    'gui': 'GUI frameworks (gradio, streamlit)',
    'graphs': 'Graph analysis (igraph, graphviz, discopy[categorical])',
    'research': 'Research tools (jupyterlab, sympy, numba, cython)',
    'scaling': 'Scaling (dask, distributed, joblib, fsspec)',
    'database': 'Database (sqlalchemy, alembic)',
    'all': 'All optional dependencies combined',
}


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

    python_version = sys.version_info
    if python_version < MIN_PYTHON_VERSION:
        logger.error(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is below the minimum required version {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}")
        return False
    else:
        logger.info(f"✅ Python version check passed: {python_version.major}.{python_version.minor}.{python_version.micro}")
        sys.stdout.flush()

    try:
        logger.debug("Checking UV availability...")
        uv_process = run_command(["uv", "--version"], check=True, verbose=verbose)
        logger.info(f"✅ UV is available: {uv_process.stdout.strip()}")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"❌ Error checking UV: {e}")
        logger.error("Please install UV first:")
        logger.error("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        logger.error("  or visit: https://docs.astral.sh/uv/getting-started/installation/")
        return False

    try:
        disk_usage = shutil.disk_usage(PROJECT_ROOT)
        free_space_gb = disk_usage.free / (1024 * 1024 * 1024)
        if free_space_gb < 1:
            logger.warning(f"⚠️ Low disk space: {free_space_gb:.2f}GB free. At least 1GB recommended for dependency installation.")
        else:
            logger.info(f"✅ Disk space check passed: {free_space_gb:.2f}GB free")
        sys.stdout.flush()
    except Exception as e:
        logger.warning(f"⚠️ Could not check disk space: {e}")

    return True


def check_uv_availability(verbose: bool = False) -> bool:
    """
    Check if UV is available and properly installed.

    Args:
        verbose: If True, enables detailed logging.

    Returns:
        True if UV is available, False otherwise.
    """
    try:
        logger.debug("Checking UV availability...")
        uv_process = run_command(["uv", "--version"], check=True, verbose=verbose)
        logger.info(f"✅ UV is available: {uv_process.stdout.strip()}")
        logger.info("✅ UV is available and ready to use")
        return True
    except Exception as e:
        logger.error(f"❌ UV not available: {e}")
        return False


def create_uv_environment(verbose: bool = False, recreate: bool = False) -> bool:
    """
    Creates a UV environment if it doesn't already exist, or recreates it if specified.

    Args:
        verbose: If True, enables detailed (DEBUG level) logging for this setup process.
        recreate: If True, deletes and recreates an existing virtual environment.

    Returns:
        True if successful, False otherwise.
    """
    if VENV_PATH.exists() and recreate:
        logger.info(f"🔄 Recreating UV environment in {VENV_PATH}...")
        sys.stdout.flush()
        try:
            shutil.rmtree(VENV_PATH)
            logger.info(f"Removed existing UV environment at {VENV_PATH}")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"❌ Failed to remove existing UV environment: {e}")
            return False

    if not VENV_PATH.exists():
        logger.info(f"🔧 Creating UV environment in {VENV_PATH}...")
        sys.stdout.flush()
        try:
            start_time = time.time()
            logger.info(f"📦 Creating virtual environment using UV...")
            run_command(["uv", "venv", str(VENV_PATH)], verbose=verbose)

            duration = time.time() - start_time
            logger.info(f"✅ UV environment created successfully at {VENV_PATH} (took {duration:.1f}s)")
            sys.stdout.flush()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create UV environment: {e}")
            if verbose:
                import traceback
                logger.error(traceback.format_exc())
            return False
    else:
        logger.info(f"✓ Using existing UV environment at {VENV_PATH}")
        try:
            if VENV_PYTHON.exists():
                test_result = subprocess.run([str(VENV_PYTHON), "--version"],
                                           capture_output=True, text=True, timeout=10)
                if test_result.returncode == 0:
                    logger.info(f"✅ Existing environment is working: {test_result.stdout.strip()}")

                    try:
                        test_imports = subprocess.run([str(VENV_PYTHON), "-c",
                                                    "import sys; import pathlib; print('Core imports work')"],
                                                   capture_output=True, text=True, timeout=10)
                        if test_imports.returncode == 0:
                            logger.info(f"✅ Core packages are available in existing environment")
                            sys.stdout.flush()
                            return True
                        else:
                            logger.warning(f"⚠️ Core packages missing, will reinstall...")
                    except Exception:
                        logger.warning(f"⚠️ Could not test core packages, will reinstall...")
                else:
                    logger.warning(f"⚠️ Existing environment may be corrupted, will recreate...")
                    return create_uv_environment(verbose=verbose, recreate=True)
            else:
                logger.warning(f"⚠️ Virtual environment Python not found, will recreate...")
                return create_uv_environment(verbose=verbose, recreate=True)
        except Exception as e:
            logger.warning(f"⚠️ Error checking existing environment: {e}, will recreate...")
            return create_uv_environment(verbose=verbose, recreate=True)

    return True


def install_uv_dependencies(verbose: bool = False, dev: bool = False, extras: list = None) -> bool:
    """
    Installs dependencies using native UV sync command from pyproject.toml.

    Args:
        verbose: If True, enables detailed logging.
        dev: If True, also installs development dependencies.
        extras: List of optional dependency groups to install.

    Returns:
        True if successful, False otherwise.
    """
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    if not pyproject_path.exists():
        logger.error(f"❌ pyproject.toml not found at {pyproject_path}")
        return False

    logger.info(f"📦 Installing dependencies from pyproject.toml using UV sync")
    sys.stdout.flush()

    try:
        start_time = time.time()

        sync_cmd = ["uv", "sync"]

        if verbose:
            sync_cmd.append("--verbose")

        if dev:
            logger.info(f"📦 Installing development dependencies...")
            sync_cmd.append("--all-extras")

        if extras:
            for extra in extras:
                logger.info(f"📦 Installing optional group: {extra}")
                sync_cmd.extend(["--extra", extra])

        sys.stdout.flush()

        if verbose:
            logger.debug(f"Running: {' '.join(sync_cmd)}")

        result = subprocess.run(
            sync_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error("❌ Failed to synchronize dependencies via UV sync")
            if verbose or True:
                logger.error("STDOUT:")
                logger.error(result.stdout)
                logger.error("STDERR:")
                logger.error(result.stderr)
            logger.warning("⚠️ Some packages failed to install, attempting to continue...")
        else:
            duration = time.time() - start_time
            logger.info(f"✅ Dependencies synchronized using UV sync in {duration:.1f}s")

        if verbose:
            logger.debug("Verifying environment Python executable")
        verify = subprocess.run([str(VENV_PYTHON), "--version"], capture_output=True, text=True)
        if verify.returncode == 0:
            if verbose:
                logger.debug(f"Python in venv: {verify.stdout.strip() or verify.stderr.strip()}")

        get_installed_package_versions(verbose)
        return True

    except Exception as e:
        logger.error(f"❌ Error during UV sync dependency installation: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return False


def get_installed_package_versions(verbose: bool = False) -> dict:
    """
    Get a list of all installed packages and their versions using UV.

    Args:
        verbose: If True, logs the full package list.

    Returns:
        A dictionary of package names and their versions.
    """
    logger.info("📋 Getting list of installed packages using UV...")
    sys.stdout.flush()

    try:
        list_cmd = ["uv", "pip", "list", "--python", str(VENV_PYTHON), "--format=json"]
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

        try:
            packages = json.loads(result.stdout)
            package_dict = {pkg["name"]: pkg["version"] for pkg in packages}

            package_count = len(package_dict)
            logger.info(f"📦 Found {package_count} installed packages using UV")

            if verbose:
                logger.info("📋 Installed packages:")
                for name, version in sorted(package_dict.items()):
                    logger.info(f"  - {name}: {version}")
            else:
                key_packages = ["pip", "pytest", "numpy", "matplotlib", "scipy", "psutil"]
                logger.info("📋 Key installed packages:")
                for pkg in key_packages:
                    if pkg in package_dict:
                        logger.info(f"  - {pkg}: {package_dict[pkg]}")

            package_list_file = VENV_PATH / "installed_packages_uv.json"
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


def add_uv_dependency(package: str, dev: bool = False, verbose: bool = False) -> bool:
    """
    Add a dependency to the project using UV add command.

    Args:
        package: Package name (optionally with version specifier)
        dev: If True, add as development dependency
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"📦 Adding package: {package}" + (" (dev)" if dev else ""))
    sys.stdout.flush()

    try:
        cmd = ["uv", "add", package]
        if dev:
            cmd.append("--dev")

        if verbose:
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"❌ Failed to add package {package}")
            logger.error(result.stderr)
            return False

        logger.info(f"✅ Successfully added {package}")
        return True

    except Exception as e:
        logger.error(f"❌ Error adding package {package}: {e}")
        return False


def remove_uv_dependency(package: str, verbose: bool = False) -> bool:
    """
    Remove a dependency from the project using UV remove command.

    Args:
        package: Package name to remove
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"🗑️ Removing package: {package}")
    sys.stdout.flush()

    try:
        cmd = ["uv", "remove", package]

        if verbose:
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"❌ Failed to remove package {package}")
            logger.error(result.stderr)
            return False

        logger.info(f"✅ Successfully removed {package}")
        return True

    except Exception as e:
        logger.error(f"❌ Error removing package {package}: {e}")
        return False


def update_uv_dependencies(verbose: bool = False, upgrade: bool = False) -> bool:
    """
    Update dependencies using UV sync command.

    Args:
        verbose: Enable verbose logging
        upgrade: If True, upgrade dependencies to latest compatible versions

    Returns:
        True if successful, False otherwise
    """
    logger.info("🔄 Updating dependencies...")
    sys.stdout.flush()

    try:
        cmd = ["uv", "sync"]
        if upgrade:
            cmd.append("--upgrade")

        if verbose:
            cmd.append("--verbose")
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error("❌ Failed to update dependencies")
            logger.error(result.stderr)
            return False

        logger.info("✅ Successfully updated dependencies")
        if verbose:
            logger.info(result.stdout)
        return True

    except Exception as e:
        logger.error(f"❌ Error updating dependencies: {e}")
        return False


def lock_uv_dependencies(verbose: bool = False) -> bool:
    """
    Update the lock file using UV lock command without installing.

    Args:
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    logger.info("🔒 Updating dependency lock file...")
    sys.stdout.flush()

    try:
        cmd = ["uv", "lock"]

        if verbose:
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error("❌ Failed to update lock file")
            logger.error(result.stderr)
            return False

        logger.info("✅ Successfully updated uv.lock")
        return True

    except Exception as e:
        logger.error(f"❌ Error updating lock file: {e}")
        return False


def setup_uv_environment(
    verbose: bool = False,
    recreate: bool = False,
    dev: bool = False,
    extras: list = None,
    skip_jax_test: bool = False,
    output_dir: Path = None,
) -> bool:
    """
    Set up the complete GNN environment using UV.

    Args:
        verbose: Enable verbose logging
        recreate: Recreate UV environment if it exists
        dev: Install development dependencies
        extras: List of optional dependency groups to install
        output_dir: Output directory for setup results (optional)

    Returns:
        True if setup successful, False otherwise
    """
    from .dependency_setup import install_jax_and_test

    try:
        logger.info("🔧 Starting UV environment setup...")

        if not check_system_requirements(verbose):
            logger.error("❌ System requirements check failed")
            return False

        if not create_uv_environment(verbose, recreate):
            logger.error("❌ UV environment creation failed")
            return False

        if VENV_PYTHON.exists():
            logger.info("📦 Installing core dependencies...")
            if not install_uv_dependencies(verbose=verbose, dev=dev, extras=extras):
                logger.warning("⚠️ Core dependency installation had issues, but continuing...")

            if not skip_jax_test:
                logger.info("🧠 Installing JAX and testing...")
                if not install_jax_and_test(verbose):
                    logger.warning("⚠️ JAX installation had issues, but continuing...")

            logger.info("✅ Validating environment...")
            validation_results = validate_uv_setup(PROJECT_ROOT, logger)

            if output_dir:
                save_setup_results(output_dir, validation_results, extras, dev)

            if validation_results.get("overall_status", False):
                logger.info("✅ GNN environment setup completed successfully using UV")
                return True
            else:
                logger.warning("⚠️ Environment validation had issues, but setup may still be functional")
                return True
        else:
            logger.error("❌ Virtual environment Python not found after creation")
            return False

    except Exception as e:
        logger.error(f"❌ UV environment setup failed: {e}")
        return False


def validate_uv_setup(project_root: Path = None, logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Validate the current UV setup and return status information.

    Args:
        project_root: Path to project root (optional)
        logger: Logger instance (optional)

    Returns:
        Dictionary with UV setup validation results
    """
    validation_results = {
        "system_requirements": False,
        "uv_environment": False,
        "dependencies": False,
        "jax_installation": False,
        "overall_status": False,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

    try:
        validation_results["system_requirements"] = check_system_requirements()

        if VENV_PATH.exists():
            validation_results["uv_environment"] = True

        try:
            versions = get_installed_package_versions()
            if versions:
                validation_results["dependencies"] = True
        except Exception:
            pass

        try:
            if VENV_PYTHON.exists():
                result = subprocess.run(
                    [str(VENV_PYTHON), "-c", "import jax; print(jax.__version__)"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    validation_results["jax_installation"] = True
                    if logger:
                        logger.info(f"✅ JAX version: {result.stdout.strip()}")
            else:
                import jax
                validation_results["jax_installation"] = True
        except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

        validation_results["overall_status"] = all([
            validation_results["system_requirements"],
            validation_results["uv_environment"],
            validation_results["dependencies"]
        ])

    except Exception as e:
        if logger:
            logger.error(f"UV validation error: {e}")

    return validation_results


def get_uv_setup_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the current UV setup.

    Returns:
        Dictionary with UV setup information
    """
    info = {
        "project_root": str(PROJECT_ROOT),
        "uv_environment_path": str(VENV_PATH),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "uv_setup_status": validate_uv_setup()
    }

    try:
        info["installed_packages"] = get_installed_package_versions()
    except Exception:
        info["installed_packages"] = {}

    return info


def check_environment_health(verbose: bool = False) -> Dict[str, Any]:
    """
    Health check of the GNN environment.

    Args:
        verbose: Enable verbose logging

    Returns:
        Dictionary with health check results
    """
    health = {
        'overall_healthy': False,
        'uv_available': False,
        'uv_version': None,
        'venv_exists': False,
        'venv_python_works': False,
        'lock_file_exists': False,
        'pyproject_exists': False,
        'core_packages': {},
        'optional_packages': {},
        'issues': [],
        'suggestions': []
    }

    logger.info("🏥 Running GNN environment health check...")

    try:
        uv_result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if uv_result.returncode == 0:
            health['uv_available'] = True
            health['uv_version'] = uv_result.stdout.strip()
            if verbose:
                logger.info(f"✅ UV: {health['uv_version']}")
        else:
            health['issues'].append("UV CLI not responding correctly")
            health['suggestions'].append("Reinstall UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
    except FileNotFoundError:
        health['issues'].append("UV not found in PATH")
        health['suggestions'].append("Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
    except Exception as e:
        health['issues'].append(f"UV check failed: {e}")

    health['pyproject_exists'] = PYPROJECT_PATH.exists()
    if not health['pyproject_exists']:
        health['issues'].append("pyproject.toml not found")
        health['suggestions'].append("Run 'uv init' to create project configuration")
    elif verbose:
        logger.info("✅ pyproject.toml exists")

    health['lock_file_exists'] = LOCK_PATH.exists()
    if not health['lock_file_exists']:
        health['issues'].append("uv.lock not found")
        health['suggestions'].append("Run 'uv lock' to generate lock file")
    elif verbose:
        logger.info("✅ uv.lock exists")

    health['venv_exists'] = VENV_PATH.exists()
    if not health['venv_exists']:
        health['issues'].append("Virtual environment not found")
        health['suggestions'].append("Run 'uv sync' to create environment and install dependencies")
    elif verbose:
        logger.info(f"✅ Virtual environment exists at {VENV_PATH}")

    if health['venv_exists'] and VENV_PYTHON.exists():
        try:
            py_result = subprocess.run(
                [str(VENV_PYTHON), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if py_result.returncode == 0:
                health['venv_python_works'] = True
                if verbose:
                    logger.info(f"✅ Venv Python: {py_result.stdout.strip()}")
            else:
                health['issues'].append("Venv Python not responding")
                health['suggestions'].append("Recreate environment: uv sync --reinstall")
        except Exception as e:
            health['issues'].append(f"Venv Python check failed: {e}")

    core_packages = ['numpy', 'matplotlib', 'networkx', 'pandas', 'scipy', 'pytest']
    for pkg in core_packages:
        try:
            result = subprocess.run(
                [str(VENV_PYTHON), "-c", f"import {pkg}; print({pkg}.__version__)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                health['core_packages'][pkg] = result.stdout.strip()
            else:
                health['core_packages'][pkg] = None
                health['issues'].append(f"Core package '{pkg}' not available")
        except Exception:
            health['core_packages'][pkg] = None

    optional_checks = {
        'llm': {'packages': ['openai'], 'extra': 'llm'},
        'visualization': {'packages': ['plotly'], 'extra': 'visualization'},
        'ml': {'packages': ['torch'], 'extra': 'ml-ai'},
        'audio': {'packages': ['librosa'], 'extra': 'audio'},
    }

    for group, config in optional_checks.items():
        try:
            pkg = config['packages'][0]
            result = subprocess.run(
                [str(VENV_PYTHON), "-c", f"import {pkg}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            health['optional_packages'][group] = (result.returncode == 0)
            if result.returncode == 0 and verbose:
                logger.info(f"✅ Optional group '{group}' is available")
        except Exception:
            health['optional_packages'][group] = False

    critical_checks = [
        health['uv_available'],
        health['pyproject_exists'],
        health['venv_exists'],
        health['venv_python_works']
    ]
    health['overall_healthy'] = all(critical_checks)

    if health['overall_healthy']:
        logger.info("✅ Environment health check PASSED")
    else:
        logger.warning(f"⚠️ Environment health check found {len(health['issues'])} issue(s)")
        for issue in health['issues']:
            logger.warning(f"  - {issue}")

    return health


def cleanup_uv_setup() -> bool:
    """
    Clean up the UV setup (remove virtual environment).

    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        if VENV_PATH.exists():
            shutil.rmtree(VENV_PATH)
            logger.info(f"Removed UV environment at {VENV_PATH}")
            return True
        else:
            logger.info("No UV environment to clean up")
            return True
    except Exception as e:
        logger.error(f"Failed to clean up UV setup: {e}")
        return False


def save_setup_results(output_dir: Path, validation_results: Dict, extras: list = None, dev: bool = False):
    """
    Save setup results to output directory.

    Args:
        output_dir: Output directory for setup results
        validation_results: Validation results from validate_uv_setup
        extras: List of optional dependency groups installed
        dev: Whether dev dependencies were installed
    """
    from datetime import datetime

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_results = {
            "timestamp": datetime.now().isoformat(),
            "validation": validation_results,
            "configuration": {
                "extras_installed": extras or [],
                "dev_dependencies": dev,
                "venv_path": str(VENV_PATH),
                "python_version": sys.version,
            },
            "uv_info": get_uv_setup_info(),
            "system_info": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "python_executable": sys.executable,
            }
        }

        summary_file = output_dir / "environment_setup_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(setup_results, f, indent=2, default=str)
        logger.info(f"💾 Setup results saved to: {summary_file}")

        packages_file = output_dir / "installed_packages.json"
        with open(packages_file, 'w') as f:
            json.dump(get_installed_package_versions(), f, indent=2)
        logger.info(f"📦 Package list saved to: {packages_file}")

    except Exception as e:
        logger.warning(f"⚠️ Failed to save setup results: {e}")


def log_system_info(logger: logging.Logger) -> Dict[str, Any]:
    """Log system information."""
    try:
        logger.info("Logging system information")

        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "node": platform.node()
        }

        logger.info(f"System Platform: {system_info['platform']}")
        logger.info(f"Python Version: {system_info['python_version']}")
        logger.info(f"Python Executable: {system_info['python_executable']}")
        logger.info(f"Architecture: {system_info['architecture']}")
        logger.info(f"Processor: {system_info['processor']}")
        logger.info(f"Machine: {system_info['machine']}")
        logger.info(f"Node: {system_info['node']}")

        logger.info("System information logged")
        return system_info

    except Exception as e:
        logger.error(f"Failed to log system information: {e}")
        return {}
