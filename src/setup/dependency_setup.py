"""
Dependency setup functions for the GNN project.

This module handles JAX installation and testing, Julia environment setup,
optional package group installation, and project structure creation.
"""

import subprocess
import sys
import shutil
from pathlib import Path
import logging
from typing import Dict, Any, List

from .uv_management import (
    PROJECT_ROOT,
    VENV_PATH,
    VENV_PYTHON,
    OPTIONAL_GROUPS,
    run_command,
    check_system_requirements,
    create_uv_environment,
    install_uv_dependencies,
    validate_uv_setup,
    save_setup_results,
)

logger = logging.getLogger(__name__)


def install_jax_and_test(verbose: bool = False) -> bool:
    """
    Ensure JAX, Optax, and Flax are installed and working using UV.
    After install, run a self-test: import JAX, print device info, check Optax/Flax, log results.

    This uses a progressive testing approach - if basic tests pass, return success even if
    advanced tests fail (which can happen due to version incompatibilities or platform issues).

    This function now tests JAX using the venv Python to avoid import issues.
    """
    import importlib.util
    import platform

    # Prevent infinite recursion by tracking attempts
    if hasattr(install_jax_and_test, '_attempts'):
        install_jax_and_test._attempts += 1
    else:
        install_jax_and_test._attempts = 0

    if install_jax_and_test._attempts > 2:
        logger.warning("JAX installation attempts exceeded limit, skipping")
        return False

    basic_tests_passed = False
    advanced_tests_passed = False

    if not VENV_PYTHON.exists():
        logger.error("Venv Python not found, cannot test JAX")
        return False

    try:
        # PHASE 1: Import test (most critical)
        test_script = """
import jax
import optax
import flax
print(f"JAX version: {jax.__version__}")
print(f"Optax version: {optax.__version__}")
print(f"Flax version: {flax.__version__}")
"""
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"✅ {line}")
        else:
            raise ImportError("JAX imports failed")

        # PHASE 2: Basic functionality test
        try:
            test_basic = """
import jax
devices = jax.devices()
print(f"Available JAX devices: {[str(d) for d in devices]}")
x = jax.numpy.array([1.0, 2.0, 3.0])
y = jax.numpy.sin(x)
sum_result = jax.numpy.sum(y)
print("JAX basic operations test passed")
"""
            result = subprocess.run(
                [str(VENV_PYTHON), "-c", test_basic],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"✅ {line}")
                basic_tests_passed = True
                logger.info("✅ JAX basic functionality verified - packages are working")
            else:
                logger.warning(f"⚠️ JAX basic tests failed: {result.stderr}")
                if verbose:
                    logger.debug(result.stderr)

        except Exception as basic_error:
            logger.warning(f"⚠️ JAX basic tests failed: {basic_error}")
            if verbose:
                import traceback
                logger.debug(traceback.format_exc())

        # PHASE 3: Advanced functionality test (non-critical)
        try:
            test_advanced = """
import jax
import optax
import flax

@jax.jit
def test_jit(x):
    return jax.numpy.sum(jax.numpy.sin(x))

result = test_jit(jax.numpy.array([1.0, 2.0, 3.0]))
print("JAX JIT compilation test passed")

def test_vmap(x):
    return jax.numpy.sin(x)

vmapped_fn = jax.vmap(test_vmap)
vmap_result = vmapped_fn(jax.numpy.array([[1.0, 2.0], [3.0, 4.0]]))
print("JAX vmap test passed")

optimizer = optax.adam(0.01)
params = {"w": jax.numpy.ones((2, 2))}
opt_state = optimizer.init(params)
print("Optax optimizer test passed")

class SimpleModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        return flax.linen.Dense(1)(x)

model = SimpleModel()
variables = model.init(jax.random.PRNGKey(0), jax.numpy.ones((1, 2)))
output = model.apply(variables, jax.numpy.ones((1, 2)))
print("Flax neural network test passed")

def test_pomdp_ops():
    belief = jax.numpy.array([0.5, 0.5])
    transition = jax.numpy.array([[0.8, 0.2], [0.2, 0.8]])
    observation = jax.numpy.array([0.9, 0.1])
    belief_pred = transition @ belief
    numerator = observation * belief_pred
    denominator = jax.numpy.sum(numerator)
    updated_belief = numerator / denominator
    return updated_belief

pomdp_result = test_pomdp_ops()
print("POMDP operations test passed")
"""
            result = subprocess.run(
                [str(VENV_PYTHON), "-c", test_advanced],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"✅ {line}")
                advanced_tests_passed = True
                logger.info("✅ JAX, Optax, and Flax advanced functionality verified")
            else:
                logger.warning(f"⚠️ JAX advanced tests failed (non-critical): {result.stderr}")
                if verbose:
                    logger.debug(result.stderr)

        except Exception as advanced_error:
            logger.warning(f"⚠️ JAX advanced tests failed (non-critical): {advanced_error}")
            if verbose:
                import traceback
                logger.debug(traceback.format_exc())

        if basic_tests_passed:
            logger.info("✅ JAX ecosystem is functional")
            return True
        else:
            logger.warning("⚠️ JAX imports succeeded but basic tests failed")
            return False

    except ImportError as e:
        logger.warning(f"JAX, Optax, or Flax not installed: {e}")

        try:
            logger.info("Attempting to repair JAX installation using UV sync...")
            install_cmd = ["uv", "sync", "--verbose"]

            if verbose:
                logger.info(f"Running: {' '.join(install_cmd)}")

            result = subprocess.run(install_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("UV sync completed successfully")
                try:
                    import jax
                    import optax
                    import flax
                    logger.info("JAX installation verified successfully")
                    return True
                except ImportError:
                    logger.warning("JAX installation succeeded but import still fails")
                    return False
            else:
                logger.error(f"Failed to install JAX using UV: {result.stderr}")
                return False

        except Exception as install_error:
            logger.error(f"Failed to install JAX using UV: {install_error}")
            return False

    except Exception as e:
        logger.error(f"JAX test failed: {e}")
        return False


def setup_julia_environment(verbose: bool = False) -> bool:
    """
    Set up Julia environment for GNN execution frameworks.

    Args:
        verbose: Enable verbose logging

    Returns:
        True if setup succeeded, False otherwise
    """
    logger.info("🔧 Setting up Julia environment for GNN execution")

    try:
        julia_path = shutil.which("julia")
        if not julia_path:
            logger.error("❌ Julia not found in PATH")
            logger.info("💡 Install Julia from: https://julialang.org/downloads/")
            return False

        version_result = subprocess.run(
            [julia_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if version_result.returncode == 0:
            version_line = version_result.stdout.strip().split('\n')[0]
            logger.info(f"✅ Julia found: {version_line}")
        else:
            logger.warning(f"⚠️ Could not determine Julia version: {version_result.stderr}")
            logger.info("✅ Julia is available, proceeding with setup")

        setup_scripts = [
            PROJECT_ROOT / "src" / "execute" / "rxinfer" / "setup_environment.jl",
            PROJECT_ROOT / "src" / "execute" / "activeinference_jl" / "setup_environment.jl"
        ]

        success_count = 0

        for script_path in setup_scripts:
            if script_path.exists():
                logger.info(f"Running Julia setup script: {script_path}")
                try:
                    result = subprocess.run(
                        [julia_path, str(script_path)],
                        cwd=script_path.parent,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode == 0:
                        logger.info(f"✅ Julia setup completed for {script_path.name}")
                        success_count += 1
                    else:
                        logger.error(f"❌ Julia setup failed for {script_path.name}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    logger.error(f"⏰ Julia setup timed out for {script_path.name}")
                except Exception as e:
                    logger.error(f"❌ Error running Julia setup for {script_path.name}: {e}")
            else:
                logger.warning(f"⚠️ Julia setup script not found: {script_path}")

        if success_count > 0:
            logger.info(f"✅ Julia environment setup completed ({success_count} frameworks configured)")
            return True
        else:
            logger.warning("⚠️ No Julia frameworks were successfully configured")
            return False

    except Exception as e:
        logger.error(f"❌ Error setting up Julia environment: {e}")
        return False


def install_optional_package_group(group_name: str, verbose: bool = False) -> bool:
    """
    Install a specific optional package group using UV sync with extras.

    Args:
        group_name: Name of the package group (see OPTIONAL_GROUPS constant)
        verbose: Enable verbose logging

    Returns:
        True if installation succeeded, False otherwise
    """
    if group_name.lower() == 'julia':
        return setup_julia_environment(verbose=verbose)

    group_aliases = {
        'ml': 'ml-ai',
        'jax': 'active-inference',
        'pymdp': 'active-inference',
    }
    normalized_name = group_aliases.get(group_name.lower(), group_name.lower())

    if normalized_name not in OPTIONAL_GROUPS:
        logger.error(f"❌ Unknown package group: {group_name}")
        logger.info(f"ℹ️ Available groups: {', '.join(OPTIONAL_GROUPS.keys())}")
        return False

    group_description = OPTIONAL_GROUPS[normalized_name]
    logger.info(f"📦 Installing '{normalized_name}' package group: {group_description}")

    try:
        sync_cmd = ["uv", "sync", "--extra", normalized_name]

        if verbose:
            sync_cmd.append("--verbose")
            logger.debug(f"Running: {' '.join(sync_cmd)}")

        result = subprocess.run(
            sync_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=600
        )

        if result.returncode == 0:
            logger.info(f"✅ Successfully installed '{normalized_name}' package group")
            if verbose and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.debug(f"  {line}")
            return True
        else:
            logger.error(f"❌ Failed to install '{normalized_name}' package group")
            logger.error(f"STDERR: {result.stderr}")
            if verbose:
                logger.error(f"STDOUT: {result.stdout}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Installation timed out for '{normalized_name}' package group")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing '{normalized_name}' package group: {e}")
        return False


def install_all_optional_packages(verbose: bool = False) -> dict:
    """
    Install all optional package groups using UV sync with 'all' extra.

    Args:
        verbose: Enable verbose logging

    Returns:
        Dictionary with installation status
    """
    logger.info("🚀 Installing ALL optional package groups via 'uv sync --extra all'...")

    try:
        sync_cmd = ["uv", "sync", "--extra", "all"]

        if verbose:
            sync_cmd.append("--verbose")
            logger.debug(f"Running: {' '.join(sync_cmd)}")

        result = subprocess.run(
            sync_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800
        )

        if result.returncode == 0:
            logger.info("✅ All optional packages installed successfully")
            if verbose and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.debug(f"  {line}")
            return {"all": True, "success": True}
        else:
            logger.error("❌ Failed to install all optional packages")
            logger.error(f"STDERR: {result.stderr}")
            return {"all": False, "success": False, "error": result.stderr}

    except subprocess.TimeoutExpired:
        logger.error("⏰ Installation timed out for all optional packages")
        return {"all": False, "success": False, "error": "timeout"}
    except Exception as e:
        logger.error(f"❌ Error installing all optional packages: {e}")
        return {"all": False, "success": False, "error": str(e)}


def install_optional_dependencies(project_root: Path, logger: logging.Logger,
                                package_groups: List[str] = None) -> bool:
    """Install optional dependencies for the project using UV sync with extras."""
    try:
        logger.info("Installing optional dependencies using UV sync")

        if not package_groups:
            package_groups = ["dev", "test", "docs"]

        for group in package_groups:
            try:
                logger.info(f"Installing {group} dependencies via UV sync")
                result = run_command(["uv", "sync", "--extra", group],
                                    cwd=project_root, check=False, verbose=True)
                if result.returncode == 0:
                    logger.info(f"✅ {group} dependencies installed successfully")
                else:
                    logger.warning(f"⚠️ {group} dependencies installation failed (exit {result.returncode})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to install {group} dependencies: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to install optional dependencies: {e}")
        return False


def create_project_structure(output_dir: Path, logger: logging.Logger) -> bool:
    """Create the standard project structure."""
    try:
        logger.info("Creating project structure")

        directories = [
            "input/gnn_files",
            "output",
            "output/logs",
            "output/temp",
            "doc",
            "tests"
        ]

        for directory in directories:
            dir_path = output_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

        config_files = {
            "input/config.yaml": "# GNN Pipeline Configuration\n",
            "output/.gitkeep": "",
            "tests/__init__.py": "# Tests package\n"
        }

        for file_path, content in config_files.items():
            full_path = output_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            logger.debug(f"Created file: {full_path}")

        logger.info("Project structure created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to create project structure: {e}")
        return False


def setup_gnn_project(project_path: str, verbose: bool = False) -> bool:
    """
    Set up a new GNN project at the specified path using UV.

    Args:
        project_path: Path where the project should be set up
        verbose: Enable verbose logging

    Returns:
        True if setup successful, False otherwise
    """
    try:
        project_path = Path(project_path)
        project_path.mkdir(parents=True, exist_ok=True)

        (project_path / "input" / "gnn_files").mkdir(parents=True, exist_ok=True)
        (project_path / "output").mkdir(parents=True, exist_ok=True)
        (project_path / "src").mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(["uv", "init"], cwd=project_path, check=True)
            logger.info(f"UV project initialized at {project_path}")
        except Exception as e:
            logger.warning(f"Could not initialize UV project: {e}")

        logger.info(f"GNN project structure created at {project_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to set up GNN project with UV: {e}")
        return False


def setup_complete_environment(
    verbose: bool = False,
    recreate: bool = False,
    install_optional: bool = False,
    optional_groups: list = None,
    output_dir: Path = None
) -> bool:
    """
    Complete environment setup with optional dependencies.

    Args:
        verbose: Enable verbose logging
        recreate: Recreate environment if it exists
        install_optional: Install optional dependencies
        optional_groups: Specific groups to install (None = all groups)
        output_dir: Output directory for setup logs

    Returns:
        True if setup successful, False otherwise
    """
    try:
        logger.info("🚀 Starting COMPLETE GNN environment setup from cold start...")

        if not check_system_requirements(verbose):
            logger.error("❌ System requirements check failed")
            return False

        if not create_uv_environment(verbose, recreate):
            logger.error("❌ UV environment creation failed")
            return False

        logger.info("📦 Installing core dependencies...")
        if not install_uv_dependencies(verbose=verbose):
            logger.warning("⚠️ Core dependency installation had issues, but continuing...")

        if install_optional:
            logger.info("\n🎁 Installing optional dependency groups...")

            if optional_groups:
                for group in optional_groups:
                    install_optional_package_group(group, verbose=verbose)
            else:
                install_all_optional_packages(verbose=verbose)

        logger.info("\n✅ Validating complete environment...")
        validation_results = validate_uv_setup(PROJECT_ROOT, logger)

        if output_dir:
            save_setup_results(output_dir, validation_results, optional_groups or [], True)

        logger.info("\n🎉 COMPLETE environment setup finished!")
        return True

    except Exception as e:
        logger.error(f"❌ Complete environment setup failed: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return False
