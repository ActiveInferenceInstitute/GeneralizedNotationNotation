"""
JAX Runner for Executing Rendered JAX POMDP Scripts

Discovers and runs JAX-generated scripts, manages device selection, logs hardware/software info, and benchmarks performance.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
@Web: https://pfjax.readthedocs.io
"""
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Union, Optional

logger = logging.getLogger(__name__)

def initialize_jax_devices() -> list:
    """Initialize and return available JAX devices, falling back to CPU on errors.

    This matches tests expecting a callable that always returns at least one device-like object.
    """
    try:
        import jax
        try:
            return jax.devices()
        except Exception:
            # Fallback to CPU-like stub
            return [type("Device", (), {"platform": "cpu", "__str__": lambda self: "cpu"})()]  # type: ignore
    except Exception:
        return [type("Device", (), {"platform": "cpu", "__str__": lambda self: "cpu"})()]  # type: ignore

def is_jax_available() -> bool:
    """Check if JAX is importable and print device info."""
    try:
        import jax
        # Handle different JAX versions
        try:
            version = jax.__version__
        except AttributeError:
            # For older JAX versions, try alternative version attributes
            try:
                import jaxlib
                version = jaxlib.__version__
            except (ImportError, AttributeError):
                version = "unknown"

        logger.info(f"JAX version: {version}")

        try:
            devices = jax.devices()
            logger.info(f"JAX devices: {[str(d) for d in devices]}")
        except Exception as e:
            logger.warning(f"Could not get JAX devices: {e}")

        return True
    except ImportError as e:
        logger.error(f"JAX not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking JAX: {e}")
        return False

def find_jax_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Find JAX scripts in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Directory not found: {base_path}")
        return []
    pattern = "**/*.py" if recursive else "*.py"
    return [f for f in base_path.glob(pattern) if "jax" in f.name.lower() or f.parent.name == "jax"]

def execute_jax_script(script_path: Path, verbose: bool = False, device: Optional[str] = None, output_dir: Optional[Path] = None, timeout: int = 300) -> bool:
    """Execute a single JAX script with enhanced dependency checking, error handling, and log persistence.
    
    Args:
        script_path: Path to the JAX script to execute
        verbose: Enable verbose output logging
        device: JAX device to use ('cpu', 'gpu', 'tpu')
        output_dir: Directory for execution logs (stdout.txt, stderr.txt, execution_log.json)
        timeout: Execution timeout in seconds (default: 300)
    """
    import json as json_mod
    import time as time_mod

    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    logger.info(f"Executing JAX script: {script_path}")

    # Check JAX and related dependencies
    # Only jax and numpy are required — generated scripts use pure JAX
    required_deps = ["jax", "numpy"]
    optional_deps = ["flax", "optax"]
    missing_deps = []

    for dep in required_deps:
        try:
            __import__(dep)
            logger.debug(f"✅ Dependency available: {dep}")
        except ImportError:
            missing_deps.append(dep)
            logger.warning(f"⚠️ Missing required dependency: {dep}")

    for dep in optional_deps:
        try:
            __import__(dep)
            logger.debug(f"✅ Optional dependency available: {dep}")
        except ImportError:
            logger.info(f"ℹ️ Optional dependency not installed: {dep}")

    if missing_deps:
        logger.error(f"Missing required JAX dependencies: {', '.join(missing_deps)}")
        logger.error("Please install missing dependencies:")
        logger.error(f"uv pip install {' '.join(missing_deps)}")
        return False

    # Validate script syntax
    try:
        with open(script_path, 'r') as f:
            content = f.read()
            compile(content, script_path.name, 'exec')
        logger.debug(f"✅ Script syntax valid: {script_path.name}")
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {script_path.name}: {e}")
        return False

    env = os.environ.copy()
    if device:
        env["JAX_PLATFORM_NAME"] = device
        logger.info(f"Using JAX device: {device}")

    # Set JAX_OUTPUT_DIR environment variable (matching PYMDP_OUTPUT_DIR pattern)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env["JAX_OUTPUT_DIR"] = str(output_dir)
        logger.debug(f"Set JAX_OUTPUT_DIR={output_dir}")

    start_time = time_mod.time()

    try:
        # Execute with enhanced error capture
        # Convert to absolute path to avoid path resolution issues
        abs_script_path = script_path.resolve()
        result = subprocess.run(
            [sys.executable, str(abs_script_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=abs_script_path.parent,
            timeout=timeout
        )

        elapsed = time_mod.time() - start_time
        success = result.returncode == 0

        if success:
            logger.info(f"✅ Script executed successfully: {script_path.name} ({elapsed:.1f}s)")
            if verbose and result.stdout.strip():
                logger.debug(f"Output from {script_path.name}:\n{result.stdout}")
        else:
            logger.error(f"❌ Script execution failed: {script_path.name}")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr.strip():
                logger.error(f"Error output:\n{result.stderr}")
            if result.stdout.strip():
                logger.debug(f"Standard output:\n{result.stdout}")

        # Save execution logs (E-1/J-1)
        log_dir = output_dir if output_dir else abs_script_path.parent
        try:
            log_dir.mkdir(parents=True, exist_ok=True)

            # Save stdout and stderr
            stdout_file = log_dir / "stdout.txt"
            with open(stdout_file, 'w') as f:
                f.write(result.stdout or "")

            stderr_file = log_dir / "stderr.txt"
            with open(stderr_file, 'w') as f:
                f.write(result.stderr or "")

            # Save execution log JSON
            execution_log = {
                "script": str(abs_script_path),
                "return_code": result.returncode,
                "success": success,
                "elapsed_seconds": round(elapsed, 2),
                "device": device or "default",
                "timeout": timeout,
                "timestamp": time_mod.strftime("%Y-%m-%d %H:%M:%S")
            }

            log_file = log_dir / "execution_log.json"
            with open(log_file, 'w') as f:
                json_mod.dump(execution_log, f, indent=2)

            logger.debug(f"Execution logs saved to: {log_dir}")
        except Exception as log_err:
            logger.warning(f"Could not save execution logs: {log_err}")

        return success
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Script execution timed out after {timeout}s: {script_path.name}")
        return False
    except Exception as e:
        logger.error(f"❌ Error executing script {script_path.name}: {e}")
        return False

def run_jax_scripts(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False, device: Optional[str] = None) -> bool:
    """Find and run JAX scripts on rendered models."""
    if not is_jax_available():
        logger.error("JAX is not available, cannot execute JAX scripts")
        return False

    # Set up execution output directory
    if execution_output_dir:
        exec_output_dir = Path(execution_output_dir)
        exec_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JAX execution outputs will be saved to: {exec_output_dir}")

    jax_dir = Path(rendered_simulators_dir) / "jax"
    logger.info(f"Looking for JAX scripts in: {jax_dir}")
    script_files = find_jax_scripts(jax_dir, recursive_search)
    if not script_files:
        logger.info("No JAX scripts found")
        return True  # Consider this success if no scripts to run
    success_count = 0
    failure_count = 0
    for script_file in script_files:
        if execute_jax_script(script_file, verbose, device):
            success_count += 1
        else:
            failure_count += 1
    logger.info(f"JAX script execution summary: {success_count} succeeded, {failure_count} failed, {success_count + failure_count} total")
    # Return True only if no failures occurred (success means zero failures)
    return failure_count == 0

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Execute JAX scripts generated by the GNN rendering step")
    parser.add_argument("--output-dir", type=Path, default="../output", help="Main pipeline output directory")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursively search for scripts in the output directory")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Enable verbose output")
    parser.add_argument("--device", choices=["cpu", "gpu", "tpu"], default=None, help="Device to run JAX scripts on")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    success = run_jax_scripts(rendered_simulators_dir=args.output_dir, recursive_search=args.recursive, verbose=args.verbose, device=args.device)
    sys.exit(0 if success else 1)
