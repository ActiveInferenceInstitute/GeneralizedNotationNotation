#!/usr/bin/env python3
"""
NumPyro Runner for Executing Rendered NumPyro POMDP Scripts

Discovers and runs NumPyro-generated scripts via subprocess with dependency
checking, syntax validation, log persistence, and execution timing.

@Web: https://num.pyro.ai/
"""
import json as json_mod
import logging
import os
import subprocess  # nosec B404 -- subprocess calls with controlled/trusted input
import sys
import tempfile
import time as time_mod
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def is_numpyro_available() -> bool:
    """Check if NumPyro (and JAX backend) is importable."""
    try:
        import jax
        import numpyro
        logger.info(f"NumPyro version: {numpyro.__version__} (JAX {jax.__version__})")
        return True
    except ImportError as e:
        logger.error(f"NumPyro not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking NumPyro: {e}")
        return False


def find_numpyro_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Find NumPyro scripts in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Directory not found: {base_path}")
        return []
    pattern = "**/*.py" if recursive else "*.py"
    return [
        f for f in base_path.glob(pattern)
        if "numpyro" in f.name.lower() or f.parent.name == "numpyro"
    ]


def execute_numpyro_script(
    script_path: Path,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Execute a single NumPyro script with log persistence.

    Args:
        script_path: Path to the NumPyro script.
        verbose: Enable verbose output.
        output_dir: Directory for execution logs.
        timeout: Execution timeout in seconds.
    """
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    logger.info(f"Executing NumPyro script: {script_path}")

    # Dependency check
    for dep in ("jax", "numpyro", "numpy"):
        try:
            __import__(dep)
            logger.debug(f"✅ Dependency available: {dep}")
        except ImportError:
            logger.error(f"❌ Missing required dependency: {dep}")
            logger.error("Install with: uv sync --extra probabilistic-programming")
            return False

    # Syntax validation
    try:
        content = script_path.read_text()
        compile(content, script_path.name, "exec")
        logger.debug(f"✅ Script syntax valid: {script_path.name}")
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {script_path.name}: {e}")
        return False

    env = os.environ.copy()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env["NUMPYRO_OUTPUT_DIR"] = str(output_dir)

    start_time = time_mod.time()

    try:
        abs_path = script_path.resolve()
        result = subprocess.run(  # nosec B603 -- subprocess calls with controlled/trusted input
            [sys.executable, str(abs_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=abs_path.parent,
            timeout=timeout,
        )

        elapsed = time_mod.time() - start_time
        success = result.returncode == 0

        if success:
            logger.info(f"✅ Script executed: {script_path.name} ({elapsed:.1f}s)")
            if verbose and result.stdout.strip():
                logger.debug(f"Output:\n{result.stdout}")
        else:
            logger.error(f"❌ Script failed: {script_path.name}")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr.strip():
                logger.error(f"Error:\n{result.stderr}")

        # Save execution logs
        log_dir = output_dir if output_dir else abs_path.parent
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / "stdout.txt"
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=stdout_path.parent, delete=False) as tmp_f:
                tmp_f.write(result.stdout or "")
            os.replace(tmp_f.name, str(stdout_path))
            stderr_path = log_dir / "stderr.txt"
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=stderr_path.parent, delete=False) as tmp_f:
                tmp_f.write(result.stderr or "")
            os.replace(tmp_f.name, str(stderr_path))
            execution_log = {
                "script": str(abs_path),
                "return_code": result.returncode,
                "success": success,
                "elapsed_seconds": round(elapsed, 2),
                "timeout": timeout,
                "timestamp": time_mod.strftime("%Y-%m-%d %H:%M:%S"),
            }
            exec_log_path = log_dir / "execution_log.json"
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=exec_log_path.parent, delete=False) as tmp_f:
                tmp_f.write(json_mod.dumps(execution_log, indent=2))
            os.replace(tmp_f.name, str(exec_log_path))
            logger.debug(f"Execution logs saved to: {log_dir}")
        except Exception as log_err:
            logger.warning(f"Could not save execution logs: {log_err}")

        return success

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Script timed out after {timeout}s: {script_path.name}")
        return False
    except Exception as e:
        logger.error(f"❌ Error executing {script_path.name}: {e}")
        return False


def run_numpyro_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Optional[Union[str, Path]] = None,
    recursive_search: bool = True,
    verbose: bool = False,
) -> bool:
    """Find and run NumPyro scripts on rendered models."""
    if not is_numpyro_available():
        logger.error("NumPyro not available, cannot execute NumPyro scripts")
        return False

    if execution_output_dir:
        exec_out = Path(execution_output_dir)
        exec_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"NumPyro execution outputs → {exec_out}")

    numpyro_dir = Path(rendered_simulators_dir) / "numpyro"
    logger.info(f"Looking for NumPyro scripts in: {numpyro_dir}")
    scripts = find_numpyro_scripts(numpyro_dir, recursive_search)
    if not scripts:
        logger.info("No NumPyro scripts found")
        return True

    success_count = 0
    failure_count = 0
    for script in scripts:
        out = Path(execution_output_dir) / script.stem if execution_output_dir else None
        if execute_numpyro_script(script, verbose, out):
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        f"NumPyro execution summary: {success_count} succeeded, "
        f"{failure_count} failed, {success_count + failure_count} total"
    )
    return failure_count == 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Execute NumPyro scripts generated by GNN rendering step"
    )
    parser.add_argument("--output-dir", type=Path, default="../output")
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    ok = run_numpyro_scripts(
        rendered_simulators_dir=args.output_dir,
        recursive_search=args.recursive,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)
