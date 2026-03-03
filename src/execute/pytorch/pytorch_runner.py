#!/usr/bin/env python3
"""
PyTorch Runner for Executing Rendered PyTorch POMDP Scripts

Discovers and runs PyTorch-generated scripts via subprocess with syntax validation,
dependency checking, log persistence, and execution timing.

@Web: https://pytorch.org/docs/stable/
"""
import json as json_mod
import logging
import os
import subprocess
import sys
import time as time_mod
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def is_pytorch_available() -> bool:
    """Check if PyTorch is importable and log version/device info."""
    try:
        import torch
        version = torch.__version__
        logger.info(f"PyTorch version: {version}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available — using CPU")
        return True
    except ImportError as e:
        logger.error(f"PyTorch not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking PyTorch: {e}")
        return False


def find_pytorch_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Find PyTorch scripts in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Directory not found: {base_path}")
        return []
    pattern = "**/*.py" if recursive else "*.py"
    return [
        f for f in base_path.glob(pattern)
        if "pytorch" in f.name.lower() or f.parent.name == "pytorch"
    ]


def execute_pytorch_script(
    script_path: Path,
    verbose: bool = False,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Execute a single PyTorch script with log persistence.

    Args:
        script_path: Path to the PyTorch script to execute.
        verbose: Enable verbose output logging.
        device: Device hint ('cpu' or 'cuda').
        output_dir: Directory for execution logs.
        timeout: Execution timeout in seconds.
    """
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    logger.info(f"Executing PyTorch script: {script_path}")

    # Dependency check
    for dep in ("torch", "numpy"):
        try:
            __import__(dep)
            logger.debug(f"✅ Dependency available: {dep}")
        except ImportError:
            logger.error(f"❌ Missing required dependency: {dep}")
            logger.error(f"Install with: uv sync --extra ml-ai")
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
    if device:
        env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else "0"
        logger.info(f"Using device: {device}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env["PYTORCH_OUTPUT_DIR"] = str(output_dir)

    start_time = time_mod.time()

    try:
        abs_path = script_path.resolve()
        result = subprocess.run(
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
            logger.info(f"✅ Script executed successfully: {script_path.name} ({elapsed:.1f}s)")
            if verbose and result.stdout.strip():
                logger.debug(f"Output:\n{result.stdout}")
        else:
            logger.error(f"❌ Script execution failed: {script_path.name}")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr.strip():
                logger.error(f"Error output:\n{result.stderr}")

        # Save execution logs
        log_dir = output_dir if output_dir else abs_path.parent
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "stdout.txt").write_text(result.stdout or "")
            (log_dir / "stderr.txt").write_text(result.stderr or "")
            execution_log = {
                "script": str(abs_path),
                "return_code": result.returncode,
                "success": success,
                "elapsed_seconds": round(elapsed, 2),
                "device": device or "default",
                "timeout": timeout,
                "timestamp": time_mod.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (log_dir / "execution_log.json").write_text(
                json_mod.dumps(execution_log, indent=2)
            )
            logger.debug(f"Execution logs saved to: {log_dir}")
        except Exception as log_err:
            logger.warning(f"Could not save execution logs: {log_err}")

        return success

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Script timed out after {timeout}s: {script_path.name}")
        return False
    except Exception as e:
        logger.error(f"❌ Error executing script {script_path.name}: {e}")
        return False


def run_pytorch_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Optional[Union[str, Path]] = None,
    recursive_search: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
) -> bool:
    """Find and run PyTorch scripts on rendered models."""
    if not is_pytorch_available():
        logger.error("PyTorch is not available, cannot execute PyTorch scripts")
        return False

    if execution_output_dir:
        exec_out = Path(execution_output_dir)
        exec_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"PyTorch execution outputs → {exec_out}")

    pytorch_dir = Path(rendered_simulators_dir) / "pytorch"
    logger.info(f"Looking for PyTorch scripts in: {pytorch_dir}")
    scripts = find_pytorch_scripts(pytorch_dir, recursive_search)
    if not scripts:
        logger.info("No PyTorch scripts found")
        return True

    success_count = 0
    failure_count = 0
    for script in scripts:
        out = Path(execution_output_dir) / script.stem if execution_output_dir else None
        if execute_pytorch_script(script, verbose, device, out):
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        f"PyTorch execution summary: {success_count} succeeded, "
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
        description="Execute PyTorch scripts generated by GNN rendering step"
    )
    parser.add_argument("--output-dir", type=Path, default="../output")
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    ok = run_pytorch_scripts(
        rendered_simulators_dir=args.output_dir,
        recursive_search=args.recursive,
        verbose=args.verbose,
        device=args.device,
    )
    sys.exit(0 if ok else 1)
