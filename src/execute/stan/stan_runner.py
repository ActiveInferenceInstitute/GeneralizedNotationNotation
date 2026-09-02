#!/usr/bin/env python3
"""
Stan Runner for executing rendered Stan drivers.

Step 11's Stan renderer emits, per model, a ``.stan`` program and a sibling
``<stem>_stan.py`` cmdstanpy driver (simulate → compile → sample → results).
Step 12 executes the driver like any other Python framework script; this
module provides the dependency probe and a direct runner used by tests and
callers outside the pipeline.
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def is_stan_available() -> bool:
    """True when ``cmdstanpy`` imports and a CmdStan toolchain is installed."""
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
        return True
    except ImportError:
        logger.info("cmdstanpy not installed (uv sync --extra stan)")
        return False
    except Exception as exc:  # ValueError when CmdStan is missing
        logger.info(f"CmdStan toolchain not available: {exc}")
        return False


def find_stan_scripts(render_output_dir: Union[str, Path]) -> List[Path]:
    """Return every rendered Stan driver (``*_stan.py``) under a render tree."""
    root = Path(render_output_dir)
    return sorted(p for p in root.rglob("*_stan.py") if p.parent.name == "stan")


def execute_stan_script(
    script_path: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one Stan driver with ``STAN_OUTPUT_DIR`` set; return a result dict."""
    script = Path(script_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["STAN_OUTPUT_DIR"] = str(out_dir)
    start = time.time()
    proc = subprocess.run(  # nosec B603
        [python_executable or sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(out_dir),
    )
    result: Dict[str, Any] = {
        "script": str(script),
        "framework": "stan",
        "return_code": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "execution_time_seconds": round(time.time() - start, 3),
        "results_file": str(out_dir / "simulation_results.json"),
    }
    if not result["success"]:
        logger.error(f"Stan driver failed ({proc.returncode}): {script.name}")
    return result


def run_stan_scripts(
    render_output_dir: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
) -> List[Dict[str, Any]]:
    """Execute every rendered Stan driver; skip all with a reason if unavailable."""
    scripts = find_stan_scripts(render_output_dir)
    if not is_stan_available():
        return [
            {
                "script": str(s),
                "framework": "stan",
                "success": False,
                "skipped": True,
                "reason": "cmdstanpy/CmdStan not installed (uv sync --extra stan)",
            }
            for s in scripts
        ]
    return [
        execute_stan_script(s, Path(output_dir) / s.parent.parent.name, timeout)
        for s in scripts
    ]
