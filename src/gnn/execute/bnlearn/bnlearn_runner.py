"""bnlearn Runner for executing rendered bnlearn scripts.

Step 11's generator-backed bnlearn renderer emits, per model, a Python
program under ``<model>/bnlearn/`` (``import bnlearn as bn`` +
``bn.make_DAG`` + ``bn.parameter_learning.fit``). Step 12 discovers and runs
it like any other Python framework script (framework directory ``bnlearn/``,
output env var ``BNLEARN_OUTPUT_DIR``).

This module provides the dependency probes and a language-aware direct
runner used by tests and callers outside the pipeline. The execution
language is derived from each emitted file's suffix — ``.py`` runs under a
Python interpreter with the ``bnlearn`` module, ``.R`` runs under Rscript
with the R ``bnlearn`` package — never assumed. Missing runtimes produce an
explicit ``skipped`` record (mirroring the Stan executor's
skip-on-missing-toolchain semantics via ``gnn.utils.runtime_safety.framework_availability``).
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.execute.security_gate import check_script_allowed
from gnn.execute.subprocess_envelope import run_subprocess_envelope
from gnn.utils.runtime_safety.framework_availability import is_framework_available

logger = logging.getLogger(__name__)

FRAMEWORK: str = "bnlearn"
OUTPUT_ENV_VAR: str = "BNLEARN_OUTPUT_DIR"

_PYTHON_SUFFIXES = frozenset({".py"})
_R_SUFFIXES = frozenset({".r"})

_PYTHON_SKIP_REASON = "bnlearn module not installed (uv sync --extra bnlearn)"
_R_SKIP_REASON = "Rscript/R bnlearn package not available (install.packages('bnlearn'))"


def is_bnlearn_available(python_executable: Optional[str] = None) -> bool:
    """True when the Python ``bnlearn`` module is importable.

    Delegates to the shared ``gnn.utils.runtime_safety.framework_availability`` probe. With
    ``python_executable=None`` the check is a cheap in-process
    ``importlib.util.find_spec``; with an interpreter path it shells out so
    the answer reflects the target interpreter's environment.
    """
    return is_framework_available(FRAMEWORK, executor=python_executable, logger=logger)


def is_r_bnlearn_available(rscript_executable: str = "Rscript") -> bool:
    """True when Rscript exists and the R ``bnlearn`` package loads."""
    rscript = shutil.which(rscript_executable)
    if rscript is None:
        logger.info("Rscript not found on PATH (R bnlearn lane unavailable)")
        return False
    probe = run_subprocess_envelope(
        [rscript, "-e", "suppressMessages(library(bnlearn))"],
        timeout=60,
        sandbox=False,
    )
    if not probe["success"]:
        logger.info("R bnlearn package probe failed: %s", probe.get("error"))
    return bool(probe["success"])


def script_language(script_path: Union[str, Path]) -> str:
    """Return ``"python"``, ``"r"``, or ``"unknown"`` from the file suffix."""
    suffix = Path(script_path).suffix.lower()
    if suffix in _PYTHON_SUFFIXES:
        return "python"
    if suffix in _R_SUFFIXES:
        return "r"
    return "unknown"


def find_bnlearn_scripts(render_output_dir: Union[str, Path]) -> List[Path]:
    """Return every rendered bnlearn script (``.py``/``.R`` in ``bnlearn/`` dirs)."""
    root = Path(render_output_dir)
    found: set[Path] = set()
    for pattern in ("*.py", "*.R"):
        for path in root.rglob(pattern):
            if path.parent.name.lower() == "bnlearn":
                found.add(path)
    return sorted(found)


def execute_bnlearn_script(
    script_path: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
    python_executable: Optional[str] = None,
    rscript_executable: str = "Rscript",
) -> Dict[str, Any]:
    """Run one rendered bnlearn script; return a structured result dict.

    The execution lane is derived from the script's suffix. Missing runtimes
    produce a ``skipped`` record without spawning a subprocess.
    """
    script = Path(script_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    language = script_language(script)
    record: Dict[str, Any] = {
        "script": str(script),
        "framework": FRAMEWORK,
        "language": language,
        "return_code": None,
        "success": False,
        "skipped": False,
        "stdout": "",
        "stderr": "",
        "execution_time_seconds": 0.0,
        # Informational pointer to the conventional results path; rendered
        # bnlearn programs are not required to write it.
        "results_file": str(out_dir / "simulation_results.json"),
    }

    # Shared pre-execution security gate (fail closed; GNN_ALLOW_UNSAFE_EXEC
    # is the only operator opt-out). Runs before any lane probe, command
    # construction, or subprocess spawn.
    gate_verdict = check_script_allowed(script)
    if gate_verdict["overridden"]:
        logger.warning(
            "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate "
            "bypassed for %s (trusted-local use only)",
            script,
        )
    if not gate_verdict["ok"]:
        record["error_type"] = gate_verdict.get("error_type", "SecurityGateBlocked")
        record["security_findings"] = gate_verdict["blocked"]
        record["error"] = (
            f"Pre-execution security gate blocked {script_path}: "
            f"{gate_verdict['reason']}"
        )
        logger.error(record["error"])
        return record

    if language == "python":
        if not is_bnlearn_available(python_executable):
            record["skipped"] = True
            record["reason"] = _PYTHON_SKIP_REASON
            logger.info(
                "Skipping bnlearn script (Python lane unavailable): %s", script.name
            )
            return record
        command: List[str] = [python_executable or sys.executable, str(script)]
    elif language == "r":
        if not is_r_bnlearn_available(rscript_executable):
            record["skipped"] = True
            record["reason"] = _R_SKIP_REASON
            logger.info("Skipping bnlearn script (R lane unavailable): %s", script.name)
            return record
        command = [rscript_executable, str(script)]
    else:
        record["skipped"] = True
        record["reason"] = (
            f"Unsupported bnlearn script language: {script.suffix or '<none>'}"
        )
        logger.info("Skipping bnlearn script (unknown language): %s", script.name)
        return record

    envelope = run_subprocess_envelope(
        command,
        timeout=timeout,
        env={OUTPUT_ENV_VAR: str(out_dir)},
        cwd=str(out_dir),
    )
    record["return_code"] = envelope["return_code"]
    record["success"] = envelope["success"]
    record["stdout"] = envelope["stdout"]
    record["stderr"] = envelope["stderr"]
    record["execution_time_seconds"] = round(envelope["duration_seconds"], 3)
    if not envelope["success"]:
        error_type = envelope.get("error_type")
        if error_type == "TimeoutExpired":
            record["error_type"] = "TimeoutExpired"
            record["error"] = (
                f"bnlearn script timed out after {timeout}s: {script.name}"
            )
        else:
            record["error_type"] = error_type or "RuntimeError"
            record["error"] = envelope.get("error") or (
                f"bnlearn script failed ({envelope['return_code']}): {script.name}"
            )
        logger.error(record["error"])
    return record


def run_bnlearn_scripts(
    render_output_dir: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
    python_executable: Optional[str] = None,
    rscript_executable: str = "Rscript",
) -> List[Dict[str, Any]]:
    """Execute every rendered bnlearn script; each records skip/fail/explicitly."""
    scripts = find_bnlearn_scripts(render_output_dir)
    return [
        execute_bnlearn_script(
            s,
            Path(output_dir) / s.parent.parent.name,
            timeout,
            python_executable=python_executable,
            rscript_executable=rscript_executable,
        )
        for s in scripts
    ]
