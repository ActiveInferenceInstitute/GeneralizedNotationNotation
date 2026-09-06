#!/usr/bin/env python3
"""
Julia environment resolution for GNN Step 12.

Owns the committed Julia project environments for Julia-backed frameworks,
the rendered-script command builder, and the Julia package availability
check. Extracted from ``execute.processor``.
"""

import logging
import subprocess  # nosec B404
from pathlib import Path
from typing import List, Optional

from .types import ScriptExecutionContext

logger = logging.getLogger(__name__)


def _julia_project_for_framework(framework: str) -> Optional[Path]:
    """Return the committed Julia environment for a supported framework."""
    execute_dir = Path(__file__).resolve().parent
    projects = {
        "rxinfer": execute_dir / "rxinfer",
        "activeinference_jl": execute_dir / "activeinference_jl",
    }
    return projects.get(framework)


def _build_script_execution_command(
    context: ScriptExecutionContext, sandbox_prefix: List[str]
) -> List[str]:
    """Build an explicit, reproducible command for a rendered script."""
    command = list(sandbox_prefix)
    if context.executor == "julia":
        project_dir = _julia_project_for_framework(context.framework)
        if project_dir is None:
            raise ValueError(
                f"No committed Julia project is registered for {context.framework}"
            )
        command.extend(
            [
                context.executor,
                "--startup-file=no",
                f"--project={project_dir}",
                context.script_path.name,
            ]
        )
        return command
    command.extend([context.executor, context.script_path.name])
    return command


def check_julia_dependencies(
    verbose: bool,
    log: Optional[logging.Logger] = None,
    frameworks: Optional[List[str]] = None,
) -> bool:
    """Check if required Julia packages are available.

    Args:
        verbose: Enable verbose logging.
        log: Optional logger instance; defaults to module logger if not provided.

    Returns:
        True if dependencies ok, False otherwise.
    """
    if log is None:
        log = logger
    try:
        # check basic julia availability
        subprocess.run(
            ["julia", "--version"], capture_output=True, check=True, timeout=10
        )  # nosec B607 B603

        requested = set(frameworks or ["rxinfer", "activeinference_jl"])

        # Each Julia-backed framework ships its own committed project
        # environment (Project.toml + Manifest.toml), so the package check must
        # run against that environment's ``--project``. A bare ``julia -e
        # "using ..."`` resolves against the global depot and always fails on a
        # clean machine, which previously skipped every Julia script.
        framework_projects = {
            "rxinfer": (
                _julia_project_for_framework("rxinfer"),
                ["JSON", "Distributions", "StatsBase", "RxInfer"],
            ),
            "activeinference_jl": (
                _julia_project_for_framework("activeinference_jl"),
                ["JSON", "Distributions", "StatsBase", "ActiveInference"],
            ),
        }

        for framework in sorted(requested):
            entry = framework_projects.get(framework)
            if entry is None:
                log.warning(f"Unknown Julia framework '{framework}'; skipping check")
                continue
            project_dir, packages = entry
            if project_dir is None:
                return False
            using_clause = ", ".join(packages)
            check_script = f"using {using_clause}"
            result = subprocess.run(  # nosec B607 B603
                [
                    "julia",
                    "--startup-file=no",
                    f"--project={project_dir}",
                    "-e",
                    check_script,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                if verbose:
                    log.warning(
                        f"Julia package check failed for {framework}: {result.stderr}"
                    )
                return False

        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False
