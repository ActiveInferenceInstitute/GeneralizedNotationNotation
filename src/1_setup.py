#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation with UV (Thin Orchestrator)

Handles argument wiring and delegates UV setup, dependency installation, and
environment validation to `src/setup/`. Keep usage examples and operational
details in `src/setup/README.md` so this numbered step stays orchestration-only.
"""

import logging
import sys
from pathlib import Path
from typing import Any, cast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from setup import (
    setup_complete_environment,
    setup_uv_environment,
)
from setup.constants import SETUP_DEFAULT_PIPELINE_EXTRAS
from utils.pipeline_template import create_standardized_pipeline_script


def setup_orchestrator(
    target_dir: str, output_dir: str, logger: "logging.Logger", **kwargs: Any
) -> Any:
    """Orchestrate setup based on provided arguments."""
    output_path = Path(output_dir)
    verbose = kwargs.get("verbose", False)
    recreate = kwargs.get("recreate_venv", False)
    install_optional_raw = kwargs.get("install_optional", False)
    optional_groups_str = kwargs.get("optional_groups", None)
    setup_core_only = bool(kwargs.get("setup_core_only", False))

    # Parse optional groups if provided
    optional_groups = None
    if isinstance(install_optional_raw, str) and install_optional_raw.strip():
        optional_groups_str = optional_groups_str or install_optional_raw
        install_optional = True
    else:
        install_optional = bool(install_optional_raw)

    if optional_groups_str:
        optional_groups = [g.strip() for g in optional_groups_str.split(",")]

    dev = bool(kwargs.get("dev", False))
    install_all_extras = bool(kwargs.get("install_all_extras", False))

    # Use full setup if optional packages requested
    if install_optional or optional_groups:
        return setup_complete_environment(
            verbose=verbose,
            recreate=recreate,
            install_optional=True,
            optional_groups=optional_groups,
            output_dir=output_path,
        )
    else:
        # Default: core deps (Step 12 backends are in core). Optional ``extras`` from constants.
        # ``--setup-core-only`` skips the JAX import self-test only.
        extras: list[str] = (
            [] if setup_core_only else list(SETUP_DEFAULT_PIPELINE_EXTRAS)
        )
        skip_jax = setup_core_only
        return setup_uv_environment(
            verbose=verbose,
            recreate=recreate,
            dev=dev,
            extras=extras,
            install_all_extras=install_all_extras,
            skip_jax_test=skip_jax,
            output_dir=output_path,
        )


run_script = create_standardized_pipeline_script(
    "1_setup.py",
    setup_orchestrator,
    "Project setup and environment validation with UV",
    additional_arguments={
        "recreate_venv": {
            "flag": "--recreate-uv-env",
            "action": "store_true",
            "help": "Recreate virtual environment",
        },
        "dev": {
            "action": "store_true",
            "help": "Install development dependencies (uv sync --extra dev)",
        },
        "install_all_extras": {
            "action": "store_true",
            "help": "Install all optional groups (uv sync --all-extras)",
        },
        "setup_core_only": {
            "action": "store_true",
            "help": "Skip post-install JAX/Optax/Flax/pymdp functional self-test (deps still installed)",
        },
        "install_optional": {
            "action": "store_true",
            "help": "Install optional dependencies",
        },
        "optional_groups": {
            "type": str,
            "help": "Comma-separated list of optional groups (jax,pymdp,visualization,audio,llm,ml)",
        },
    },
)


def main() -> int:
    """Main entry point for the setup step."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())
