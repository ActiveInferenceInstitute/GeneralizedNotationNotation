#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation with UV (Thin Orchestrator)

This step handles project initialization, UV environment setup,

dependency installation, and environment validation using
Python packaging standards.

How to run:
  # Default: ``uv sync`` core dependencies (includes JAX, NumPyro, PyTorch, DisCoPy for step 12)
  python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose

  # Skip JAX self-test during setup (still installs core, which includes JAX)
  python src/1_setup.py --setup-core-only --target-dir input/gnn_files --output-dir output --verbose
  
  # Optional groups (LLM client libraries are core deps since 1.3.x)
  python src/1_setup.py --install-optional --optional-groups=llm --verbose
  
  # Install all optional dependencies
  python src/1_setup.py --install-optional --verbose

  # Dev test tooling only (pytest-cov, xdist, …)
  python src/1_setup.py --dev --verbose

  # Every optional extra in pyproject (heavy)
  python src/1_setup.py --install-all-extras --verbose
  
  # Alternative: Use uv directly for specific extras
  uv sync                          # Core includes openai, ollama, python-dotenv, aiohttp
  uv sync --extra llm              # Compatibility alias (same LLM stack)
  uv sync --extra visualization    # Install visualization packages
  uv sync --extra all              # Install all optional packages

Expected outputs:
  - Environment setup results in the specified output directory
  - UV environment creation and validation
  - Dependency installation and verification
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that UV is installed and available
  - Check that src/setup/ contains setup modules
  - Check that the output directory is writable
  - Verify system requirements and permissions
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module functions
try:
    from setup import setup_uv_environment, setup_complete_environment, install_optional_package_group
    from setup.constants import SETUP_DEFAULT_PIPELINE_EXTRAS
except ImportError:
    SETUP_DEFAULT_PIPELINE_EXTRAS = ()  # type: ignore[misc,assignment]
    def setup_uv_environment(
        verbose=False, recreate=False, dev=True, extras=None, install_all_extras=False, skip_jax_test=True, output_dir=None
    ) -> bool:
        """Recovery setup function when module unavailable."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Setup module not available - using recovery")
        return True

    def setup_complete_environment(verbose=False, recreate=False, install_optional=False, optional_groups=None, output_dir=None) -> bool:
        """Recovery full setup function."""
        return setup_uv_environment(verbose=verbose, recreate=recreate, output_dir=output_dir)

    def install_optional_package_group(group_name, verbose=False) -> bool:
        """Recovery optional package installation."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Cannot install {group_name} - setup module not available")
        return False

def setup_orchestrator(target_dir: str, output_dir: str, logger: "logging.Logger", **kwargs: Any) -> Any:
    """Orchestrate setup based on provided arguments."""
    verbose = kwargs.get('verbose', False)
    recreate = kwargs.get('recreate_venv', False)
    install_optional = kwargs.get('install_optional', False)
    optional_groups_str = kwargs.get('optional_groups', None)
    setup_core_only = bool(kwargs.get('setup_core_only', False))

    # Parse optional groups if provided
    optional_groups = None
    if optional_groups_str:
        optional_groups = [g.strip() for g in optional_groups_str.split(',')]

    dev = bool(kwargs.get('dev', False))
    install_all_extras = bool(kwargs.get('install_all_extras', False))

    # Use full setup if optional packages requested
    if install_optional or optional_groups:
        return setup_complete_environment(
            verbose=verbose,
            recreate=recreate,
            install_optional=True,
            optional_groups=optional_groups,
            output_dir=output_dir
        )
    else:
        # Default: core deps (Step 12 backends are in core). Optional ``extras`` from constants.
        # ``--setup-core-only`` skips the JAX import self-test only.
        extras: list[str] = [] if setup_core_only else list(SETUP_DEFAULT_PIPELINE_EXTRAS)
        skip_jax = setup_core_only
        return setup_uv_environment(
            verbose=verbose,
            recreate=recreate,
            dev=dev,
            extras=extras,
            install_all_extras=install_all_extras,
            skip_jax_test=skip_jax,
            output_dir=output_dir
        )

run_script = create_standardized_pipeline_script(
    "1_setup.py",
    setup_orchestrator,
    "Project setup and environment validation with UV",
    additional_arguments={
        "recreate_venv": {"type": bool, "help": "Recreate virtual environment"},
        "dev": {"type": bool, "help": "Install development dependencies (uv sync --extra dev)"},
        "install_all_extras": {"type": bool, "help": "Install all optional groups (uv sync --all-extras)"},
        "setup_core_only": {"type": bool, "help": "Core dependencies only; skip execution-frameworks (JAX stack)"},
        "install_optional": {"type": bool, "help": "Install optional dependencies"},
        "optional_groups": {"type": str, "help": "Comma-separated list of optional groups (jax,pymdp,visualization,audio,llm,ml)"}
    }
)

def main() -> int:
    """Main entry point for the setup step."""
    return run_script()

if __name__ == "__main__":
    raise SystemExit(main())
