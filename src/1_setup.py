#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation with UV (Thin Orchestrator)

This step handles project initialization, UV environment setup,
dependency installation, and environment validation using modern
Python packaging standards.

How to run:
  python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

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

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from setup import setup_uv_environment
except ImportError:
    def setup_uv_environment(verbose=False, recreate=False, dev=True, extras=[], skip_jax_test=True):
        """Fallback setup function when module unavailable."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Setup module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "1_setup.py",
    lambda target_dir, output_dir, logger, **kwargs: setup_uv_environment(
        verbose=kwargs.get('verbose', False),
        recreate=kwargs.get('recreate_venv', False),
        dev=kwargs.get('dev', True),
        extras=["llm", "visualization", "audio", "gui"],
        skip_jax_test=True,
        output_dir=output_dir
    ),
    "Project setup and environment validation with UV",
    additional_arguments={
        "recreate_venv": {"type": bool, "help": "Recreate virtual environment"},
        "dev": {"type": bool, "help": "Install development dependencies"}
    }
)

def main() -> int:
    """Main entry point for the setup step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 