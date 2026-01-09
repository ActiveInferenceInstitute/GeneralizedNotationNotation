#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation with UV (Thin Orchestrator)

This step handles project initialization, UV environment setup,

dependency installation, and environment validation using
Python packaging standards.

How to run:
  # Basic setup (core dependencies only)
  python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose
  
  # Install with LLM support (OpenAI, Anthropic, Ollama)
  python src/1_setup.py --install-optional --optional-groups=llm --verbose
  
  # Install all optional dependencies
  python src/1_setup.py --install-optional --verbose
  
  # Alternative: Use uv directly for specific extras
  uv sync --extra llm              # Install LLM packages (openai, anthropic, etc.)
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

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module functions
try:
    from setup import setup_uv_environment, setup_complete_environment, install_optional_package_group
except ImportError:
    def setup_uv_environment(verbose=False, recreate=False, dev=True, extras=[], skip_jax_test=True):
        """Fallback setup function when module unavailable."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Setup module not available - using fallback")
        return True
    
    def setup_complete_environment(verbose=False, recreate=False, install_optional=False, optional_groups=None, output_dir=None):
        """Fallback full setup function."""
        return setup_uv_environment(verbose=verbose, recreate=recreate, output_dir=output_dir)
    
    def install_optional_package_group(group_name, verbose=False):
        """Fallback optional package installation."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Cannot install {group_name} - setup module not available")
        return False

def setup_orchestrator(target_dir, output_dir, logger, **kwargs):
    """Orchestrate setup based on provided arguments."""
    verbose = kwargs.get('verbose', False)
    recreate = kwargs.get('recreate_venv', False)
    install_optional = kwargs.get('install_optional', False)
    optional_groups_str = kwargs.get('optional_groups', None)
    
    # Parse optional groups if provided
    optional_groups = None
    if optional_groups_str:
        optional_groups = [g.strip() for g in optional_groups_str.split(',')]
    
    # Handle dev argument default (ArgumentParser may return None for defaults not in argv)
    dev = kwargs.get('dev')
    if dev is None:
        dev = True
    
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
        # Use basic setup for core dependencies only
        return setup_uv_environment(
            verbose=verbose,
            recreate=recreate,
            dev=dev,
            extras=[],
            skip_jax_test=False,  # Test JAX functionality
            output_dir=output_dir
        )

run_script = create_standardized_pipeline_script(
    "1_setup.py",
    setup_orchestrator,
    "Project setup and environment validation with UV",
    additional_arguments={
        "recreate_venv": {"type": bool, "help": "Recreate virtual environment"},
        "dev": {"type": bool, "help": "Install development dependencies"},
        "install_optional": {"type": bool, "help": "Install optional dependencies"},
        "optional_groups": {"type": str, "help": "Comma-separated list of optional groups (jax,pymdp,visualization,audio,llm,ml)"}
    }
)

def main() -> int:
    """Main entry point for the setup step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 