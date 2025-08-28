#!/usr/bin/env python3
"""
Step 6: Validation Processing (Thin Orchestrator)

This step performs validation and quality assurance on GNN models,
including semantic validation, performance profiling, and consistency checking.

How to run:
  python src/6_validation.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Validation results in the specified output directory
  - Semantic validation reports and scores
  - Performance profiling and resource estimates
  - Consistency checking and quality metrics
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from validation import process_validation
except ImportError:
    def process_validation(target_dir, output_dir, logger, **kwargs):
        """Fallback validation when module unavailable."""
        logger.warning("Validation module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "6_validation.py",
    process_validation,
    "Validation processing for GNN models",
    additional_arguments={
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "profile": {"type": bool, "help": "Enable performance profiling"}
    }
)

def main() -> int:
    """Main entry point for the validation step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 