#!/usr/bin/env python3
"""
Step 6: Validation Processing (Thin Orchestrator)

This step performs validation and quality assurance on GNN models,
including semantic validation, performance profiling, and consistency checking.

How to run:
  python src/gnn/6_validation.py --target-dir input/gnn_files --output-dir output --verbose
  python src/gnn/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Validation results in the specified output directory
  - Semantic validation reports and scores (``--strict`` raises the semantic
    validation level to "strict")
  - Performance profiling and resource estimates
  - Consistency checking and quality metrics
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths
"""

import sys
from pathlib import Path
from typing import cast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.utils.pipeline_orchestration.pipeline_template import (
    create_standardized_pipeline_script,
)

# Hard import: validation is a core module and must always be available.
from gnn.validation import process_validation

run_script = create_standardized_pipeline_script(
    "6_validation.py",
    process_validation,
    "Validation processing for GNN models",
    additional_arguments={
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "profile": {"type": bool, "help": "Enable performance profiling"},
    },
)


def main() -> int:
    """Main entry point for the validation step."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())
