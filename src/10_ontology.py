#!/usr/bin/env python3
"""
Step 10: Ontology Processing (Thin Orchestrator)

This step orchestrates ontology processing and validation for GNN models.
It is a thin orchestrator that delegates core functionality to the ontology module.

How to run:
  python src/10_ontology.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Ontology processing results in the specified output directory
  - Ontology validation and compliance reports
  - Term mapping and relationship analysis
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that ontology dependencies are installed
  - Check that src/ontology/ contains ontology modules
  - Check that the output directory is writable
  - Verify ontology configuration and term mapping setup
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from ontology import process_ontology
except ImportError:
    def process_ontology(target_dir, output_dir, logger=None, **kwargs):
        """Fallback ontology processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning("Ontology module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "10_ontology.py",
    process_ontology,
    "Ontology processing and validation for GNN models",
    additional_arguments={
        "ontology_terms_file": {"type": Path, "help": "Path to ontology terms JSON file", "flag": "--ontology-terms-file"}
    }
)

def main() -> int:
    """Main entry point for the ontology step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
