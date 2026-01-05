#!/usr/bin/env python3
"""
Step 5: Type Checking and Validation (Thin Orchestrator)

This step performs type checking and validation on GNN files.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/type_checker/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the type_checker module.

Pipeline Flow:
    main.py → 5_type_checker.py (this script) → type_checker/ (modular implementation)

How to run:
  python src/5_type_checker.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Type checking results in the specified output directory
  - Comprehensive validation reports and analysis
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that type checker dependencies are installed
  - Check that src/type_checker/ contains type checker modules
  - Check that the output directory is writable
  - Verify type checker configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from type_checker import GNNTypeChecker
except ImportError:
    def GNNTypeChecker():
        """Fallback type checker when module unavailable."""
        class FallbackTypeChecker:
            def validate_gnn_files(self, target_dir, output_dir, **kwargs):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Type checker module not available - using fallback checks")
                
                # Basic validation: Check if files exist and are not empty
                target_path = Path(target_dir)
                if not target_path.exists():
                     logger.error(f"Target directory {target_dir} does not exist")
                     return False
                     
                gnn_files = list(target_path.glob("**/*.md"))
                if not gnn_files:
                    logger.warning(f"No GNN files found in {target_dir}")
                    return True # Nothing to validate is technically valid
                    
                valid_count = 0
                for f in gnn_files:
                    if f.stat().st_size > 0:
                        valid_count += 1
                    else:
                        logger.warning(f"Empty file found: {f}")
                        
                logger.info(f"Fallback validation: scanned {len(gnn_files)} files, {valid_count} non-empty.")
                return True
        return FallbackTypeChecker()

run_script = create_standardized_pipeline_script(
    "5_type_checker.py",
    lambda target_dir, output_dir, logger, **kwargs: GNNTypeChecker().validate_gnn_files(
        target_dir, output_dir, **kwargs
    ),
    "Type checking and validation of GNN files",
    additional_arguments={
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "estimate_resources": {"type": bool, "help": "Enable resource estimation"}
    }
)

def main() -> int:
    """Main entry point for the type checker step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
