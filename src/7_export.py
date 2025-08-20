#!/usr/bin/env python3
"""
Step 7: Multi-format Export Generation (Thin Orchestrator)

This step generates exports in multiple formats (JSON, XML, GraphML, GEXF, Pickle).

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/export/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the export module.

Pipeline Flow:
    main.py → 7_export.py (this script) → export/ (modular implementation)

How to run:
  python src/7_export.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Multi-format exports in the specified output directory
  - JSON, XML, GraphML, GEXF, and Pickle format files
  - Comprehensive export reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that export dependencies are installed
  - Check that src/export/ contains export modules
  - Check that the output directory is writable
  - Verify export configuration and format requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "7_export.py",
    process_export,
    "Multi-format export generation"
)

def main() -> int:
    """Main entry point for the export step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
