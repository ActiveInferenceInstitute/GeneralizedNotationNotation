#!/usr/bin/env python3
"""
Step 9: Advanced Visualization (Thin Orchestrator)

This step provides advanced visualization capabilities for GNN models with
comprehensive safe-to-fail patterns and robust output management.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/advanced_visualization/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the advanced_visualization module.

Pipeline Flow:
    main.py → 9_advanced_viz.py (this script) → advanced_visualization/ (modular implementation)

How to run:
  python src/9_advanced_viz.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Advanced visualization results in the specified output directory
  - Interactive visualizations, dashboards, and analysis plots
  - Comprehensive visualization reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that advanced visualization dependencies are installed
  - Check that src/advanced_visualization/ contains visualization modules
  - Check that the output directory is writable
  - Verify visualization configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from advanced_visualization import process_advanced_viz_standardized

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "9_advanced_viz.py",
    process_advanced_viz_standardized,
    "Advanced visualization and exploration with safe-to-fail patterns",
    additional_arguments={
        "viz_type": {
            "type": str,
            "choices": ["all", "3d", "interactive", "dashboard"],
            "default": "all",
            "help": "Type of visualization to generate"
        },
        "interactive": {"type": bool, "default": True, "help": "Generate interactive visualizations"},
        "export_formats": {"type": str, "nargs": "+", "default": ["html", "json"], "help": "Export formats"}
    }
)

def main() -> int:
    """Main entry point for the advanced visualization step."""
    return run_script()
