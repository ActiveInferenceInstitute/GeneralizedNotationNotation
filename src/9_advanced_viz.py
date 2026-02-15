#!/usr/bin/env python3
"""
Step 9: Advanced Visualization (Thin Orchestrator)

This step delegates advanced visualization processing to the module implementation.

How to run:
  python src/9_advanced_viz.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from advanced_visualization.processor import process_advanced_viz_standardized_impl

run_script = create_standardized_pipeline_script(
    "9_advanced_viz.py",
    process_advanced_viz_standardized_impl,
    "Advanced visualization and exploration with safe-to-fail patterns",
    additional_arguments={
        "viz_type": {
            "type": str,
            "choices": ["all", "3d", "interactive", "dashboard", "d2", "diagrams", "pipeline", "statistical", "pomdp", "network"],
            "default": "all",
            "help": "Type of visualization to generate: all, 3d, interactive, dashboard, d2/diagrams (D2 diagrams), pipeline (pipeline D2 diagrams), statistical (statistical plots and correlations), pomdp (POMDP-specific visualizations), network (network analysis)",
        },
        "interactive": {
            "type": bool,
            "default": True,
            "help": "Generate interactive visualizations",
        },
        "export_formats": {
            "type": str,
            "nargs": "+",
            "default": ["html", "json"],
            "help": "Export formats",
        },
    },
)


def main() -> int:
    """Main entry point for advanced visualization step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main()) 