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
from advanced_visualization import process_advanced_visualization

run_script = create_standardized_pipeline_script(
    "9_advanced_viz.py",
    process_advanced_visualization,
    "Advanced visualization and exploration with safe-to-fail patterns",
    additional_arguments={
        "viz_type": {
            "type": str,
            "choices": ["all", "3d", "interactive", "dashboard"],
            "default": "all",
            "help": "Type of visualization to generate",
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


if __name__ == "__main__":
    sys.exit(run_script()) 