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


if __name__ == "__main__":
    # Explicitly handle --help to ensure it doesn't execute the full pipeline
    if "--help" in sys.argv or "-h" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description="Advanced visualization and exploration with safe-to-fail patterns")
        parser.add_argument("--target-dir", type=str, help="Target directory containing files to process")
        parser.add_argument("--output-dir", type=str, default="output", help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true", help="Process files recursively")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument("--viz_type", default="all", help="Type of visualization to generate")
        parser.add_argument("--interactive", default=True, help="Generate interactive visualizations")
        parser.add_argument("--export_formats", nargs="+", default=["html", "json"], help="Export formats")
        parser.print_help()
        sys.exit(0)

    sys.exit(run_script()) 