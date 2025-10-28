#!/usr/bin/env python3
"""
Step 22: GUI Processing (Thin Orchestrator)

This step orchestrates GUI processing for GNN models.
Provides multiple interactive GUI implementations and headless artifact generation.

Available GUI Options:
    - gui_1: Form-based Interactive GNN Constructor (Port 7860)
    - gui_2: Visual Matrix Editor (Port 7861)
    - gui_3: State Space Design Studio (Port 7862)
    - oxdraw: Visual diagram-as-code interface with Mermaid (Port 5151)

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/gui/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the gui module.

Pipeline Flow:
    main.py → 22_gui.py (this script) → gui/ (modular implementation)

How to run:
  # Headless mode (pipeline default - generates artifacts only)
  python src/22_gui.py --target-dir input/gnn_files --output-dir output --headless
  
  # Interactive mode (launch GUI servers)
  python src/22_gui.py --target-dir input/gnn_files --output-dir output --interactive
  
  # Specific GUI types
  python src/22_gui.py --gui-types "gui_1,oxdraw" --interactive
  
  # As part of pipeline (automatically runs in headless mode)
  python src/main.py  # runs all steps including GUI in headless mode

Expected outputs:
  - GUI artifacts in output/22_gui_output/ (headless mode)
  - Running GUI servers on ports 7860-7862, 5151 (interactive mode)
  - gui_processing_summary.json with execution status
  - Constructed/designed GNN models in markdown format
  - GUI status and metadata files

If you encounter errors:
  - Check that Gradio is installed: uv pip install -e .[gui]
  - Check that src/gui/ contains GUI modules
  - Check that the output directory is writable
  - For interactive mode, ensure ports 7860-7862, 5151 are available
  - Verify GUI dependencies (gradio, plotly, numpy, pandas)
  - For oxdraw: Install with `cargo install oxdraw` (optional for headless mode)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from gui import process_gui
except ImportError:
    def process_gui(target_dir, output_dir, **kwargs):
        """Fallback GUI processing when module unavailable."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("GUI module not available - using fallback")
        logger.info("Install GUI support with: uv pip install -e .[gui]")
        return True

run_script = create_standardized_pipeline_script(
    "22_gui.py",
    process_gui,
    "GUI processing for GNN models (Interactive Constructor)",
    additional_arguments={
        "headless": {
            "action": "store_true",
            "default": False,  # Will be set to True in process_gui if not --interactive
            "help": "Run in headless mode (artifacts only, no GUI servers)"
        },
        "interactive": {
            "action": "store_true",
            "default": False,
            "help": "Launch interactive GUI servers (overrides headless)"
        },
        "gui_types": {
            "type": str,
            "default": "gui_1,gui_2",
            "help": "Comma-separated list of GUI types to run (gui_1, gui_2, gui_3, oxdraw)"
        },
        "open_browser": {
            "action": "store_true",
            "default": False,
            "help": "Automatically open browser for interactive GUIs"
        },
    },
)

def main() -> int:
    """Main entry point for the GUI step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
