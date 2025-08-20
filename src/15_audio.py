#!/usr/bin/env python3
"""
Step 15: Audio Processing (Thin Orchestrator)

This step orchestrates audio processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/audio/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the audio module.

Pipeline Flow:
    main.py → 15_audio.py (this script) → audio/ (modular implementation)

How to run:
  python src/15_audio.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Audio processing results in the specified output directory
  - Comprehensive audio reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that audio dependencies are installed
  - Check that src/audio/ contains audio modules
  - Check that the output directory is writable
  - Verify audio configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from audio import process_audio

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "15_audio.py",
    process_audio,
    "Audio processing"
)

def main() -> int:
    """Main entry point for the 15_audio step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
