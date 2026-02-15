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

# Import module function
try:
    from audio import process_audio
except ImportError as e:
    def process_audio(target_dir, output_dir, logger=None, **kwargs):
        """Fallback audio processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning(f"Audio module not available - using fallback: {e}")
        logger.info("Install audio support with: uv pip install -e .[audio]")
        
        # Create a fallback result file to let downstream steps know we skipped
        try:
            from pathlib import Path
            import json
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "audio_processing_skipped.json", "w") as f:
                json.dump({"status": "skipped", "reason": str(e)}, f, indent=2)
        except Exception:
            pass
            
        return True

run_script = create_standardized_pipeline_script(
    "15_audio.py",
    process_audio,
    "Audio processing for GNN models",
    additional_arguments={
        "duration": {"type": float, "default": 30.0, "help": "Audio duration in seconds"},
        "audio_backend": {"type": str, "default": "auto", "help": "Audio backend to use (auto, sapf, pedalboard)"},
        "sonification": {"type": bool, "default": True, "help": "Generate sonification", "flag": "--sonification"},
        "full_analysis": {"type": bool, "default": False, "help": "Run full audio analysis", "flag": "--full-analysis"}
    }
)

def main() -> int:
    """Main entry point for the audio step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())