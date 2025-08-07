#!/usr/bin/env python3
"""
Step 15: Audio Processing (Thin Orchestrator)

This step orchestrates audio generation and processing for GNN models.
It is a thin orchestrator that delegates core functionality to the audio module.

How to run:
  python src/15_audio.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Audio files generated from GNN models in the specified output directory
  - Sonification results and audio analysis
  - Audio visualization and metadata
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that audio dependencies are installed (librosa, soundfile, pedalboard, etc.)
  - Check that src/audio/ contains audio modules
  - Check that the output directory is writable
  - Verify audio backend configuration (SAPF, Pedalboard, etc.)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

# Import the audio processor
from audio import process_audio

def process_audio_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized audio processing function.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for audio results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("15_audio.py")
        
        # Set up output directory
        step_output_dir = get_output_dir_for_script("15_audio.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")
        logger.info(f"Recursive processing: {recursive}")
        
        # Extract audio-specific parameters
        duration = kwargs.get('duration', 30.0)
        audio_backend = kwargs.get('audio_backend', 'auto')
        
        logger.info(f"Audio duration: {duration}s")
        logger.info(f"Audio backend: {audio_backend}")
        
        # Validate input directory
        if not target_dir.exists():
            log_step_error(logger, f"Input directory does not exist: {target_dir}")
            return False
        
        # Find GNN files
        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True  # Not an error, just no files to process
        
        logger.info(f"Found {len(gnn_files)} GNN files to process for audio generation")
        
        # Process files using the audio module
        success = process_audio(
            target_dir=target_dir,
            output_dir=step_output_dir,
            verbose=verbose,
            **kwargs
        )
        
        if success:
            log_step_success(logger, f"Successfully generated audio for {len(gnn_files)} GNN models")
        else:
            log_step_error(logger, "Audio processing failed")
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Audio processing failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main audio processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("15_audio")
    
    # Setup logging
    logger = setup_step_logging("audio", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("15_audio.py", Path(args.output_dir))
        
        # Process using standardized pattern
        success = process_audio_standardized(
            target_dir=Path(args.target_dir) if hasattr(args, 'target_dir') else Path("input/gnn_files"),
            output_dir=output_dir,
            logger=logger,
            recursive=getattr(args, 'recursive', False),
            verbose=getattr(args, 'verbose', False),
            duration=getattr(args, 'duration', 30.0),
            audio_backend=getattr(args, 'audio_backend', 'auto')
        )
        
        return 0 if success else 1
        
    except Exception as e:
        log_step_error(logger, f"Audio processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
