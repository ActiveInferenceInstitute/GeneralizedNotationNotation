#!/usr/bin/env python3
"""
Step 13: LLM Processing (Thin Orchestrator)

This step orchestrates LLM processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/llm/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the llm module.

Pipeline Flow:
    main.py → 13_llm.py (this script) → llm/ (modular implementation)

How to run:
  python src/13_llm.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - LLM processing results in the specified output directory
  - Comprehensive LLM reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that LLM dependencies are installed
  - Check that src/llm/ contains LLM modules
  - Check that the output directory is writable
  - Verify LLM configuration and requirements
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning,
    create_standardized_pipeline_script,
)
from utils.argument_utils import ArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

from llm import (
    process_llm,
    analyze_gnn_model,
    get_module_info,
    get_available_providers,
)

run_script = create_standardized_pipeline_script(
    "13_llm.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_llm_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "LLM processing for GNN analysis",
)


def _run_llm_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized LLM processing function with improved timeout handling and logging.

    Args:
        target_dir: Directory containing GNN files to analyze
        output_dir: Output directory for LLM results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing LLM")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("13_llm") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("13_llm.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract LLM-specific parameters
        analysis_type = kwargs.get('analysis_type', 'comprehensive')
        provider = kwargs.get('provider', 'auto')

        if analysis_type:
            logger.info(f"Analysis type: {analysis_type}")
        if provider:
            logger.info(f"LLM provider: {provider}")

        # Validate input directory
        if not target_dir.exists():
            log_step_error(logger, f"Input directory does not exist: {target_dir}")
            return False

        # Find GNN files
        pattern = "**/*.md" if kwargs.get('recursive', False) else "*.md"
        gnn_files = list(target_dir.glob(pattern))

        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True  # Not an error, just no files to process

        logger.info(f"Found {len(gnn_files)} GNN files to analyze")

        # Check for timeout configuration
        llm_timeout = kwargs.get('llm_timeout', 360)  # Default 6 minutes
        logger.info(f"LLM processing timeout: {llm_timeout}s")

        # Add timeout wrapper for LLM processing
        import signal
        import time

        def timeout_handler(signum, frame):
            logger.warning(f"LLM processing timed out after {llm_timeout} seconds")
            log_step_warning(logger, "LLM processing timed out - completing with partial results")
            raise TimeoutError(f"LLM processing timed out after {llm_timeout} seconds")

        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(llm_timeout)

        try:
            logger.info("🚀 Starting LLM processing with timeout protection...")
            logger.info("💡 If processing seems stuck, it will timeout and provide partial results")

            # Process LLM via module API with progress tracking
            logger.info("LLM module available, processing files...")

            # Add periodic progress logging
            def log_progress():
                logger.info("⏱️ LLM processing in progress... (this may take several minutes for complex models)")

            # Start progress logging
            import threading
            progress_timer = threading.Timer(30.0, log_progress)  # Log progress every 30 seconds
            progress_timer.start()

            try:
                result = process_llm(target_dir=target_dir, output_dir=step_output_dir, **kwargs)
                progress_timer.cancel()  # Cancel the progress timer if we complete normally
                logger.info("✅ LLM processing completed normally")
                return result
            except Exception as e:
                progress_timer.cancel()  # Cancel the progress timer on error
                logger.error(f"❌ LLM processing failed: {e}")
                log_step_error(logger, f"LLM processing failed: {e}")
                return False
            finally:
                progress_timer.cancel()  # Always cancel the timer

        except TimeoutError as te:
            logger.warning(f"⚠️ LLM processing timed out: {te}")
            log_step_warning(logger, f"LLM processing timed out - may have partial results: {te}")
            return True  # Return success with warnings for timeout
        except Exception as e:
            logger.error(f"❌ LLM processing failed: {e}")
            log_step_error(logger, f"LLM processing failed: {e}")
            return False
        finally:
            # Restore original signal handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    except Exception as e:
        log_step_error(logger, f"LLM processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the LLM step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
