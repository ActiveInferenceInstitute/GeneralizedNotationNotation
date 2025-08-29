#!/usr/bin/env python3
"""
Step 22: GUI Processing (Thin Orchestrator)

This step orchestrates GUI processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/gui/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the gui module.

Pipeline Flow:
    main.py → 22_gui.py (this script) → gui/ (modular implementation)

How to run:
  python src/22_gui.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - GUI processing results in the specified output directory
  - Comprehensive GUI reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that GUI dependencies are installed
  - Check that src/gui/ contains GUI modules
  - Check that the output directory is writable
  - Verify GUI configuration and requirements
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

from gui import (
    process_gui,
    gui_1,
    gui_2,
    gui_3,
    get_available_guis,
)

run_script = create_standardized_pipeline_script(
    "22_gui.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_gui_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "GUI processing for interactive GNN construction",
    additional_arguments={
        "gui_mode": {
            "dest": "gui_mode",
            "default": "all",
            "help": "GUI mode: 'all', 'gui_1', 'gui_2', 'gui_3', or comma-separated list"
        },
        "interactive_mode": {
            "dest": "interactive_mode", 
            "action": "store_true",
            "help": "Run GUIs in interactive mode (default: headless artifact generation)"
        }
    }
)


def _run_gui_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized GUI processing function.

    Args:
        target_dir: Directory containing GNN files for GUI processing
        output_dir: Output directory for GUI results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing GUI")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("22_gui") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("22_gui.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract GUI-specific parameters
        gui_mode = kwargs.get('gui_mode', 'all')  # 'all', 'gui_1', 'gui_2', or specific list
        interactive_mode = kwargs.get('interactive_mode', False)
        headless = kwargs.get('headless', False)

        logger.info(f"GUI mode: {gui_mode}")
        if interactive_mode:
            logger.info("Running in interactive mode")
        if headless:
            logger.info("Running in headless mode (generating artifacts only)")

        # List available GUIs
        available_guis = get_available_guis()
        logger.info(f"Available GUI implementations: {list(available_guis.keys())}")
        
        # Determine which GUIs to run
        if gui_mode == 'all':
            gui_types = list(available_guis.keys())
        elif gui_mode in available_guis:
            gui_types = [gui_mode]
        else:
            try:
                # Try to parse as comma-separated list
                gui_types = [g.strip() for g in str(gui_mode).split(',')]
                # Filter to only valid GUI types
                gui_types = [g for g in gui_types if g in available_guis]
                if not gui_types:
                    logger.warning(f"No valid GUI types found in '{gui_mode}', defaulting to all")
                    gui_types = list(available_guis.keys())
            except:
                logger.warning(f"Invalid GUI mode '{gui_mode}', defaulting to all")
                gui_types = list(available_guis.keys())

        logger.info(f"Will run GUI types: {gui_types}")

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

        logger.info(f"Found {len(gnn_files)} GNN files for GUI processing")

        # Process GUIs individually
        results = {}
        overall_success = True
        
        for gui_type in gui_types:
            try:
                logger.info(f"🚀 Starting {gui_type.upper()}: {available_guis[gui_type]['name']}")
                
                # Create GUI-specific output directory
                gui_output_dir = step_output_dir / f"{gui_type}_output"
                gui_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run the specific GUI
                if gui_type == 'gui_1':
                    result = gui_1(
                        target_dir=target_dir,
                        output_dir=gui_output_dir,
                        logger=logger,
                        verbose=kwargs.get('verbose', False),
                        headless=headless,
                        export_filename=f"constructed_model_{gui_type}.md",
                        open_browser=interactive_mode and not headless,
                    )
                elif gui_type == 'gui_2':
                    result = gui_2(
                        target_dir=target_dir,
                        output_dir=gui_output_dir,
                        logger=logger,
                        verbose=kwargs.get('verbose', False),
                        headless=headless,
                        export_filename=f"visual_model_{gui_type}.md",
                        open_browser=interactive_mode and not headless,
                    )
                elif gui_type == 'gui_3':
                    result = gui_3(
                        target_dir=target_dir,
                        output_dir=gui_output_dir,
                        logger=logger,
                        verbose=kwargs.get('verbose', False),
                        headless=headless,
                        export_filename=f"designed_model_{gui_type}.md",
                        open_browser=interactive_mode and not headless,
                    )
                else:
                    result = {
                        "gui_type": gui_type,
                        "success": False,
                        "error": f"Unknown GUI type: {gui_type}"
                    }
                
                results[gui_type] = result
                
                if result.get('success', False):
                    logger.info(f"✅ {gui_type.upper()} completed successfully")
                else:
                    logger.error(f"❌ {gui_type.upper()} failed: {result.get('error', 'Unknown error')}")
                    overall_success = False
                    
            except Exception as e:
                logger.error(f"❌ {gui_type.upper()} failed with exception: {e}")
                results[gui_type] = {
                    "gui_type": gui_type,
                    "success": False,
                    "error": str(e)
                }
                overall_success = False
        
        # Save overall results summary
        summary_file = step_output_dir / "gui_processing_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump({
                "gui_types_requested": gui_types,
                "gui_types_available": list(available_guis.keys()),
                "results": results,
                "overall_success": overall_success,
                "total_guis_run": len(results),
                "successful_guis": sum(1 for r in results.values() if r.get('success', False)),
                "processing_mode": "headless" if headless else "interactive"
            }, f, indent=2)
        
        logger.info(f"📊 GUI processing summary saved to: {summary_file}")
        
        # Keep the process alive if GUIs were launched successfully (non-headless mode)
        if not headless and overall_success and any(r.get('success', False) for r in results.values()):
            import time
            logger.info("🌐 GUIs are running! Access them at:")
            for gui_type, result in results.items():
                if result.get('success', False):
                    if gui_type == 'gui_1':
                        port = 7860
                    elif gui_type == 'gui_2':
                        port = 7861
                    elif gui_type == 'gui_3':
                        port = 7862
                    else:
                        port = 7860  # fallback
                    logger.info(f"  • {gui_type.upper()}: http://localhost:{port}")
            
            logger.info("💡 Press Ctrl+C to stop all GUIs and exit")
            
            try:
                # Keep the main process alive to maintain GUI threads
                while True:
                    time.sleep(10)  # Check every 10 seconds
                    # You could add health checks here if needed
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down GUIs...")
                return overall_success
        
        return overall_success

    except Exception as e:
        log_step_error(logger, f"GUI processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the GUI step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
