"""
GUI module for Interactive GNN Constructors.

This module provides multiple GUI implementations:
- GUI 1: Form-based Interactive GNN Constructor  
- GUI 2: Visual Matrix Editor with drag-and-drop
- GUI 3: State Space Design Studio
- oxdraw: Visual diagram-as-code interface

Public API:
- process_gui: main processing function (runs all available GUIs)
- gui_1: form-based GUI with component management
- gui_2: visual matrix editor with drag-and-drop
- gui_3: state space design studio
- oxdraw: visual diagram-as-code with Mermaid
- get_available_guis: list all available GUI implementations
"""

# Import GUI runners
from .gui_1 import gui_1, get_gui_1_info
from .gui_2 import gui_2, get_gui_2_info
from .gui_3 import gui_3, get_gui_3_info
from .oxdraw import oxdraw_gui, get_oxdraw_info

# Import GUI 1 utilities for backward compatibility
from .gui_1 import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_components_from_markdown,
    parse_state_space_from_markdown,
    add_state_space_entry,
    update_state_space_entry,
    remove_state_space_entry,
)

def get_available_guis():
    """Get list of available GUI implementations with their info"""
    return {
        "gui_1": get_gui_1_info(),
        "gui_2": get_gui_2_info(),
        "gui_3": get_gui_3_info(),
        "oxdraw": get_oxdraw_info(),
    }

def process_gui(target_dir, output_dir, verbose=False, **kwargs):
    """
    Main processing function for GUI module.
    
    By default, runs all available GUI implementations in headless mode.
    Can be restricted using gui_types parameter.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
            - gui_types: List of GUI types to run (default: gui_1, gui_2)
            - headless: Run in headless mode (default: True for pipeline)
            - interactive: Launch interactive GUI servers (overrides headless)
            - open_browser: Whether to open browser for interactive GUIs
        
    Returns:
        Boolean indicating success of all GUI runs
    """
    import logging
    from pathlib import Path
    import json
    
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Handle interactive vs headless mode
    # Interactive mode overrides headless
    interactive = kwargs.get('interactive', False)
    if interactive:
        kwargs['headless'] = False
        logger.info("🎮 Running in INTERACTIVE mode - will launch GUI servers")
    else:
        # Default to headless mode for pipeline integration
        kwargs['headless'] = kwargs.get('headless', True)
        if kwargs['headless']:
            logger.info("📦 Running in HEADLESS mode - generating artifacts only (fast)")
    
    # Determine which GUIs to run
    gui_types = kwargs.get('gui_types', 'gui_1,gui_2')
    if isinstance(gui_types, str):
        gui_types = [g.strip() for g in gui_types.split(',')]
    
    # Prepare kwargs for GUI functions - remove keys that are passed explicitly
    gui_kwargs = {k: v for k, v in kwargs.items() 
                  if k not in ['logger', 'target_dir', 'output_dir', 'verbose']}
    
    results = {}
    overall_success = True
    
    try:
        logger.info(f"Processing GUI module for files in {target_dir}")
        logger.info(f"Running GUI types: {gui_types}")
        logger.info(f"Mode: {'INTERACTIVE' if not kwargs['headless'] else 'HEADLESS'}")
        
        # Run each requested GUI
        for gui_type in gui_types:
            try:
                if gui_type == 'gui_1':
                    result = gui_1(
                        target_dir=Path(target_dir), 
                        output_dir=Path(output_dir),
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                elif gui_type == 'gui_2':
                    result = gui_2(
                        target_dir=Path(target_dir),
                        output_dir=Path(output_dir), 
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                elif gui_type == 'gui_3':
                    result = gui_3(
                        target_dir=Path(target_dir),
                        output_dir=Path(output_dir),
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                elif gui_type == 'oxdraw':
                    result = oxdraw_gui(
                        target_dir=Path(target_dir),
                        output_dir=Path(output_dir),
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                else:
                    logger.warning(f"Unknown GUI type: {gui_type}")
                    result = {
                        "gui_type": gui_type,
                        "success": False,
                        "error": f"Unknown GUI type: {gui_type}"
                    }
                
                results[gui_type] = result
                if not result.get('success', False):
                    overall_success = False
                    
            except Exception as e:
                logger.error(f"GUI {gui_type} failed: {e}")
                results[gui_type] = {
                    "gui_type": gui_type,
                    "success": False,
                    "error": str(e)
                }
                overall_success = False
        
        # Save processing summary
        try:
            output_path = Path(output_dir)
            summary_file = output_path / "gui_processing_summary.json"
            summary_file.write_text(json.dumps({
                "mode": "interactive" if not kwargs['headless'] else "headless",
                "gui_types": gui_types,
                "results": results,
                "overall_success": overall_success
            }, indent=2))
            logger.info(f"📊 GUI processing summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"Failed to save GUI processing summary: {e}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"GUI processing failed: {e}")
        return False

# Legacy compatibility - point to GUI 1
def run_gui(target_dir, output_dir, logger, **kwargs):
    """Legacy compatibility function - delegates to GUI 1"""
    return gui_1(target_dir, output_dir, logger, **kwargs)

__all__ = [
    "process_gui",
    "gui_1", 
    "gui_2",
    "gui_3",
    "oxdraw_gui",
    "get_available_guis",
    "get_gui_1_info",
    "get_gui_2_info",
    "get_gui_3_info",
    "get_oxdraw_info",
    # Legacy GUI 1 utilities
    "run_gui",
    "add_component_to_markdown",
    "update_component_states", 
    "remove_component_from_markdown",
    "parse_components_from_markdown",
    "parse_state_space_from_markdown",
    "add_state_space_entry",
    "update_state_space_entry",
    "remove_state_space_entry",
]


