"""
GUI module for the Interactive GNN Constructor.

Public API:
- run_gui: launch or generate GUI artifacts
- add_component_to_markdown: add a component block to markdown
- update_component_states: append or replace states for a component
- remove_component_from_markdown: remove a component block by name
- parse_components_from_markdown: parse components section to structured list
"""

from .processor import run_gui
from .markdown import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_components_from_markdown,
    parse_state_space_from_markdown,
    add_state_space_entry,
    update_state_space_entry,
    remove_state_space_entry,
)


def process_gui(target_dir, output_dir, verbose=False, **kwargs):
    """
    Main processing function for gui.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Processing gui for files in {target_dir}")
        # Placeholder implementation - delegate to actual module functions
        # This would be replaced with actual implementation
        logger.info(f"Gui processing completed")
        return True
    except Exception as e:
        logger.error(f"Gui processing failed: {e}")
        return False


__all__ = [
    "run_gui",
    "add_component_to_markdown",
    "update_component_states",
    "remove_component_from_markdown",
    "parse_components_from_markdown",
    "parse_state_space_from_markdown",
    "add_state_space_entry",
    "update_state_space_entry",
    "remove_state_space_entry",
, 'process_gui']


