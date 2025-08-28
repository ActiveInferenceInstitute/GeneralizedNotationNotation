"""
GUI 1: Interactive GNN Constructor (Form-based Interface)

This is the original form-based GUI implementation with:
- Component management via forms
- State space editing
- Live markdown synchronization
"""

from pathlib import Path
from typing import Dict, Any
import logging

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


def gui_1(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]:
    """
    Run GUI 1: Form-based Interactive GNN Constructor
    
    Args:
        target_dir: Directory containing GNN files to load
        output_dir: Output directory for results
        logger: Logger instance
        **kwargs: Additional options (headless, export_filename, etc.)
        
    Returns:
        Dictionary with execution results and status
    """
    try:
        # Extract GUI 1 specific parameters
        headless = kwargs.get('headless', False)
        export_filename = kwargs.get('export_filename', 'constructed_model_gui1.md')
        open_browser = kwargs.get('open_browser', True)
        verbose = kwargs.get('verbose', False)
        
        logger.info("🎮 Starting GUI 1: Form-based Interactive GNN Constructor")
        
        success = run_gui(
            target_dir=target_dir,
            output_dir=output_dir, 
            logger=logger,
            verbose=verbose,
            headless=headless,
            export_filename=export_filename,
            open_browser=open_browser,
        )
        
        result = {
            "gui_type": "gui_1",
            "description": "Form-based Interactive GNN Constructor",
            "success": success,
            "output_file": str(output_dir / export_filename) if success else None,
            "backend": "gradio" if not headless else "headless",
            "features": [
                "Component management via forms",
                "State space editing", 
                "Live markdown synchronization",
                "Two-pane interface"
            ]
        }
        
        if success:
            logger.info("✅ GUI 1 completed successfully")
        else:
            logger.error("❌ GUI 1 failed")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ GUI 1 failed with exception: {e}")
        return {
            "gui_type": "gui_1",
            "description": "Form-based Interactive GNN Constructor", 
            "success": False,
            "error": str(e),
            "backend": "error"
        }


def get_gui_1_info() -> Dict[str, Any]:
    """Get information about GUI 1 capabilities"""
    return {
        "name": "Form-based Interactive GNN Constructor",
        "description": "Original GUI with component forms and state space editing",
        "features": [
            "Component management via forms",
            "State space editing with live validation",
            "Two-pane interface (controls + markdown)",
            "Real-time markdown synchronization",
            "Gradio-based web interface"
        ],
        "requirements": ["gradio"],
        "export_format": "GNN Markdown (.md)",
        "interface_type": "form-based"
    }


__all__ = [
    "gui_1",
    "get_gui_1_info",
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
