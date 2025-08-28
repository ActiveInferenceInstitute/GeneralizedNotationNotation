"""
GUI 2: Visual Matrix Editor for GNN Models

This is a click-and-drag visual interface for editing GNN models with:
- Visual matrix representation and editing
- Drag-and-drop state space modification
- Real-time GNN markdown generation
- POMDP template-based initialization
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .processor import run_gui
from .matrix_editor import (
    create_matrix_from_gnn,
    update_gnn_from_matrix,
    get_pomdp_template,
)


def gui_2(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]:
    """
    Run GUI 2: Visual Matrix Editor for GNN Models
    
    Args:
        target_dir: Directory containing GNN files to load (prefers POMDP templates)
        output_dir: Output directory for results  
        logger: Logger instance
        **kwargs: Additional options (headless, export_filename, etc.)
        
    Returns:
        Dictionary with execution results and status
    """
    try:
        # Extract GUI 2 specific parameters
        headless = kwargs.get('headless', False)
        export_filename = kwargs.get('export_filename', 'visual_model_gui2.md')
        open_browser = kwargs.get('open_browser', True)
        verbose = kwargs.get('verbose', False)
        
        logger.info("🎯 Starting GUI 2: Visual Matrix Editor")
        
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
            "gui_type": "gui_2",
            "description": "Visual Matrix Editor for GNN Models",
            "success": success,
            "output_file": str(output_dir / export_filename) if success else None,
            "backend": "gradio+plotly" if not headless else "headless",
            "features": [
                "Visual matrix representation and editing",
                "Drag-and-drop state space modification",
                "Real-time GNN markdown generation",
                "POMDP template-based initialization",
                "Interactive heatmaps and plots",
                "Matrix dimension validation"
            ]
        }
        
        if success:
            logger.info("✅ GUI 2 completed successfully")
        else:
            logger.error("❌ GUI 2 failed")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ GUI 2 failed with exception: {e}")
        return {
            "gui_type": "gui_2",
            "description": "Visual Matrix Editor for GNN Models",
            "success": False,
            "error": str(e),
            "backend": "error"
        }


def get_gui_2_info() -> Dict[str, Any]:
    """Get information about GUI 2 capabilities"""
    return {
        "name": "Visual Matrix Editor for GNN Models",
        "description": "Interactive visual interface for matrix editing with drag-and-drop functionality",
        "features": [
            "Visual matrix representation with heatmaps",
            "Interactive drag-and-drop editing",
            "Real-time GNN markdown synchronization",
            "POMDP template initialization",
            "Matrix validation and consistency checking",
            "Multi-tab interface (A, B, C, D matrices)",
            "Vector and tensor visualization"
        ],
        "requirements": ["gradio", "plotly", "numpy"],
        "export_format": "GNN Markdown (.md)",
        "interface_type": "visual-drag-drop",
        "template_base": "Active Inference POMDP"
    }


__all__ = [
    "gui_2",
    "get_gui_2_info",
    "run_gui",
    "create_matrix_from_gnn", 
    "update_gnn_from_matrix",
    "get_pomdp_template",
]
