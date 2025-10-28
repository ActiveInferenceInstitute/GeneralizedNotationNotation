"""
oxdraw GUI Integration Module

Visual diagram-as-code interface for GNN Active Inference model construction
through bidirectional GNN ↔ Mermaid ↔ oxdraw synchronization.

This module is part of the GUI module (Step 22) and provides oxdraw as one
of the available GUI options alongside gui_1, gui_2, and gui_3.

Public API:
- oxdraw_gui: main function for oxdraw GUI processing
- get_oxdraw_info: get oxdraw module information
- process_oxdraw: core processing function
- gnn_to_mermaid: convert GNN to Mermaid format
- mermaid_to_gnn: parse Mermaid back to GNN
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Import core oxdraw functions
from .processor import process_oxdraw
from .mermaid_converter import gnn_to_mermaid
from .mermaid_parser import mermaid_to_gnn
from .utils import (
    infer_node_shape,
    infer_edge_style,
    validate_mermaid_syntax
)


def get_oxdraw_info() -> Dict[str, Any]:
    """Get oxdraw GUI information"""
    return {
        "name": "oxdraw",
        "description": "Visual diagram-as-code interface with Mermaid synchronization",
        "port": 5151,  # Default oxdraw server port
        "category": "Visual Diagram Editor",
        "features": [
            "GNN ↔ Mermaid bidirectional conversion",
            "Interactive visual model editing",
            "Real-time diagram-as-code synchronization",
            "Ontology term preservation",
            "Headless batch conversion support"
        ],
        "dependencies": ["oxdraw CLI (optional for interactive mode)"],
        "modes": ["interactive", "headless"]
    }


def oxdraw_gui(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    oxdraw GUI processing function (GUI module interface).
    
    This is the main entry point when oxdraw is selected as a GUI type
    in the GUI module (step 22).
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for oxdraw results
        logger: Logger instance
        verbose: Enable verbose logging
        **kwargs: Additional options including:
            - headless (bool): Run in headless mode (default: True for pipeline)
            - mode (str): "interactive" or "headless" (alternative to headless flag)
            - launch_editor (bool): Launch oxdraw editor (requires interactive mode)
            - port (int): oxdraw server port (default: 5151)
            - host (str): oxdraw server host (default: 127.0.0.1)
            - auto_convert (bool): Auto-convert GNN files to Mermaid
            - validate_on_save (bool): Validate conversions
    
    Returns:
        Dictionary with processing results:
        {
            "gui_type": "oxdraw",
            "success": bool,
            "mode": "interactive" or "headless",
            "files_processed": int,
            "outputs": List[str],
            "duration": float
        }
    """
    import time
    start_time = time.time()
    
    # Determine mode - headless by default for pipeline integration
    headless = kwargs.get('headless', True)
    mode = kwargs.get('mode', 'headless' if headless else 'interactive')
    
    # Override headless if explicitly set to interactive
    if mode == 'interactive':
        headless = False
    
    logger.info(f"🎨 oxdraw GUI - Mode: {mode.upper()}")
    
    # Create oxdraw-specific output directory
    oxdraw_output = output_dir / "oxdraw_output"
    oxdraw_output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Call core oxdraw processing function
        success = process_oxdraw(
            target_dir=target_dir,
            output_dir=oxdraw_output,
            logger=logger,
            mode=mode,
            **kwargs
        )
        
        # Collect outputs
        outputs = []
        if oxdraw_output.exists():
            outputs = [str(f) for f in oxdraw_output.glob("*.mmd")]
            outputs.extend([str(f) for f in oxdraw_output.glob("*.json")])
        
        duration = time.time() - start_time
        
        result = {
            "gui_type": "oxdraw",
            "success": success,
            "mode": mode,
            "files_processed": len(list(Path(target_dir).glob("*.md"))),
            "outputs": outputs,
            "duration": duration,
            "output_dir": str(oxdraw_output)
        }
        
        if success:
            logger.info(f"✅ oxdraw processing complete in {duration:.2f}s")
        else:
            logger.warning(f"⚠️ oxdraw processing completed with warnings")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ oxdraw processing failed: {e}")
        duration = time.time() - start_time
        return {
            "gui_type": "oxdraw",
            "success": False,
            "mode": mode,
            "error": str(e),
            "duration": duration,
            "output_dir": str(oxdraw_output)
        }


__all__ = [
    "oxdraw_gui",
    "get_oxdraw_info",
    "process_oxdraw",
    "gnn_to_mermaid",
    "mermaid_to_gnn",
    "infer_node_shape",
    "infer_edge_style",
    "validate_mermaid_syntax",
]
