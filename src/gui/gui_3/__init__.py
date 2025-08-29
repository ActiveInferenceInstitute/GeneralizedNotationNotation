#!/usr/bin/env python3
"""
GUI 3: State Space Design Studio
Low-dependency visual design experience for Active Inference models
"""

import logging
from pathlib import Path
from typing import Any, Dict

def gui_3(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]:
    """
    Launch GUI 3: State Space Design Studio
    
    A low-dependency visual design experience focused on:
    - State space architecture design
    - Ontology term assertion editing  
    - Visual connection modeling
    - Parameter tuning interface
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for GUI results
        logger: Logger instance
        **kwargs: Additional arguments (headless, verbose, etc.)
        
    Returns:
        Dict with success status and metadata
    """
    try:
        logger.info("🎨 Starting GUI 3: State Space Design Studio")
        
        from .processor import run_gui as run_design_studio
        
        result = run_design_studio(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            **kwargs
        )
        
        return {
            "gui_type": "gui_3",
            "name": "State Space Design Studio",
            "success": result,
            "features": [
                "Visual state space designer",
                "Ontology term editor", 
                "Connection graph interface",
                "Parameter tuning controls",
                "Low-dependency HTML/CSS design"
            ],
            "port": 7862,
            "url": "http://localhost:7862"
        }
        
    except Exception as e:
        logger.error(f"❌ GUI 3 failed: {e}")
        return {
            "gui_type": "gui_3", 
            "name": "State Space Design Studio",
            "success": False,
            "error": str(e)
        }


def get_gui_3_info() -> Dict[str, Any]:
    """Get metadata about GUI 3"""
    return {
        "name": "State Space Design Studio",
        "description": "Low-dependency visual design experience for state spaces and ontology",
        "features": [
            "Visual state space architecture",
            "Ontology term assertions", 
            "Connection graph design",
            "Parameter tuning interface",
            "Export to GNN format"
        ],
        "dependencies": ["gradio"],
        "port": 7862
    }
