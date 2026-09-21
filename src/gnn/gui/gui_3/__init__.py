#!/usr/bin/env python3
"""
GUI 3: State Space Design Studio
Low-dependency visual design experience for Active Inference models
"""

import logging
from pathlib import Path
from typing import Any, Dict


def gui_3(
    target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs: Any
) -> Dict[str, Any]:
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
        **kwargs: Extra pipeline kwargs (e.g. recursive, verbose) are
            tolerated and ignored.

    Returns:
        Dict with success status and metadata
    """
    try:
        logger.info("🎨 Starting GUI 3: State Space Design Studio")

        # Extract GUI 3 specific parameters (the pipeline forwards extra
        # kwargs, e.g. recursive, that run_gui does not accept).
        headless = kwargs.get("headless", False)
        export_filename = kwargs.get("export_filename", "designed_model_gui_3.md")
        open_browser = kwargs.get("open_browser", False)

        from . import processor as gui_3_processor
        from .processor import run_gui as run_design_studio

        success = run_design_studio(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            headless=headless,
            export_filename=export_filename,
            open_browser=open_browser,
        )

        result: Dict[str, Any] = {
            "gui_type": "gui_3",
            "name": "State Space Design Studio",
            "success": success,
            "features": [
                "Visual state space designer",
                "Ontology term editor",
                "Connection graph interface",
                "Parameter tuning controls",
                "Low-dependency HTML/CSS design",
            ],
        }

        # Port/url only when the interactive server was actually launched.
        if (
            success is True
            and not headless
            and gui_3_processor._GUI_BACKEND is not None
        ):
            result["port"] = gui_3_processor._GUI3_PORT
            result["url"] = f"http://localhost:{gui_3_processor._GUI3_PORT}"

        return result

    except Exception as e:
        logger.error(f"❌ GUI 3 failed: {e}")
        return {
            "gui_type": "gui_3",
            "name": "State Space Design Studio",
            "success": False,
            "error": str(e),
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
            "Export to GNN format",
        ],
        "dependencies": ["gradio"],
        "port": 7862,
    }
