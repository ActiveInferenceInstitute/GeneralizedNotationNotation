"""
Processor for GUI 2: Visual Matrix Editor

Handles the main processing logic for the visual matrix editing interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:
    import gradio as gr
    _GUI_BACKEND = "gradio"
except ImportError:
    gr = None  # type: ignore
    _GUI_BACKEND = None

from utils.pipeline_template import (
    log_step_success,
    log_step_error,
)

from .matrix_editor import (
    get_pomdp_template,
    create_matrix_from_gnn,
    update_gnn_from_matrix,
    validate_matrix_dimensions,
)


def run_gui(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    verbose: bool = False,
    headless: bool = False,
    export_filename: str = "visual_model.md",
    open_browser: bool = True,
) -> bool:
    """
    Launch the visual matrix editor GUI or generate artifacts in headless mode.
    
    Features:
    - Visual matrix representation with drag-and-drop editing
    - Real-time GNN markdown generation
    - POMDP template initialization
    - Interactive state space modification
    """
    try:
        # Setup output directory
        try:
            from pipeline.config import get_output_dir_for_script
            output_root = get_output_dir_for_script("22_gui.py", output_dir)
        except Exception:
            output_root = output_dir

        output_root.mkdir(parents=True, exist_ok=True)

        # Load starting template - prefer POMDP template for GUI 2
        starter_md = _load_template_markdown(target_dir)
        if starter_md is None:
            starter_md = get_pomdp_template()

        starter_path = output_root / export_filename
        starter_path.write_text(starter_md)

        if headless or _GUI_BACKEND is None:
            # Generate headless artifacts for GUI 2
            gui_output_dir = output_root
            gui_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visual matrix representation
            visual_data = create_matrix_from_gnn(starter_md)
            
            # Save visual data as JSON for potential future loading
            (gui_output_dir / "visual_matrices.json").write_text(
                json.dumps(visual_data, indent=2)
            )
            
            # Generate status artifact
            (gui_output_dir / "gui_status.json").write_text(json.dumps({
                "backend": _GUI_BACKEND or "none",
                "launched": False,
                "export_file": str(starter_path),
                "gui_type": "visual_matrix_editor",
                "visual_data_file": str(gui_output_dir / "visual_matrices.json"),
                "features": [
                    "Visual matrix editing",
                    "Drag-and-drop interface", 
                    "Real-time GNN generation",
                    "POMDP template-based"
                ]
            }, indent=2))
            
            log_step_success(logger, f"GUI 2 artifacts generated (headless). Export: {starter_path}")
            return True

        # Build visual matrix editor GUI
        from .ui import build_visual_gui
        demo = build_visual_gui(markdown_text=starter_md, export_path=starter_path, logger=logger)

        # Launch GUI
        demo.launch(
            share=False,
            prevent_thread_lock=not open_browser,
            server_name="0.0.0.0",
            inbrowser=open_browser,
        )

        # Record launch status
        (output_root / "gui_status.json").write_text(json.dumps({
            "backend": _GUI_BACKEND,
            "launched": True,
            "export_file": str(starter_path),
            "gui_type": "visual_matrix_editor",
            "features": [
                "Visual matrix editing",
                "Drag-and-drop interface",
                "Real-time GNN generation", 
                "POMDP template-based"
            ]
        }, indent=2))

        log_step_success(logger, "GUI 2 (Visual Matrix Editor) launched")
        return True

    except Exception as e:
        log_step_error(logger, f"GUI 2 launch failed: {e}")
        return False


def _load_template_markdown(target_dir: Path) -> Optional[str]:
    """
    Load template markdown, preferring the POMDP template for visual editing.
    """
    try:
        # Look for specific POMDP files first
        pomdp_files = list(target_dir.glob("*pomdp*.md")) + list(target_dir.glob("*POMDP*.md"))
        if pomdp_files:
            return pomdp_files[0].read_text()
        
        # Fall back to any markdown file
        for p in sorted(target_dir.glob("*.md")):
            return p.read_text()
        
        for p in sorted(target_dir.glob("**/*.md")):
            return p.read_text()
            
    except Exception:
        return None
        
    return None
