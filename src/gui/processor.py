from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:
    import gradio as gr  # Lightweight, widely available
    _GUI_BACKEND = "gradio"
except Exception:
    gr = None  # type: ignore
    _GUI_BACKEND = None

from utils.pipeline_template import (
    log_step_success,
    log_step_error,
)

from .markdown import parse_state_space_from_markdown  # noqa: F401


def run_gui(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    verbose: bool = False,
    headless: bool = False,
    export_filename: str = "constructed_model.md",
    open_browser: bool = True,
) -> bool:
    """
    Launch the interactive GUI or generate GUI artifacts in headless mode.

    - Left: controls to insert/edit GNN components
    - Right: synchronized plaintext GNN markdown editor
    
    If headless=True, generate a minimal HTML description and save a starter
    GNN markdown file to output_dir without launching a server.
    """
    try:
        # Normalize output directory to the pipeline standard (nest under step output) when available
        try:
            from pipeline.config import get_output_dir_for_script
            output_root = get_output_dir_for_script("22_gui.py", output_dir)
        except Exception:
            output_root = output_dir

        output_root.mkdir(parents=True, exist_ok=True)

        # Starter markdown assembled from any existing file in target_dir if present
        starter_md = _load_first_markdown(target_dir)
        if starter_md is None:
            starter_md = """# GNN Model\n\ncomponents:\n  - name: example_component\n    type: observation\n    states: [s1, s2]\n\n"""

        starter_path = output_root / export_filename
        starter_path.write_text(starter_md)

        if headless or _GUI_BACKEND is None:
            # Persist a small artifact describing GUI availability
            (output_root / "22_gui_output" / "gui_status.json").parent.mkdir(parents=True, exist_ok=True)
            (output_root / "22_gui_output" / "gui_status.json").write_text(json.dumps({
                "backend": _GUI_BACKEND or "none",
                "launched": False,
                "export_file": str(starter_path)
            }, indent=2))
            log_step_success(logger, f"GUI artifacts generated (headless). Export: {starter_path}")
            return True

        # Build Gradio UI via modular builder
        from .ui import build_gui
        demo = build_gui(markdown_text=starter_md, export_path=starter_path)

        # Launch returns a server; prevent block in pipelines by non-blocking share=False
        demo.launch(
            share=False,
            prevent_thread_lock=not open_browser,
            server_name="0.0.0.0",
            inbrowser=open_browser,
        )

        # Record availability artifact
        (output_root / "22_gui_output" / "gui_status.json").parent.mkdir(parents=True, exist_ok=True)
        (output_root / "22_gui_output" / "gui_status.json").write_text(json.dumps({
            "backend": _GUI_BACKEND,
            "launched": True,
            "export_file": str(starter_path)
        }, indent=2))

        log_step_success(logger, "GUI launched")
        return True
    except Exception as e:
        log_step_error(logger, f"GUI launch failed: {e}")
        return False


def _load_first_markdown(target_dir: Path) -> Optional[str]:
    try:
        for p in sorted(target_dir.glob("*.md")):
            return p.read_text()
        for p in sorted(target_dir.glob("**/*.md")):
            return p.read_text()
    except Exception:
        return None
    return None


