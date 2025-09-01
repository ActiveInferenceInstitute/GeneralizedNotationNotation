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
            # Enhanced fallback handling for missing GUI backend
            if _GUI_BACKEND is None:
                logger.warning("⚠️ Gradio not available - generating fallback artifacts only")
                logger.info("💡 Install GUI support with: uv pip install -e .[gui]")
            
            # Persist enhanced artifact describing GUI availability
            # `output_root` is already the step-specific output directory
            gui_output_dir = output_root
            gui_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive fallback status report
            fallback_status = {
                "backend": _GUI_BACKEND or "none",
                "launched": False,
                "export_file": str(starter_path),
                "gui_type": "form_based_constructor",
                "status": "fallback_mode" if _GUI_BACKEND is None else "headless_mode",
                "reason": "gradio_not_available" if _GUI_BACKEND is None else "headless_requested",
                "recommendations": [
                    "Install gradio with: uv pip install gradio>=4.0.0",
                    "Run with --interactive-mode for full GUI experience",
                    "Use generated template as starting point for manual editing"
                ] if _GUI_BACKEND is None else [
                    "Generated template available for manual editing"
                ]
            }
            
            (gui_output_dir / "gui_status.json").write_text(json.dumps(fallback_status, indent=2))
            
            log_step_success(logger, f"GUI artifacts generated ({'fallback' if _GUI_BACKEND is None else 'headless'}). Export: {starter_path}")
            return True

        # Build Gradio UI via modular builder
        logger.info("🔧 Building Form-based Interactive GNN Constructor...")
        from .ui import build_gui
        demo = build_gui(markdown_text=starter_md, export_path=starter_path, logger=logger)
        logger.info("✅ Form-based Constructor UI built successfully")

        # Launch GUI with proper blocking behavior
        if headless:
            # Generate artifacts without launching server in headless mode
            pass  # Already handled above
        else:
            # Launch GUI server - stay open but allow other GUIs to launch too
            logger.info(f"🌐 Launching GUI 1 on http://localhost:7860 (open_browser={open_browser})")
            import threading
            import time
            
            def launch_gui():
                logger.info("🎮 Form-based Constructor starting...")
                demo.launch(
                    share=False,
                    prevent_thread_lock=False,  # Let the thread properly block on the server
                    server_name="0.0.0.0",
                    server_port=7860,
                    inbrowser=open_browser,
                    show_error=True,
                    quiet=False,  # Show server startup messages
                )
            
            # Launch in a separate thread to allow multiple GUIs
            gui_thread = threading.Thread(target=launch_gui, daemon=False)
            gui_thread.start()
            
            # Give it a moment to start and verify
            time.sleep(3)
            logger.info("🎮 GUI 1 is running on http://localhost:7860")
            logger.info("🔍 Features: Component management, state space editing, live markdown sync")

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


