#!/usr/bin/env python3
"""
GUI 3: State Space Design Studio Processor
Low-dependency visual design interface for Active Inference models
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..backend import (
    detect_gradio_backend,
    wait_for_server_launch,
    write_json_atomically,
    write_text_atomically,
)
from ..runner import launch_gradio_in_thread, load_first_markdown, resolve_output_root

# Shared backend detection (same recovery semantics as GUI 1 / GUI 2).
_GUI_STATUS = detect_gradio_backend()
_GUI_BACKEND = _GUI_STATUS.name

_GUI_BACKEND_REASON = _GUI_STATUS.reason
_GUI3_PORT = 7862


def run_gui(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    headless: bool = False,
    export_filename: str = "designed_model_gui_3.md",
    open_browser: bool = False,
) -> bool:
    """
    Launch the State Space Design Studio GUI.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for GUI results
        logger: Logger instance
        headless: Run without launching the browser GUI
        export_filename: Filename for the exported model
        open_browser: Open browser automatically on launch

    Returns:
        True if GUI launched successfully, False otherwise
    """

    try:
        output_root = resolve_output_root(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        starter_path = output_root / export_filename

        # Load starter GNN content
        starter_md = _load_starter_content(target_dir, logger)

        if headless or _GUI_BACKEND is None:
            # Generate design artifacts without launching GUI
            if _GUI_BACKEND is None:
                logger.info(
                    "Gradio not available; generating static headless GUI 3 "
                    "artifacts (expected for default pipeline; use "
                    "uv sync --extra gui for interactive UI)"
                )
                logger.info("Install GUI support with: uv sync --extra gui")
            else:
                logger.info(
                    "📦 Running GUI 3 in HEADLESS mode - generating artifacts only"
                )

            design_analysis = _analyze_gnn_design(starter_md)

            # Write starter model to file
            write_text_atomically(starter_path, starter_md)

            # Save design analysis
            analysis_file = output_root / "design_analysis.json"
            write_json_atomically(
                analysis_file,
                {
                    "gui_type": "design_studio",
                    "backend": _GUI_BACKEND or "none",
                    "status": "headless_mode"
                    if _GUI_BACKEND
                    else "static_headless_mode",
                    "analysis": design_analysis,
                    "export_path": str(starter_path),
                    "headless_mode": True,
                    "recommendations": [
                        "Run with --interactive to launch GUI server on port 7862"
                    ]
                    if _GUI_BACKEND
                    else [
                        "Install with: uv sync --extra gui",
                        "Run with --interactive for full GUI experience",
                    ],
                },
            )

            logger.info(f"🎨 Design analysis saved to: {analysis_file}")

            status_file = output_root / "design_studio_status.json"
            write_json_atomically(
                status_file,
                {
                    "gui_type": "design_studio",
                    "backend": _GUI_BACKEND or "none",
                    "launched": False,
                    "export_file": str(starter_path),
                    "status": "headless_mode"
                    if _GUI_BACKEND
                    else "static_headless_mode",
                    "reason": "headless_requested"
                    if _GUI_BACKEND
                    else "gradio_not_available",
                    "backend_reason": _GUI_BACKEND_REASON,
                    "analysis_file": str(analysis_file),
                    "recommendations": [
                        "Run with --interactive to launch GUI server on port 7862"
                    ]
                    if _GUI_BACKEND
                    else [
                        "Install with: uv sync --extra gui",
                        "Run with --interactive for full GUI experience",
                    ],
                },
            )

            return True

        # Interactive mode - build the Design Studio GUI
        logger.info("🔧 Building State Space Design Studio...")
        from .ui_designer import build_design_studio

        demo = build_design_studio(
            markdown_text=starter_md, export_path=starter_path, logger=logger
        )
        logger.info("✅ State Space Design Studio UI built successfully")

        # Launch GUI
        logger.info(
            f"🌐 Launching GUI 3 on http://localhost:{_GUI3_PORT} (open_browser={open_browser})"
        )
        thread = launch_gradio_in_thread(
            demo, port=_GUI3_PORT, open_browser=open_browser
        )
        status_file = output_root / "design_studio_status.json"
        launch_failure = wait_for_server_launch(thread, _GUI3_PORT)
        if launch_failure is not None:
            write_json_atomically(
                status_file,
                {
                    "gui_type": "design_studio",
                    "backend": _GUI_BACKEND,
                    "launched": False,
                    "export_file": str(starter_path),
                    "status": "launch_failed",
                    "reason": launch_failure,
                    "backend_reason": _GUI_BACKEND_REASON,
                },
            )
            logger.error(f"GUI 3 launch verification failed: {launch_failure}")
            return False
        logger.info(f"🎨 Design Studio is running on http://localhost:{_GUI3_PORT}")
        logger.info(
            "🔍 Features: Visual state space design, ontology editing, connection graphs, low-dependency approach"
        )

        # Save launch status
        write_json_atomically(
            status_file,
            {
                "gui_type": "design_studio",
                "backend": _GUI_BACKEND,
                "launched": True,
                "export_file": str(starter_path),
                "status": "interactive_mode",
                "reason": "gradio_launched",
                "backend_reason": _GUI_BACKEND_REASON,
                "port": _GUI3_PORT,
                "url": f"http://localhost:{_GUI3_PORT}",
                "features": [
                    "State space visual designer",
                    "Ontology term editor",
                    "Connection graph interface",
                    "Parameter tuning controls",
                ],
            },
        )

        return True

    except Exception as e:
        logger.error(f"❌ GUI 3 launch failed: {e}")
        return False


def _load_starter_content(target_dir: Path, logger: logging.Logger) -> str:
    """Load starter GNN content from target directory"""

    content = load_first_markdown(
        target_dir, prefer_patterns=("actinf_pomdp_agent.md",)
    )
    if content is not None:
        logger.info(f"📖 Loaded starter GNN content from {target_dir}")
        return content
    logger.warning("⚠️ No readable GNN files found, using default POMDP template")
    return _get_default_pomdp_template()


def _get_default_pomdp_template() -> str:
    """Get default POMDP template for design studio"""
    return """# GNN Example: Active Inference POMDP Agent (Design Studio)
# GNN Version: 1.0

## ModelName
Active Inference POMDP Agent - Design Studio Template

## StateSpaceBlock
A[3,3,type=float]   # Likelihood matrix
B[3,3,3,type=float] # Transition matrix
C[3,type=float]     # Preference vector
D[3,type=float]     # Prior vector
s[3,1,type=float]   # Current hidden-state distribution
o[3,1,type=int]     # Current observation
u[1,type=int]       # Selected action

## Connections
D>s
s-A
A-o
s-B
B>u

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates

## ModelParameters
num_hidden_states: 3
num_obs: 3
num_actions: 3
"""


def _analyze_gnn_design(gnn_content: str) -> dict[str, Any]:
    """Analyze GNN content for design studio insights"""

    analysis: dict[str, Any] = {
        "state_spaces": [],
        "ontology_terms": {},
        "connections": [],
        "parameters": {},
    }

    lines = gnn_content.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("## "):
            current_section = line[3:]
            continue

        if current_section == "StateSpaceBlock":
            if "[" in line and "]" in line and not line.startswith("#"):
                # Extract state space definitions
                var_name = line.split("[")[0]
                dimensions = line.split("[")[1].split("]")[0]
                analysis["state_spaces"].append(
                    {"variable": var_name, "dimensions": dimensions}
                )

        elif current_section == "ActInfOntologyAnnotation":
            if "=" in line and not line.startswith("#"):
                var, concept = line.split("=", 1)
                analysis["ontology_terms"][var.strip()] = concept.split("#", 1)[
                    0
                ].strip()

        elif current_section == "Connections":
            if line and not line.startswith("#"):
                analysis["connections"].append(line)

        elif current_section == "ModelParameters":
            if ":" in line and not line.startswith("#"):
                param, value = line.split(":", 1)
                analysis["parameters"][param.strip()] = value.split("#", 1)[0].strip()

    return analysis
