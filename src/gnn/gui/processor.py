"""
GUI Processor - Core GUI processing logic for GNN Pipeline.

This module contains the main processing functions for the GUI module,
extracted from __init__.py to follow the thin orchestrator pattern.

Composable pieces:
- ``normalize_gui_types``: parse/validate the ``gui_types`` option.
- ``summarize_gui_results``: aggregate per-GUI results into a typed summary.
- ``collect_pipeline_outputs``: discover pipeline artifacts for navigation.
- ``generate_html_navigation``: render the navigation page from discovery.
"""

from __future__ import annotations

import html
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote

from gnn.pipeline.step_registry import STEPS, StepInfo

from .backend import write_json_atomically, write_text_atomically
from .runner import interactive_servers_running

logger = logging.getLogger(__name__)


class GUISummary(TypedDict):
    """Aggregated outcome of a ``process_gui`` run."""

    total: int
    succeeded: int
    failed: int
    failed_guis: list[str]
    overall_success: bool


DEFAULT_GUI_TYPES: tuple[str, ...] = ("gui_1", "gui_2")

# Files listed per pipeline step in navigation.html. ``file_count`` still
# reports the full number of discovered files; only the listing is capped.
MAX_FILES_PER_SECTION = 20

# (display name, output directory, glob patterns) for every pipeline step,
# derived from the canonical step registry (``gnn.pipeline.step_registry`` —
# the same source the website catalogue uses).

#: Display-name fixes for stems whose title-cased form differs from the
#: established navigation labels.
_SECTION_TITLE_OVERRIDES: dict[str, str] = {
    "gnn": "GNN Processing",
    "advanced_viz": "Advanced Visualization",
}

#: Per-step glob patterns, keyed by registry stem; stems without an entry
#: fall back to :data:`_DEFAULT_SECTION_PATTERNS`.
_SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "0_template": ("*.json", "*.md"),
    "1_setup": ("*.json",),
    "2_tests": ("*.txt", "*.json"),
    "3_gnn": ("*.json", "*.md", "*.pkl"),
    "4_model_registry": ("*.json",),
    "5_type_checker": ("*.json", "*.md"),
    "6_validation": ("*.json",),
    "7_export": ("*.json", "*.xml", "*.pkl"),
    "8_visualization": ("*.png", "*.svg", "*.csv", "*.json"),
    "9_advanced_viz": ("*.png", "*.svg", "*.json", "*.html", "*.csv"),
    "10_ontology": ("*.json",),
    "11_render": ("*.py", "*.jl", "*.stan", "*.md", "*.json", "*.png"),
    "12_execute": ("*.txt", "*.json", "*.md", "*.png", "*.svg", "*.html", "*.csv"),
    "13_llm": ("*.md", "*.json"),
    "14_ml_integration": ("*.json",),
    "15_audio": (
        "*.json",
        "*.wav",
        "*.aiff",
        "*.mp3",
        "*.ogg",
        "*.flac",
        "*.png",
        "*.sc",
        "*.scd",
    ),
    "16_analysis": ("*.json", "*.md", "*.html", "*.csv", "*.png", "*.svg"),
    "17_integration": ("*.json",),
    "18_security": ("*.json",),
    "19_research": ("*.json",),
    "20_website": ("*.html", "*.json"),
    "21_mcp": ("*.json",),
    "22_gui": ("*.md", "*.json", "*.mmd"),
    "23_report": ("*.html", "*.md", "*.json"),
    "24_intelligent_analysis": ("*.json", "*.md", "*.html"),
}

_DEFAULT_SECTION_PATTERNS: tuple[str, ...] = ("*.json", "*.md")

# Acronym casing for stem words that must not be title-cased (mirrors the
# website generator's registry-stem display logic; ``"llm".title()`` would
# yield "Llm", breaking the established labels).
_ACRONYM_DISPLAY: dict[str, str] = {
    "gnn": "GNN",
    "gui": "GUI",
    "llm": "LLM",
    "mcp": "MCP",
    "ml": "ML",
}


def _display_name_from_stem(stem: str) -> str:
    """Display name for a registry stem (``"9_advanced_viz"`` → ``"Advanced Viz"``)."""
    suffix = stem.partition("_")[2]
    return " ".join(
        _ACRONYM_DISPLAY.get(word, word.title()) for word in suffix.split("_")
    )


def derive_pipeline_output_sections(
    steps: Iterable[StepInfo],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Derive ``(display name, output dir, glob patterns)`` sections from registry steps.

    A step added to ``gnn.pipeline.step_registry.STEPS`` flows through here
    automatically: its output dir is the registry ``output_dir_name``, its
    display name is the title-cased stem suffix (unless overridden), and its
    patterns default to :data:`_DEFAULT_SECTION_PATTERNS`.
    """
    return tuple(
        (
            _SECTION_TITLE_OVERRIDES.get(
                step.script_stem.partition("_")[2],
                _display_name_from_stem(step.script_stem),
            ),
            step.output_dir_name,
            _SECTION_PATTERNS.get(step.script_stem, _DEFAULT_SECTION_PATTERNS),
        )
        for step in steps
    )


PIPELINE_OUTPUT_SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    derive_pipeline_output_sections(STEPS)
)


def normalize_gui_types(value: str | Sequence[str] | None) -> list[str]:
    """Normalize a ``gui_types`` option into a clean list of GUI identifiers.

    Accepts a comma-separated string, a sequence of identifiers, or ``None``
    (the pipeline default ``gui_1,gui_2``). Blank entries are dropped.
    """
    if value is None:
        return list(DEFAULT_GUI_TYPES)
    if isinstance(value, str):
        value = value.split(",")
    return [item.strip() for item in value if item.strip()]


def summarize_gui_results(
    results: Mapping[str, Mapping[str, Any]],
) -> GUISummary:
    """Aggregate per-GUI result mappings into a typed summary.

    A GUI counts as failed when its ``success`` key is missing or falsy.
    """
    failed = sorted(
        gui_type
        for gui_type, result in results.items()
        if not result.get("success", False)
    )
    total = len(results)
    return {
        "total": total,
        "succeeded": total - len(failed),
        "failed": len(failed),
        "failed_guis": failed,
        "overall_success": not failed,
    }


def process_gui(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Main processing function for GUI module.

    By default, runs all available GUI implementations in headless mode.
    Can be restricted using gui_types parameter.

    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
            - logger: Optional caller-provided logger (honored when passed)
            - gui_types: List of GUI types to run (default: gui_1, gui_2)
            - headless: Derived as ``not interactive`` — every non-interactive
              run is headless (explicit ``headless=True`` stays headless).
            - interactive: Launch interactive GUI servers and, once any is
              live, keep the process serving until they stop (CLI behavior).

    Returns:
        Boolean indicating success of all GUI runs
    """
    from .gui_1 import gui_1
    from .gui_2 import gui_2
    from .gui_3 import gui_3
    from .oxdraw import oxdraw_gui

    caller_logger = kwargs.get("logger")
    if isinstance(caller_logger, logging.Logger):
        logger = caller_logger
    else:
        logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Derive headless from interactive: interactive wins (launches servers);
    # every other combination runs headless. The pipeline step config and the
    # 22_gui CLI both default headless=False, so headless must be derived —
    # a literal ``headless=False`` default would silently force interactive.
    interactive = bool(kwargs.get("interactive", False))
    kwargs["headless"] = not interactive
    if interactive:
        logger.info("🎮 Running in INTERACTIVE mode - will launch GUI servers")
    else:
        logger.info("📦 Running in HEADLESS mode - generating artifacts only (fast)")

    # Determine which GUIs to run
    gui_types = normalize_gui_types(kwargs.get("gui_types"))

    # Prepare kwargs for GUI functions
    gui_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ["logger", "target_dir", "output_dir", "verbose"]
    }

    results: dict[str, dict[str, Any]] = {}
    overall_success = True

    try:
        logger.info(f"Processing GUI module for files in {target_dir}")
        logger.info(f"Running GUI types: {gui_types}")
        logger.info(f"Mode: {'INTERACTIVE' if not kwargs['headless'] else 'HEADLESS'}")

        # Map GUI types to functions
        gui_functions: dict[str, Any] = {
            "gui_1": gui_1,
            "gui_2": gui_2,
            "gui_3": gui_3,
            "oxdraw": oxdraw_gui,
        }

        # Run each requested GUI
        for index, gui_type in enumerate(gui_types, 1):
            try:
                logger.debug(
                    f"[{index}/{len(gui_types)}] Initializing pipeline for GUI component: {gui_type}..."
                )
                if gui_type in gui_functions:
                    logger.info(f"🎨 Generating visual assets via {gui_type} engine...")
                    result = gui_functions[gui_type](
                        target_dir=Path(target_dir),
                        output_dir=Path(output_dir),
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs,
                    )
                    if result.get("success", False):
                        logger.info(
                            f"✅ Successfully compiled {gui_type} visual DOM artifacts."
                        )
                    else:
                        logger.warning(
                            f"⚠️ Warning: {gui_type} returned non-success parsing state."
                        )
                else:
                    logger.warning(
                        f"Unknown GUI type configuration: '{gui_type}'. Skipping."
                    )
                    result = {
                        "gui_type": gui_type,
                        "success": False,
                        "error": f"Unknown GUI type: {gui_type}",
                    }

                results[gui_type] = result
                if not result.get("success", False):
                    overall_success = False

            except Exception as e:
                logger.error(
                    f"🚨 GUI render exception in {gui_type}: {e}", exc_info=verbose
                )
                results[gui_type] = {
                    "gui_type": gui_type,
                    "success": False,
                    "error": str(e),
                }
                overall_success = False

        # Save processing summary
        _save_processing_summary(
            output_dir, kwargs, gui_types, results, overall_success, logger
        )

        # Generate HTML navigation page for all outputs
        _generate_navigation_page(output_dir, logger)

        # Keep the process alive while interactive servers are serving: the
        # launcher threads are daemons, so without this gate ``--interactive``
        # would exit right after the runner returns. Gated strictly on
        # liveness, so gradio-less runs never block.
        if interactive and interactive_servers_running():
            logger.info(
                "🖥️ Interactive GUI server(s) running - waiting for them to "
                "stop (Ctrl+C to exit)"
            )
            try:
                while interactive_servers_running():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                logger.info("Interactive session interrupted; shutting down")

        return overall_success

    except Exception as e:
        logger.error(f"GUI processing failed: {e}")
        return False


def _save_processing_summary(
    output_dir: Path,
    kwargs: Mapping[str, Any],
    gui_types: list[str],
    results: Mapping[str, Mapping[str, Any]],
    overall_success: bool,
    logger: logging.Logger,
) -> None:
    """Save GUI processing summary to JSON file."""
    try:
        summary_file = Path(output_dir) / "gui_processing_summary.json"
        write_json_atomically(
            summary_file,
            {
                "mode": "interactive"
                if not kwargs.get("headless", True)
                else "headless",
                "gui_types": gui_types,
                "results": dict(results),
                "overall_success": overall_success,
            },
        )
        logger.info(f"📊 GUI processing summary saved to: {summary_file}")
    except Exception as e:
        logger.warning(f"Failed to save GUI processing summary: {e}")


def _generate_navigation_page(output_dir: Path, logger: logging.Logger) -> None:
    """Generate HTML navigation page for pipeline outputs."""
    try:
        pipeline_output_dir = Path(output_dir).parent
        nav_success = generate_html_navigation(pipeline_output_dir, output_dir, logger)
        if nav_success:
            logger.info("✅ HTML navigation page generated")
    except Exception as e:
        logger.warning(f"Failed to generate HTML navigation: {e}")


def generate_html_navigation(
    pipeline_output_dir: Path, output_dir: Path, logger: logging.Logger
) -> bool:
    """
    Generate HTML navigation page that links to all pipeline output types.

    Args:
        pipeline_output_dir: Directory containing all pipeline outputs (typically output/)
        output_dir: GUI output directory where navigation.html will be created
        logger: Logger instance

    Returns:
        True if navigation page generated successfully, False otherwise
    """
    try:
        logger.info("Generating HTML navigation page for pipeline outputs")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect output information
        output_sections, total_files = collect_pipeline_outputs(
            pipeline_output_dir, max_files_per_section=None
        )

        # Generate HTML content
        html_content = _build_navigation_html(
            output_sections, total_files, pipeline_output_dir, output_dir
        )

        # Write HTML file
        nav_file = output_dir / "navigation.html"
        write_text_atomically(nav_file, html_content)

        logger.info(f"✅ HTML navigation page generated: {nav_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate HTML navigation: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def collect_pipeline_outputs(
    pipeline_output_dir: Path,
    max_files_per_section: int | None = MAX_FILES_PER_SECTION,
) -> tuple[list[dict[str, Any]], int]:
    """Discover pipeline artifacts grouped by step output directory.

    Returns ``(sections, total_files)`` where each section is
    ``{step_name, step_dir, file_count, files}``. ``file_count`` reports every
    discovered file while ``files`` is capped at ``max_files_per_section``
    (``None`` returns all files). An active run admits only indexed artifacts
    belonging to the GUI's selected models. Standalone use is a filesystem census.
    """
    from gnn.pipeline.artifact_ownership import current_artifact_files
    from gnn.pipeline.run_context import current_run_context

    pipeline_output_dir = pipeline_output_dir.resolve()
    context = current_run_context()
    selected = (
        {model.model_id for model in context.selected_models(22)}
        if context is not None
        else set()
    )
    sections: list[dict[str, Any]] = []
    total_files = 0

    for step_name, step_dir, patterns in PIPELINE_OUTPUT_SECTIONS:
        step_path = pipeline_output_dir / step_dir
        if not step_path.exists():
            continue

        step_files: list[dict[str, Any]] = []
        for pattern in patterns:
            for file_path in current_artifact_files(step_path, pattern):
                if context is not None:
                    context.raise_if_expired()
                    owners = [
                        model
                        for model in context.models
                        if model.artifact_stem in file_path.parts
                        or file_path.stem == model.artifact_stem
                        or file_path.stem.startswith(model.artifact_stem + "_")
                    ]
                    if any(model.model_id not in selected for model in owners):
                        continue
                if file_path.is_file():
                    try:
                        rel_path = file_path.relative_to(pipeline_output_dir).as_posix()
                        file_size = file_path.stat().st_size
                        file_size_mb = file_size / (1024 * 1024)

                        step_files.append(
                            {
                                "name": file_path.name,
                                "path": rel_path,
                                "size_mb": round(file_size_mb, 3),
                                "type": file_path.suffix.lower(),
                            }
                        )
                        total_files += 1
                    except OSError as e:
                        logger.debug(f"Could not read file {file_path}: {e}")

        if step_files:
            step_files.sort(key=lambda x: (x["type"], x["name"], x["path"]))
            sections.append(
                {
                    "step_name": step_name,
                    "step_dir": step_dir,
                    "file_count": len(step_files),
                    "files": step_files[:max_files_per_section],
                }
            )

    return sections, total_files


def _build_navigation_html(
    output_sections: list[dict[str, Any]],
    total_files: int,
    pipeline_output_dir: Path,
    output_dir: Path | None = None,
) -> str:
    """Build the HTML content for the navigation page."""
    from gnn.pipeline.run_context import current_run_context

    output_dir = output_dir or pipeline_output_dir / "22_gui_output"
    context = current_run_context()
    scope_note = (
        "Indexed artifacts recorded so far for this run and the selected GUI models. "
        "File availability does not establish step completion."
        if context is not None
        else "Filesystem census of available artifacts; files may belong to earlier runs. "
        "File availability does not establish current-run completion."
    )
    file_types = sorted(
        {item["type"] for section in output_sections for item in section["files"]}
    )
    type_options = "".join(
        f'<option value="{html.escape(kind, quote=True)}">{html.escape(kind)}</option>'
        for kind in file_types
    )
    pipeline_dir_name = html.escape(pipeline_output_dir.name)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Output Navigation</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            color: #2c3e50;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid rgba(0, 123, 255, 0.2);
        }}
        h1 {{ color: #1a202c; margin: 0; font-size: 2.8em; font-weight: 800; letter-spacing: -1px; }}
        h2 {{ color: #2d3748; margin-top: 40px; margin-bottom: 20px; font-size: 1.8em; font-weight: 600; border-left: 5px solid #4299e1; padding-left: 15px; }}
        .summary {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.4);
            color: white; padding: 30px; border-radius: 16px;
            text-align: center; margin: 30px 0;
            box-shadow: 0 8px 20px rgba(118, 75, 162, 0.2);
        }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }}
        .summary-card {{
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px; padding: 20px; text-align: center;
            transition: transform 0.3s ease;
        }}
        .summary-card:hover {{ transform: translateY(-5px); background: rgba(255, 255, 255, 0.25); }}
        .summary-card .value {{ font-size: 2.2em; font-weight: 800; color: #fff; margin: 10px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .step-section {{
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 16px; padding: 25px; margin: 25px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }}
        .step-section:hover {{ background: rgba(255, 255, 255, 0.8); box-shadow: 0 8px 15px rgba(0,0,0,0.05); }}
        .step-header {{ font-weight: 700; color: #2d3748; margin-bottom: 15px; font-size: 1.3em; }}
        .file-list {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-top: 20px; }}
        .file-item {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 10px; padding: 15px;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .file-item:hover {{ transform: translateY(-3px); box-shadow: 0 6px 12px rgba(0,0,0,0.08); border-color: #cbd5e0; }}
        .file-item a {{ color: #3182ce; text-decoration: none; font-weight: 600; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .file-item a:hover {{ color: #2b6cb0; text-decoration: underline; }}
        .file-meta {{ color: #718096; font-size: 12px; margin-top: 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
        .link {{ color: #3182ce; text-decoration: none; font-weight: 600; transition: color 0.2s; }}
        .link:hover {{ color: #2b6cb0; text-decoration: underline; }}
        [hidden] {{ display: none !important; }}
        .filters {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 24px 0; }}
        .filters label {{ display: grid; gap: 6px; font-weight: 600; }}
        .filters input, .filters select {{ padding: 10px; border: 1px solid #718096; border-radius: 6px; font: inherit; max-width: 100%; box-sizing: border-box; }}
        .filters input {{ width: 360px; }}
        .file-path {{ overflow-wrap: anywhere; color: #4a5568; font-size: 12px; margin-top: 8px; }}
        .more-files {{ margin-top: 20px; }}
        summary {{ cursor: pointer; font-weight: 600; padding: 8px 0; }}
        :focus-visible {{ outline: 3px solid #1a56db; outline-offset: 3px; }}
        @media (max-width: 600px) {{
            body {{ padding: 10px; }}
            .container {{ padding: 18px; }}
            h1 {{ font-size: 2em; }}
            .file-list {{ grid-template-columns: minmax(0, 1fr); }}
            .filters label, .filters input {{ width: 100%; }}
        }}
        @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
    </style>
</head>
<body>
    <main class="container">
        <div class="header">
            <h1>🎯 GNN Pipeline Output Navigation</h1>
            <p>Comprehensive navigation to all pipeline outputs and artifacts</p>
            <p>{scope_note}</p>
        </div>

        <div class="summary">
            <h2>Pipeline Overview</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div>Steps with listed artifacts</div>
                    <div class="value">{len(output_sections)}</div>
                </div>
                <div class="summary-card">
                    <div>Total Files</div>
                    <div class="value">{total_files}</div>
                </div>
                <div class="summary-card">
                    <div>Output Directory</div>
                    <div class="value" style="font-size: 0.8em;">{pipeline_dir_name}</div>
                </div>
            </div>
        </div>

        <h2>📁 Output Sections</h2>
        <div class="filters">
            <label for="artifact-search">Search filenames, paths, or steps
                <input id="artifact-search" type="search" autocomplete="off">
            </label>
            <label for="artifact-type">File type
                <select id="artifact-type"><option value="">All types</option>{type_options}</select>
            </label>
        </div>
        <p id="artifact-count" role="status" aria-live="polite">{total_files} files available</p>
        <p id="artifact-empty" hidden>No artifacts match these filters. Clear the search or select another file type.</p>
        <noscript><p>Search requires JavaScript. Every file remains available in the sections below; expand “More files” to see longer lists.</p></noscript>
"""

    # Add each step section
    for section in output_sections:
        step_name = html.escape(str(section["step_name"]))
        step_dir = html.escape(str(section["step_dir"]))
        html_content += f"""
        <div class="step-section">
            <div class="step-header">📂 {step_name} ({step_dir})</div>
            <p><strong>Files:</strong> {section["file_count"]}</p>
            <div class="file-list">
"""
        for index, file_info in enumerate(section["files"]):
            if index == MAX_FILES_PER_SECTION:
                html_content += f"""
            </div>
            <details class="more-files"><summary>More files ({len(section["files"]) - MAX_FILES_PER_SECTION})</summary>
            <div class="file-list">
"""
            file_name = html.escape(str(file_info["name"]))
            file_path = html.escape(str(file_info["path"]))
            relative = Path(
                os.path.relpath(pipeline_output_dir / file_info["path"], output_dir)
            ).as_posix()
            file_url = html.escape(quote(relative, safe="/"), quote=True)
            file_type = html.escape(str(file_info["type"]), quote=True)
            html_content += f"""
                <div class="file-item" data-file-type="{file_type}">
                    <a href="{file_url}" target="_blank" rel="noopener">{file_name}</a>
                    <div class="file-path">{file_path}</div>
                    <div class="file-meta">{file_type} • {file_info["size_mb"]} MB</div>
                </div>
"""
        if section["file_count"] > len(section["files"]):
            html_content += f"""
                <div class="file-item" style="opacity: 0.7; font-style: italic;">
                    ... and {section["file_count"] - len(section["files"])} more files
                </div>
"""
        html_content += """
            </div>
"""
        if len(section["files"]) > MAX_FILES_PER_SECTION:
            html_content += "</details>\n"
        html_content += """
        </div>
"""

    crosslinks = []
    available_paths = {
        item["path"] for section in output_sections for item in section["files"]
    }
    for relative_path, label in (
        (
            "23_report_output/comprehensive_analysis_report.html",
            "View Comprehensive Report",
        ),
        ("20_website_output/index.html", "Open Website Dashboard"),
    ):
        if relative_path in available_paths:
            relative_url = Path(
                os.path.relpath(pipeline_output_dir / relative_path, output_dir)
            ).as_posix()
            url = html.escape(quote(relative_url, safe="/"), quote=True)
            crosslinks.append(f'<p><a href="{url}" class="link">{label}</a></p>')
    crosslink_html = "\n".join(crosslinks)
    html_content += f"""
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d;">
            <p>Generated by GNN Pipeline GUI Module</p>
            {crosslink_html}
        </div>
    </main>
    <script>
    (() => {{
        const search = document.getElementById('artifact-search');
        const type = document.getElementById('artifact-type');
        const count = document.getElementById('artifact-count');
        const empty = document.getElementById('artifact-empty');
        const sections = [...document.querySelectorAll('.step-section')];
        function filterArtifacts() {{
            const query = search.value.trim().toLowerCase();
            let matched = 0;
            for (const section of sections) {{
                const heading = section.querySelector('.step-header').textContent.toLowerCase();
                let sectionMatches = 0;
                for (const file of section.querySelectorAll('[data-file-type]')) {{
                    const text = heading + ' ' + file.textContent.toLowerCase();
                    const visible = (!query || text.includes(query)) && (!type.value || file.dataset.fileType === type.value);
                    file.hidden = !visible;
                    if (visible) sectionMatches++;
                }}
                section.hidden = sectionMatches === 0;
                const more = section.querySelector('.more-files');
                if (more) {{
                    const overflowMatches = [...more.querySelectorAll('[data-file-type]')].some(file => !file.hidden);
                    more.hidden = !overflowMatches;
                    more.open = !!(query || type.value) && overflowMatches;
                }}
                matched += sectionMatches;
            }}
            count.textContent = matched + ' of {total_files} files match';
            empty.hidden = matched !== 0;
        }}
        search.addEventListener('input', filterArtifacts);
        type.addEventListener('change', filterArtifacts);
    }})();
    </script>
</body>
</html>
"""

    return html_content
