"""Shared runner plumbing for the Step 22 GUI processors.

Extracted from ``gui_1``/``gui_2``/``gui_3`` processors so each GUI keeps only
its own domain logic: output-root normalization, starter-markdown discovery,
and the background Gradio launch pattern are identical across GUIs.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_ORCHESTRATOR_SCRIPT = "22_gui.py"


def resolve_output_root(output_dir: Path) -> Path:
    """Normalize ``output_dir`` to the pipeline-standard step output root.

    Uses ``pipeline.config.get_output_dir_for_script`` when the pipeline
    package is importable and falls back to the caller-supplied directory
    otherwise (e.g. when the GUI runners are used outside the pipeline).
    """
    try:
        from gnn.pipeline.config import get_output_dir_for_script

        return Path(get_output_dir_for_script(_ORCHESTRATOR_SCRIPT, output_dir))
    except Exception:  # pragma: no cover - depends on pipeline availability
        return Path(output_dir)


def load_first_markdown(
    target_dir: Path, prefer_patterns: Sequence[str] = ()
) -> str | None:
    """Return the content of the first markdown file found in ``target_dir``.

    Patterns in ``prefer_patterns`` are consulted first (in order), then any
    top-level ``*.md`` file, then a recursive ``**/*.md`` walk. Returns
    ``None`` when nothing readable exists.
    """
    try:
        for pattern in prefer_patterns:
            matches = sorted(target_dir.glob(pattern))
            if matches:
                return matches[0].read_text(encoding="utf-8")
        for path in sorted(target_dir.glob("*.md")):
            return path.read_text(encoding="utf-8")
        for path in sorted(target_dir.glob("**/*.md")):
            return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def launch_gradio_in_thread(
    demo: Any, *, port: int, open_browser: bool
) -> threading.Thread:
    """Launch a Gradio Blocks app on a non-daemon background thread.

    Non-daemon so multiple GUIs can launch servers concurrently within one
    pipeline process; callers decide how long to keep the process alive.
    """

    def _launch() -> None:
        demo.launch(
            share=False,
            prevent_thread_lock=False,  # Let the thread block on the server
            server_name="0.0.0.0",  # nosec B104
            server_port=port,
            inbrowser=open_browser,
            show_error=True,
            quiet=False,
        )

    thread = threading.Thread(target=_launch, daemon=False)
    thread.start()
    return thread


__all__ = ["launch_gradio_in_thread", "load_first_markdown", "resolve_output_root"]
