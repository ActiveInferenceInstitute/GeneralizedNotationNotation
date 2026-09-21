"""Shared runner plumbing for the Step 22 GUI processors.

Extracted from ``gui_1``/``gui_2``/``gui_3`` processors so each GUI keeps only
its own domain logic: output-root normalization, starter-markdown discovery,
and the background Gradio launch pattern are identical across GUIs.

Server-thread contract: ``launch_gradio_in_thread`` returns a *daemon* thread,
so a launched Gradio server can never block process/step exit. Callers decide
how long to keep the process alive — the CLI interactive path keeps it alive
via ``process_gui`` polling ``interactive_servers_running()``.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_LAUNCHED_SERVER_THREADS: list[threading.Thread] = []
_LAUNCHED_SERVER_THREADS_LOCK = threading.Lock()


def launch_gradio_in_thread(
    demo: Any, *, port: int, open_browser: bool, server_name: str = "127.0.0.1"
) -> threading.Thread:
    """Launch a Gradio Blocks app on a daemon background thread.

    Daemon so a launched server can never block process/step exit; interactive
    callers keep the process alive themselves (``process_gui`` does this for
    the CLI by polling ``interactive_servers_running()``). The thread is
    registered so the keep-alive gate can observe liveness.
    """

    def _launch() -> None:
        demo.launch(
            share=False,
            prevent_thread_lock=False,  # Let the thread block on the server
            server_name=server_name,
            server_port=port,
            inbrowser=open_browser,
            show_error=True,
            quiet=False,
        )

    thread = threading.Thread(target=_launch, daemon=True)
    with _LAUNCHED_SERVER_THREADS_LOCK:
        _LAUNCHED_SERVER_THREADS.append(thread)
    thread.start()
    return thread


def registered_server_threads() -> tuple[threading.Thread, ...]:
    """Return a snapshot of the threads registered by ``launch_gradio_in_thread``."""
    with _LAUNCHED_SERVER_THREADS_LOCK:
        return tuple(_LAUNCHED_SERVER_THREADS)


def interactive_servers_running() -> bool:
    """Return True while any launched Gradio server thread is still alive."""
    with _LAUNCHED_SERVER_THREADS_LOCK:
        return any(thread.is_alive() for thread in _LAUNCHED_SERVER_THREADS)


def clear_launched_server_threads() -> None:
    """Forget all registered server threads (test-isolation helper)."""
    with _LAUNCHED_SERVER_THREADS_LOCK:
        _LAUNCHED_SERVER_THREADS.clear()


__all__ = [
    "clear_launched_server_threads",
    "interactive_servers_running",
    "launch_gradio_in_thread",
    "load_first_markdown",
    "registered_server_threads",
    "resolve_output_root",
]


def resolve_output_root(output_dir: Path) -> Path:
    """Normalize ``output_dir`` to the pipeline-standard step output root.

    Thin delegate to ``pipeline.config.resolve_step_output_dir``; the shared
    fallback policy (standalone recovery = caller-supplied directory, fail
    loud on an unimportable package) is documented there. This wrapper keeps
    no private ``except ImportError`` fallback of its own.
    """
    from gnn.pipeline.config import resolve_step_output_dir

    return Path(resolve_step_output_dir("22_gui", output_dir))


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
