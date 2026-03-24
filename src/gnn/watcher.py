#!/usr/bin/env python3
"""
GNN File Watcher — Live re-validation on file change.

Monitors a directory for GNN file changes and auto-validates.
Uses watchdog if available, otherwise falls back to polling.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class GNNWatcher:
    """
    Watches a directory for GNN file changes and runs validation.

    Args:
        watch_dir: Directory to monitor.
        on_change: Callback invoked with (file_path, content) on change.
        debounce_ms: Minimum interval between callbacks for same file.
        extensions: File extensions to watch.
    """

    def __init__(
        self,
        watch_dir: Path,
        on_change: Optional[Callable] = None,
        debounce_ms: int = 250,
        extensions: tuple = (".md",),
    ):
        self.watch_dir = Path(watch_dir)
        self.on_change = on_change or self._default_callback
        self.debounce_ms = debounce_ms
        self.extensions = extensions
        self._last_fire: dict = {}
        self._running = False

    def start(self):
        """Start watching (blocking)."""
        # Try watchdog first
        try:
            self._start_watchdog()
        except ImportError:
            logger.info("watchdog not installed — falling back to polling")
            self._start_polling()

    def stop(self):
        """Signal watcher to stop."""
        self._running = False

    def _start_watchdog(self):
        """Use watchdog for efficient filesystem monitoring."""
        from watchdog.events import FileModifiedEvent, FileSystemEventHandler
        from watchdog.observers import Observer

        class _GNNChangeHandler(FileSystemEventHandler):
            def __init__(self, gnn_watcher):
                super().__init__()
                self._watcher = gnn_watcher

            def on_modified(self, event):
                if isinstance(event, FileModifiedEvent):
                    path = Path(event.src_path)
                    if path.suffix in self._watcher.extensions:
                        self._watcher._debounced_fire(path)

        observer = Observer()
        observer.schedule(_GNNChangeHandler(self), str(self.watch_dir), recursive=True)
        observer.start()
        self._running = True

        logger.info(f"👁️ Watching {self.watch_dir} (watchdog)")
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.debug("Watcher stopped by user (KeyboardInterrupt)")
        finally:
            observer.stop()
            observer.join()

    def _start_polling(self, interval: float = 1.0):
        """Recovery polling-based watcher."""
        self._running = True
        snapshots: dict = {}

        logger.info(f"👁️ Watching {self.watch_dir} (polling, {interval}s)")
        try:
            while self._running:
                for f in self.watch_dir.rglob("*"):
                    if f.suffix in self.extensions and f.is_file():
                        mtime = f.stat().st_mtime
                        if f not in snapshots or snapshots[f] != mtime:
                            snapshots[f] = mtime
                            self._debounced_fire(f)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.debug("Polling watcher stopped by user (KeyboardInterrupt)")

    def _debounced_fire(self, path: Path):
        """Fire callback with debouncing."""
        now = time.time() * 1000
        last = self._last_fire.get(str(path), 0)

        if now - last < self.debounce_ms:
            return

        self._last_fire[str(path)] = now

        try:
            content = path.read_text(encoding="utf-8")
            self.on_change(path, content)
        except OSError as e:
            logger.warning(f"Could not read {path}: {e}")

    @staticmethod
    def _default_callback(path: Path, content: str):
        """Default callback: run GNN validation."""
        import sys
        src_dir = str(Path(__file__).parent.parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        try:
            from gnn.schema import parse_state_space, validate_required_sections

            errors = validate_required_sections(content, file_path=str(path))
            variables, var_errors = parse_state_space(content, file_path=str(path))
            errors.extend(var_errors)

            if errors:
                print(f"\n⚠️ {path.name}: {len(errors)} issue(s)")
                for e in errors[:5]:
                    print(f"  {e}")
            else:
                print(f"\n✅ {path.name}: valid ({len(variables)} variables)")
        except Exception as e:
            print(f"\n❌ {path.name}: validation error: {e}")
