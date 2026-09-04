"""Shared, typed module loader for tests that exercise standalone scripts.

Several root-level test files load ``scripts/*.py`` (and ``doc/development/*.py``)
helpers by file path because those tools are standalone CLIs, not importable
packages. This helper centralizes the ``importlib`` boilerplate, including the
optional sibling-directory ``sys.path`` injection some scripts need
(they import siblings via bare module names, which plain ``importlib`` does
not resolve).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_path(
    module_name: str,
    file_path: str | Path,
    *,
    sys_path: str | Path | None = None,
) -> ModuleType:
    """Load a Python file as a fresh module under ``module_name``.

    Args:
        module_name: Name to register the module under (also the key used in
            ``sys.modules``; a repeat call replaces the previous entry).
        file_path: Path to the ``.py`` file to execute.
        sys_path: Optional directory to prepend to ``sys.path`` while the
            module executes, for scripts that import sibling modules by bare
            name. Removed afterwards unless it was already present.

    Returns:
        The executed module.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such script: {path}")

    added_path = False
    if sys_path is not None:
        sys_path_str = str(Path(sys_path))
        if sys_path_str not in sys.path:
            sys.path.insert(0, sys_path_str)
            added_path = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if added_path:
            try:
                sys.path.remove(str(Path(sys_path)))  # type: ignore[arg-type]
            except ValueError:
                pass


__all__ = ["load_module_from_path"]
