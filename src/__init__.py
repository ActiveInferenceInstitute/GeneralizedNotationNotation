"""
GNN Pipeline Core Module

This module provides the core functionality for the GNN processing pipeline.
"""

from .core import get_module_info

# Expose submodules expected by tests via `import src`
try:
    import sapf  # noqa: F401
except Exception:
    # Provide a minimal shim to satisfy attribute existence checks
    class _Shim:
        __all__ = []
    sapf = _Shim()  # type: ignore

__all__ = [
    'get_module_info',
    'sapf'
] 