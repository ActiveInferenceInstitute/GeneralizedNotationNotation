"""Lean execution backend: fep_lean bridge document verification (v0.5)."""

from .lean_runner import (
    FEP_LEAN_ROOT_ENV,
    lean_toolchain_available,
    resolve_fep_lean_root,
    run_lean_scripts,
    verify_document,
)

__all__ = [
    "FEP_LEAN_ROOT_ENV",
    "lean_toolchain_available",
    "resolve_fep_lean_root",
    "run_lean_scripts",
    "verify_document",
]
