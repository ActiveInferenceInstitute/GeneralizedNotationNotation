#!/usr/bin/env python3
"""
Execution type definitions for GNN Step 12.

Shared types, literals, and constants for the execute module. This module is
a leaf: it must not import from ``execute.processor`` or any other execute
sibling, to keep the facade import graph acyclic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# File suffixes Step 12 can execute; companion artifacts (``.stan``, ``.toml``,
# ``.json``) listed in a render summary are not scripts.
_EXECUTABLE_SUFFIXES = frozenset({".py", ".jl"})

ExecutionFrameworkName = Literal[
    "pymdp",
    "rxinfer",
    "jax",
    "discopy",
    "activeinference_jl",
    "pytorch",
    "numpyro",
    "stan",
]


@dataclass(frozen=True)
class ScriptExecutionContext:
    """Normalized execution metadata for one rendered script."""

    script_path: Path
    script_name: str
    framework: str
    model_name: str
    executor: str
