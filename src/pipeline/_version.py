#!/usr/bin/env python3
"""Single source of truth for the ``pipeline`` package version.

Kept dependency-free so both ``pipeline/__init__.py`` and leaf modules (e.g.
``pipeline/execution.get_pipeline_info``) can import it without cycles or
version drift.
"""

__version__ = "1.6.0"

__all__ = ["__version__"]
