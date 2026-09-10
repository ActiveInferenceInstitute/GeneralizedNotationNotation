#!/usr/bin/env python3
"""Single source of truth for the ``pipeline`` package version.

Kept so both ``pipeline/__init__.py`` and leaf modules (e.g.
``pipeline/execution.get_pipeline_info``) can import the version without
cycles; the value itself now comes from the canonical ``gnn.__version__``.
"""

from gnn import __version__

__all__ = ["__version__"]
