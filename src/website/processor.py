"""
Website processor facade for GNN Processing Pipeline.

Delegates to renderer.py which contains the actual implementation.
This facade exists for architectural consistency: the documented pattern
expects a processor.py in every module directory.
"""

from typing import Any

from website.renderer import process_website

__all__: list[Any] = ["process_website"]
