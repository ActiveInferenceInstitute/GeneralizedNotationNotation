"""
Website processor shim for GNN Processing Pipeline.

Delegates to renderer.py which contains the actual implementation.
This shim exists for architectural consistency: the documented pattern
expects a processor.py in every module directory.
"""

from website.renderer import process_website

__all__ = ["process_website"]
