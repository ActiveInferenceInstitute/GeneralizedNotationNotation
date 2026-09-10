"""Documentation subtree under src/gnn/doc/.

This module holds static Markdown documentation and its MCP discovery marker.
It has no processor.py because documentation is served as files, not code.
"""

from gnn import __version__

__all__: list[str] = ["__version__"]
