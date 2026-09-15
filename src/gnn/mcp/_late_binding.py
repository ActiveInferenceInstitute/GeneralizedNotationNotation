"""Late-binding references into the ``gnn.mcp.mcp`` namespace.

The MCP responsibility split (MAJ-04 sibling-mixin extraction) moved method
bodies into sibling modules. Those bodies read module globals such as
``time``; tests swap ``gnn.mcp.mcp.time`` for a fake clock (see
``tests/mcp/test_registry_internals.py``), so the binding must be resolved
through the host module's namespace at call time — a plain ``import time``
in a sibling module would freeze the stdlib module and silently miss the
swap.
"""

from __future__ import annotations

import sys
from typing import Any

_HOST_MODULE = __package__ + ".mcp"


class _MCPModuleRef:
    """Forward attribute access to a name currently bound in ``gnn.mcp.mcp``.

    Two-level resolution: first fetch the host module's *current* binding of
    the referenced name (so patched seams like the fake test clock are
    honored), then fetch the requested attribute from that object. With no
    patch this resolves to the same object a plain import would have bound.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        """Resolve ``attr`` on the host's current ``self._name`` binding."""
        host = sys.modules.get(_HOST_MODULE)
        if host is None:
            raise AttributeError(
                f"{_HOST_MODULE} is not initialized; cannot resolve {self._name}.{attr}"
            )
        return getattr(getattr(host, self._name), attr)
