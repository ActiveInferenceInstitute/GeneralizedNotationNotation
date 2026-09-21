"""MCP dispatch concern package (S2-33 Step 6): canonical home of the MCP
server leaf (``mcp/server.py``) and the generic pipeline-step dispatcher
(``mcp/dispatch.py``).

Import-weight invariant (I1/R2): this ``__init__`` executes nothing heavy at
import time — the server leaf's module-scope ``import psutil`` is paid only
when ``register_tools`` actually runs, and ``import gnn.utils`` never touches
this package. ``import gnn.utils.mcp.dispatch`` stays import-light.

The MCP auto-discovery (``gnn/mcp/mcp.py``) resolves ``gnn.utils.mcp`` to
this package and needs a callable ``register_tools``: the def below is the
live entry point and lazily delegates to the server leaf. The shadowed file
``src/gnn/utils/mcp.py`` carries the same def for the repo-wide static audit
(``tests/mcp/test_mcp_audit.py`` / ``test_mcp_module_presence.py`` scan for a
file named ``mcp.py``); Python's import system resolves ``gnn.utils.mcp`` to
this package, never to that file.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ["register_tools"]


def register_tools(server: object) -> None:
    """Register the utils MCP tools (delegates to ``gnn.utils.mcp.server``).

    The server import — and its module-scope psutil — is deferred to call
    time, mirroring the old monolithic ``gnn/utils/mcp.py`` import cost
    (I1/R2: registry discovery paths unchanged).
    """
    from gnn.utils.mcp.server import register_tools as _register_tools

    logger.info("utils MCP entry point delegating to gnn.utils.mcp.server")
    _register_tools(server)


def __dir__() -> list[str]:
    return sorted(globals())
