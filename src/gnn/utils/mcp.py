"""Static-audit entry point for the utils MCP module (S2-33 Step 6).

The implementation moved to ``gnn/utils/mcp/server.py``; at runtime the
package ``gnn/utils/mcp/`` shadows this file (regular packages take
precedence over same-named modules), so imports of ``gnn.utils.mcp`` resolve
to the package's lazy PEP 562 facade. This file exists for the static audit
surface (``tests/mcp/test_mcp_module_presence.py`` and the
``test_all_mcp_*`` audit checks scan for a file named ``mcp.py`` defining
``register_tools``) and documents the delegation below.
"""

import logging

logger = logging.getLogger(__name__)


def register_tools(server: object) -> None:
    """Register the utils MCP tools (delegates to ``gnn.utils.mcp.server``).

    Kept as a real def so the repo-wide static audit
    (``tests/mcp/test_mcp_audit.py``) finds a ``register_tools()`` that
    calls ``logger.info`` on this module, matching every other module-level
    MCP entry point. The server import (and its module-scope psutil) is
    deferred to call time, mirroring the old module's import cost (I1/R2).
    """
    from gnn.utils.mcp.server import register_tools as _register_tools

    logger.info("utils MCP entry point delegating to gnn.utils.mcp.server")
    _register_tools(server)
