"""MCP dispatch concern package (S2-33 Step 6): canonical home of the MCP
server leaf (``mcp/server.py``) and the generic pipeline-step dispatcher
(``mcp/dispatch.py``).

Import-weight invariant (I1/R2): this ``__init__`` executes nothing heavy at
import time — the server leaf's module-scope ``import psutil`` is paid only
when ``register_tools`` actually runs (or an earlier-name server attribute is
resolved through the PEP 562 ``__getattr__`` below), and ``import gnn.utils``
never touches this package. ``import gnn.utils.mcp.dispatch`` therefore stays
as light as the old ``import gnn.utils.mcp_dispatch`` was.

The MCP auto-discovery (``gnn/mcp/mcp.py``) resolves ``gnn.utils.mcp`` to
this package and needs a callable ``register_tools``: the def below is the
live entry point and lazily delegates to the server leaf. The shadowed file
``src/gnn/utils/mcp.py`` carries the same def for the repo-wide static audit
(``tests/mcp/test_mcp_audit.py`` / ``test_mcp_module_presence.py`` scan for a
file named ``mcp.py``); Python's import system resolves ``gnn.utils.mcp`` to
this package, never to that file.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SERVER_EXPORTS = (
    "SENSITIVE_ENV_KEY_MARKERS",
    "get_environment_info",
    "get_file_info",
    "get_logging_info",
    "get_system_info",
    "is_sensitive_env_key",
    "redact_environment",
    "register_tools",
    "validate_dependencies",
)

__all__ = list(_SERVER_EXPORTS)


def register_tools(server: object) -> None:
    """Register the utils MCP tools (delegates to ``gnn.utils.mcp.server``).

    The server import — and its module-scope psutil — is deferred to call
    time, mirroring the old monolithic ``gnn/utils/mcp.py`` import cost
    (I1/R2: registry discovery paths unchanged).
    """
    from gnn.utils.mcp.server import register_tools as _register_tools

    logger.info("utils MCP entry point delegating to gnn.utils.mcp.server")
    _register_tools(server)


def __getattr__(name: str) -> Any:
    """Resolve the remaining earlier-name server attributes (PEP 562).

    Stragglers reading ``gnn.utils.mcp.get_environment_info``-style names get
    the historical surface with a migration warning; the ImportError
    propagates unchanged (no silent fallback — repo rule).
    """
    if name not in _SERVER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import warnings
    from importlib import import_module

    warnings.warn(
        "importing server names through gnn.utils.mcp is the earlier shape; "
        "import gnn.utils.mcp.server instead",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(import_module("gnn.utils.mcp.server"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SERVER_EXPORTS))
