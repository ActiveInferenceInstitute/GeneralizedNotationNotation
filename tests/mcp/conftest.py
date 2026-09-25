"""Order-isolation for the global MCP singleton (t-0069).

Some tests in this directory call ``gnn.mcp.mcp.initialize()`` on the
module-global singleton with a ``modules_allowlist`` (e.g. the result-cache
TTL-knob test, the backend-module tests). That call latches the shared
singleton's ``_modules_discovered`` flag while registering only the
allowlisted subset (~28 of the census tools). Any later test that
reaches the global registry without ``force_refresh=True`` then sees the
degraded 28-tool registry — an order-dependent failure class first caught
by the ARCH-3 census work (``test_meta_tools_registration`` fails after the
TTL-knob test).

This fixture snapshots the global singleton before each test and restores
it after, so singleton mutations cannot leak across test boundaries. It is
test-side only: production registration semantics are unchanged, and the
census pins in ``tests/helpers/mcp_census.py`` still assert the exact
tool audit (ARCH-3 untouched).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

import gnn.mcp.mcp as mcp_module

# Instance attributes holding registration state that discovery and
# registration mutate in place; shallow-copy these so a restore undoes
# registrations even when the singleton object itself is reused.
_REGISTRATION_DICTS = ("tools", "resources", "modules")


def _snapshot(instance: Any) -> Dict[str, Any]:
    """Return a restorable shallow snapshot of the singleton's state."""
    state: Dict[str, Any] = {}
    for key, value in instance.__dict__.items():
        if key in _REGISTRATION_DICTS and isinstance(value, dict):
            state[key] = dict(value)
        else:
            state[key] = value
    return state


@pytest.fixture(autouse=True)
def _isolate_global_mcp_singleton() -> Any:
    """Snapshot the global MCP singleton per test; restore it afterwards."""
    saved_ref: Optional[Any] = mcp_module._mcp_instance
    saved_state: Optional[Dict[str, Any]] = (
        _snapshot(saved_ref) if saved_ref is not None else None
    )
    yield
    mcp_module._mcp_instance = saved_ref
    if saved_ref is not None and saved_state is not None:
        saved_ref.__dict__.clear()
        saved_ref.__dict__.update(saved_state)
