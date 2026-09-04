"""Tests for visualization.backends.backend_status and the theme SSOT invariant."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.backends import backend_status
from visualization.graph.network_visualizations import _determine_connection_type
from visualization.theme import EDGE_STYLES


class TestBackendStatus:
    def test_returns_all_backend_keys_as_bools(self) -> None:
        status = backend_status()
        assert set(status) == {"matplotlib", "numpy", "seaborn", "networkx", "plotly"}
        assert all(isinstance(v, bool) for v in status.values())

    def test_core_backends_available_in_dev_env(self) -> None:
        # matplotlib/numpy/networkx are required core deps per pyproject.
        status = backend_status()
        assert status["matplotlib"] is True
        assert status["numpy"] is True
        assert status["networkx"] is True

    def test_module_info_reports_backends(self) -> None:
        import visualization

        info = visualization.get_module_info()
        assert info["backends"] == backend_status()


class TestThemeEdgeStyleSSOT:
    """graph must not carry a private copy of the theme palette (regression)."""

    def test_every_determined_connection_type_is_styled(self) -> None:
        # _determine_connection_type can emit these named types; each must
        # resolve to a real theme style (previously three fell back to gray).
        determined = {
            _determine_connection_type("s", "o", "hidden_state", "observation"),
            _determine_connection_type("s", "B", "hidden_state", "transition_matrix"),
            _determine_connection_type("s", "A", "hidden_state", "likelihood_matrix"),
            _determine_connection_type("u", "s", "action", "hidden_state"),
            _determine_connection_type("π", "u", "policy", "action"),
            _determine_connection_type("C", "G"),
            _determine_connection_type("E", "π"),
            _determine_connection_type("π", "u"),
        }
        assert determined
        for conn_type in determined:
            style = EDGE_STYLES.get(conn_type)
            assert style is not None, f"unstyled connection type: {conn_type}"
            assert {"color", "width", "alpha", "style"} <= set(style)
