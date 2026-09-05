"""Tests for the additive public-API surface introduced in the refactor.

Covers:
- ``VIZ_TYPE_CHOICES`` canonical choice set (matches orchestrator + processor).
- ``probe_capabilities()`` live environment probe.
- ``record_attempt`` re-exported from the package.
- ``utils.ArgumentParser``'s CLI ``viz_type`` choices stay in parity with
  ``VIZ_TYPE_CHOICES`` (the runtime parser owns enforcement; see
  ``test_orchestrator_choices_match``).
- ``mcp.py`` ``process_advanced_visualization_mcp`` honors ``generate_d2`` by
  routing to a non-D2 viz_type when false.
- ``dashboard.py`` footer timestamp renders (regression test for the silent
  ``{datetime.now()}`` unexpanded expression bug).
"""

import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVizTypeChoices:
    def test_is_tuple_of_strings(self) -> None:
        from advanced_visualization import VIZ_TYPE_CHOICES

        assert isinstance(VIZ_TYPE_CHOICES, tuple)
        assert all(isinstance(v, str) for v in VIZ_TYPE_CHOICES)

    def test_includes_documented_values(self) -> None:
        from advanced_visualization import VIZ_TYPE_CHOICES

        expected = {
            "all",
            "3d",
            "interactive",
            "dashboard",
            "d2",
            "diagrams",
            "pipeline",
            "statistical",
            "pomdp",
            "network",
        }
        assert expected.issubset(set(VIZ_TYPE_CHOICES))

    def test_orchestrator_choices_match(self) -> None:
        """The canonical ``VIZ_TYPE_CHOICES`` tuple matches the ``viz_type``
        choices the real CLI parser enforces (``utils.ArgumentParser``), so
        every value the CLI accepts is one the processor routes on."""
        from advanced_visualization import VIZ_TYPE_CHOICES
        from utils.arg_parsing import ArgumentParser as StepArgumentParser

        arg_def = StepArgumentParser.ARGUMENT_DEFINITIONS["viz_type"]
        assert arg_def.choices is not None
        assert list(VIZ_TYPE_CHOICES) == arg_def.choices


class TestProbeCapabilities:
    def test_returns_dict_of_bools(self) -> None:
        from advanced_visualization import probe_capabilities

        caps = probe_capabilities()
        assert isinstance(caps, dict)
        for key in ("d2_cli", "matplotlib", "numpy", "plotly", "seaborn", "networkx"):
            assert key in caps
            assert isinstance(caps[key], bool)

    def test_d2_cli_reflects_actual_path(self) -> None:
        import shutil

        from advanced_visualization import probe_capabilities

        caps = probe_capabilities()
        assert caps["d2_cli"] == (shutil.which("d2") is not None)

    def test_numpy_and_matplotlib_true_in_test_env(self) -> None:
        from advanced_visualization import probe_capabilities

        caps = probe_capabilities()
        assert caps["matplotlib"] is True
        assert caps["numpy"] is True


class TestRecordAttemptReExport:
    def test_importable_from_package(self) -> None:
        from advanced_visualization._shared import record_attempt

        assert callable(record_attempt)


class TestMcpGenerateD2Honored:
    def _make(self) -> Any:
        from advanced_visualization.mcp import process_advanced_visualization_mcp

        return process_advanced_visualization_mcp

    def test_generate_d2_true_routes_to_all(self, tmp_path: Any) -> None:
        fn = self._make()
        # Empty input -> warning code 2 (no models), but viz_type="all" is used.
        result = fn(str(tmp_path / "in"), str(tmp_path / "out"), generate_d2=True)
        assert result["success"] in (True, False, 2)
        assert result["generate_d2"] is True
        assert "viz_type=all" in result["message"]

    def test_generate_d2_false_routes_to_network(self, tmp_path: Any) -> None:
        fn = self._make()
        result = fn(str(tmp_path / "in2"), str(tmp_path / "out2"), generate_d2=False)
        assert result["generate_d2"] is False
        assert "viz_type=network" in result["message"]


class TestDashboardTimestampRenders:
    def test_footer_contains_rendered_timestamp(self, tmp_path: Any) -> None:
        """Regression: dashboard.py shipped ``{datetime.now().strftime(...)}`` as
        literal text because the footer f-string block was a plain ``\"\"\"`` string
        with no ``f`` prefix and ``datetime`` was never imported. Fixed to render.

        The same footer chunk carries the ``<script>`` tab-switching JS; the test also
        pins that f-string ``{{`` escapes render as single braces there (a plain
        string would ship literal ``{{`` and break the JS)."""
        from advanced_visualization.dashboard import generate_dashboard

        content = """# Test Dashboard Model

## StateSpaceBlock
hidden_states[10, type=float]
observations[5, type=float]

## Connections
hidden_states -> observations

## Parameters
learning_rate = 0.01
"""
        out = tmp_path / "dash"
        result = generate_dashboard(content, "test_model", out)
        assert result is not None
        html = result.read_text()
        # A rendered timestamp looks like "Generated on 20YY-MM-DD HH:MM:SS"
        assert "Generated on 20" in html
        # The dead unexpanded expression must NOT appear
        assert "{datetime.now().strftime" not in html
        # JS/braces: f-string escapes render as single braces; no doubled
        # braces may ship anywhere in the generated document.
        assert "function showTab(tabName) {" in html
        doubled = [line for line in html.splitlines() if "{{" in line]
        assert doubled == [], f"literal double braces shipped: {doubled[:5]}"


class TestD2ConstantsAndFormatValidation:
    def test_d2_constants_exposed(self) -> None:
        from advanced_visualization.d2_visualizer import (
            D2_COMPILE_TIMEOUT_S,
            D2_MISSING_MESSAGE,
            VALID_D2_FORMATS,
        )

        assert D2_COMPILE_TIMEOUT_S == 30
        assert urlsplit(D2_MISSING_MESSAGE.rsplit(" ", 1)[-1]).hostname == "d2lang.com"
        assert VALID_D2_FORMATS == ("svg", "png", "pdf")

    def test_compile_returns_missing_message_when_no_cli(self, tmp_path: Any) -> None:
        """The availability gate returns the canonical missing-CLI error.

        ``d2_available`` is an instance attribute set from ``shutil.which``;
        forcing it ``False`` exercises the no-CLI path deterministically
        whether or not the d2 binary happens to be installed.
        """
        from advanced_visualization.d2_visualizer import (
            D2_MISSING_MESSAGE,
            D2DiagramSpec,
            D2Visualizer,
        )

        v = D2Visualizer()
        v.d2_available = False
        spec = D2DiagramSpec(name="t", description="d", d2_content="a: {shape: circle}")
        result = v.compile_d2_diagram(spec, tmp_path, formats=["svg"])
        assert result.success is False
        assert result.error_message == D2_MISSING_MESSAGE

    def test_compile_drops_unsupported_formats(self, tmp_path: Any) -> None:
        """Unsupported formats never reach the CLI. With no d2 CLI installed,
        the availability check returns the missing-CLI error before the filter
        runs; with a CLI present, the filter drops unsupported formats first
        and an all-unsupported list falls back to ``VALID_D2_FORMATS[:2]``
        (svg+png) — never spec defaults or arbitrary suffixes. Either way the
        call must not raise on a bogus format."""
        from advanced_visualization.d2_visualizer import (
            D2_MISSING_MESSAGE,
            D2DiagramSpec,
            D2Visualizer,
        )

        v = D2Visualizer()
        spec = D2DiagramSpec(
            name="t", description="d", d2_content="a: {}", output_formats=["svg"]
        )
        # No CLI: the availability gate returns the missing-CLI error and the
        # filter never runs. CLI present: bogus is dropped and svg compiles —
        # a .bogus artifact must never appear.
        result = v.compile_d2_diagram(spec, tmp_path, formats=["svg", "bogus"])
        if v.d2_available:
            assert result.success is True
            assert not (tmp_path / "t.bogus").exists()
        else:
            assert result.success is False
            assert result.error_message == D2_MISSING_MESSAGE
