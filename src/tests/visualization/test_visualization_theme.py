"""Tests for visualization/theme.py public helpers.

Covers the colour-palette and edge-style lookup helpers plus the module-level
constants that act as the single source of truth for Steps 8/9 rendering.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVarTypeColor:
    """Tests for get_var_type_color."""

    def _fn(self) -> Any:
        from visualization.theme import get_var_type_color

        return get_var_type_color

    def test_known_type_returns_2d_palette_color(self) -> None:
        from visualization.theme import VAR_TYPE_COLORS

        assert self._fn()("hidden_state") == VAR_TYPE_COLORS["hidden_state"]
        assert self._fn()("observation") == VAR_TYPE_COLORS["observation"]

    def test_unknown_type_falls_back_to_unknown(self) -> None:
        from visualization.theme import VAR_TYPE_COLORS

        assert self._fn()("not_a_real_var_type") == VAR_TYPE_COLORS["unknown"]

    def test_3d_palette_used_when_flagged(self) -> None:
        from visualization.theme import VAR_TYPE_COLORS_3D

        assert self._fn()("action", palette_3d=True) == VAR_TYPE_COLORS_3D["action"]

    def test_3d_unknown_type_falls_back(self) -> None:
        from visualization.theme import VAR_TYPE_COLORS_3D

        assert self._fn()("bogus", palette_3d=True) == VAR_TYPE_COLORS_3D["unknown"]


class TestEdgeStyle:
    """Tests for get_edge_style."""

    def _fn(self) -> Any:
        from visualization.theme import get_edge_style

        return get_edge_style

    def test_known_connection_type(self) -> None:
        from visualization.theme import EDGE_STYLES

        assert self._fn()("state_transition") == EDGE_STYLES["state_transition"]

    def test_unknown_connection_type_falls_back(self) -> None:
        from visualization.theme import EDGE_STYLES

        assert self._fn()("unknown_edge") == EDGE_STYLES["generic_causal"]

    def test_known_style_has_color_and_width(self) -> None:
        style = self._fn()("observation_generation")
        assert "color" in style
        assert "width" in style


class TestThemeConstants:
    """Test the module-level constant surfaces."""

    def test_palettes_are_populated(self) -> None:
        from visualization.theme import (
            GENERATIVE_MODEL_COLORS,
            VAR_TYPE_COLORS,
            VAR_TYPE_COLORS_3D,
        )

        assert len(VAR_TYPE_COLORS) >= 8
        assert len(VAR_TYPE_COLORS_3D) >= 8
        assert "D" in GENERATIVE_MODEL_COLORS
        assert "A" in GENERATIVE_MODEL_COLORS

    def test_edge_styles_reference_reasonable_values(self) -> None:
        from visualization.theme import EDGE_STYLES

        # Every edge style must carry rendering kwargs.
        for style in EDGE_STYLES.values():
            assert "color" in style
            assert "width" in style
            assert "alpha" in style

    def test_figure_defaults(self) -> None:
        from visualization.theme import FIGURE_DEFAULTS

        assert FIGURE_DEFAULTS["dpi"] > 100
        assert (10, 8) in (FIGURE_DEFAULTS["figsize"],)
        assert "title_fontsize" in FIGURE_DEFAULTS

    def test_colormap_presets(self) -> None:
        from visualization.theme import COLORMAP_PRESETS

        for key in ("heatmap", "transition", "correlation", "diverging"):
            assert key in COLORMAP_PRESETS

    def test_max_figure_dimension_guard(self) -> None:
        from visualization.theme import MAX_FIGURE_DIMENSION

        assert MAX_FIGURE_DIMENSION >= 100
