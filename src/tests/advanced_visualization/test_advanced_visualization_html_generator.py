"""Tests for advanced_visualization/html_generator.py.

Covers the HTMLVisualizationGenerator success and error paths using real
structured model data (no mocks).
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestHTMLVisualizationGenerator:
    def _make_generator(self) -> Any:
        from advanced_visualization.html_generator import HTMLVisualizationGenerator

        return HTMLVisualizationGenerator()

    def test_success_page_includes_model_name(self) -> None:
        gen = self._make_generator()
        data: dict[str, Any] = {
            "success": True,
            "blocks": [{"name": "s", "type": "hidden_state"}],
            "connections": [],
            "parameters": [],
            "equations": [],
            "model_info": {"name": "MyModel"},
        }
        html = gen.generate_advanced_visualization(data, "MyModel")
        assert "<!DOCTYPE html>" in html
        assert "MyModel" in html
        assert "s" in html

    def test_success_page_renders_blocks_connections_parameters_equations(self) -> None:
        gen = self._make_generator()
        data: dict[str, Any] = {
            "success": True,
            "blocks": [{"name": "o", "type": "observation"}],
            "connections": [{"from": ["s"], "to": ["o"], "type": "generative"}],
            "parameters": [{"name": "A", "value": [[0.5, 0.5]]}],
            "equations": [{"label": "FEP", "content": "F = -ln P(o,s)"}],
            "model_info": {},
        }
        html = gen.generate_advanced_visualization(data, "M")
        assert "o" in html
        assert "s" in html
        assert "A" in html
        assert "FEP" in html

    def test_parameters_and_equations_omit_section_when_empty(self) -> None:
        gen = self._make_generator()
        data: dict[str, Any] = {
            "success": True,
            "blocks": [],
            "connections": [],
            "parameters": [],
            "equations": [],
            "model_info": {},
        }
        html = gen.generate_advanced_visualization(data, "M")
        assert "Model Parameters" not in html
        assert "Model Equations" not in html

    def test_error_page_generated_on_failure_data(self) -> None:
        gen = self._make_generator()
        data: dict[str, Any] = {"success": False, "errors": ["boom"]}
        html = gen.generate_advanced_visualization(data, "Broken")
        assert "Error" in html
        assert "Broken" in html

    def test_error_page_shows_error_messages(self) -> None:
        from advanced_visualization.html_generator import (
            HTMLVisualizationGenerator as G,
        )

        gen = G()
        html = gen._generate_error_page("M", ["first error", "second error"])
        assert "first error" in html
        assert "second error" in html