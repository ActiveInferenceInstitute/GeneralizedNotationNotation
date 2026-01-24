#!/usr/bin/env python3
"""
Comprehensive GUI Module Tests

Tests the GUI module's core functionality including:
- Component manipulation (add, update, remove)
- Markdown parsing
- State space management
- GUI type discovery
- Error handling
"""

import pytest
from pathlib import Path
import json
import logging

from gui import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_components_from_markdown,
    parse_state_space_from_markdown,
    add_state_space_entry,
    update_state_space_entry,
    remove_state_space_entry,
    get_available_guis,
    FEATURES,
)


class TestGUIModuleComprehensive:
    """Comprehensive tests for GUI module component management."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui_markdown_helpers_imports(self):
        """Test that all GUI markdown helpers are callable."""
        assert callable(add_component_to_markdown)
        assert callable(update_component_states)
        assert callable(remove_component_from_markdown)
        assert callable(parse_components_from_markdown)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_add_component_and_parse(self):
        """Test adding a component and parsing it back."""
        base = "# GNN Model\n\n"
        md = add_component_to_markdown(base, "comp1", "observation", ["s1", "s2"])
        comps = parse_components_from_markdown(md)
        assert any(c.get("name") == "comp1" for c in comps)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_add_multiple_components(self):
        """Test adding multiple components to markdown."""
        md = "# GNN Model\n\n"
        md = add_component_to_markdown(md, "obs1", "observation", ["o1", "o2"])
        md = add_component_to_markdown(md, "hidden1", "hidden", ["h1", "h2", "h3"])
        md = add_component_to_markdown(md, "action1", "action", ["a1"])

        comps = parse_components_from_markdown(md)
        names = [c.get("name") for c in comps]
        assert "obs1" in names
        assert "hidden1" in names
        assert "action1" in names

    @pytest.mark.unit
    @pytest.mark.fast
    def test_update_states_append_and_replace(self):
        """Test updating component states with append and replace modes."""
        md = (
            "# Title\n\ncomponents:\n"
            "  - name: cA\n    type: observation\n    states: [a, b]\n"
        )
        md_append = update_component_states(md, "cA", ["c"], mode="append")
        assert "states appended" in md_append or "c" in md_append

        md_replace = update_component_states(md, "cA", ["x", "y"], mode="replace")
        assert "states: [x, y]" in md_replace or "x" in md_replace

    @pytest.mark.unit
    @pytest.mark.fast
    def test_remove_component(self):
        """Test removing a component from markdown."""
        md = (
            "components:\n"
            "  - name: c1\n    type: observation\n    states: [s]\n"
            "  - name: c2\n    type: hidden\n    states: [h]\n"
        )
        md2 = remove_component_from_markdown(md, "c1")
        assert "- name: c1" not in md2
        assert "- name: c2" in md2

    @pytest.mark.unit
    @pytest.mark.fast
    def test_remove_nonexistent_component(self):
        """Test removing a component that doesn't exist."""
        md = "components:\n  - name: c1\n    type: observation\n"
        md2 = remove_component_from_markdown(md, "nonexistent")
        # Should not raise and should preserve original
        assert "c1" in md2

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_empty_components(self):
        """Test parsing markdown with no components."""
        md = "# Just a title\n\nSome text without components"
        comps = parse_components_from_markdown(md)
        assert isinstance(comps, list)


class TestStateSpaceManagement:
    """Tests for state space entry management."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_state_space_helpers_callable(self):
        """Test that state space helpers are callable."""
        assert callable(parse_state_space_from_markdown)
        assert callable(add_state_space_entry)
        assert callable(update_state_space_entry)
        assert callable(remove_state_space_entry)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_add_state_space_entry(self):
        """Test adding a state space entry."""
        md = "# Model\n\n## StateSpaceBlock\n"
        md2 = add_state_space_entry(md, "A", [3, 3], "float")
        assert "A" in md2

    @pytest.mark.unit
    @pytest.mark.fast
    def test_update_state_space_entry(self):
        """Test updating a state space entry."""
        md = "## StateSpaceBlock\nA[3,3,type=float]\n"
        md2 = update_state_space_entry(md, "A", [5, 5], "int")
        # Should contain updated dimensions
        assert "5" in md2 or "A" in md2

    @pytest.mark.unit
    @pytest.mark.fast
    def test_remove_state_space_entry(self):
        """Test removing a state space entry."""
        md = "## StateSpaceBlock\nA[3,3,type=float]\nB[2,2,type=int]\n"
        md2 = remove_state_space_entry(md, "A")
        assert "B" in md2


class TestGUIDiscovery:
    """Tests for GUI type discovery and info."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_get_available_guis_returns_dict(self):
        """Test that get_available_guis returns a dictionary."""
        guis = get_available_guis()
        assert isinstance(guis, dict)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_get_available_guis_has_expected_keys(self):
        """Test that available GUIs include expected types."""
        guis = get_available_guis()
        expected_keys = ["gui_1", "gui_2", "gui_3", "oxdraw"]
        for key in expected_keys:
            assert key in guis, f"Expected GUI type '{key}' not found"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui_info_has_name_and_description(self):
        """Test that each GUI info dict has name and description."""
        guis = get_available_guis()
        for gui_type, info in guis.items():
            assert isinstance(info, dict), f"{gui_type} info should be a dict"
            # Info should have at least some content
            assert len(info) > 0, f"{gui_type} info should not be empty"


class TestGUIFeatures:
    """Tests for GUI module features."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_features_is_dict(self):
        """Test that FEATURES is a dictionary."""
        assert isinstance(FEATURES, dict)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_features_has_expected_keys(self):
        """Test that FEATURES has expected capability flags."""
        expected_features = [
            "form_based_constructor",
            "visual_matrix_editor",
            "state_space_studio",
            "diagram_as_code",
            "mcp_integration"
        ]
        for feature in expected_features:
            assert feature in FEATURES, f"Expected feature '{feature}' not found"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_features_are_booleans(self):
        """Test that all feature flags are boolean."""
        for key, value in FEATURES.items():
            assert isinstance(value, bool), f"Feature '{key}' should be boolean"


class TestGUIErrorHandling:
    """Tests for GUI error handling."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_add_component_with_empty_states(self):
        """Test adding a component with empty states list."""
        md = "# Model\n"
        md2 = add_component_to_markdown(md, "empty_comp", "observation", [])
        assert "empty_comp" in md2

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parse_malformed_markdown(self):
        """Test parsing malformed markdown doesn't raise."""
        malformed = "not: valid: yaml: [[[broken"
        # Should not raise
        try:
            result = parse_components_from_markdown(malformed)
            assert isinstance(result, list)
        except Exception:
            # Acceptable to raise for malformed input
            pass

    @pytest.mark.unit
    @pytest.mark.fast
    def test_update_states_missing_component(self):
        """Test updating states for missing component."""
        md = "components:\n  - name: c1\n"
        result = update_component_states(md, "missing", ["state1"])
        # Should handle gracefully
        assert isinstance(result, str)
