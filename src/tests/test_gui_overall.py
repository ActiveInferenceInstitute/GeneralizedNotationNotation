#!/usr/bin/env python3
import pytest
from pathlib import Path

from gui import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_components_from_markdown,
)


class TestGUIModuleComprehensive:
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_gui_markdown_helpers_imports(self):
        assert callable(add_component_to_markdown)
        assert callable(update_component_states)
        assert callable(remove_component_from_markdown)
        assert callable(parse_components_from_markdown)

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_add_component_and_parse(self):
        base = "# GNN Model\n\n"
        md = add_component_to_markdown(base, "comp1", "observation", ["s1", "s2"]) 
        comps = parse_components_from_markdown(md)
        assert any(c.get("name") == "comp1" for c in comps)

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_update_states_append_and_replace(self):
        md = (
            "# Title\n\ncomponents:\n"
            "  - name: cA\n    type: observation\n    states: [a, b]\n"
        )
        md_append = update_component_states(md, "cA", ["c"], mode="append")
        assert "states appended" in md_append
        md_replace = update_component_states(md, "cA", ["x", "y"], mode="replace")
        assert "states: [x, y]" in md_replace

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_remove_component(self):
        md = (
            "components:\n"
            "  - name: c1\n    type: observation\n    states: [s]\n"
            "  - name: c2\n    type: hidden\n    states: [h]\n"
        )
        md2 = remove_component_from_markdown(md, "c1")
        assert "- name: c1" not in md2
        assert "- name: c2" in md2


