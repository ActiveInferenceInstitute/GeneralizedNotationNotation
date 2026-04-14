#!/usr/bin/env python3
"""
Property-based tests for the GNN Parser using Hypothesis (v1.4.0 Milestone).
Ensures resilience against broken, non-standard, and adversarial notation formatting.
"""

import math
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from gnn.processor import (
    _extract_sections_lightweight,
    _extract_variables_lightweight,
    parse_gnn_file,
    validate_gnn_structure,
)


# -- Strategies --

# Generate sensible text blocks that might appear in markdown/GNN
st_text_blocks = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc"), blacklist_characters=["\x00"]),
    min_size=0,
    max_size=200,
)

# Generate headings like "# Heading" or "### Title   "
@st.composite
def st_markdown_headings(draw):
    level = draw(st.integers(min_value=1, max_value=6))
    title = draw(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")), min_size=1, max_size=20).map(str.strip)).strip()
    if not title:
        title = "DefaultHeading"
    return f"{'#' * level} {title}"

# Generate valid variable definitions
@st.composite
def st_variable_definitions(draw):
    var_format = draw(st.sampled_from(["annotation", "assignment", "array", "invalid"]))
    name = draw(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=1, max_size=10))
    val = draw(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1, max_size=10))
    
    if var_format == "annotation":
        return f"{name}: {val}"
    elif var_format == "assignment":
        return f"{name} = {val}"
    elif var_format == "array":
        return f"{name}[{val}]"
    else:
        # Just random variable-looking garbage
        return f"{name} {val}"

# Generate complete but potentially chaotic GNN file content
@st.composite
def st_gnn_content(draw):
    num_headings = draw(st.integers(min_value=0, max_value=5))
    num_vars = draw(st.integers(min_value=0, max_value=10))
    
    parts = []
    
    # Mix of headings, vars, and raw text
    for _ in range(num_headings):
        parts.append(draw(st_markdown_headings()))
        parts.append(draw(st_text_blocks))
        
    for _ in range(num_vars):
        parts.append(draw(st_variable_definitions()))
        
    # Append random raw text
    parts.append(draw(st_text_blocks))
    
    # Shuffle parts and join with newlines
    draw(st.permutations(parts))
    return "\n\n".join(parts)


# -- Tests --

@pytest.mark.unit
@given(content=st_text_blocks)
@settings(max_examples=100)
def test_extract_sections_lightweight_resilience(content: str):
    """Ensure section extraction never crashes on completely arbitrary text."""
    sections = _extract_sections_lightweight(content)
    assert isinstance(sections, list)
    for sec in sections:
        assert isinstance(sec, str)


@pytest.mark.unit
@given(content=st_text_blocks)
@settings(max_examples=100)
def test_extract_variables_lightweight_resilience(content: str):
    """Ensure variable extraction never crashes on completely arbitrary text."""
    variables = _extract_variables_lightweight(content)
    assert isinstance(variables, list)
    for var in variables:
        assert isinstance(var, str)


@pytest.mark.unit
@given(content=st_gnn_content())
@settings(max_examples=50)
def test_parse_gnn_file_resilience(content: str):
    """Ensure full GNN parsing never crashes on pseudo-valid/chaotic content."""
    dummy_file = Path("dummy.gnn")
    # Even if file doesn't exist, passing content bypasses read
    result = parse_gnn_file(dummy_file, content=content)
    
    assert isinstance(result, dict)
    assert "success" in result
    
    if result["success"]:
        assert "sections" in result
        assert "variables" in result
        assert "structure_info" in result
        info = result["structure_info"]
        assert info["line_count"] == len(content.splitlines())
        assert info["char_count"] == len(content)
    else:
        # If it failed to parse for an unexpected reason, it should cleanly return error strings
        assert "error" in result
        assert "errors" in result


@pytest.mark.unit
@given(content=st_gnn_content())
@settings(max_examples=50)
def test_validate_gnn_structure_resilience(content: str):
    """Ensure validation logic handles adversarial brackets and missing properties cleanly."""
    dummy_file = Path("dummy.gnn")
    result = validate_gnn_structure(dummy_file, content=content)
    
    assert isinstance(result, dict)
    assert "file_path" in result
    assert isinstance(result.get("valid", False), bool)
    assert isinstance(result.get("errors", []), list)
    assert isinstance(result.get("warnings", []), list)

    # If the content is totally empty after stripping, valid must be False
    if not content.strip():
        assert result["valid"] is False
        assert any("empty" in e.lower() for e in result["errors"])
