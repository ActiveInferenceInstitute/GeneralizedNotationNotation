#!/usr/bin/env python3
"""
Adversarial tests for the GNN Parser.
Ensures resilience against broken, non-standard, and adversarial notation formatting.
Refactored to zero-mock pytest parametrizations replacing deprecated Hypothesis routines.
"""

from pathlib import Path

import pytest

from gnn.processor import (
    _extract_sections_lightweight,
    _extract_variables_lightweight,
    parse_gnn_file,
    validate_gnn_structure,
)

# Deterministic adversarial permutations to emulate Hypothesis coverage
ADVERSARIAL_PAYLOADS = [
    "",  # Empty
    "   \n  \t ",  # Whitespace only
    "# Single Heading",  # Single headless
    "Some raw text without headings",  # Raw text
    "## Missing Body:",  # Unfinished section
    "## ValidSection\nVar = ",  # Broken assignment
    "### Another\nVar[3, 3, type=float]\n\n# H\nVar2: 3",  # Normal looking
    "##\n#\n###\n",  # Pure broken headings
    "\x00\x01\xFF\xFE",  # Invalid bytes / non-printable
    "## Section\nA = B = C = D",  # Chained assignment
    "## Variables:\nvar1 = [1, 2, [3, 4",  # Unclosed bracket
    "\n\n\n## Section\n\n\nvar: 1\n\n\n",  # Heavy spacing
]

@pytest.mark.unit
@pytest.mark.parametrize("content", ADVERSARIAL_PAYLOADS)
def test_extract_sections_lightweight_resilience(content: str):
    """Ensure section extraction never crashes on completely arbitrary text."""
    sections = _extract_sections_lightweight(content)
    assert isinstance(sections, list)
    for sec in sections:
        assert isinstance(sec, str)

@pytest.mark.unit
@pytest.mark.parametrize("content", ADVERSARIAL_PAYLOADS)
def test_extract_variables_lightweight_resilience(content: str):
    """Ensure variable extraction never crashes on completely arbitrary text."""
    variables = _extract_variables_lightweight(content)
    assert isinstance(variables, list)
    for var in variables:
        assert isinstance(var, str)

@pytest.mark.unit
@pytest.mark.parametrize("content", ADVERSARIAL_PAYLOADS)
def test_parse_gnn_file_resilience(content: str):
    """Ensure full GNN parsing never crashes on pseudo-valid/chaotic content."""
    dummy_file = Path("dummy.gnn")
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
        assert "error" in result
        assert "errors" in result

@pytest.mark.unit
@pytest.mark.parametrize("content", ADVERSARIAL_PAYLOADS)
def test_validate_gnn_structure_resilience(content: str):
    """Ensure validation logic handles adversarial brackets and missing properties cleanly."""
    dummy_file = Path("dummy.gnn")
    result = validate_gnn_structure(dummy_file, content=content)
    
    assert isinstance(result, dict)
    assert "file_path" in result
    assert isinstance(result.get("valid", False), bool)
    assert isinstance(result.get("errors", []), list)
    assert isinstance(result.get("warnings", []), list)

    if not content.strip():
        assert result["valid"] is False
        assert any("empty" in e.lower() for e in result["errors"])
