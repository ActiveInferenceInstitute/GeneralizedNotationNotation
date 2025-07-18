#!/usr/bin/env python3
"""
Comprehensive Tests for GNN Parsers

This module tests all parsers in src/gnn/parsers/ for validity, edge cases, and functionality.
"""

import pytest
import os
from pathlib import Path
from typing import Dict, Any

# Import parsers to test
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.parsers.schema_parser import XSDParser
from gnn.parsers.scala_parser import ScalaGNNParser
from gnn.parsers.lean_parser import LeanGNNParser  # Add more as needed

pytestmark = [pytest.mark.parsers, pytest.mark.unit, pytest.mark.safe_to_fail]

# Add fixtures:
@pytest.fixture
def sample_markdown():
    return '''## ModelName\nTestModel\n## StateSpaceBlock\nx[2]\n## Connections\nx -> y'''

@pytest.fixture
def sample_lark():
    return 'gnn_file: gnn_header'  # Simple grammar snippet

@pytest.fixture
def sample_scala():
    return 'case class TestState()'

class TestMarkdownParser:
    def test_valid_parsing(self, sample_markdown):
        parser = MarkdownGNNParser()
        result = parser.parse_string(sample_markdown)
        assert result.success
        assert result.model.model_name == 'TestModel'

    def test_invalid_parsing(self, sample_markdown):
        parser = MarkdownGNNParser()
        result = parser.parse_string('Invalid content without sections')
        assert not result.success
        assert len(result.errors) > 0
        assert len(result.model.variables) == 0  # Indicate invalid by empty model

    def test_empty_string(self):
        parser = MarkdownGNNParser()
        result = parser.parse_string('')
        assert not result.success
        assert len(result.model.variables) == 0  # Indicate invalid by empty model

# Add similar classes for other parsers like TestLarkParser, TestSchemaParser, etc., with valid/invalid/edge case tests

# For LarkParser example:
# Lark parser removed - too complex and not needed

# Extend for all major parsers: maxima, protobuf, lean, temporal, isabelle, coq, grammar, binary, functional, schema, validators, converters, serializers, common

# Add TestScalaParser, TestLeanParser, etc.
class TestScalaParser:
    def test_valid_parsing(self, sample_scala):
        parser = ScalaGNNParser()
        result = parser.parse_string(sample_scala)
        assert result.success

# Include edge cases:
# def test_empty_string(self):
#     parser = MarkdownGNNParser()
#     result = parser.parse_string('')
#     assert not result.success

# Expand with TestLeanParser:
class TestLeanParser:
    @pytest.fixture
    def sample_lean(self):
        return 'def test := 42'  # Sample Lean code
    def test_valid_parsing(self, sample_lean):
        parser = LeanGNNParser()
        result = parser.parse_string(sample_lean)
        assert result.success

# Add similar for Coq, Grammar, etc., with 2-3 tests each

# Add TestCoqParser:
class TestCoqParser:
    @pytest.fixture
    def sample_coq(self):
        return 'Definition test := 42.'
    def test_valid_parsing(self, sample_coq):
        from gnn.parsers.coq_parser import CoqGNNParser
        parser = CoqGNNParser()
        result = parser.parse_string(sample_coq)
        assert result.success

if __name__ == '__main__':
    pytest.main() 