#!/usr/bin/env python3
"""
Test Gnn Parsing Tests

This file contains tests migrated from test_gnn_core_modules.py.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.parsers.scala_parser import ScalaGNNParser
from gnn.parsers.lean_parser import LeanGNNParser


# Migrated from test_gnn_core_modules.py
class TestGNNDiscovery:
    """Test gnn discovery functionality."""
    
    @pytest.mark.unit
    def test_discovery_imports(self):
        """Test that discovery function can be imported from gnn package."""
        from gnn import discover_gnn_files
        assert callable(discover_gnn_files)
    
    @pytest.mark.unit  
    def test_discover_gnn_files(self, isolated_temp_dir, sample_gnn_files):
        """Test GNN file discovery functionality."""
        from gnn import discover_gnn_files
        
        # Create test directory with GNN files
        test_dir = isolated_temp_dir / "test_models"
        test_dir.mkdir()
        
        # Create various file types
        (test_dir / "model1.md").write_text("# GNN Model 1")
        (test_dir / "model2.gnn").write_text("# GNN Model 2") 
        (test_dir / "not_gnn.txt").write_text("Not a GNN file")
        (test_dir / "model3.yaml").write_text("# GNN Model 3")
        
        discovered_files = discover_gnn_files(test_dir)
        
        # Verify discovery results
        assert isinstance(discovered_files, list)
        # Should find at least the .md files
        assert len(discovered_files) >= 1
        
        # Verify file paths are returned
        for file_path in discovered_files:
            assert isinstance(file_path, (str, Path))


# Migrated from test_parsers.py
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


# Migrated from test_parsers.py
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


# Migrated from test_parsers.py
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


# Migrated from test_parsers.py
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
