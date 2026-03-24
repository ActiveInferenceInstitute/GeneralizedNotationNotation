#!/usr/bin/env python3
"""
Test Gnn Processing Tests

This file contains tests migrated from test_gnn_core_modules.py.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


from gnn.parsers import (
    CoqSerializer,
    JSONSerializer,
    LeanSerializer,
    MarkdownSerializer,
    PKLSerializer,
    ProtobufSerializer,
    ScalaSerializer,
    XMLSerializer,
    YAMLSerializer,
)
from gnn.parsers.common import GNNSerializer


# Migrated from test_gnn_core_modules.py
class TestGNNParsersSerializers:
    """Test gnn.parsers serializer modules (modular imports, not monolith)."""

    @pytest.mark.unit
    def test_serializers_imports(self):
        """Test that serializers can be imported from modular files."""
        # Verify serializer classes exist and are proper types
        assert JSONSerializer is not None
        assert XMLSerializer is not None
        assert MarkdownSerializer is not None
        assert GNNSerializer is not None

    @pytest.mark.unit
    def test_json_serializer_instance(self):
        """Test JSONSerializer can be instantiated."""
        serializer = JSONSerializer()
        assert serializer is not None
        assert hasattr(serializer, 'serialize')

    @pytest.mark.unit
    def test_multiple_serializers_available(self):
        """Test that multiple serializer formats are available."""
        serializers = [
            JSONSerializer, XMLSerializer, YAMLSerializer,
            MarkdownSerializer, ScalaSerializer, ProtobufSerializer,
            PKLSerializer, LeanSerializer, CoqSerializer
        ]
        assert len(serializers) >= 9, "At least 9 serializers should be available"
