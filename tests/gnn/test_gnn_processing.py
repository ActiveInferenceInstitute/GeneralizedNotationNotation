#!/usr/bin/env python3
"""
Test Gnn Processing Tests

This file contains tests migrated from test_gnn_core_modules.py.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
    def test_serializers_imports(self) -> Any:
        """Serializer classes import from modular files and define serialize."""
        for cls in (JSONSerializer, XMLSerializer, MarkdownSerializer):
            assert callable(getattr(cls, "serialize", None)), (
                f"{cls.__name__} must define a callable serialize method"
            )
        assert isinstance(GNNSerializer, type)

    @pytest.mark.unit
    def test_json_serializer_instance(self) -> Any:
        """Test JSONSerializer can be instantiated."""
        serializer = JSONSerializer()
        assert isinstance(serializer, JSONSerializer)
        assert hasattr(serializer, "serialize")

    @pytest.mark.unit
    def test_multiple_serializers_available(self) -> Any:
        """Test that multiple serializer formats are available."""
        serializers: list[Any] = [
            JSONSerializer,
            XMLSerializer,
            YAMLSerializer,
            MarkdownSerializer,
            ScalaSerializer,
            ProtobufSerializer,
            PKLSerializer,
            LeanSerializer,
            CoqSerializer,
        ]
        for serializer_cls in serializers:
            assert callable(getattr(serializer_cls, "serialize", None)), (
                f"{serializer_cls.__name__} must define a callable serialize method"
            )
        # Concrete serializers must actually inherit the shared ABC base, not
        # just be listed here by name.
        from gnn.parsers.base_serializer import BaseGNNSerializer

        assert all(
            issubclass(cls, BaseGNNSerializer)
            for cls in (JSONSerializer, XMLSerializer)
        )
