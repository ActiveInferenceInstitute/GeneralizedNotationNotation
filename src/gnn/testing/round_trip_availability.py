#!/usr/bin/env python3
"""
Serializer and validator availability probe for the GNN round-trip test suite.

Extracted from ``testing.test_round_trip``.
"""

from .round_trip_config import LOGGING_CONFIG

try:
    # Use proper absolute imports from src
    # Add src directory to path if not already there using safer path operations
    import os
    import sys
    from pathlib import Path

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_file_dir, "..", "..", "..", "src")
    src_path = os.path.normpath(src_path)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from gnn.parsers import GNNParsingSystem
    from gnn.parsers.alloy_serializer import AlloySerializer
    from gnn.parsers.asn1_serializer import ASN1Serializer
    from gnn.parsers.binary_serializer import BinarySerializer
    from gnn.parsers.common import (
        Connection,
        ConnectionType,
        DataType,
        GNNFormat,
        GNNInternalRepresentation,
        ParseResult,
        Variable,
    )
    from gnn.parsers.coq_serializer import CoqSerializer
    from gnn.parsers.functional_serializer import FunctionalSerializer
    from gnn.parsers.grammar_serializer import GrammarSerializer
    from gnn.parsers.isabelle_serializer import IsabelleSerializer

    # Update serializer imports
    from gnn.parsers.json_serializer import JSONSerializer
    from gnn.parsers.lean_serializer import LeanSerializer
    from gnn.parsers.pkl_serializer import PKLSerializer
    from gnn.parsers.protobuf_serializer import ProtobufSerializer
    from gnn.parsers.python_serializer import PythonSerializer
    from gnn.parsers.scala_serializer import ScalaSerializer
    from gnn.parsers.xml_serializer import XMLSerializer
    from gnn.parsers.xsd_serializer import XSDSerializer
    from gnn.parsers.yaml_serializer import YAMLSerializer
    from gnn.parsers.znotation_serializer import ZNotationSerializer

    GNN_AVAILABLE = True

    from gnn.schema_validator import (
        CrossFormatValidator,
        GNNValidator,
        validate_cross_format_consistency,
    )
    from gnn.types import ParsedGNN, ValidationResult

    CROSS_FORMAT_AVAILABLE = True

except ImportError as e:
    if LOGGING_CONFIG["enable_debug"]:
        print(f"GNN module not available: {e}")
    GNN_AVAILABLE = False
