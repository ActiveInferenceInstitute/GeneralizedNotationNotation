"""
GNN Parsers - Comprehensive Multi-Format Parsing System

This module provides unified parsing capabilities for all GNN specification formats,
including categorical specifications, formal proofs, schemas, and neural networks.

Supported Formats:
- Markdown (.md) - Standard GNN format
- Scala (.scala) - Categorical specifications  
- Lean (.lean) - Category theory proofs
- Coq (.v) - Formal verification
- Python (.py) - Geometric/neural implementations
- BNF/EBNF (.bnf/.ebnf) - Grammar specifications
- Isabelle/HOL (.thy) - Theorem proving
- Maxima (.mac) - Symbolic computation
- XML (.xml) - Petri nets and schemas
- PNML (.pnml) - Petri net markup
- JSON (.json) - Data interchange
- Protocol Buffers (.proto) - Serialization
- YAML (.yaml) - Configuration
- XSD (.xsd) - XML schemas
- ASN.1 (.asn1) - Data structure definitions
- Alloy (.als) - Model checking
- Z notation (.zed) - Formal specifications
- TLA+ (.tla) - Temporal logic
- Agda (.agda) - Type theory
- Haskell (.hs) - Functional specifications
- Pickle (.pkl) - Python serialization

Features:
- Unicode support for mathematical symbols (e.g., π, σ, μ) in variable names
- Special handling for Active Inference models with standard variables (A, B, C, D, E, F, G)
- Comprehensive validation for model consistency and correctness
- Cross-format conversion while preserving semantics

Author: @docxology
Date: 2025-01-11
License: MIT
"""

# system.py is the single source of truth for parser/serializer registration.
# Re-export everything from there to avoid double-importing all 46 concrete classes.
from pathlib import Path as _Path
from typing import Optional as _Optional
from typing import Union

from .common import ASTNode, ValidationError, ValidationWarning
from .converters import ConversionError, FormatConverter
from .schema_serializer import SchemaSerializer
from .system import (
    PARSER_REGISTRY,
    SERIALIZER_REGISTRY,
    AgdaParser,
    AgdaSerializer,
    AlloyParser,
    AlloySerializer,
    ASN1Parser,
    ASN1Serializer,
    BinarySerializer,
    BNFParser,
    CoqGNNParser,
    CoqSerializer,
    EBNFParser,
    FunctionalSerializer,
    # Parsers
    GNNFormat,
    # Common
    GNNInternalRepresentation,
    GNNParser,
    GNNParsingSystem,
    GrammarSerializer,
    HaskellGNNParser,
    IsabelleParser,
    IsabelleSerializer,
    JSONGNNParser,
    JSONSerializer,
    LeanGNNParser,
    LeanSerializer,
    MarkdownGNNParser,
    # Serializers
    MarkdownSerializer,
    MaximaParser,
    MaximaSerializer,
    ParseError,
    ParseResult,
    PickleGNNParser,
    PKLParser,
    PKLSerializer,
    PNMLParser,
    ProtobufGNNParser,
    ProtobufSerializer,
    PythonGNNParser,
    PythonSerializer,
    ScalaGNNParser,
    ScalaSerializer,
    TLAParser,
    TLASerializer,
    XMLGNNParser,
    XMLSerializer,
    XSDParser,
    XSDSerializer,
    YAMLGNNParser,
    YAMLSerializer,
    ZNotationParser,
    ZNotationSerializer,
)
from .temporal_serializer import TemporalSerializer
from .unified_parser import UnifiedGNNParser
from .validators import GNNValidator


def parse_gnn_file_structured(file_path: Union[str, _Path],
                              format_hint: _Optional[GNNFormat] = None,
                              strict_validation: bool = True) -> ParseResult:
    """Parse a GNN file using the unified parsing system, returning a structured ParseResult."""
    system = GNNParsingSystem(strict_validation=strict_validation)
    return system.parse_file(file_path, format_hint)


def convert_gnn_format(input_file: Union[str, _Path],
                       output_file: Union[str, _Path],
                       target_format: _Optional[GNNFormat] = None) -> None:
    """Convert a GNN file from one format to another."""
    system = GNNParsingSystem()
    result = system.parse_file(input_file)

    if target_format is None:
        output_path = _Path(output_file)
        target_format = system._detect_format(output_path)

    system.serialize_to_file(result.model, output_file, target_format)

__all__ = [
    # Core classes
    'GNNParsingSystem',
    'UnifiedGNNParser',
    'GNNFormat',
    'ParseResult',
    'GNNInternalRepresentation',
    'ASTNode',
    'ParseError',
    'GNNParser',

    # Parsers
    'MarkdownGNNParser',
    'ScalaGNNParser',
    'LeanGNNParser',
    'CoqGNNParser',
    'PythonGNNParser',
    'BNFParser',
    'EBNFParser',
    'IsabelleParser',
    'MaximaParser',
    'XMLGNNParser',
    'PNMLParser',
    'JSONGNNParser',
    'ProtobufGNNParser',
    'YAMLGNNParser',
    'XSDParser',
    'ASN1Parser',
    'PKLParser',
    'AlloyParser',
    'ZNotationParser',
    'TLAParser',
    'AgdaParser',
    'HaskellGNNParser',
    'PickleGNNParser',

    # Serializers
    'MarkdownSerializer',
    'JSONSerializer',
    'XMLSerializer',
    'YAMLSerializer',
    'ScalaSerializer',
    'ProtobufSerializer',
    'PKLSerializer',
    'XSDSerializer',
    'ASN1Serializer',
    'LeanSerializer',
    'CoqSerializer',
    'PythonSerializer',
    'GrammarSerializer',
    'IsabelleSerializer',
    'MaximaSerializer',
    'AlloySerializer',
    'ZNotationSerializer',
    'SchemaSerializer',
    'TemporalSerializer',
    'TLASerializer',
    'AgdaSerializer',
    'FunctionalSerializer',
    'BinarySerializer',

    # Utilities
    'FormatConverter',
    'ConversionError',
    'GNNValidator',
    'ValidationError',
    'ValidationWarning',
    'parse_gnn_file_structured',
    'convert_gnn_format',

    # Registries
    'PARSER_REGISTRY',
    'SERIALIZER_REGISTRY'
]
