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

# Import all parser modules
from .unified_parser import UnifiedGNNParser, GNNFormat, ParseResult
from .markdown_parser import MarkdownGNNParser
from .scala_parser import ScalaGNNParser
from .lean_parser import LeanGNNParser
from .coq_parser import CoqGNNParser
from .python_parser import PythonGNNParser
from .grammar_parser import BNFParser, EBNFParser
from .isabelle_parser import IsabelleParser
from .maxima_parser import MaximaParser
from .xml_parser import XMLGNNParser, PNMLParser
from .json_parser import JSONGNNParser
from .protobuf_parser import ProtobufGNNParser
from .yaml_parser import YAMLGNNParser
from .schema_parser import XSDParser, ASN1Parser, PKLParser, AlloyParser, ZNotationParser
from .temporal_parser import TLAParser, AgdaParser
from .functional_parser import HaskellGNNParser
from .binary_parser import PickleGNNParser

# Import serializers from individual files
from .markdown_serializer import MarkdownSerializer
from .json_serializer import JSONSerializer
from .xml_serializer import XMLSerializer
from .yaml_serializer import YAMLSerializer
from .scala_serializer import ScalaSerializer
from .protobuf_serializer import ProtobufSerializer
from .pkl_serializer import PKLSerializer
from .xsd_serializer import XSDSerializer
from .asn1_serializer import ASN1Serializer
from .lean_serializer import LeanSerializer
from .coq_serializer import CoqSerializer
from .python_serializer import PythonSerializer
from .grammar_serializer import GrammarSerializer
from .isabelle_serializer import IsabelleSerializer
from .maxima_serializer import MaximaSerializer
from .alloy_serializer import AlloySerializer
from .znotation_serializer import ZNotationSerializer
from .schema_serializer import SchemaSerializer
from .temporal_serializer import TemporalSerializer, TLASerializer, AgdaSerializer
from .functional_serializer import FunctionalSerializer
from .binary_serializer import BinarySerializer

# Import converters and validators
from .converters import FormatConverter, ConversionError
from .validators import GNNValidator, ValidationError, ValidationWarning
from .common import GNNInternalRepresentation, ASTNode, ParseError, GNNParser

# Import system and utilities
from .system import GNNParsingSystem, PARSER_REGISTRY, SERIALIZER_REGISTRY
from .utils import parse_gnn_file, convert_gnn_format

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
    'parse_gnn_file',
    'convert_gnn_format',
    
    # Registries
    'PARSER_REGISTRY',
    'SERIALIZER_REGISTRY'
] 