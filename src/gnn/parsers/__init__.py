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

from typing import Dict, Any, List, Optional, Union, Type, Protocol
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Import all parser modules (Lark parser removed)
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
from .temporal_serializer import TemporalSerializer
from .functional_serializer import FunctionalSerializer
from .binary_serializer import BinarySerializer

# Import converters and validators
from .converters import FormatConverter, ConversionError
from .validators import GNNValidator, ValidationError, ValidationWarning
from .common import GNNInternalRepresentation, ASTNode, ParseError

# Configure logging
logger = logging.getLogger(__name__)

# Registry of all available parsers and serializers
PARSER_REGISTRY: Dict[GNNFormat, Type['GNNParser']] = {
    GNNFormat.MARKDOWN: MarkdownGNNParser,
    GNNFormat.SCALA: ScalaGNNParser,
    GNNFormat.LEAN: LeanGNNParser,
    GNNFormat.COQ: CoqGNNParser,
    GNNFormat.PYTHON: PythonGNNParser,
    GNNFormat.BNF: BNFParser,
    GNNFormat.EBNF: EBNFParser,
    GNNFormat.ISABELLE: IsabelleParser,
    GNNFormat.MAXIMA: MaximaParser,
    GNNFormat.XML: XMLGNNParser,
    GNNFormat.PNML: PNMLParser,
    GNNFormat.JSON: JSONGNNParser,
    GNNFormat.PROTOBUF: ProtobufGNNParser,
    GNNFormat.YAML: YAMLGNNParser,
    GNNFormat.XSD: XSDParser,
    GNNFormat.ASN1: ASN1Parser,
    GNNFormat.PKL: PKLParser,
    GNNFormat.ALLOY: AlloyParser,
    GNNFormat.Z_NOTATION: ZNotationParser,
    GNNFormat.TLA_PLUS: TLAParser,
    GNNFormat.AGDA: AgdaParser,
    GNNFormat.HASKELL: HaskellGNNParser,
    GNNFormat.PICKLE: PickleGNNParser
}

SERIALIZER_REGISTRY: Dict[GNNFormat, Type['GNNSerializer']] = {
    GNNFormat.MARKDOWN: MarkdownSerializer,
    GNNFormat.JSON: JSONSerializer,
    GNNFormat.XML: XMLSerializer,
    GNNFormat.YAML: YAMLSerializer,
    GNNFormat.SCALA: ScalaSerializer,
    GNNFormat.PROTOBUF: ProtobufSerializer,
    GNNFormat.PKL: PKLSerializer,
    GNNFormat.XSD: XSDSerializer,
    GNNFormat.ASN1: ASN1Serializer,
    GNNFormat.LEAN: LeanSerializer,
    GNNFormat.COQ: CoqSerializer,
    GNNFormat.PYTHON: PythonSerializer,
    GNNFormat.BNF: GrammarSerializer,
    GNNFormat.EBNF: GrammarSerializer,
    GNNFormat.ISABELLE: IsabelleSerializer,
    GNNFormat.MAXIMA: MaximaSerializer,
    GNNFormat.ALLOY: AlloySerializer,
    GNNFormat.Z_NOTATION: ZNotationSerializer,
    GNNFormat.TLA_PLUS: TemporalSerializer,
    GNNFormat.AGDA: TemporalSerializer,  # Assuming Agda uses Temporal
    GNNFormat.HASKELL: FunctionalSerializer,
    GNNFormat.PICKLE: BinarySerializer
}

# Main parsing interface
class GNNParsingSystem:
    """
    Unified GNN parsing system supporting all specification formats.
    
    This class provides the main interface for parsing, converting, and
    serializing GNN models across all supported formats.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the GNN parsing system.
        
        Args:
            strict_validation: Whether to enforce strict validation rules
        """
        self.strict_validation = strict_validation
        self.validator = GNNValidator(strict=strict_validation)
        self.converter = FormatConverter()
        self.unified_parser = UnifiedGNNParser()
        
        # Initialize format-specific parsers
        self._parsers: Dict[GNNFormat, 'GNNParser'] = {}
        self._serializers: Dict[GNNFormat, 'GNNSerializer'] = {}
        
        self._initialize_parsers()
        self._initialize_serializers()
        
        logger.info(f"GNN Parsing System initialized with {len(self._parsers)} parsers")
    
    def _initialize_parsers(self):
        """Initialize all format-specific parsers."""
        for fmt, parser_class in PARSER_REGISTRY.items():
            try:
                self._parsers[fmt] = parser_class()
                logger.debug(f"Initialized parser for {fmt.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {fmt.value}: {e}")
    
    def _initialize_serializers(self):
        """Initialize all format-specific serializers."""
        for fmt, serializer_class in SERIALIZER_REGISTRY.items():
            try:
                self._serializers[fmt] = serializer_class()
                logger.debug(f"Initialized serializer for {fmt.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize serializer for {fmt.value}: {e}")
    
    def parse_file(self, file_path: Union[str, Path], 
                   format_hint: Optional[GNNFormat] = None) -> ParseResult:
        """
        Parse a GNN specification file.
        
        Args:
            file_path: Path to the file to parse
            format_hint: Optional format hint if file extension is ambiguous
            
        Returns:
            ParseResult containing the parsed model and metadata
            
        Raises:
            ParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine format
        detected_format = format_hint or self._detect_format(file_path)
        
        if detected_format not in self._parsers:
            raise ParseError(f"No parser available for format: {detected_format}")
        
        logger.info(f"Parsing {file_path} as {detected_format.value}")
        
        # Parse using format-specific parser
        parser = self._parsers[detected_format]
        result = parser.parse_file(file_path)
        
        # Validate if required
        if self.strict_validation:
            validation_result = self.validator.validate(result.model)
            result.validation_result = validation_result
            
            if validation_result.has_errors():
                logger.warning(f"Validation errors found in {file_path}")
        
        return result
    
    def parse_string(self, content: str, 
                     format: GNNFormat) -> ParseResult:
        """
        Parse GNN specification from string content.
        
        Args:
            content: String content to parse
            format: Format of the content
            
        Returns:
            ParseResult containing the parsed model
        """
        if format not in self._parsers:
            raise ParseError(f"No parser available for format: {format}")
        
        parser = self._parsers[format]
        result = parser.parse_string(content)
        
        if self.strict_validation:
            validation_result = self.validator.validate(result.model)
            result.validation_result = validation_result
        
        return result
    
    def convert(self, model: GNNInternalRepresentation,
                from_format: GNNFormat,
                to_format: GNNFormat) -> GNNInternalRepresentation:
        """
        Convert a GNN model between formats.
        
        Args:
            model: GNN model in internal representation
            from_format: Source format
            to_format: Target format
            
        Returns:
            Converted model
        """
        return self.converter.convert(model, from_format, to_format)
    
    def serialize(self, model: GNNInternalRepresentation,
                  format: GNNFormat) -> str:
        """
        Serialize a GNN model to string format.
        
        Args:
            model: GNN model to serialize
            format: Target format
            
        Returns:
            Serialized string representation
        """
        if format not in self._serializers:
            raise ValueError(f"No serializer available for format: {format}")
        
        serializer = self._serializers[format]
        return serializer.serialize(model)
    
    def serialize_to_file(self, model: GNNInternalRepresentation,
                          file_path: Union[str, Path],
                          format: Optional[GNNFormat] = None):
        """
        Serialize a GNN model to file.
        
        Args:
            model: GNN model to serialize
            file_path: Output file path
            format: Target format (inferred from extension if not provided)
        """
        file_path = Path(file_path)
        
        if format is None:
            format = self._detect_format(file_path)
        
        content = self.serialize(model, format)
        
        # Write to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Serialized model to {file_path} as {format.value}")
    
    def _detect_format(self, file_path: Path) -> GNNFormat:
        """
        Detect GNN format from file extension and content.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Detected format
        """
        extension = file_path.suffix.lower()
        
        format_map = {
            '.md': GNNFormat.MARKDOWN,
            '.scala': GNNFormat.SCALA,
            '.lean': GNNFormat.LEAN,
            '.v': GNNFormat.COQ,
            '.py': GNNFormat.PYTHON,
            '.bnf': GNNFormat.BNF,
            '.ebnf': GNNFormat.EBNF,
            '.thy': GNNFormat.ISABELLE,
            '.mac': GNNFormat.MAXIMA,
            '.xml': GNNFormat.XML,
            '.pnml': GNNFormat.PNML,
            '.json': GNNFormat.JSON,
            '.proto': GNNFormat.PROTOBUF,
            '.yaml': GNNFormat.YAML,
            '.yml': GNNFormat.YAML,
            '.xsd': GNNFormat.XSD,
            '.asn1': GNNFormat.ASN1,
            '.als': GNNFormat.ALLOY,
            '.zed': GNNFormat.Z_NOTATION,
            '.tla': GNNFormat.TLA_PLUS,
            '.agda': GNNFormat.AGDA,
            '.hs': GNNFormat.HASKELL,
            '.pkl': GNNFormat.PICKLE
        }
        
        if extension in format_map:
            return format_map[extension]
        
        # Try content-based detection for ambiguous files
        if file_path.exists():
            content = file_path.read_text(encoding='utf-8', errors='ignore')[:1000]
            return self._detect_format_from_content(content)
        
        # Default to markdown for GNN files
        return GNNFormat.MARKDOWN
    
    def _detect_format_from_content(self, content: str) -> GNNFormat:
        """
        Detect format from file content analysis.
        
        Args:
            content: File content sample
            
        Returns:
            Detected format
        """
        content = content.strip()
        
        # Check for specific format indicators
        if content.startswith('<?xml'):
            if 'pnml' in content.lower():
                return GNNFormat.PNML
            return GNNFormat.XML
        elif content.startswith('{') or content.startswith('['):
            return GNNFormat.JSON
        elif '##' in content and any(section in content for section in ['GNNSection', 'ModelName', 'StateSpaceBlock']):
            return GNNFormat.MARKDOWN
        elif 'package' in content and 'import cats' in content:
            return GNNFormat.SCALA
        elif content.startswith('theory') or 'imports Main' in content:
            return GNNFormat.ISABELLE
        elif 'Require Import' in content:
            return GNNFormat.COQ
        elif 'import' in content and ('.lean' in content or 'Mathlib' in content):
            return GNNFormat.LEAN
        elif content.startswith('module') and 'where' in content:
            return GNNFormat.HASKELL
        elif 'EXTENDS' in content or 'VARIABLES' in content:
            return GNNFormat.TLA_PLUS
        elif 'data' in content and ':' in content and 'Set' in content:
            return GNNFormat.AGDA
        
        # Default fallback
        return GNNFormat.MARKDOWN
    
    def get_supported_formats(self) -> List[GNNFormat]:
        """Get list of all supported formats."""
        return list(self._parsers.keys())
    
    def get_available_parsers(self) -> Dict[GNNFormat, str]:
        """Get information about available parsers."""
        return {fmt: parser.__class__.__name__ for fmt, parser in self._parsers.items()}
    
    def get_available_serializers(self) -> Dict[GNNFormat, str]:
        """Get information about available serializers."""
        return {fmt: serializer.__class__.__name__ for fmt, serializer in self._serializers.items()}

# Convenience functions for direct usage
def parse_gnn_file(file_path: Union[str, Path], 
                   format_hint: Optional[GNNFormat] = None,
                   strict_validation: bool = True) -> ParseResult:
    """
    Convenience function to parse a GNN file.
    
    Args:
        file_path: Path to the file to parse
        format_hint: Optional format hint
        strict_validation: Whether to enforce strict validation
        
    Returns:
        ParseResult containing the parsed model
    """
    parser_system = GNNParsingSystem(strict_validation=strict_validation)
    return parser_system.parse_file(file_path, format_hint)

def convert_gnn_format(input_file: Union[str, Path],
                       output_file: Union[str, Path],
                       target_format: Optional[GNNFormat] = None) -> None:
    """
    Convenience function to convert between GNN formats.
    
    Args:
        input_file: Source file path
        output_file: Target file path
        target_format: Target format (inferred if not provided)
    """
    parser_system = GNNParsingSystem()
    
    # Parse input
    result = parser_system.parse_file(input_file)
    
    # Determine target format
    if target_format is None:
        target_format = parser_system._detect_format(Path(output_file))
    
    # Serialize to output
    parser_system.serialize_to_file(result.model, output_file, target_format)

# Export main classes and functions
__all__ = [
    'GNNParsingSystem',
    'GNNFormat',
    'ParseResult',
    'GNNInternalRepresentation',
    'ParseError',
    'ValidationError',
    'ValidationWarning',
    'ConversionError',
    'parse_gnn_file',
    'convert_gnn_format',
    'PARSER_REGISTRY',
    'SERIALIZER_REGISTRY'
] 