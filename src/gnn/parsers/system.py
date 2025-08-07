#!/usr/bin/env python3
"""
GNN Parser System module for GNN Processing Pipeline.

This module provides the main parsing system functionality.
"""

from typing import Dict, Any, List, Optional, Union, Type
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

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

SERIALIZER_REGISTRY: Dict[GNNFormat, Type] = {
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
    GNNFormat.TLA_PLUS: TLASerializer,
    GNNFormat.AGDA: AgdaSerializer,
    GNNFormat.HASKELL: FunctionalSerializer,
    GNNFormat.PICKLE: BinarySerializer
}

class GNNParsingSystem:
    """
    Unified GNN Parsing System
    
    Provides comprehensive parsing, serialization, and conversion capabilities
    for all supported GNN specification formats.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the parsing system.
        
        Args:
            strict_validation: Whether to perform strict validation
        """
        self.strict_validation = strict_validation
        self.parsers: Dict[GNNFormat, GNNParser] = {}
        self.serializers: Dict[GNNFormat, Any] = {}
        self.converter = FormatConverter()
        self.validator = GNNValidator()
        
        self._initialize_parsers()
        self._initialize_serializers()
    
    def _initialize_parsers(self):
        """Initialize all available parsers."""
        for format_type, parser_class in PARSER_REGISTRY.items():
            try:
                self.parsers[format_type] = parser_class()
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {format_type}: {e}")
    
    def _initialize_serializers(self):
        """Initialize all available serializers."""
        for format_type, serializer_class in SERIALIZER_REGISTRY.items():
            try:
                self.serializers[format_type] = serializer_class()
            except Exception as e:
                logger.warning(f"Failed to initialize serializer for {format_type}: {e}")
    
    def parse_file(self, file_path: Union[str, Path], 
                   format_hint: Optional[GNNFormat] = None) -> ParseResult:
        """
        Parse a GNN file.
        
        Args:
            file_path: Path to the file to parse
            format_hint: Optional format hint
            
        Returns:
            ParseResult containing the parsed model
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect format if not provided
        if format_hint is None:
            format_hint = self._detect_format(file_path)
        
        # Delegate to the concrete parser's file method to avoid double-reading
        parser = self.parsers.get(format_hint)
        if parser is None:
            raise ValueError(f"Unsupported format: {format_hint}")
        return parser.parse_file(str(file_path))
    
    def parse_string(self, content: str, 
                     format: GNNFormat) -> ParseResult:
        """
        Parse GNN content from string.
        
        Args:
            content: String content to parse
            format: Format of the content
            
        Returns:
            ParseResult containing the parsed model
        """
        if format not in self.parsers:
            raise ValueError(f"Unsupported format: {format}")
        
        parser = self.parsers[format]
        
        try:
            # Parse the content using the parser's string API
            result = parser.parse_string(content)
            
            # Validate if strict validation is enabled
            if self.strict_validation:
                validation_result = self.validator.validate(result.model)
                if not validation_result.is_valid:
                    logger.warning(f"Validation warnings: {validation_result.warnings}")
            
            return result
            
        except Exception as e:
            logger.error(f"Parsing failed for format {format}: {e}")
            raise ParseError(f"Failed to parse {format} content: {e}")
    
    def convert(self, model: GNNInternalRepresentation,
                from_format: GNNFormat,
                to_format: GNNFormat) -> GNNInternalRepresentation:
        """
        Convert a model from one format to another.
        
        Args:
            model: The model to convert
            from_format: Source format
            to_format: Target format
            
        Returns:
            Converted model
        """
        return self.converter.convert(model, from_format, to_format)
    
    def serialize(self, model: GNNInternalRepresentation,
                  format: GNNFormat) -> str:
        """
        Serialize a model to string.
        
        Args:
            model: The model to serialize
            format: Target format
            
        Returns:
            Serialized string
        """
        if format not in self.serializers:
            raise ValueError(f"Unsupported format: {format}")
        
        serializer = self.serializers[format]
        
        try:
            return serializer.serialize(model)
        except Exception as e:
            logger.error(f"Serialization failed for format {format}: {e}")
            raise ParseError(f"Failed to serialize to {format}: {e}")
    
    def serialize_to_file(self, model: GNNInternalRepresentation,
                          file_path: Union[str, Path],
                          format: Optional[GNNFormat] = None):
        """
        Serialize a model to file.
        
        Args:
            model: The model to serialize
            file_path: Output file path
            format: Optional format hint
        """
        file_path = Path(file_path)
        
        # Detect format from file extension if not provided
        if format is None:
            format = self._detect_format(file_path)
        
        # Serialize to string
        content = self.serialize(model, format)
        
        # Write to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _detect_format(self, file_path: Path) -> GNNFormat:
        """
        Detect format from file extension.
        
        Args:
            file_path: Path to the file
            
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
            '.pkl': GNNFormat.PKL,
            '.als': GNNFormat.ALLOY,
            '.zed': GNNFormat.Z_NOTATION,
            '.tla': GNNFormat.TLA_PLUS,
            '.agda': GNNFormat.AGDA,
            '.hs': GNNFormat.HASKELL
        }
        
        if extension in format_map:
            return format_map[extension]
        else:
            raise ValueError(f"Unknown file extension: {extension}")
    
    def _detect_format_from_content(self, content: str) -> GNNFormat:
        """
        Detect format from content.
        
        Args:
            content: File content
            
        Returns:
            Detected format
        """
        # Simple heuristics for format detection
        if content.startswith('<?xml'):
            return GNNFormat.XML
        elif content.startswith('{') or content.startswith('['):
            return GNNFormat.JSON
        elif '---' in content[:100]:
            return GNNFormat.YAML
        elif 'syntax' in content.lower() or 'grammar' in content.lower():
            return GNNFormat.BNF
        elif 'theorem' in content.lower() or 'lemma' in content.lower():
            return GNNFormat.LEAN
        elif 'import' in content.lower() and 'scala' in content.lower():
            return GNNFormat.SCALA
        elif 'import' in content.lower() and 'python' in content.lower():
            return GNNFormat.PYTHON
        else:
            # Default to markdown for unknown formats
            return GNNFormat.MARKDOWN
    
    def get_supported_formats(self) -> List[GNNFormat]:
        """Get list of supported formats that can be parsed and serialized."""
        return [fmt for fmt in self.parsers.keys() if fmt in self.serializers]
    
    def get_available_parsers(self) -> Dict[GNNFormat, str]:
        """Get available parsers with their descriptions."""
        return {format: parser.__class__.__name__ 
                for format, parser in self.parsers.items()}
    
    def get_available_serializers(self) -> Dict[GNNFormat, str]:
        """Get available serializers with their descriptions."""
        return {format: serializer.__class__.__name__ 
                for format, serializer in self.serializers.items()}
