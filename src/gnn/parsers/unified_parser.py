"""
Unified GNN Parser

This module provides the main unified parser that coordinates all format-specific
parsers and provides a single interface for parsing any GNN format.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type
import logging

from .common import (
    GNNFormat, ParseResult, GNNInternalRepresentation, ParseError, 
    BaseGNNParser, normalize_variable_name
)

logger = logging.getLogger(__name__)

class UnifiedGNNParser:
    """
    Unified parser that can handle all GNN formats.
    
    This parser automatically detects the format and delegates to the
    appropriate format-specific parser.
    """
    
    def __init__(self):
        """Initialize the unified parser."""
        self.format_parsers: Dict[GNNFormat, BaseGNNParser] = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize all format-specific parsers on demand."""
        # Parsers will be initialized lazily when needed
        pass
    
    def parse_file(self, file_path: Union[str, Path], 
                   format_hint: Optional[GNNFormat] = None) -> ParseResult:
        """
        Parse a GNN file automatically detecting or using the provided format.
        
        Args:
            file_path: Path to the file to parse
            format_hint: Optional format hint to skip auto-detection
            
        Returns:
            ParseResult with the parsed model
            
        Raises:
            ParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Detect format if not provided
            if format_hint is None:
                format_hint = self._detect_format(file_path)
                logger.info(f"Detected format: {format_hint.value} for {file_path}")
            
            # Get or create format-specific parser
            try:
                parser = self._get_parser(format_hint)
            except ValueError as e:
                # Handle parser initialization error
                parse_time = time.time() - start_time
                result = ParseResult(
                    model=GNNInternalRepresentation(model_name=f"Failed {format_hint.value} Parse"),
                    success=False,
                    parse_time=parse_time,
                    source_file=str(file_path)
                )
                result.add_error(f"Parser initialization failed: {str(e)}")
                return result
            
            # Parse the file
            logger.info(f"Parsing {file_path} as {format_hint.value}")
            result = parser.parse_file(str(file_path))
            
            # Add metadata
            result.parse_time = time.time() - start_time
            result.source_file = str(file_path)
            result.model.source_format = format_hint
            
            # Compute checksum
            try:
                content = file_path.read_text(encoding='utf-8')
                result.model.checksum = hashlib.md5(content.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"Failed to compute checksum: {e}")
            
            logger.info(f"Successfully parsed {file_path} in {result.parse_time:.3f}s")
            return result
            
        except Exception as e:
            parse_time = time.time() - start_time
            logger.error(f"Failed to parse {file_path}: {e}")
            
            # Return failed result
            result = ParseResult(
                model=GNNInternalRepresentation(model_name="Failed Parse"),
                success=False,
                parse_time=parse_time,
                source_file=str(file_path)
            )
            result.add_error(str(e))
            return result
    
    def parse_string(self, content: str, 
                     format: GNNFormat) -> ParseResult:
        """
        Parse GNN content from a string.
        
        Args:
            content: String content to parse
            format: Format of the content
            
        Returns:
            ParseResult with the parsed model
        """
        start_time = time.time()
        
        try:
            # Get format-specific parser
            try:
                parser = self._get_parser(format)
            except ValueError as e:
                # Handle parser initialization error
                parse_time = time.time() - start_time
                result = ParseResult(
                    model=GNNInternalRepresentation(model_name=f"Failed {format.value} Parse"),
                    success=False,
                    parse_time=parse_time
                )
                result.add_error(f"Parser initialization failed: {str(e)}")
                return result
            
            # Parse the content
            logger.info(f"Parsing string content as {format.value}")
            result = parser.parse_string(content)
            
            # Add metadata
            result.parse_time = time.time() - start_time
            result.model.source_format = format
            
            try:
                result.model.checksum = hashlib.md5(content.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"Failed to compute checksum: {e}")
            
            logger.info(f"Successfully parsed string content in {result.parse_time:.3f}s")
            return result
            
        except Exception as e:
            parse_time = time.time() - start_time
            logger.error(f"Failed to parse string content: {e}")
            
            # Return failed result
            result = ParseResult(
                model=GNNInternalRepresentation(model_name="Failed Parse"),
                success=False,
                parse_time=parse_time
            )
            result.add_error(str(e))
            return result
    
    def _detect_format(self, file_path: Path) -> GNNFormat:
        """
        Detect the format of a file based on extension and content.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Detected GNNFormat
        """
        extension = file_path.suffix.lower()
        
        # Extension-based detection
        extension_map = {
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
        
        if extension in extension_map:
            detected = extension_map[extension]
            
            # For ambiguous extensions, use content-based detection
            if extension in ['.xml', '.py']:
                content_detected = self._detect_format_from_content(file_path)
                if content_detected != detected:
                    return content_detected
            
            return detected
        
        # Content-based detection for unknown extensions
        return self._detect_format_from_content(file_path)
    
    def _detect_format_from_content(self, file_path: Path) -> GNNFormat:
        """
        Detect format from file content analysis.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Detected GNNFormat
        """
        try:
            # Read a sample of the file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')[:2000]
            content_lower = content.lower()
            
            # Check for specific format indicators
            if content.strip().startswith('<?xml'):
                if 'pnml' in content_lower or 'petri' in content_lower:
                    return GNNFormat.PNML
                elif 'xsd' in content_lower or 'schema' in content_lower:
                    return GNNFormat.XSD
                else:
                    return GNNFormat.XML
            
            elif content.strip().startswith('{') or content.strip().startswith('['):
                return GNNFormat.JSON
            
            elif ('##' in content and 
                  any(section in content for section in ['GNNSection', 'ModelName', 'StateSpaceBlock'])):
                return GNNFormat.MARKDOWN
            
            elif 'package' in content and 'import cats' in content:
                return GNNFormat.SCALA
            
            elif content.startswith('theory') or 'imports Main' in content_lower:
                return GNNFormat.ISABELLE
            
            elif 'Require Import' in content:
                return GNNFormat.COQ
            
            elif ('import' in content and 
                  ('mathlib' in content_lower or '.lean' in content_lower)):
                return GNNFormat.LEAN
            
            elif content.startswith('module') and 'where' in content:
                return GNNFormat.HASKELL
            
            elif 'EXTENDS' in content or 'VARIABLES' in content:
                return GNNFormat.TLA_PLUS
            
            elif 'data' in content and ':' in content and 'Set' in content:
                return GNNFormat.AGDA
            
            elif 'def ' in content and 'jax' in content_lower:
                return GNNFormat.PYTHON
            
            elif 'load(' in content and ('maxima' in content_lower or '.mac' in content_lower):
                return GNNFormat.MAXIMA
            
            elif ('::=' in content or '<rule>' in content_lower) and 'grammar' in content_lower:
                if 'ebnf' in content_lower:
                    return GNNFormat.EBNF
                else:
                    return GNNFormat.BNF
            
            elif 'syntax' in content_lower and ('yaml' in content_lower or '---' in content):
                return GNNFormat.YAML
            
            elif ('message' in content and 'protobuf' in content_lower) or '.proto' in content_lower:
                return GNNFormat.PROTOBUF
            
            elif 'sig' in content and 'alloy' in content_lower:
                return GNNFormat.ALLOY
            
            elif 'schema' in content_lower and ('\\begin' in content or '\\end' in content):
                return GNNFormat.Z_NOTATION
            
            elif 'ASN1' in content or 'BEGIN' in content and 'END' in content:
                return GNNFormat.ASN1
            
            # Default to markdown for GNN files
            return GNNFormat.MARKDOWN
            
        except Exception:
            # Default fallback
            return GNNFormat.MARKDOWN
    
    def _get_parser(self, format: GNNFormat) -> BaseGNNParser:
        """
        Get or create a format-specific parser.
        
        Args:
            format: Format to get parser for
            
        Returns:
            BaseGNNParser for the format
            
        Raises:
            ValueError: If format not supported
        """
        if format not in self.format_parsers:
            try:
                # Get the parser class
                parser_class = self._get_parser_class(format)
                
                # Create and initialize the parser
                self.format_parsers[format] = parser_class()
                logger.debug(f"Initialized parser for {format.value}")
                
            except ImportError as e:
                logger.error(f"Failed to import parser for {format.value}: {e}")
                raise ValueError(f"Parser for {format.value} is not available. Required module missing: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize parser for {format.value}: {e}")
                raise ValueError(f"Failed to initialize parser for {format.value}: {e}")
        
        return self.format_parsers[format]
    
    def _get_parser_class(self, format: GNNFormat) -> Type[BaseGNNParser]:
        """
        Get the parser class for a specific format.
        
        Args:
            format: Format to get parser class for
            
        Returns:
            Parser class
            
        Raises:
            ParseError: If no parser class is available
        """
        # Import parsers dynamically to avoid circular imports
        if format == GNNFormat.MARKDOWN:
            from .markdown_parser import MarkdownGNNParser
            return MarkdownGNNParser
        
        elif format == GNNFormat.SCALA:
            from .scala_parser import ScalaGNNParser
            return ScalaGNNParser
        
        elif format == GNNFormat.LEAN:
            from .lean_parser import LeanGNNParser
            return LeanGNNParser
        
        elif format == GNNFormat.COQ:
            from .coq_parser import CoqGNNParser
            return CoqGNNParser
        
        elif format == GNNFormat.PYTHON:
            from .python_parser import PythonGNNParser
            return PythonGNNParser
        
        elif format == GNNFormat.BNF:
            from .grammar_parser import BNFParser
            return BNFParser
        
        elif format == GNNFormat.EBNF:
            from .grammar_parser import EBNFParser
            return EBNFParser
        
        elif format == GNNFormat.ISABELLE:
            from .isabelle_parser import IsabelleParser
            return IsabelleParser
        
        elif format == GNNFormat.MAXIMA:
            from .maxima_parser import MaximaParser
            return MaximaParser
        
        elif format == GNNFormat.XML:
            from .xml_parser import XMLGNNParser
            return XMLGNNParser
        
        elif format == GNNFormat.PNML:
            from .xml_parser import PNMLParser
            return PNMLParser
        
        elif format == GNNFormat.JSON:
            from .json_parser import JSONGNNParser
            return JSONGNNParser
        
        elif format == GNNFormat.PROTOBUF:
            from .protobuf_parser import ProtobufGNNParser
            return ProtobufGNNParser
        
        elif format == GNNFormat.YAML:
            from .yaml_parser import YAMLGNNParser
            return YAMLGNNParser
        
        elif format == GNNFormat.XSD:
            from .schema_parser import XSDParser
            return XSDParser
        
        elif format == GNNFormat.ASN1:
            from .schema_parser import ASN1Parser
            return ASN1Parser
        
        elif format == GNNFormat.ALLOY:
            from .schema_parser import AlloyParser
            return AlloyParser
        
        elif format == GNNFormat.Z_NOTATION:
            from .schema_parser import ZNotationParser
            return ZNotationParser
        
        elif format == GNNFormat.TLA_PLUS:
            from .temporal_parser import TLAParser
            return TLAParser
        
        elif format == GNNFormat.AGDA:
            from .temporal_parser import AgdaParser
            return AgdaParser
        
        elif format == GNNFormat.HASKELL:
            from .functional_parser import HaskellGNNParser
            return HaskellGNNParser
        
        elif format == GNNFormat.PICKLE:
            from .binary_parser import PickleGNNParser
            return PickleGNNParser
        
        else:
            raise ParseError(f"No parser available for format: {format.value}")
    
    def get_supported_formats(self) -> List[GNNFormat]:
        """Get list of all supported formats."""
        return list(GNNFormat)
    
    def clear_parser_cache(self):
        """Clear the parser cache to free memory."""
        self.format_parsers.clear()
        logger.info("Parser cache cleared")

# Re-export for convenience
__all__ = ['UnifiedGNNParser', 'GNNFormat', 'ParseResult'] 