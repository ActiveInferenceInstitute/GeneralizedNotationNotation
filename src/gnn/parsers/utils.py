#!/usr/bin/env python3
"""
GNN Parser Utils module for GNN Processing Pipeline.

This module provides parser utility functions.
"""

from typing import Union, Optional
from pathlib import Path
from .system import GNNParsingSystem
from .unified_parser import GNNFormat, ParseResult

def parse_gnn_file(file_path: Union[str, Path], 
                   format_hint: Optional[GNNFormat] = None,
                   strict_validation: bool = True) -> ParseResult:
    """
    Parse a GNN file using the unified parsing system.
    
    Args:
        file_path: Path to the file to parse
        format_hint: Optional format hint
        strict_validation: Whether to perform strict validation
        
    Returns:
        ParseResult containing the parsed model
    """
    system = GNNParsingSystem(strict_validation=strict_validation)
    return system.parse_file(file_path, format_hint)

def convert_gnn_format(input_file: Union[str, Path],
                       output_file: Union[str, Path],
                       target_format: Optional[GNNFormat] = None) -> None:
    """
    Convert a GNN file from one format to another.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        target_format: Optional target format (detected from extension if not provided)
    """
    system = GNNParsingSystem()
    
    # Parse the input file
    result = system.parse_file(input_file)
    
    # Determine target format
    if target_format is None:
        output_path = Path(output_file)
        target_format = system._detect_format(output_path)
    
    # Serialize to the target format
    system.serialize_to_file(result.model, output_file, target_format)
