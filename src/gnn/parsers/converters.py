"""
GNN Format Converters Module

This module provides format conversion capabilities for GNN specifications.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConversionError(Exception):
    """Raised when format conversion fails."""
    pass

class FormatConverter:
    """
    Converts GNN models between different formats.
    """
    
    def __init__(self):
        """Initialize the format converter."""
        self.supported_conversions = {}
    
    def convert(self, model: Any, from_format: str, to_format: str) -> Any:
        """
        Convert a GNN model between formats.
        
        Args:
            model: The model to convert
            from_format: Source format
            to_format: Target format
            
        Returns:
            Converted model
            
        Raises:
            ConversionError: If conversion is not supported or fails
        """
        conversion_key = (from_format, to_format)
        
        if conversion_key not in self.supported_conversions:
            raise ConversionError(f"Conversion from {from_format} to {to_format} not supported")
        
        try:
            return self.supported_conversions[conversion_key](model)
        except Exception as e:
            raise ConversionError(f"Conversion failed: {e}")
    
    def register_conversion(self, from_format: str, to_format: str, converter_func):
        """
        Register a conversion function.
        
        Args:
            from_format: Source format
            to_format: Target format
            converter_func: Function to perform the conversion
        """
        self.supported_conversions[(from_format, to_format)] = converter_func
        logger.debug(f"Registered conversion: {from_format} -> {to_format}")
    
    def is_supported(self, from_format: str, to_format: str) -> bool:
        """
        Check if a conversion is supported.
        
        Args:
            from_format: Source format
            to_format: Target format
            
        Returns:
            True if conversion is supported
        """
        return (from_format, to_format) in self.supported_conversions 