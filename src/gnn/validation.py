#!/usr/bin/env python3
"""
GNN Validation Strategy Module

This module provides comprehensive validation strategies for GNN models
with multiple validation levels and extensible validation rules.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .types import ValidationLevel, ValidationResult
from .schema_validator import GNNValidator

logger = logging.getLogger(__name__)


class ValidationStrategy:
    """
    Comprehensive validation with multiple levels and strategies.
    
    Supports validation levels from basic syntax checking to
    research-grade semantic validation with round-trip testing.
    """
    
    def __init__(self):
        self.validation_level = "standard"
        self.enable_strict_checking = False
        self.validators = {}
        self._initialize_validators()
    
    def configure(self, validation_level: str = "standard",
                 enable_strict_checking: bool = False):
        """Configure validation parameters."""
        self.validation_level = validation_level
        self.enable_strict_checking = enable_strict_checking
        
        # Update validator configurations
        for validator in self.validators.values():
            if hasattr(validator, 'validation_level'):
                validator.validation_level = ValidationLevel(validation_level.upper())
    
    def _initialize_validators(self):
        """Initialize validation components."""
        try:
            # Primary GNN validator
            self.validators['gnn'] = GNNValidator(
                validation_level=ValidationLevel.STANDARD,
                enable_round_trip_testing=False
            )
            logger.debug("GNN validator initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize GNN validator: {e}")
            self.validators['gnn'] = None
    
    def validate_files(self, files: List[Path]) -> Dict[Path, ValidationResult]:
        """
        Validate multiple files based on configured level.
        
        Args:
            files: List of file paths to validate
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        
        logger.info(f"Validating {len(files)} files at level: {self.validation_level}")
        
        for file_path in files:
            try:
                result = self.validate_file(file_path)
                results[file_path] = result
                
                # Log validation outcome
                if result.is_valid:
                    logger.debug(f"✓ {file_path.name} - Valid")
                else:
                    logger.warning(f"✗ {file_path.name} - Invalid ({len(result.errors)} errors)")
                    
            except Exception as e:
                # Create error result for files that couldn't be validated
                error_result = ValidationResult(
                    is_valid=False,
                    validation_level=ValidationLevel(self.validation_level.upper()),
                    format_tested="unknown"
                )
                error_result.errors.append(f"Validation failed: {e}")
                results[file_path] = error_result
                logger.error(f"Failed to validate {file_path}: {e}")
        
        # Log summary
        valid_count = sum(1 for r in results.values() if r.is_valid)
        logger.info(f"Validation complete: {valid_count}/{len(files)} files valid")
        
        return results
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a single file based on configured level.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            ValidationResult with comprehensive validation data
        """
        if not file_path.exists():
            result = ValidationResult(
                is_valid=False,
                validation_level=ValidationLevel(self.validation_level.upper())
            )
            result.errors.append(f"File not found: {file_path}")
            return result
        
        # Determine validation approach based on level
        validation_level_enum = ValidationLevel(self.validation_level.upper())
        
        if validation_level_enum == ValidationLevel.BASIC:
            return self._validate_basic(file_path)
        elif validation_level_enum == ValidationLevel.STANDARD:
            return self._validate_standard(file_path)
        elif validation_level_enum == ValidationLevel.STRICT:
            return self._validate_strict(file_path)
        elif validation_level_enum == ValidationLevel.RESEARCH:
            return self._validate_research(file_path)
        elif validation_level_enum == ValidationLevel.ROUND_TRIP:
            return self._validate_round_trip(file_path)
        else:
            return self._validate_standard(file_path)  # Default fallback
    
    def _validate_basic(self, file_path: Path) -> ValidationResult:
        """Basic validation - file accessibility and format detection."""
        result = ValidationResult(
            is_valid=True,
            validation_level=ValidationLevel.BASIC
        )
        
        try:
            # Check file accessibility
            file_size = file_path.stat().st_size
            if file_size == 0:
                result.warnings.append("File is empty")
            
            # Detect format
            format_detected = self._detect_file_format(file_path)
            result.format_tested = format_detected
            
            # Basic content check
            if format_detected in ['json', 'xml', 'yaml']:
                self._validate_structured_format_basic(file_path, result)
            else:
                self._validate_text_format_basic(file_path, result)
            
            result.metadata['file_size'] = file_size
            result.metadata['format_detected'] = format_detected
            
        except Exception as e:
            result.errors.append(f"Basic validation failed: {e}")
            result.is_valid = False
        
        return result
    
    def _validate_standard(self, file_path: Path) -> ValidationResult:
        """Standard validation - structure and basic semantics."""
        if self.validators['gnn']:
            return self.validators['gnn'].validate_file(file_path, ValidationLevel.STANDARD)
        else:
            # Fallback to basic validation
            result = self._validate_basic(file_path)
            result.validation_level = ValidationLevel.STANDARD
            result.warnings.append("Using basic validation - GNN validator unavailable")
            return result
    
    def _validate_strict(self, file_path: Path) -> ValidationResult:
        """Strict validation - enhanced semantics and consistency."""
        if self.validators['gnn']:
            return self.validators['gnn'].validate_file(file_path, ValidationLevel.STRICT)
        else:
            result = self._validate_basic(file_path)
            result.validation_level = ValidationLevel.STRICT
            result.warnings.append("Using basic validation - GNN validator unavailable")
            return result
    
    def _validate_research(self, file_path: Path) -> ValidationResult:
        """Research-grade validation - comprehensive analysis."""
        if self.validators['gnn']:
            return self.validators['gnn'].validate_file(file_path, ValidationLevel.RESEARCH)
        else:
            result = self._validate_basic(file_path)
            result.validation_level = ValidationLevel.RESEARCH
            result.warnings.append("Using basic validation - GNN validator unavailable")
            return result
    
    def _validate_round_trip(self, file_path: Path) -> ValidationResult:
        """Round-trip validation - semantic preservation testing."""
        if self.validators['gnn']:
            # Enable round-trip testing for this validation
            original_setting = self.validators['gnn'].enable_round_trip_testing
            self.validators['gnn'].enable_round_trip_testing = True
            
            try:
                result = self.validators['gnn'].validate_file(file_path, ValidationLevel.ROUND_TRIP)
            finally:
                # Restore original setting
                self.validators['gnn'].enable_round_trip_testing = original_setting
            
            return result
        else:
            result = self._validate_basic(file_path)
            result.validation_level = ValidationLevel.ROUND_TRIP
            result.warnings.append("Round-trip testing unavailable - GNN validator missing")
            return result
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from extension and content."""
        ext = file_path.suffix.lower()
        
        format_map = {
            '.md': 'markdown',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.gnn': 'gnn'
        }
        
        return format_map.get(ext, 'unknown')
    
    def _validate_structured_format_basic(self, file_path: Path, result: ValidationResult):
        """Basic validation for structured formats (JSON, XML, YAML)."""
        try:
            content = file_path.read_text(encoding='utf-8')
            format_type = result.format_tested
            
            if format_type == 'json':
                import json
                json.loads(content)
                result.suggestions.append("JSON format is valid")
                
            elif format_type == 'xml':
                import xml.etree.ElementTree as ET
                ET.fromstring(content)
                result.suggestions.append("XML format is valid")
                
            elif format_type == 'yaml':
                try:
                    import yaml
                    yaml.safe_load(content)
                    result.suggestions.append("YAML format is valid")
                except ImportError:
                    result.warnings.append("Cannot validate YAML - PyYAML not available")
                    
        except Exception as e:
            result.errors.append(f"Format validation failed: {e}")
            result.is_valid = False
    
    def _validate_text_format_basic(self, file_path: Path, result: ValidationResult):
        """Basic validation for text formats (Markdown, GNN)."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check for basic GNN markers
            gnn_markers = ['ModelName', 'StateSpaceBlock', 'Connections', 'Parameters']
            found_markers = [marker for marker in gnn_markers if marker in content]
            
            if found_markers:
                result.suggestions.append(f"GNN markers found: {found_markers}")
            else:
                result.warnings.append("No standard GNN markers found")
            
            # Check for section structure
            if '##' in content:
                section_count = content.count('##')
                result.metadata['section_count'] = section_count
                if section_count >= 3:
                    result.suggestions.append(f"Well-structured document ({section_count} sections)")
                else:
                    result.warnings.append("Document has few sections")
            
        except UnicodeDecodeError:
            result.warnings.append("File contains non-UTF-8 content")
        except Exception as e:
            result.errors.append(f"Text validation failed: {e}")
            result.is_valid = False
    
    def get_validation_summary(self, results: Dict[Path, ValidationResult]) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        summary = {
            'total_files': len(results),
            'valid_files': 0,
            'invalid_files': 0,
            'validation_level': self.validation_level,
            'format_distribution': {},
            'error_summary': {},
            'warning_summary': {},
        }
        
        for file_path, result in results.items():
            if result.is_valid:
                summary['valid_files'] += 1
            else:
                summary['invalid_files'] += 1
            
            # Format distribution
            fmt = result.format_tested or 'unknown'
            summary['format_distribution'][fmt] = summary['format_distribution'].get(fmt, 0) + 1
            
            # Error patterns
            for error in result.errors:
                error_type = error.split(':')[0] if ':' in error else 'General'
                summary['error_summary'][error_type] = summary['error_summary'].get(error_type, 0) + 1
            
            # Warning patterns
            for warning in result.warnings:
                warning_type = warning.split(':')[0] if ':' in warning else 'General'
                summary['warning_summary'][warning_type] = summary['warning_summary'].get(warning_type, 0) + 1
        
        summary['success_rate'] = (summary['valid_files'] / summary['total_files']) * 100 if summary['total_files'] > 0 else 0
        
        return summary 