"""
Cross-Format Validation for GNN Schema Consistency

This module ensures that all format representations (JSON Schema, YAML Schema,
XSD, Protocol Buffers, etc.) maintain consistency and validate the same GNN models.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from .schema_validator import GNNValidator, ValidationResult, GNNParser

logger = logging.getLogger(__name__)


@dataclass
class CrossFormatValidationResult:
    """Results from cross-format validation."""
    is_consistent: bool
    schema_formats: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    format_results: Dict[str, ValidationResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossFormatValidator:
    """Validates consistency across multiple GNN schema formats."""
    
    def __init__(self, gnn_module_path: Optional[Path] = None):
        if gnn_module_path is None:
            gnn_module_path = Path(__file__).parent
        
        self.gnn_path = gnn_module_path
        self.format_validators = {}
        self._initialize_validators()
    
    def _initialize_validators(self):
        """Initialize validators for different schema formats."""
        # JSON Schema validator
        json_schema_path = self.gnn_path / "gnn_schema.json"
        if json_schema_path.exists():
            self.format_validators['json'] = GNNValidator(json_schema_path)
        
        # Additional format validators can be added here
        self.available_formats = list(self.format_validators.keys())
    
    def validate_cross_format_consistency(self, gnn_content: str) -> CrossFormatValidationResult:
        """Validate GNN content against all available schema formats."""
        result = CrossFormatValidationResult(is_consistent=True)
        result.schema_formats = self.available_formats.copy()
        
        # Validate against each format
        format_validation_results = {}
        
        for format_name, validator in self.format_validators.items():
            try:
                # Create temporary file for validation
                temp_file = self._create_temp_file(gnn_content)
                validation_result = validator.validate_file(temp_file)
                format_validation_results[format_name] = validation_result
                result.format_results[format_name] = validation_result
            except Exception as e:
                result.inconsistencies.append(f"Error validating {format_name} format: {str(e)}")
                result.is_consistent = False
        
        # Check for consistency between formats
        self._check_format_consistency(format_validation_results, result)
        
        # Add comprehensive metadata
        result.metadata = self._generate_metadata(format_validation_results)
        
        return result
    
    def _check_format_consistency(self, format_results: Dict[str, ValidationResult], 
                                 cross_result: CrossFormatValidationResult):
        """Check consistency between validation results from different formats."""
        if len(format_results) < 2:
            return  # Need at least 2 formats to check consistency
        
        # Get validation outcomes
        outcomes = {fmt: result.is_valid for fmt, result in format_results.items()}
        
        # Check if all formats agree on validity
        if len(set(outcomes.values())) > 1:
            valid_formats = [fmt for fmt, valid in outcomes.items() if valid]
            invalid_formats = [fmt for fmt, valid in outcomes.items() if not valid]
            
            cross_result.inconsistencies.append(
                f"Formats disagree on validity: Valid={valid_formats}, Invalid={invalid_formats}"
            )
            cross_result.is_consistent = False
        
        # Check for common error patterns
        all_errors = {}
        for fmt, result in format_results.items():
            all_errors[fmt] = set(result.errors)
        
        # Find errors that appear in some formats but not others
        if len(all_errors) > 1:
            all_error_sets = list(all_errors.values())
            common_errors = set.intersection(*all_error_sets)
            unique_errors = {}
            
            for fmt, errors in all_errors.items():
                unique_errors[fmt] = errors - common_errors
                if unique_errors[fmt]:
                    cross_result.warnings.append(
                        f"Format {fmt} reports unique errors: {list(unique_errors[fmt])}"
                    )
    
    def _generate_metadata(self, format_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive metadata about cross-format validation."""
        metadata = {
            'formats_tested': list(format_results.keys()),
            'total_errors': sum(len(result.errors) for result in format_results.values()),
            'total_warnings': sum(len(result.warnings) for result in format_results.values()),
            'format_summary': {}
        }
        
        for fmt, result in format_results.items():
            metadata['format_summary'][fmt] = {
                'valid': result.is_valid,
                'error_count': len(result.errors),
                'warning_count': len(result.warnings),
                'has_metadata': bool(result.metadata)
            }
        
        return metadata
    
    def _create_temp_file(self, content: str) -> Path:
        """Create temporary file with GNN content."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    def validate_schema_definitions_consistency(self) -> CrossFormatValidationResult:
        """Validate that schema definition files themselves are consistent."""
        result = CrossFormatValidationResult(is_consistent=True)
        
        # Load schema definitions
        schemas = {}
        
        try:
            # JSON Schema
            json_path = self.gnn_path / "gnn_schema.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    schemas['json'] = json.load(f)
            
            # YAML Schema
            yaml_path = self.gnn_path / "gnn_schema.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    schemas['yaml'] = yaml.safe_load(f)
            
            result.schema_formats = list(schemas.keys())
            
            # Check structural consistency
            self._validate_schema_structure_consistency(schemas, result)
            
        except Exception as e:
            result.inconsistencies.append(f"Error loading schema definitions: {str(e)}")
            result.is_consistent = False
        
        return result
    
    def _validate_schema_structure_consistency(self, schemas: Dict[str, Dict], 
                                             result: CrossFormatValidationResult):
        """Validate structural consistency between schema definitions."""
        if 'json' in schemas and 'yaml' in schemas:
            json_schema = schemas['json']
            yaml_schema = schemas['yaml'] 
            
            # Check required sections consistency
            json_required = set(json_schema.get('required', []))
            yaml_required = set()
            
            # Extract required sections from YAML schema
            if 'required_sections' in yaml_schema:
                yaml_required = set(yaml_schema['required_sections'])
            
            if json_required != yaml_required:
                result.inconsistencies.append(
                    f"Required sections differ: JSON={json_required}, YAML={yaml_required}"
                )
                result.is_consistent = False
            
            # Check property definitions consistency
            json_props = set(json_schema.get('properties', {}).keys())
            yaml_sections = set()
            
            if 'sections' in yaml_schema:
                yaml_sections = set(yaml_schema['sections'].keys())
            
            if json_props and yaml_sections and json_props != yaml_sections:
                result.warnings.append(
                    f"Property definitions may differ between JSON and YAML schemas"
                )


def validate_cross_format_consistency(gnn_content: str) -> CrossFormatValidationResult:
    """Convenience function for cross-format validation."""
    validator = CrossFormatValidator()
    return validator.validate_cross_format_consistency(gnn_content)


def validate_schema_consistency() -> CrossFormatValidationResult:
    """Convenience function for schema definition consistency validation."""
    validator = CrossFormatValidator()
    return validator.validate_schema_definitions_consistency()


# CLI interface for cross-format validation
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Format GNN Schema Validation")
    parser.add_argument("--file", type=str, help="GNN file to validate")
    parser.add_argument("--check-schemas", action="store_true", 
                       help="Check schema definition consistency")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.check_schemas:
        result = validate_schema_consistency()
        print(f"Schema Consistency: {'CONSISTENT' if result.is_consistent else 'INCONSISTENT'}")
        
        if result.inconsistencies:
            print("\nInconsistencies:")
            for inconsistency in result.inconsistencies:
                print(f"  - {inconsistency}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
    
    elif args.file:
        with open(args.file, 'r') as f:
            content = f.read()
        
        result = validate_cross_format_consistency(content)
        print(f"Cross-Format Validation: {'CONSISTENT' if result.is_consistent else 'INCONSISTENT'}")
        
        if args.verbose:
            print(f"\nFormats tested: {result.schema_formats}")
            print(f"Metadata: {result.metadata}")
        
        if result.inconsistencies:
            print("\nInconsistencies:")
            for inconsistency in result.inconsistencies:
                print(f"  - {inconsistency}")
    
    else:
        parser.print_help() 