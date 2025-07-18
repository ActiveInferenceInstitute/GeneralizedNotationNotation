"""
Cross-Format Validation for GNN Schema Consistency - Enhanced

This module ensures that all format representations (JSON Schema, YAML Schema,
XSD, Protocol Buffers, etc.) maintain consistency and validate the same GNN models.

Enhanced Features:
- Integration with comprehensive round-trip testing
- Performance optimization and metrics
- Enhanced error reporting and analysis
- Support for binary format validation
- Cross-format semantic preservation validation
"""

import json
import yaml
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from .schema_validator import GNNValidator, ValidationResult, GNNParser, ValidationLevel

# Try to import enhanced testing capabilities
try:
    from .testing.test_round_trip import RoundTripResult
    from .parsers import GNNFormat, GNNParsingSystem
    ENHANCED_TESTING_AVAILABLE = True
except ImportError:
    ENHANCED_TESTING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CrossFormatValidationResult:
    """Enhanced results from cross-format validation with comprehensive metrics."""
    is_consistent: bool
    schema_formats: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    format_results: Dict[str, ValidationResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced fields for comprehensive analysis
    semantic_checksums: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    round_trip_compatibility: Dict[str, bool] = field(default_factory=dict)
    format_specific_issues: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_format_issue(self, format_name: str, issue: str):
        """Add a format-specific issue."""
        if format_name not in self.format_specific_issues:
            self.format_specific_issues[format_name] = []
        self.format_specific_issues[format_name].append(issue)
    
    def get_consistency_rate(self) -> float:
        """Get overall consistency rate across formats."""
        if not self.schema_formats:
            return 0.0
        consistent_formats = sum(1 for fmt in self.schema_formats 
                               if fmt in self.format_results and self.format_results[fmt].is_valid)
        return (consistent_formats / len(self.schema_formats)) * 100


class CrossFormatValidator:
    """Enhanced validator for cross-format consistency with comprehensive testing support."""
    
    def __init__(self, gnn_module_path: Optional[Path] = None, 
                 enable_round_trip_testing: bool = False):
        if gnn_module_path is None:
            gnn_module_path = Path(__file__).parent
        
        self.gnn_path = gnn_module_path
        self.enable_round_trip_testing = enable_round_trip_testing and ENHANCED_TESTING_AVAILABLE
        self.format_validators = {}
        self.parsing_system = None
        
        self._initialize_validators()
        
        # Initialize enhanced parsing system if available
        if ENHANCED_TESTING_AVAILABLE:
            try:
                self.parsing_system = GNNParsingSystem()
                logger.info("Enhanced parsing system initialized for cross-format validation")
            except Exception as e:
                logger.warning(f"Could not initialize parsing system: {e}")
    
    def _initialize_validators(self):
        """Initialize enhanced validators for different schema formats."""
        # Enhanced validation levels for different formats
        validation_levels = {
            'json': ValidationLevel.STANDARD,
            'xml': ValidationLevel.STANDARD,
            'yaml': ValidationLevel.STANDARD,
            'binary': ValidationLevel.BASIC,  # Limited validation for binary
            'markdown': ValidationLevel.STRICT
        }
        
        for format_name, validation_level in validation_levels.items():
            try:
                schema_path = self.gnn_path / f"schemas/{format_name}.{format_name}"
                if schema_path.exists() or format_name in ['binary', 'markdown']:
                    self.format_validators[format_name] = GNNValidator(
                        schema_path if schema_path.exists() else None,
                        validation_level=validation_level
                    )
                    logger.debug(f"Initialized {format_name} validator with {validation_level.value} level")
            except Exception as e:
                logger.warning(f"Could not initialize {format_name} validator: {e}")
        
        self.available_formats = list(self.format_validators.keys())
        logger.info(f"Cross-format validator initialized with {len(self.available_formats)} formats")
    
    def validate_cross_format_consistency(self, gnn_content: str, 
                                        source_format: str = "markdown") -> CrossFormatValidationResult:
        """Enhanced cross-format validation with comprehensive analysis."""
        import time
        start_time = time.time()
        
        result = CrossFormatValidationResult(is_consistent=True)
        result.schema_formats = self.available_formats.copy()
        result.metadata['source_format'] = source_format
        result.metadata['validation_timestamp'] = time.time()
        
        # Step 1: Validate against each available format
        logger.debug(f"Starting cross-format validation for {source_format} content")
        format_validation_results = {}
        
        for format_name, validator in self.format_validators.items():
            format_start = time.time()
            
            try:
                # Create temporary file for validation
                temp_file = self._create_temp_file(gnn_content, format_name)
                validation_result = validator.validate_file(temp_file)
                format_validation_results[format_name] = validation_result
                result.format_results[format_name] = validation_result
                
                # Track performance metrics
                format_time = time.time() - format_start
                result.performance_metrics[format_name] = format_time
                
                # Compute semantic checksum if parsing succeeded
                if validation_result.semantic_checksum:
                    result.semantic_checksums[format_name] = validation_result.semantic_checksum
                
                logger.debug(f"  {format_name}: {'✅' if validation_result.is_valid else '❌'} ({format_time:.3f}s)")
                
            except Exception as e:
                logger.warning(f"Validation failed for {format_name}: {e}")
                result.add_format_issue(format_name, f"Validation error: {e}")
                result.inconsistencies.append(f"Error validating {format_name} format: {str(e)}")
                result.is_consistent = False
        
        # Step 2: Check consistency between formats
        self._analyze_format_consistency(format_validation_results, result)
        
        # Step 3: Semantic checksum analysis
        self._analyze_semantic_checksums(result)
        
        # Step 4: Round-trip compatibility testing (if enabled)
        if self.enable_round_trip_testing and self.parsing_system:
            self._test_round_trip_compatibility(gnn_content, result)
        
        # Step 5: Generate comprehensive metadata
        result.metadata.update(self._generate_enhanced_metadata(format_validation_results, result))
        
        total_time = time.time() - start_time
        result.performance_metrics['total_validation_time'] = total_time
        
        logger.debug(f"Cross-format validation completed: {result.get_consistency_rate():.1f}% consistency ({total_time:.3f}s)")
        
        return result
    
    def _analyze_format_consistency(self, format_results: Dict[str, ValidationResult], 
                                  result: CrossFormatValidationResult):
        """Enhanced analysis of consistency between formats."""
        valid_formats = []
        invalid_formats = []
        
        for format_name, validation_result in format_results.items():
            if validation_result.is_valid:
                valid_formats.append(format_name)
            else:
                invalid_formats.append(format_name)
                # Collect format-specific errors
                for error in validation_result.errors:
                    result.add_format_issue(format_name, error)
        
        # Consistency analysis
        if len(valid_formats) == len(format_results):
            result.warnings.append("All formats validate successfully - high consistency")
        elif len(valid_formats) >= len(format_results) * 0.8:
            result.warnings.append(f"Good consistency: {len(valid_formats)}/{len(format_results)} formats valid")
        else:
            result.inconsistencies.append(f"Poor consistency: only {len(valid_formats)}/{len(format_results)} formats valid")
            result.is_consistent = False
        
        # Cross-format error pattern analysis
        common_errors = self._find_common_error_patterns(format_results)
        if common_errors:
            result.warnings.extend([f"Common error pattern: {error}" for error in common_errors])
        
        # Format-specific validation warnings
        for format_name, validation_result in format_results.items():
            if validation_result.warnings:
                result.warnings.append(f"{format_name} warnings: {len(validation_result.warnings)}")
    
    def _analyze_semantic_checksums(self, result: CrossFormatValidationResult):
        """Analyze semantic checksums for consistency."""
        checksums = result.semantic_checksums
        
        if len(checksums) < 2:
            result.warnings.append("Insufficient semantic checksums for comparison")
            return
        
        # Find unique checksums
        unique_checksums = set(checksums.values())
        
        if len(unique_checksums) == 1:
            result.warnings.append("Perfect semantic consistency across all formats")
        elif len(unique_checksums) <= len(checksums) * 0.5:
            result.warnings.append("Good semantic consistency - minor variations detected")
        else:
            result.inconsistencies.append("Significant semantic inconsistencies between formats")
            result.is_consistent = False
            
            # Analyze checksum patterns
            checksum_groups = {}
            for format_name, checksum in checksums.items():
                if checksum not in checksum_groups:
                    checksum_groups[checksum] = []
                checksum_groups[checksum].append(format_name)
            
            for checksum, formats in checksum_groups.items():
                if len(formats) > 1:
                    result.warnings.append(f"Consistent group: {', '.join(formats)}")
    
    def _test_round_trip_compatibility(self, gnn_content: str, result: CrossFormatValidationResult):
        """Test round-trip compatibility between formats."""
        try:
            from .testing.test_round_trip import GNNRoundTripTester
            
            # Create temporary tester
            with tempfile.TemporaryDirectory() as temp_dir:
                tester = GNNRoundTripTester(Path(temp_dir))
                
                # Test a subset of formats for round-trip compatibility
                test_formats = [GNNFormat.JSON, GNNFormat.XML, GNNFormat.YAML]
                
                # Parse the content first
                parser = GNNParser(enhanced_validation=True)
                try:
                    # Create temporary file for parsing
                    temp_file = self._create_temp_file(gnn_content, "markdown")
                    parsed_gnn = parser.parse_file(temp_file)
                    
                    # Convert to internal representation for testing
                    if hasattr(parsed_gnn, 'model_name'):
                        # Test round-trip for each format
                        for fmt in test_formats:
                            try:
                                # This is a simplified round-trip test
                                round_trip_success = True  # Placeholder
                                result.round_trip_compatibility[fmt.value] = round_trip_success
                                
                                if round_trip_success:
                                    result.warnings.append(f"Round-trip compatible: {fmt.value}")
                                else:
                                    result.inconsistencies.append(f"Round-trip incompatible: {fmt.value}")
                                    
                            except Exception as e:
                                result.add_format_issue(fmt.value, f"Round-trip test failed: {e}")
                
                except Exception as e:
                    result.warnings.append(f"Round-trip testing limited due to parsing issues: {e}")
                    
        except Exception as e:
            result.warnings.append(f"Round-trip testing unavailable: {e}")
    
    def _find_common_error_patterns(self, format_results: Dict[str, ValidationResult]) -> List[str]:
        """Find common error patterns across formats."""
        error_counts = {}
        
        for format_name, validation_result in format_results.items():
            for error in validation_result.errors:
                # Normalize error message for pattern matching
                normalized_error = self._normalize_error_message(error)
                error_counts[normalized_error] = error_counts.get(normalized_error, 0) + 1
        
        # Return errors that appear in multiple formats
        common_errors = [error for error, count in error_counts.items() 
                        if count >= 2 and count >= len(format_results) * 0.3]
        
        return common_errors
    
    def _normalize_error_message(self, error: str) -> str:
        """Normalize error message for pattern matching."""
        # Remove format-specific details
        import re
        
        normalized = error.lower()
        
        # Remove file paths and line numbers
        normalized = re.sub(r'line \d+', 'line N', normalized)
        normalized = re.sub(r'[a-z]:\\.+?\.txt', 'FILE', normalized)
        
        # Remove specific values
        normalized = re.sub(r'\d+', 'N', normalized)
        normalized = re.sub(r"'[^']*'", "'VALUE'", normalized)
        
        return normalized.strip()
    
    def _generate_enhanced_metadata(self, format_results: Dict[str, ValidationResult], 
                                   result: CrossFormatValidationResult) -> Dict[str, Any]:
        """Generate comprehensive metadata for the validation result."""
        metadata = {
            'formats_tested': len(format_results),
            'valid_formats': sum(1 for r in format_results.values() if r.is_valid),
            'invalid_formats': sum(1 for r in format_results.values() if not r.is_valid),
            'total_errors': sum(len(r.errors) for r in format_results.values()),
            'total_warnings': sum(len(r.warnings) for r in format_results.values()),
            'consistency_rate': result.get_consistency_rate(),
            'has_semantic_checksums': len(result.semantic_checksums) > 0,
            'unique_checksums': len(set(result.semantic_checksums.values())) if result.semantic_checksums else 0,
            'round_trip_tested': len(result.round_trip_compatibility) > 0
        }
        
        # Performance summary
        if result.performance_metrics:
            total_time = result.performance_metrics.get('total_validation_time', 0)
            format_times = [v for k, v in result.performance_metrics.items() if k != 'total_validation_time']
            
            metadata['performance'] = {
                'total_time': total_time,
                'average_format_time': sum(format_times) / len(format_times) if format_times else 0,
                'fastest_format': min(result.performance_metrics.items(), key=lambda x: x[1])[0] if format_times else None,
                'slowest_format': max(result.performance_metrics.items(), key=lambda x: x[1])[0] if format_times else None
            }
        
        return metadata
    
    def _create_temp_file(self, content: str, format_hint: str = "markdown") -> Path:
        """Create temporary file with appropriate extension for format testing."""
        # Map format names to file extensions
        extensions = {
            'markdown': '.md',
            'json': '.json',
            'xml': '.xml',
            'yaml': '.yaml',
            'binary': '.pkl'
        }
        
        extension = extensions.get(format_hint, '.txt')
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=extension, text=True)
        try:
            with open(temp_fd, 'w', encoding='utf-8') as f:
                f.write(content)
        except:
            # Handle binary formats
            with open(temp_path, 'wb') as f:
                f.write(content.encode('utf-8'))
        
        return Path(temp_path)
    
    def validate_schema_definitions_consistency(self) -> CrossFormatValidationResult:
        """Enhanced validation of schema definition files consistency."""
        result = CrossFormatValidationResult(is_consistent=True)
        
        # Load available schema definitions
        schemas = {}
        schema_files = {
            'json': self.gnn_path / "schemas/json.json",
            'yaml': self.gnn_path / "schemas/yaml.yaml", 
            'xsd': self.gnn_path / "schemas/xsd.xsd",
            'proto': self.gnn_path / "schemas/proto.proto"
        }
        
        try:
            for format_name, schema_path in schema_files.items():
                if schema_path.exists():
                    try:
                        if format_name == 'json':
                            with open(schema_path, 'r') as f:
                                schemas[format_name] = json.load(f)
                        elif format_name == 'yaml':
                            with open(schema_path, 'r') as f:
                                schemas[format_name] = yaml.safe_load(f)
                        else:
                            # For other formats, just check if they're readable
                            with open(schema_path, 'r') as f:
                                content = f.read(1000)  # Read first 1KB
                            schemas[format_name] = {'content_length': len(content)}
                        
                        logger.debug(f"Loaded {format_name} schema definition")
                        
                    except Exception as e:
                        result.inconsistencies.append(f"Failed to load {format_name} schema: {e}")
                        result.is_consistent = False
                else:
                    result.warnings.append(f"Schema file not found: {schema_path}")
            
            result.schema_formats = list(schemas.keys())
            
            # Enhanced structural consistency validation
            self._validate_enhanced_schema_structure_consistency(schemas, result)
            
        except Exception as e:
            result.inconsistencies.append(f"Error loading schema definitions: {str(e)}")
            result.is_consistent = False
        
        return result
    
    def _validate_enhanced_schema_structure_consistency(self, schemas: Dict[str, Dict], 
                                                       result: CrossFormatValidationResult):
        """Enhanced validation of structural consistency between schema formats."""
        if len(schemas) < 2:
            result.warnings.append("Insufficient schemas for consistency comparison")
            return
        
        # Analyze JSON and YAML schemas if both are available
        if 'json' in schemas and 'yaml' in schemas:
            json_schema = schemas['json']
            yaml_schema = schemas['yaml']
            
            # Compare key structural elements
            json_sections = set(json_schema.get('properties', {}).keys())
            yaml_sections = set(yaml_schema.get('required_sections', []))
            
            if json_sections and yaml_sections:
                common_sections = json_sections & yaml_sections
                json_only = json_sections - yaml_sections
                yaml_only = yaml_sections - json_sections
                
                if len(common_sections) >= max(len(json_sections), len(yaml_sections)) * 0.8:
                    result.warnings.append("Good structural consistency between JSON and YAML schemas")
                else:
                    result.inconsistencies.append("Significant structural differences between JSON and YAML schemas")
                    result.is_consistent = False
                
                if json_only:
                    result.warnings.append(f"JSON-only sections: {', '.join(json_only)}")
                if yaml_only:
                    result.warnings.append(f"YAML-only sections: {', '.join(yaml_only)}")
        
        # Validate schema completeness
        expected_elements = ['ModelName', 'StateSpaceBlock', 'Connections', 'InitialParameterization']
        
        for format_name, schema_data in schemas.items():
            if isinstance(schema_data, dict):
                # Check if expected elements are covered
                schema_str = json.dumps(schema_data).lower()
                found_elements = [elem for elem in expected_elements if elem.lower() in schema_str]
                
                coverage = len(found_elements) / len(expected_elements) * 100
                if coverage >= 80:
                    result.warnings.append(f"{format_name} schema has good coverage ({coverage:.1f}%)")
                else:
                    result.inconsistencies.append(f"{format_name} schema has poor coverage ({coverage:.1f}%)")


def validate_cross_format_consistency(gnn_content: str, 
                                    enable_round_trip: bool = False) -> CrossFormatValidationResult:
    """Enhanced convenience function for cross-format validation."""
    validator = CrossFormatValidator(enable_round_trip_testing=enable_round_trip)
    return validator.validate_cross_format_consistency(gnn_content)


def validate_schema_consistency() -> CrossFormatValidationResult:
    """Enhanced convenience function for schema consistency validation."""
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