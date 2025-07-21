"""
GNN (Generalized Notation Notation) Core Module

This package provides comprehensive tools for working with GNN model specifications,
including parsing, validation, formal verification, and cross-format consistency checking.
"""

from pathlib import Path
from typing import Dict, List, Any, Union
import re # Added for lightweight processing

# Core validation and parsing
from .schema_validator import (
    GNNValidator, GNNParser, ValidationResult, ParsedGNN,
    GNNVariable, GNNConnection, ValidationLevel, GNNSyntaxError,
    validate_gnn_file
)

# Import parsers for additional functionality
try:
    from .parsers import parse_gnn_file as parse_gnn_file_parser
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False

# Cross-format validation
try:
    from .cross_format_validator import (
        CrossFormatValidator, CrossFormatValidationResult,
        validate_cross_format_consistency, validate_schema_consistency
    )
    CROSS_FORMAT_AVAILABLE = True
except ImportError:
    CROSS_FORMAT_AVAILABLE = False

# Formal parsing (optional dependency)
# Lark parser removed - too complex and not needed
FORMAL_PARSER_AVAILABLE = False

# Provide stub classes for graceful degradation
class GNNFormalParser:
    """Stub class for when Lark is not available."""
    def __init__(self): pass
    def parse_file(self, file_path): return None
    def parse_content(self, content, source_name="<string>"): return None
    def validate_syntax(self, content): return False, ["Lark not available"]
    def visualize_parse_tree(self, content): return "Lark not available"

class ParsedGNNFormal:
    """Stub class for when Lark is not available."""
    def __init__(self): pass

def parse_gnn_formal(file_path): return None
def validate_gnn_syntax_formal(content): return False, ["Lark not available"]
def get_parse_tree_visualization(content): return "Lark not available"

# MCP integration
try:
    from .mcp import (
        get_gnn_documentation, validate_gnn_content,
        parse_gnn_content, analyze_gnn_model
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Module metadata
__version__ = "1.1.0"
__author__ = "Active Inference Institute"
__description__ = "Generalized Notation Notation for Active Inference models"

# Feature availability flags
FEATURES = {
    'core_validation': True,
    'formal_parsing': FORMAL_PARSER_AVAILABLE,
    'cross_format_validation': CROSS_FORMAT_AVAILABLE,
    'mcp_integration': MCP_AVAILABLE,
    'parsers': PARSERS_AVAILABLE
}

# Main API functions
__all__ = [
    # Core validation
    'validate_gnn_file',
    'GNNValidator',
    'GNNParser',
    'ValidationResult',
    'ParsedGNN',
    
    # File discovery and processing
    'discover_gnn_files',
    'parse_gnn_file',
    'validate_gnn_structure',
    'process_gnn_directory',
    'generate_gnn_report',
    
    # Cross-format validation (if available)
    'validate_cross_format_consistency',
    'validate_schema_consistency',
    
    # Formal parsing (if available)
    'parse_gnn_formal',
    'validate_gnn_syntax_formal',
    
    # Metadata
    'FEATURES',
    '__version__',
    'get_module_info',
    'validate_gnn'
]

# Add conditional exports
if CROSS_FORMAT_AVAILABLE:
    __all__.extend(['CrossFormatValidator', 'CrossFormatValidationResult'])

if FORMAL_PARSER_AVAILABLE:
    __all__.extend(['GNNFormalParser', 'ParsedGNNFormal', 'get_parse_tree_visualization'])

if MCP_AVAILABLE:
    __all__.extend(['get_gnn_documentation', 'validate_gnn_content', 'parse_gnn_content', 'analyze_gnn_model'])

# =============================================================================
# File Discovery and Processing Functions
# =============================================================================

def discover_gnn_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Discover GNN files in a directory.
    
    Args:
        directory: Directory to search for GNN files
        recursive: Whether to search recursively
        
    Returns:
        List of paths to GNN files
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    # GNN file patterns
    gnn_patterns = ["*.md", "*.gnn", "*.txt"]
    
    files = []
    if recursive:
        for pattern in gnn_patterns:
            files.extend(directory.rglob(pattern))
    else:
        for pattern in gnn_patterns:
            files.extend(directory.glob(pattern))
    
    # Filter for actual GNN files (containing GNN sections)
    gnn_files = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '## GNNSection' in content or '## GNNVersionAndFlags' in content:
                    gnn_files.append(file_path)
        except Exception:
            continue
    
    return gnn_files

def parse_gnn_file(file_path: Union[str, Path]) -> ParsedGNN:
    """
    Parse a GNN file into structured format.
    
    Args:
        file_path: Path to GNN file
        
    Returns:
        ParsedGNN object containing structured data
    """
    file_path = Path(file_path)
    
    if PARSERS_AVAILABLE:
        # Use the unified parser if available
        try:
            result = parse_gnn_file_parser(file_path)
            if result.success:
                # Convert to ParsedGNN format
                return _convert_parse_result_to_parsed_gnn(result)
        except Exception:
            pass
    
    # Fallback to schema validator parser
    parser = GNNParser()
    return parser.parse_file(file_path)

def validate_gnn_structure(file_path: Union[str, Path]) -> ValidationResult:
    """
    Validate GNN file structure.
    
    Args:
        file_path: Path to GNN file
        
    Returns:
        ValidationResult with validation status
    """
    file_path = Path(file_path)
    validator = GNNValidator()
    return validator.validate_file(file_path)

def process_gnn_directory(directory: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
    """
    Process all GNN files in a directory with lightweight processing.
    
    Args:
        directory: Directory to process
        recursive: Whether to process recursively
        
    Returns:
        Dictionary containing processing results
    """
    directory = Path(directory)
    gnn_files = discover_gnn_files(directory, recursive)
    
    results = {
        'directory': str(directory),
        'total_files': len(gnn_files),
        'processed_files': [],
        'errors': [],
        'summary': {}
    }
    
    for file_path in gnn_files:
        try:
            # Lightweight file reading and basic validation
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Quick validation checks
            is_valid = (
                '## ModelName' in content and 
                '## StateSpaceBlock' in content and 
                '## Connections' in content
            )
            
            # Extract basic model information
            model_name_match = re.search(r'## ModelName\n(.+)', content)
            model_name = model_name_match.group(1).strip() if model_name_match else 'Unknown Model'
            
            # Count variables and connections
            variables_count = len(re.findall(r'\w+\[.*?\]', content))
            connections_count = len(re.findall(r'\w+[>-]\w+', content))
            
            file_result = {
                'file_path': str(file_path),
                'model_name': model_name,
                'variables_count': variables_count,
                'connections_count': connections_count,
                'is_valid': is_valid,
                'errors': [],
                'warnings': []
            }
            
            results['processed_files'].append(file_result)
            
        except Exception as e:
            results['errors'].append({
                'file_path': str(file_path),
                'error': str(e)
            })
    
    # Generate summary
    valid_files = [f for f in results['processed_files'] if f['is_valid']]
    results['summary'] = {
        'valid_files': len(valid_files),
        'invalid_files': len(results['processed_files']) - len(valid_files),
        'total_variables': sum(f['variables_count'] for f in results['processed_files']),
        'total_connections': sum(f['connections_count'] for f in results['processed_files'])
    }
    
    return results

def generate_gnn_report(processing_results: Dict[str, Any], output_path: Union[str, Path] = None) -> str:
    """
    Generate a comprehensive report from GNN processing results.
    
    Args:
        processing_results: Results from process_gnn_directory
        output_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    import json
    from datetime import datetime
    
    report_lines = [
        "# GNN Processing Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Directory: {processing_results['directory']}",
        "",
        "## Summary",
        f"- Total files found: {processing_results['total_files']}",
        f"- Valid files: {processing_results['summary']['valid_files']}",
        f"- Invalid files: {processing_results['summary']['invalid_files']}",
        f"- Total variables: {processing_results['summary']['total_variables']}",
        f"- Total connections: {processing_results['summary']['total_connections']}",
        "",
        "## File Details"
    ]
    
    for file_result in processing_results['processed_files']:
        status = "✓" if file_result['is_valid'] else "✗"
        report_lines.extend([
            f"### {status} {file_result['file_path']}",
            f"- Model: {file_result['model_name']}",
            f"- Variables: {file_result['variables_count']}",
            f"- Connections: {file_result['connections_count']}"
        ])
        
        if file_result['errors']:
            report_lines.append("- Errors:")
            for error in file_result['errors']:
                report_lines.append(f"  - {error}")
        
        if file_result['warnings']:
            report_lines.append("- Warnings:")
            for warning in file_result['warnings']:
                report_lines.append(f"  - {warning}")
        
        report_lines.append("")
    
    if processing_results['errors']:
        report_lines.extend([
            "## Processing Errors",
            ""
        ])
        for error in processing_results['errors']:
            report_lines.extend([
                f"### {error['file_path']}",
                f"Error: {error['error']}",
                ""
            ])
    
    report_content = "\n".join(report_lines)
    
    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    return report_content

def _convert_parse_result_to_parsed_gnn(parse_result) -> ParsedGNN:
    """Convert parse result to ParsedGNN format."""
    # This is a simplified conversion - in practice, you'd need to map
    # the parse result structure to ParsedGNN fields
    return ParsedGNN(
        gnn_section=parse_result.model.gnn_section if hasattr(parse_result.model, 'gnn_section') else "",
        version=parse_result.model.version if hasattr(parse_result.model, 'version') else "",
        model_name=parse_result.model.model_name if hasattr(parse_result.model, 'model_name') else "",
        model_annotation=parse_result.model.model_annotation if hasattr(parse_result.model, 'model_annotation') else "",
        variables=parse_result.model.variables if hasattr(parse_result.model, 'variables') else {},
        connections=parse_result.model.connections if hasattr(parse_result.model, 'connections') else [],
        parameters=parse_result.model.parameters if hasattr(parse_result.model, 'parameters') else {},
        equations=parse_result.model.equations if hasattr(parse_result.model, 'equations') else [],
        time_config=parse_result.model.time_config if hasattr(parse_result.model, 'time_config') else {},
        ontology_mappings=parse_result.model.ontology_mappings if hasattr(parse_result.model, 'ontology_mappings') else {},
        model_parameters=parse_result.model.model_parameters if hasattr(parse_result.model, 'model_parameters') else {},
        footer=parse_result.model.footer if hasattr(parse_result.model, 'footer') else ""
    )


def get_module_info():
    """Get comprehensive information about the GNN module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'available_validators': [],
        'available_parsers': [],
        'schema_formats': []
    }
    
    # Check available validators
    info['available_validators'].append('GNNValidator')
    if CROSS_FORMAT_AVAILABLE:
        info['available_validators'].append('CrossFormatValidator')
    
    # Check available parsers
    info['available_parsers'].append('GNNParser')
    if FORMAL_PARSER_AVAILABLE:
        info['available_parsers'].append('GNNFormalParser')
    
    # Check schema formats
    import os
    module_dir = os.path.dirname(__file__)
    schema_files = {
        'JSON': os.path.exists(os.path.join(module_dir, 'schemas/json.json')),
        'YAML': os.path.exists(os.path.join(module_dir, 'schemas/yaml.yaml')),
        'XSD': os.path.exists(os.path.join(module_dir, 'schemas/xsd.xsd')),
        'Protocol Buffers': os.path.exists(os.path.join(module_dir, 'schemas/proto.proto')),
        'Apple Pkl': os.path.exists(os.path.join(module_dir, 'schemas/pkl.pkl')),
        'EBNF Grammar': os.path.exists(os.path.join(module_dir, 'grammars/ebnf.ebnf')),
        'BNF Grammar': os.path.exists(os.path.join(module_dir, 'grammars/bnf.bnf')),
        'Lean': os.path.exists(os.path.join(module_dir, 'formal_specs/lean.lean')),
        'Haskell': os.path.exists(os.path.join(module_dir, 'type_systems/haskell.hs')),
        'Z Notation': os.path.exists(os.path.join(module_dir, 'formal_specs/z_notation.zed')),
        'Alloy': os.path.exists(os.path.join(module_dir, 'formal_specs/alloy.als')),
        'ASN.1': os.path.exists(os.path.join(module_dir, 'schemas/asn1.asn1')),
        'Agda': os.path.exists(os.path.join(module_dir, 'formal_specs/agda.agda'))
    }
    
    info['schema_formats'] = [fmt for fmt, exists in schema_files.items() if exists]
    
    return info


def validate_gnn(file_path_or_content, validation_level=ValidationLevel.STANDARD, 
                use_formal_parser=True, cross_format_check=False):
    """
    Comprehensive GNN validation with multiple options.
    
    Args:
        file_path_or_content: Path to GNN file or GNN content as string
        validation_level: Validation strictness level
        use_formal_parser: Whether to use formal Lark parser if available
        cross_format_check: Whether to perform cross-format validation
    
    Returns:
        ValidationResult or CrossFormatValidationResult
    """
    import os
    
    # Determine if input is file path or content
    if os.path.exists(str(file_path_or_content)):
        # File path
        if cross_format_check and CROSS_FORMAT_AVAILABLE:
            with open(file_path_or_content, 'r') as f:
                content = f.read()
            return validate_cross_format_consistency(content)
        else:
            validator = GNNValidator(use_formal_parser=use_formal_parser)
            return validator.validate_file(file_path_or_content)
    else:
        # Content string
        if cross_format_check and CROSS_FORMAT_AVAILABLE:
            return validate_cross_format_consistency(file_path_or_content)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(file_path_or_content)
                temp_path = f.name
            
            validator = GNNValidator(use_formal_parser=use_formal_parser)
            result = validator.validate_file(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            return result 