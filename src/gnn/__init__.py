"""
GNN (Generalized Notation Notation) Core Module

This package provides comprehensive tools for working with GNN model specifications,
including parsing, validation, formal verification, and cross-format consistency checking.
"""

# Core validation and parsing
from .schema_validator import (
    GNNValidator, GNNParser, ValidationResult, ParsedGNN,
    GNNVariable, GNNConnection, ValidationLevel, GNNSyntaxError,
    validate_gnn_file
)

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
try:
    from .parsers.lark_parser import (
        GNNFormalParser, ParsedGNNFormal, parse_gnn_formal,
        validate_gnn_syntax_formal, get_parse_tree_visualization
    )
    FORMAL_PARSER_AVAILABLE = True
except ImportError:
    FORMAL_PARSER_AVAILABLE = False

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
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Generalized Notation Notation for Active Inference models"

# Feature availability flags
FEATURES = {
    'core_validation': True,
    'formal_parsing': FORMAL_PARSER_AVAILABLE,
    'cross_format_validation': CROSS_FORMAT_AVAILABLE,
    'mcp_integration': MCP_AVAILABLE
}

# Main API functions
__all__ = [
    # Core validation
    'validate_gnn_file',
    'GNNValidator',
    'GNNParser',
    'ValidationResult',
    'ParsedGNN',
    
    # Cross-format validation (if available)
    'validate_cross_format_consistency',
    'validate_schema_consistency',
    
    # Formal parsing (if available)
    'parse_gnn_formal',
    'validate_gnn_syntax_formal',
    
    # Metadata
    'FEATURES',
    '__version__'
]

# Add conditional exports
if CROSS_FORMAT_AVAILABLE:
    __all__.extend(['CrossFormatValidator', 'CrossFormatValidationResult'])

if FORMAL_PARSER_AVAILABLE:
    __all__.extend(['GNNFormalParser', 'ParsedGNNFormal', 'get_parse_tree_visualization'])

if MCP_AVAILABLE:
    __all__.extend(['get_gnn_documentation', 'validate_gnn_content', 'parse_gnn_content', 'analyze_gnn_model'])


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
        'JSON': os.path.exists(os.path.join(module_dir, 'gnn_schema.json')),
        'YAML': os.path.exists(os.path.join(module_dir, 'gnn_schema.yaml')),
        'XSD': os.path.exists(os.path.join(module_dir, 'gnn_schema.xsd')),
        'Protocol Buffers': os.path.exists(os.path.join(module_dir, 'gnn_schema.proto')),
        'Apple Pkl': os.path.exists(os.path.join(module_dir, 'gnn_schema.pkl')),
        'EBNF Grammar': os.path.exists(os.path.join(module_dir, 'gnn_grammar.ebnf')),
        'BNF Grammar': os.path.exists(os.path.join(module_dir, 'gnn_grammar.bnf')),
        'Lean': os.path.exists(os.path.join(module_dir, 'gnn_category_theory.lean')),
        'Haskell': os.path.exists(os.path.join(module_dir, 'GNN.hs')),
        'Z Notation': os.path.exists(os.path.join(module_dir, 'gnn_schema.zed')),
        'Alloy': os.path.exists(os.path.join(module_dir, 'gnn_schema.als')),
        'ASN.1': os.path.exists(os.path.join(module_dir, 'gnn_schema.asn1')),
        'Agda': os.path.exists(os.path.join(module_dir, 'gnn_type_theory.agda'))
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