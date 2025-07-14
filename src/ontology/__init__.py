"""
GNN Ontology Processing Module

This package provides tools for parsing, validating, and processing Active Inference ontology
annotations in GNN model specifications.
"""

# Core ontology processing functions
from .mcp import (
    parse_gnn_ontology_section,
    load_defined_ontology_terms,
    validate_annotations,
    generate_ontology_report_for_file,
    get_mcp_interface
)

# MCP integration
try:
    from .mcp import (
        register_tools
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Active Inference ontology processing for GNN models"

# Feature availability flags
FEATURES = {
    'ontology_parsing': True,
    'term_validation': True,
    'report_generation': True,
    'mcp_integration': MCP_AVAILABLE
}

# Main API functions
__all__ = [
    # Core ontology functions
    'parse_gnn_ontology_section',
    'load_defined_ontology_terms',
    'validate_annotations',
    'generate_ontology_report_for_file',
    'get_mcp_interface',
    'process_ontology',
    'validate_ontology_terms',
    'map_gnn_to_ontology',
    'generate_ontology_report',
    
    # MCP integration (if available)
    'register_tools',
    
    # Metadata
    'FEATURES',
    '__version__'
]

# Add conditional exports
if not MCP_AVAILABLE:
    __all__.remove('register_tools')


def get_module_info():
    """Get comprehensive information about the ontology module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'processing_capabilities': [],
        'supported_formats': []
    }
    
    # Processing capabilities
    info['processing_capabilities'].extend([
        'GNN ontology section parsing',
        'Ontology term validation',
        'Markdown report generation',
        'JSON ontology term loading',
        'Annotation mapping validation'
    ])
    
    # Supported formats
    info['supported_formats'].extend(['GNN', 'JSON', 'Markdown'])
    
    return info


def process_gnn_ontology(gnn_file_path: str, ontology_terms_path: str = None, 
                        generate_report: bool = True) -> dict:
    """
    Process ontology annotations in a GNN file.
    
    Args:
        gnn_file_path: Path to the GNN file
        ontology_terms_path: Path to ontology terms JSON file (optional)
        generate_report: Whether to generate a markdown report
    
    Returns:
        Dictionary with processing result information
    """
    from pathlib import Path
    
    try:
        # Read GNN file
        gnn_path = Path(gnn_file_path)
        if not gnn_path.exists():
            return {
                "success": False,
                "error": f"GNN file not found: {gnn_file_path}"
            }
        
        with open(gnn_path, 'r', encoding='utf-8') as f:
            gnn_content = f.read()
        
        # Parse ontology section
        annotations = parse_gnn_ontology_section(gnn_content)
        
        result = {
            "success": True,
            "gnn_file": str(gnn_path),
            "annotations_found": len(annotations),
            "annotations": annotations
        }
        
        # Load and validate ontology terms if provided
        if ontology_terms_path:
            terms_path = Path(ontology_terms_path)
            if terms_path.exists():
                defined_terms = load_defined_ontology_terms(str(terms_path))
                validation_results = validate_annotations(annotations, defined_terms)
                
                result.update({
                    "defined_terms_loaded": len(defined_terms),
                    "validation_results": validation_results,
                    "valid_mappings": len(validation_results.get("valid_mappings", {})),
                    "invalid_terms": len(validation_results.get("invalid_terms", {}))
                })
            else:
                result["warning"] = f"Ontology terms file not found: {ontology_terms_path}"
        
        # Generate report if requested
        if generate_report:
            validation_results = result.get("validation_results", {})
            report = generate_ontology_report_for_file(str(gnn_path), annotations, validation_results)
            result["report"] = report
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "gnn_file": gnn_file_path,
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_ontology_processing_options() -> dict:
    """Get information about ontology processing options."""
    return {
        'parsing_options': {
            'strict_mode': 'Strict parsing with error reporting',
            'lenient_mode': 'Lenient parsing with warnings',
            'comment_handling': 'Handle inline comments in annotations'
        },
        'validation_options': {
            'case_sensitive': 'Case-sensitive term matching',
            'case_insensitive': 'Case-insensitive term matching',
            'fuzzy_matching': 'Fuzzy string matching for similar terms'
        },
        'report_formats': {
            'markdown': 'Markdown formatted reports',
            'json': 'JSON structured reports',
            'html': 'HTML formatted reports'
        },
        'output_options': {
            'detailed': 'Detailed validation reports',
            'summary': 'Summary validation reports',
            'minimal': 'Minimal validation reports'
        }
    }


# Test-compatible function alias
def process_ontology(gnn_file_path, **kwargs):
    """Process ontology (test-compatible alias)."""
    return process_gnn_ontology(gnn_file_path, **kwargs)

def validate_ontology_terms(annotations, defined_terms):
    """Validate ontology terms (test-compatible alias)."""
    return validate_annotations(annotations, defined_terms)

def map_gnn_to_ontology(gnn_file_path, ontology_terms_path=None, **kwargs):
    """Map GNN to ontology (test-compatible alias)."""
    return process_gnn_ontology(gnn_file_path, ontology_terms_path, **kwargs)

def generate_ontology_report(gnn_file_path, ontology_terms_path=None, **kwargs):
    """Generate ontology report (test-compatible alias)."""
    return generate_ontology_report_for_file(gnn_file_path, ontology_terms_path, **kwargs) 