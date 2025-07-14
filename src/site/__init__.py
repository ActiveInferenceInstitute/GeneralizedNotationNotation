"""
GNN Site Generation Module

This package provides tools for generating comprehensive HTML reports from GNN pipeline outputs,
including visualizations, exports, execution results, and other artifacts.
"""

# Core site generation functions
from .generator import (
    generate_html_report,
    main_site_generator,
    embed_image,
    embed_markdown_file,
    embed_text_file,
    embed_json_file,
    embed_html_file,
    process_directory_generic,
    make_section_id,
    add_collapsible_section
)

# MCP integration
try:
    from .mcp import (
        register_tools,
        generate_pipeline_summary_site_mcp
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "HTML site generation from GNN pipeline outputs"

# Feature availability flags
FEATURES = {
    'html_generation': True,
    'image_embedding': True,
    'markdown_rendering': True,
    'json_rendering': True,
    'text_rendering': True,
    'collapsible_sections': True,
    'mcp_integration': MCP_AVAILABLE
}

# Main API functions
__all__ = [
    # Core generation functions
    'generate_html_report',
    'main_site_generator',
    
    # Embedding functions
    'embed_image',
    'embed_markdown_file',
    'embed_text_file',
    'embed_json_file',
    'embed_html_file',
    
    # Utility functions
    'process_directory_generic',
    'make_section_id',
    'add_collapsible_section',
    
    # MCP integration (if available)
    'register_tools',
    'generate_pipeline_summary_site_mcp',
    
    # Metadata
    'FEATURES',
    '__version__'
]

# Add conditional exports
if not MCP_AVAILABLE:
    __all__.remove('register_tools')
    __all__.remove('generate_pipeline_summary_site_mcp')


def get_module_info():
    """Get comprehensive information about the site module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'supported_formats': [],
        'embedding_capabilities': []
    }
    
    # Supported formats
    info['supported_formats'].extend(['HTML', 'Markdown', 'JSON', 'Text', 'Images'])
    
    # Embedding capabilities
    info['embedding_capabilities'].extend([
        'Base64 image embedding',
        'Markdown to HTML conversion',
        'JSON pretty-printing',
        'Text file embedding',
        'HTML file embedding',
        'Collapsible sections',
        'Table of contents generation',
        'Navigation menu generation'
    ])
    
    return info


def generate_site_from_pipeline_output(pipeline_output_dir: str, output_filename: str = "pipeline_summary.html", 
                                     include_images: bool = True, include_logs: bool = True) -> dict:
    """
    Generate a comprehensive HTML site from pipeline output directory.
    
    Args:
        pipeline_output_dir: Path to the pipeline output directory
        output_filename: Name of the output HTML file
        include_images: Whether to embed images as base64
        include_logs: Whether to include log files
    
    Returns:
        Dictionary with generation result information
    """
    from pathlib import Path
    
    try:
        output_dir = Path(pipeline_output_dir)
        if not output_dir.exists():
            return {
                "success": False,
                "error": f"Pipeline output directory does not exist: {pipeline_output_dir}"
            }
        
        output_file = output_dir / output_filename
        
        # Generate the HTML report
        generate_html_report(output_dir, output_file)
        
        return {
            "success": True,
            "input_directory": str(output_dir),
            "output_file": str(output_file),
            "message": f"HTML site generated successfully at {output_file}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "input_directory": pipeline_output_dir,
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_supported_file_types() -> dict:
    """Get information about supported file types for embedding."""
    return {
        'images': {
            'extensions': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'embedding': 'Base64 encoding',
            'description': 'Images are embedded directly in HTML using base64 encoding'
        },
        'markdown': {
            'extensions': ['.md', '.markdown'],
            'embedding': 'HTML conversion',
            'description': 'Markdown files are converted to HTML using markdown library'
        },
        'json': {
            'extensions': ['.json'],
            'embedding': 'Pretty-printed',
            'description': 'JSON files are pretty-printed in pre-formatted blocks'
        },
        'text': {
            'extensions': ['.txt', '.log', '.out'],
            'embedding': 'Pre-formatted',
            'description': 'Text files are embedded in pre-formatted blocks'
        },
        'html': {
            'extensions': ['.html', '.htm'],
            'embedding': 'Iframe embedding',
            'description': 'HTML files are embedded using iframes'
        }
    } 