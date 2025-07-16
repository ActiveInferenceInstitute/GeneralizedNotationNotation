"""
Website Generation Module for GNN Pipeline

This module provides comprehensive website generation capabilities for the GNN pipeline,
including static HTML generation, MCP integration, and enhanced visualization features.

Features:
- Static HTML website generation from pipeline artifacts
- Enhanced search functionality with highlighting and real-time filtering
- Collapsible sections and dynamic navigation with smooth animations
- Comprehensive file embedding (images, markdown, JSON, text, HTML, code files)
- Advanced MCP integration with parameter validation and error handling
- Website validation and quality assessment with accessibility scoring
- Responsive design with modern CSS and mobile optimization
- Performance optimization with file size limits and lazy loading
- Comprehensive error handling and logging throughout the pipeline

MCP Tools:
- generate_pipeline_summary_website: Main website generation with enhanced features
- analyze_pipeline_outputs: Comprehensive pipeline output analysis with accessibility metrics
- validate_website_output: Advanced website quality validation with scoring
- get_website_statistics: Detailed website statistics and content analysis

Version: 2.1.0
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .generator import (
    generate_website, 
    generate_html_report,
    embed_image,
    embed_markdown_file,
    embed_text_file,
    embed_json_file,
    embed_html_file
)
from .mcp import (
    generate_pipeline_summary_website_mcp,
    analyze_pipeline_outputs_mcp,
    validate_website_output_mcp,
    get_website_statistics_mcp,
    register_tools
)

# Module metadata
__version__ = "1.1.0"
__author__ = "GNN Pipeline Team"
__description__ = "Enhanced website generation for GNN pipeline outputs with advanced features"

# Feature flags
FEATURES = {
    "search_functionality": True,
    "file_highlighting": True,
    "collapsible_sections": True,
    "dynamic_navigation": True,
    "responsive_design": True,
    "accessibility_features": True,
    "mcp_integration": True,
    "website_validation": True,
    "enhanced_file_embedding": True,
    "performance_optimization": True,
    "error_handling": True,
    "parameter_validation": True,
    "accessibility_scoring": True,
    "file_size_limits": True,
    "real_time_search": True,
    "mobile_optimization": True
}

# Supported file types for embedding
SUPPORTED_FILE_TYPES = {
    "images": [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff", ".ico"],
    "markdown": [".md", ".markdown"],
    "json": [".json"],
    "text": [".txt", ".log", ".csv", ".tsv", ".xml", ".yaml", ".yml"],
    "html": [".html", ".htm"],
    "code": [".py", ".js", ".css", ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".go", ".rs", ".swift", ".kt", ".scala", ".r", ".jl", ".m", ".sh", ".bash", ".zsh", ".fish"]
}

# MCP tool descriptions
MCP_TOOLS = {
    "generate_pipeline_summary_website": {
        "description": "Generate a comprehensive HTML website summarizing GNN pipeline outputs with enhanced features",
        "parameters": {
            "output_dir": "Path to the pipeline output directory (required)",
            "website_output_filename": "Name of the output HTML file (default: index.html)",
            "verbose": "Enable verbose logging (default: false)",
            "include_metadata": "Include generation metadata in the website (default: true)",
            "search_enabled": "Enable search functionality in the website (default: true)",
            "validate_output": "Validate the generated website after creation (default: true)",
            "max_file_size_mb": "Maximum file size in MB to embed (default: 50MB)"
        }
    },
    "analyze_pipeline_outputs": {
        "description": "Analyze pipeline outputs and provide detailed statistics with accessibility metrics",
        "parameters": {
            "output_dir": "Path to the pipeline output directory (required)"
        }
    },
    "validate_website_output": {
        "description": "Validate generated website for quality, accessibility, and performance with scoring",
        "parameters": {
            "website_path": "Path to the generated website HTML file (required)"
        }
    },
    "get_website_statistics": {
        "description": "Get detailed statistics about the generated website including file size and content analysis",
        "parameters": {
            "website_path": "Path to the generated website HTML file (required)"
        }
    }
}

def get_supported_file_types() -> Dict[str, Any]:
    """Get supported file types for website generation."""
    return SUPPORTED_FILE_TYPES.copy()

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive module information."""
    return {
        "version": __version__,
        "description": __description__,
        "features": FEATURES,
        "supported_file_types": SUPPORTED_FILE_TYPES,
        "embedding_capabilities": {
            "images": "Embed images with base64 encoding",
            "markdown": "Convert markdown to HTML and embed",
            "text": "Embed text files with syntax highlighting",
            "json": "Pretty-print and embed JSON files",
            "html": "Embed HTML files with proper escaping"
        },
        "mcp_tools": MCP_TOOLS,
        "main_functions": [
            "generate_website",
            "generate_html_report",
            "generate_pipeline_summary_website_mcp",
            "analyze_pipeline_outputs_mcp",
            "validate_website_output_mcp",
            "get_website_statistics_mcp"
        ]
    }

def validate_website_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate website generation configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validation result with success status and any errors
    """
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ["output_dir"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate output directory
    if "output_dir" in config:
        output_dir = Path(config["output_dir"])
        if not output_dir.exists():
            errors.append(f"Output directory does not exist: {output_dir}")
        elif not output_dir.is_dir():
            errors.append(f"Output path is not a directory: {output_dir}")
    
    # Validate optional fields
    if "website_output_filename" in config:
        filename = config["website_output_filename"]
        if not filename.endswith(".html"):
            warnings.append("Website output filename should end with .html")
    
    if "max_file_size_mb" in config:
        max_size = config["max_file_size_mb"]
        if not isinstance(max_size, (int, float)) or max_size <= 0:
            errors.append("max_file_size_mb must be a positive number")
        elif max_size > 100:
            warnings.append("max_file_size_mb is very large (>100MB), may cause performance issues")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

# Initialize logger
logger = logging.getLogger(__name__)

# Export main functions
__all__ = [
    "generate_website",
    "generate_html_report",
    "generate_pipeline_summary_website_mcp",
    "analyze_pipeline_outputs_mcp",
    "validate_website_output_mcp",
    "get_website_statistics_mcp",
    "register_tools",
    "get_module_info",
    "validate_website_config",
    "FEATURES",
    "SUPPORTED_FILE_TYPES",
    "MCP_TOOLS"
] 