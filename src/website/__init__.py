"""
Website module for GNN Processing Pipeline.

This module provides static HTML website generation from pipeline artifacts.
"""

from .generator import WebsiteGenerator, generate_website
from .renderer import (
    WebsiteRenderer,
    process_website,
    generate_html_report,
    embed_image,
    embed_markdown_file,
    embed_text_file,
    embed_json_file,
    embed_html_file,
    get_module_info,
    get_supported_file_types,
    validate_website_config
)

__version__ = "1.0.0"

# Feature flags/constants expected by tests
FEATURES = {
    "html": True,
    "embedding": True,
    "basic_processing": True,
}
SUPPORTED_FILE_TYPES = {
    "html": ["html", "htm", "css", "js"],
    "text": ["md", "markdown", "txt", "rst"],
    "markdown": ["md", "markdown"],
    "json": ["json"],
    "data": ["json", "yaml", "yml", "csv"],
    "images": ["png", "jpg", "jpeg", "gif", "svg"]
}

__all__ = [
    'WebsiteGenerator',
    'WebsiteRenderer',
    'generate_website',
    'process_website',
    'generate_html_report',
    'embed_image',
    'embed_markdown_file',
    'embed_text_file',
    'embed_json_file',
    'embed_html_file',
    'get_module_info',
    'get_supported_file_types',
    'validate_website_config',
    '__version__'
]

