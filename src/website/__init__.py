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
    'validate_website_config'
]
