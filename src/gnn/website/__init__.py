"""
Website module for GNN Processing Pipeline.

This module provides static HTML website generation from pipeline artifacts.
"""

from typing import Any

from gnn import __version__

from .collection import (
    PURE_DICT_KEYS,
    collect_website_data,
    website_data_from_dict,
)
from .generator import (
    PIPELINE_STEPS,
    StepInfo,
    WebsiteGenerator,
    generate_website,
    get_pipeline_steps,
)
from .inspection import inspect_website, list_website_pages
from .pages import SITE_PAGES, is_valid_page, page_count, page_names
from .renderer import (
    SUPPORTED_FILE_TYPES,
    WebsiteRenderer,
    embed_html_file,
    embed_image,
    embed_json_file,
    embed_markdown_file,
    embed_text_file,
    generate_html_report,
    get_module_info,
    get_supported_file_types,
    process_website,
    validate_website_config,
)

# Feature flags/constants expected by tests
FEATURES: dict[str, Any] = {
    "html": True,
    "embedding": True,
    "basic_processing": True,
    "mcp_integration": True,
    "multi_page": True,
    "dark_mode": True,
    "premium_design": True,
}

__all__: list[Any] = [
    "WebsiteGenerator",
    "WebsiteRenderer",
    "SUPPORTED_FILE_TYPES",
    "generate_website",
    "process_website",
    "generate_html_report",
    "embed_image",
    "embed_markdown_file",
    "embed_text_file",
    "embed_json_file",
    "embed_html_file",
    "get_module_info",
    "get_supported_file_types",
    "validate_website_config",
    "collect_website_data",
    "website_data_from_dict",
    "PURE_DICT_KEYS",
    "get_pipeline_steps",
    "PIPELINE_STEPS",
    "StepInfo",
    "inspect_website",
    "list_website_pages",
    "page_count",
    "page_names",
    "is_valid_page",
    "SITE_PAGES",
    "__version__",
]
