#!/usr/bin/env python3
"""
GNN YAML Front-Matter Parser — Structured metadata extraction.

Extracts optional YAML front-matter (between `---` delimiters) from GNN files.
Supported fields: author, version, framework_targets, tags, created, description.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Regex for YAML front-matter block (must be at the very start of file)
_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)

# Known front-matter fields with types
_KNOWN_FIELDS = {
    "author": str,
    "version": str,
    "framework_targets": list,
    "tags": list,
    "created": str,
    "updated": str,
    "description": str,
    "license": str,
    "source": str,
}


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML front-matter from the beginning of a GNN file.

    Args:
        content: Full file content string.

    Returns:
        Tuple of (metadata_dict, remaining_content).
        If no front-matter is found, metadata_dict is empty and
        remaining_content is the original content.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    yaml_block = match.group(1)
    remaining = content[match.end():]

    metadata = _parse_yaml_simple(yaml_block)
    if metadata:
        logger.debug(f"Front-matter: {list(metadata.keys())}")
    return metadata, remaining


def _parse_yaml_simple(yaml_text: str) -> Dict[str, Any]:
    """
    Simple YAML parser for front-matter (avoids PyYAML dependency).

    Handles:
      - key: value (strings)
      - key: [item1, item2] (inline lists)
      - key:
        - item1
        - item2 (block lists)
    """
    result: Dict[str, Any] = {}

    # Try PyYAML first if available
    try:
        import yaml
        parsed = yaml.safe_load(yaml_text)
        if isinstance(parsed, dict):
            return parsed
    except ImportError:
        pass  # yaml not installed, use manual parser
    except Exception as e:
        logger.debug(f"yaml.safe_load failed, falling back to manual parser: {e}")

    # Recovery: manual parsing
    current_key: Optional[str] = None
    current_list: list = []

    for line in yaml_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Block list item
        if stripped.startswith("- ") and current_key:
            current_list.append(stripped[2:].strip().strip("\"'"))
            result[current_key] = current_list
            continue

        # Key-value pair
        if ":" in stripped:
            # Flush previous list key
            current_key = None
            current_list = []

            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if not value:
                # Start of a block list
                current_key = key
                current_list = []
                result[key] = []
            elif value.startswith("[") and value.endswith("]"):
                # Inline list
                items = [item.strip().strip("\"'") for item in value[1:-1].split(",")]
                result[key] = [i for i in items if i]
            else:
                # Scalar value
                result[key] = value.strip("\"'")

    return result


def has_frontmatter(content: str) -> bool:
    """Check if content has YAML front-matter."""
    return bool(_FRONTMATTER_RE.match(content))
