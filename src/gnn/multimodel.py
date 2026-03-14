#!/usr/bin/env python3
"""
Multi-Model File Support — Split and parse multiple models per GNN file.

Provides:
  - split_models(): splits on `---` horizontal rules
  - parse_multimodel(): parses each block independently
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Horizontal rule separator between models (must be on its own line)
_MODEL_SEPARATOR = re.compile(r"^\s*---+\s*$", re.MULTILINE)


def split_models(content: str) -> List[str]:
    """
    Split a GNN file into individual model blocks.

    Models are separated by `---` horizontal rules. The first block
    may contain YAML front-matter (handled separately by frontmatter.py).

    If there's only one model (no separators beyond optional front-matter),
    returns a single-element list.

    Args:
        content: Full file content.

    Returns:
        List of model content strings.
    """
    # Strip optional front-matter first
    stripped = content
    try:
        from .frontmatter import parse_frontmatter, has_frontmatter
        if has_frontmatter(content):
            _, stripped = parse_frontmatter(content)
    except ImportError as e:
        logger.debug("Frontmatter module not available: %s", e)

    # Split on horizontal rules
    blocks = _MODEL_SEPARATOR.split(stripped)

    # Filter empty blocks
    models = [b.strip() for b in blocks if b.strip()]

    if len(models) > 1:
        logger.info(f"📄 Multi-model file: {len(models)} models found")
    else:
        logger.debug("Single-model file")

    return models


def parse_multimodel(
    content: str,
    file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse all models in a (possibly multi-model) GNN file.

    Args:
        content: Full file content.
        file_path: Optional source file for error reporting.

    Returns:
        List of parsed model dicts, each containing:
          - variables: list of parsed variables
          - connections: list of parsed connections
          - errors: list of parse errors
          - model_index: 0-based index within file
    """
    from .schema import parse_state_space, parse_connections

    model_blocks = split_models(content)
    results = []

    for i, block in enumerate(model_blocks):
        model_file = f"{file_path}#model{i}" if file_path else f"model{i}"

        variables, var_errors = parse_state_space(block, file_path=model_file)
        var_names = {v.name for v in variables}
        connections, conn_errors = parse_connections(
            block, known_variables=var_names, file_path=model_file
        )

        results.append({
            "model_index": i,
            "variables": [
                {"name": v.name, "dimensions": v.dimensions, "dtype": v.dtype, "default": v.default}
                for v in variables
            ],
            "connections": [
                {
                    "source": c.source, "target": c.target,
                    "directed": c.directed, "label": c.label, "line": c.line,
                }
                for c in connections
            ],
            "errors": [str(e) for e in var_errors + conn_errors],
            "variable_count": len(variables),
            "connection_count": len(connections),
        })

    return results
