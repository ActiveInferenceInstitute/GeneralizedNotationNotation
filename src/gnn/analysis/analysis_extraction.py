#!/usr/bin/env python3
"""
Regex-based structural extraction (variables, connections, sections) for GNN Step 16 analysis.

Extracted from ``analysis.analyzer``.
"""

import re
from typing import (
    Any,
    Dict,
    List,
)


def extract_variables(content: str) -> List[Dict[str, Any]]:
    """Extract variables for statistical analysis."""
    variables: list[dict[str, Any]] = []

    # Look for variable definitions
    var_patterns: list[str] = [
        r"(\w+)\s*:\s*(\w+)",  # name: type
        r"(\w+)\s*=\s*([^;\n]+)",  # name = value
        r"(\w+)\s*\[([^\]]+)\]",  # name[dimensions]
    ]

    for pattern in var_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            variables.append(
                {
                    "name": match.group(1),
                    "definition": match.group(0),
                    "line": content[: match.start()].count("\n") + 1,
                    "type": match.group(2) if ":" in match.group(0) else "unknown",
                }
            )

    return variables


def extract_connections(content: str) -> List[Dict[str, Any]]:
    """Extract connections for statistical analysis."""
    connections: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()  # Deduplicate connections

    # 1. Parse GNN ## Connections section directly (highest priority)
    connections_section = re.search(
        r"##\s*Connections\s*\n(.*?)(?=\n##\s|\Z)", content, re.DOTALL
    )
    if connections_section:
        section_text = connections_section.group(1)
        section_start_line = content[: connections_section.start()].count("\n") + 2
        for i, line in enumerate(section_text.strip().split("\n")):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # GNN connection operators: > (directional), - (bidirectional), < (reverse)
            gnn_match = re.match(r"(\w+)\s*([>\-<])\s*(\w+)", line)
            if gnn_match:
                src, op, tgt = (
                    gnn_match.group(1),
                    gnn_match.group(2),
                    gnn_match.group(3),
                )
                conn_type = (
                    "directional"
                    if op == ">"
                    else "bidirectional"
                    if op == "-"
                    else "reverse"
                )
                key = (src, tgt, conn_type)
                if key not in seen:
                    seen.add(key)
                    connections.append(
                        {
                            "source": src,
                            "target": tgt,
                            "connection": line,
                            "connection_type": conn_type,
                            "line": section_start_line + i,
                        }
                    )

    # 2. Also look for generic connection patterns outside the section
    conn_patterns: list[tuple[str, str]] = [
        (r"(\w+)\s*->\s*(\w+)", "directional"),  # source -> target
        (r"(\w+)\s*→\s*(\w+)", "directional"),  # source → target
        (r"(\w+)\s*connects\s*(\w+)", "association"),  # source connects target
    ]

    for pattern, conn_type in conn_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            key = (match.group(1), match.group(2), conn_type)
            if key not in seen:
                seen.add(key)
                connections.append(
                    {
                        "source": match.group(1),
                        "target": match.group(2),
                        "connection": match.group(0),
                        "connection_type": conn_type,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

    return connections


def extract_sections(content: str) -> List[Dict[str, Any]]:
    """Extract sections for statistical analysis."""
    sections: list[Any] = []

    # Look for section headers
    section_patterns: list[Any] = [
        r"^#+\s+(.+)$",  # Markdown headers
        r"^(\w+):\s*$",  # Section labels
    ]

    for pattern in section_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            sections.append(
                {
                    "name": match.group(1),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

    return sections
