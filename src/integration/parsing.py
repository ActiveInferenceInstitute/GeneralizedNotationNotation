"""Pure text-extraction primitives for GNN system integration.

Every function here is I/O-light and side-effect free: GNN markdown content
(or a directory to scan) goes in, plain data comes out. These primitives are
the single source of truth for integration parsing — the graph builder, the
processor, and tests all consume them, so a format tweak happens in one place.

Extraction surfaces (mirrors the GNN markdown conventions used across the
pipeline's ``input/gnn_files`` exemplars):

- ``## StateSpaceBlock`` variable declarations (``A[3,3,type=float]``)
- YAML-style component lists (``- name: alpha``)
- ``## Connections`` edges with ``>`` (directional), ``-`` (bidirectional),
  and ``<`` (reverse) operators
- ``$ref: name`` cross-file references
- ``type: Name`` declarations (for undefined-type heuristics)
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Section + token patterns ────────────────────────────────────────────────

_STATE_SECTION_RE = re.compile(
    r"##\s*StateSpaceBlock\s*\n(.*?)(?=\n##\s|\Z)", re.DOTALL
)
_CONNECTIONS_SECTION_RE = re.compile(
    r"##\s*Connections\s*\n(.*?)(?=\n##\s|\Z)", re.DOTALL
)

# Variable declarations like: A[3,3,type=float]  or  f(x)
_VAR_DECL_RE = re.compile(r"(\w+)\s*[\[\(]")
# YAML-style component entries: "- name: alpha"
_YAML_NAME_RE = re.compile(r"^\s*-\s*name:\s*(\w+)", re.MULTILINE)
# Connection operators: > (directional), - (bidirectional), < (reverse)
_CONNECTION_RE = re.compile(r"(\w+)\s*([>\-<])\s*(\w+)")
# Cross-file references: $ref: name
_REF_RE = re.compile(r"\$ref:\s*(\w+)")
# Inline type annotations: type: Name
_TYPE_REF_RE = re.compile(r"type:\s*(\w+)")

#: Built-in scalar/collection type names never reported as undefined types.
BUILTIN_TYPE_NAMES: frozenset[str] = frozenset(
    {"String", "Integer", "Float", "Boolean", "Array", "Object"}
)


def parse_state_components(content: str) -> list[str]:
    """Return component names declared in ``## StateSpaceBlock`` sections."""
    components: list[str] = []
    for section in _STATE_SECTION_RE.findall(content):
        for line in section.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _VAR_DECL_RE.match(line)
            if match:
                components.append(match.group(1))
    return components


def parse_yaml_components(content: str) -> list[str]:
    """Return component names from YAML-style ``- name:`` definitions."""
    return [match.group(1) for match in _YAML_NAME_RE.finditer(content)]


def parse_connections(content: str) -> list[tuple[str, str, str]]:
    """Return ``(source, operator, target)`` triples from ``## Connections``."""
    connections: list[tuple[str, str, str]] = []
    for section in _CONNECTIONS_SECTION_RE.findall(content):
        for line in section.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _CONNECTION_RE.match(line)
            if match:
                connections.append((match.group(1), match.group(2), match.group(3)))
    return connections


def parse_references(content: str) -> list[str]:
    """Return names referenced via ``$ref: name``."""
    return _REF_RE.findall(content)


def parse_type_references(content: str) -> list[str]:
    """Return names referenced via ``type: Name`` annotations."""
    return _TYPE_REF_RE.findall(content)


def undefined_type_names(
    type_names: list[str], known_components: set[str] | frozenset[str]
) -> list[str]:
    """Return capitalized ``type:`` names that resolve to no known component.

    Built-in scalar names (see :data:`BUILTIN_TYPE_NAMES`) and lowercase
    annotations are ignored — the heuristic only flags CamelCase types that
    look like model component references.
    """
    issues: list[str] = []
    for name in type_names:
        if not name or not name[0].isupper():
            continue
        if name in known_components or name in BUILTIN_TYPE_NAMES:
            continue
        issues.append(name)
    return issues


def discover_gnn_files(target_dir: Path) -> list[Path]:
    """Discover GNN markdown files for integration analysis.

    Scans ``target_dir`` for ``*.md`` files, then walks upward (at most five
    levels) looking for a sibling ``input/gnn_files/`` directory whose files
    are appended as well. Results are deduplicated by filename, first
    occurrence winning.
    """
    gnn_files: list[Path] = list(Path(target_dir).glob("*.md"))

    project_root = Path(target_dir)
    for _ in range(5):  # Walk up to find project root
        input_dir = project_root / "input" / "gnn_files"
        if input_dir.exists():
            gnn_files.extend(input_dir.glob("*.md"))
            break
        parent = project_root.parent
        if parent == project_root:
            break
        project_root = parent

    # Deduplicate by filename (first occurrence wins)
    seen_names: set[str] = set()
    unique_files: list[Path] = []
    for gnn_file in gnn_files:
        if gnn_file.name not in seen_names:
            seen_names.add(gnn_file.name)
            unique_files.append(gnn_file)
    return unique_files
