"""Single source of truth for export formats.

This registry is the canonical mapping between export format names, the
writer functions that produce them (from :mod:`export.formatters`), their
file extensions, and their category (``data`` / ``graph`` / ``text``).

Consumers:
    - :mod:`export.processor` (pipeline dispatch in ``process_export``)
    - :mod:`export.utils` (module introspection surfaces)
    - :mod:`export` package helpers (``get_supported_formats*``,
      ``validate_export_format``)

Public functions: get_export_registry, resolve_format_writer,
get_format_categories, is_supported_format, DEFAULT_PIPELINE_FORMATS
"""

from pathlib import Path
from typing import Any, Callable, Dict, Literal, Tuple, TypedDict

from .formatters import (
    export_to_gexf,
    export_to_graphml,
    export_to_json,
    export_to_pickle,
    export_to_plaintext_dsl,
    export_to_plaintext_summary,
    export_to_xml,
)

#: A format writer takes the model/parse dict and the output path and
#: returns ``True`` on success. This is the ``formatters``-family contract
#: (the ``format_exporters``-family returns ``tuple[bool, str]`` instead).
FormatWriter = Callable[[Dict[str, Any], Path], bool]

#: Categories used by ``get_supported_formats_dict`` and ``get_module_info``.
Category = Literal["data", "graph", "text"]


class ExportFormatSpec(TypedDict):
    """Description of one supported export format."""

    name: str
    category: Category
    extension: str
    writer: FormatWriter
    description: str


#: Pipeline-facing formats (the five written by ``process_export`` and the
#: default for ``export_model``), in canonical output order.
DEFAULT_PIPELINE_FORMATS: Tuple[str, ...] = ("json", "xml", "graphml", "gexf", "pickle")

_FORMAT_SPECS: Tuple[ExportFormatSpec, ...] = (
    ExportFormatSpec(
        name="json",
        category="data",
        extension=".json",
        writer=export_to_json,
        description="Human-readable JSON serialization of the parsed model",
    ),
    ExportFormatSpec(
        name="xml",
        category="data",
        extension=".xml",
        writer=export_to_xml,
        description="Hierarchical XML serialization of the parsed model",
    ),
    ExportFormatSpec(
        name="graphml",
        category="graph",
        extension=".graphml",
        writer=export_to_graphml,
        description="GraphML graph format for network-analysis tools",
    ),
    ExportFormatSpec(
        name="gexf",
        category="graph",
        extension=".gexf",
        writer=export_to_gexf,
        description="GEXF graph format for Gephi-style visualization",
    ),
    ExportFormatSpec(
        name="pickle",
        category="data",
        extension=".pkl",
        writer=export_to_pickle,
        description="Python pickle binary serialization",
    ),
    ExportFormatSpec(
        name="txt",
        category="text",
        extension=".txt",
        writer=export_to_plaintext_summary,
        description="Human-readable plaintext model summary",
    ),
    ExportFormatSpec(
        name="dsl",
        category="text",
        extension=".dsl",
        writer=export_to_plaintext_dsl,
        description="Round-trip GNN-like DSL text",
    ),
)

_REGISTRY: Dict[str, ExportFormatSpec] = {spec["name"]: spec for spec in _FORMAT_SPECS}


def get_export_registry() -> Dict[str, ExportFormatSpec]:
    """Return a copy of the canonical format registry keyed by format name."""
    return dict(_REGISTRY)


def resolve_format_writer(format_name: str) -> FormatWriter | None:
    """Return the writer for ``format_name``, or ``None`` when unsupported."""
    spec = _REGISTRY.get(format_name)
    return spec["writer"] if spec else None


def get_format_categories() -> Dict[str, list[str]]:
    """Return format names grouped by category (data, graph, text)."""
    grouped: Dict[str, list[str]] = {"data": [], "graph": [], "text": []}
    for spec in _FORMAT_SPECS:
        grouped[spec["category"]].append(spec["name"])
    return grouped


def is_supported_format(format_name: str) -> bool:
    """Return ``True`` when ``format_name`` is in the registry."""
    return format_name in _REGISTRY


def get_format_spec(format_name: str) -> ExportFormatSpec | None:
    """Return the full spec for ``format_name``, or ``None`` when unknown."""
    spec = _REGISTRY.get(format_name)
    return spec
