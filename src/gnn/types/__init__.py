"""Shared dataclass types for the GNN module (circular-import safe).

Public re-exports live here; definitions live in :mod:`gnn.types.definitions`.
``GNNFormat`` and ``GNNInternalRepresentation`` are re-exported from
:mod:`gnn.parsers.common`, their single authoritative definition.

The import graph is acyclic at runtime: ``gnn.parsers.basic`` imports the
names it needs from :mod:`gnn.types.definitions` (the implementation module)
rather than from this package facade, so no import order between this package
and ``gnn.parsers`` can produce a partially-initialized cycle.
"""

# Single authoritative definitions live in parsers/common.py.
from gnn.parsers.common import (  # noqa: E402  re-export for types consumers
    GNNFormat,
    GNNInternalRepresentation,
)
from gnn.types.definitions import (
    ComprehensiveTestReport,
    GNNConnection,
    GNNSyntaxError,
    GNNVariable,
    ParsedGNN,
    ParseResult,
    RoundTripResult,
    ValidationLevel,
    ValidationResult,
)

__all__ = [
    "ComprehensiveTestReport",
    "GNNConnection",
    "GNNFormat",
    "GNNInternalRepresentation",
    "GNNSyntaxError",
    "GNNVariable",
    "ParseResult",
    "ParsedGNN",
    "RoundTripResult",
    "ValidationLevel",
    "ValidationResult",
]
