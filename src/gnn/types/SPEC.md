# Types Module Specification

## Overview
Single authoritative home for the shared dataclass types used across the GNN pipeline — the parsed-document model plus validation and round-trip results — kept dependency-light so any consumer can import it safely.

## Components
- `definitions.py` - `ValidationLevel` (BASIC, STANDARD, STRICT, RESEARCH, ROUND_TRIP), `GNNSyntaxError` (line/column context plus `format_message()`), `ValidationResult` (round-trip aggregation with success-rate helper), `GNNVariable` / `GNNConnection` / `ParsedGNN` (the parsed-document model), `ParseResult` (model plus errors and warnings), `RoundTripResult` (per-format pair outcome), `ComprehensiveTestReport` (format-matrix keyed test report)
- `__init__.py` - Public re-export surface (`__all__`, eleven names); runtime re-exports `GNNFormat` / `GNNInternalRepresentation` from `gnn.parsers.common`, their single authoritative definition

## Invariants
- The runtime import graph stays acyclic: `gnn.parsers.basic` imports the names it needs from `definitions.py` (the implementation module) rather than from this package facade, so no import order between `gnn.types` and `gnn.parsers` can produce a partially-initialized cycle.
- Stdlib only at runtime (`dataclasses`, `datetime`, `enum`, `typing`); parser names appear in `definitions.py` only under `TYPE_CHECKING`, with quoted annotations.
- `gnn.parsers.common` is a runtime re-export target only and never runtime-imports back.

## Key Exports
```python
from gnn.types import (
    ComprehensiveTestReport,
    GNNConnection,
    GNNFormat,
    GNNInternalRepresentation,
    GNNSyntaxError,
    GNNVariable,
    ParseResult,
    ParsedGNN,
    RoundTripResult,
    ValidationLevel,
    ValidationResult,
)
```

## Receipts
```bash
uv run --extra dev python -m pytest tests/types/test_definitions.py -q
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
