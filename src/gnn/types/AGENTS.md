# gnn.types — Shared GNN Dataclass Types

## Overview

**Purpose:** Single authoritative home for the shared dataclass types used
across the GNN pipeline (parsing, validation, rendering, round-trip testing),
kept dependency-light so any consumer can import it without dragging
pipeline weight.

**Pipeline step:** Cross-cutting (consumed by steps 3, 5, 6, 7, 11 and the
schema validator).

**Category:** Core types.

**Status:** Maintained.

**Version:** 1.0.0

## Core Functionality

1. `ValidationLevel` — enum of validation depths (BASIC, STANDARD, STRICT,
   RESEARCH, ROUND_TRIP).
2. `GNNSyntaxError` — syntax error with line/column and `format_message()`.
3. `ValidationResult` — round-trip aggregation with success-rate helpers.
4. `GNNVariable`, `GNNConnection`, `ParsedGNN` — the parsed-document model.
5. `ParseResult` — parse outcome wrapper (model, errors, warnings).
6. `RoundTripResult` — per-format pair round-trip outcome.
7. `ComprehensiveTestReport` — format-matrix keyed test report.
8. `GNNFormat`, `GNNInternalRepresentation` — re-exported from
   `gnn.parsers.common`, their single authoritative definition.

## Module Structure

- `__init__.py` — public re-export surface (`__all__`), order-agnostic:
  `gnn.parsers.basic` imports from `definitions.py` directly, keeping the
  parsers ↔ types import graph acyclic at runtime.
- `definitions.py` — the dataclass definitions themselves; parser names appear
  only under `TYPE_CHECKING` (annotations are quoted strings).

## Dependencies

- Stdlib only at runtime (`dataclasses`, `datetime`, `enum`, `typing`).
- `gnn.parsers.common` (runtime re-export only, resolved after definitions
  bind; `common` is stdlib-light and never runtime-imports back).

## Testing

```bash
uv run --extra dev python -m pytest tests/gnn/ -q
```
