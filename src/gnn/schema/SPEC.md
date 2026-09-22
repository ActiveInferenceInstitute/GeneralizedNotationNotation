# gnn.schema — Lightweight GNN Document Schema and Validation

## Overview

A second, deliberately lightweight parsing/validation surface for GNN
documents, independent of the formal parser in `gnn.schema_validator`:
section checks, connection-edge extraction, state-space parsing, and
parameterization-vs-declaration shape cross-checks, plus a JSON Schema for
parsed model objects. Import-light by design (stdlib only at module scope)
so headless consumers (`lsp`, `cli` watcher, `extract`) can use it without
pulling pipeline weight.

## Components

- `parser.py` — implementation module:
  - `GNNParseError` (dataclass: `code`, `message`, optional `line`) and
    `GNNVariable` (name/type/dimensions/params), `GNNConnectionEdge`
    (src/tgt + optional annotations)
  - `REQUIRED_SECTIONS` and `validate_required_sections(content)` — section
    presence checks with error codes (`GNN-E001`-style)
  - `GNN_MODEL_SCHEMA` — JSON Schema dict for a single parsed GNN model
  - `validate_gnn_object(obj)` — validate a dict against the schema
  - `parse_connections(content)` — extract edges with optional annotations
  - `parse_state_space(content)` — state-space block parsing
  - `validate_matrix_dimensions(...)` — cross-check parameterization vs
    dimension declarations
- `__init__.py` — thin re-export surface (no logic)

## Invariants

- Import-light: module scope stays stdlib-only; heavier dependencies are
  never pulled in transitively through `gnn.schema`.
- Independent of `gnn.schema_validator`/`gnn.parsers`: no import in either
  direction (this is the lightweight surface; the formal parser lives
  elsewhere).
- All validators return structured results (`GNNParseError` lists) rather
  than raising on the first problem.

## Key Exports

```python
from gnn.schema import (
    GNN_MODEL_SCHEMA,
    REQUIRED_SECTIONS,
    GNNConnectionEdge,
    GNNParseError,
    GNNVariable,
    parse_connections,
    parse_state_space,
    validate_gnn_object,
    validate_matrix_dimensions,
    validate_required_sections,
)
```

## Consumers

- `gnn.lsp` (watcher/diagnostics), `gnn.cli`, `gnn.extract` — headless
  consumers importing the light surface
- Downstream tooling validating GNN documents without the full pipeline