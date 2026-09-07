# gnn.schema

Lightweight parsing and validation for GNN documents: required-section
checks, connection-edge and state-space parsing, matrix-dimension
cross-validation, and JSON Schema validation of parsed model objects.

```python
from gnn.schema import parse_state_space, validate_required_sections
```

Importing `gnn.schema` is stdlib-light at module scope (headless-safe); see
`AGENTS.md` for the contract and `parser.py` for the implementation.
