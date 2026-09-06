# Schema Validator Module

Syntax-level validation and parsing for GNN model files: a regex-based parser, a
multi-level validator with round-trip and schema checks, and cross-format
consistency verification. This package was split in 3.2.0 from the former
single-file `src/gnn/schema_validator.py` (parser half → `syntax.py`,
validation half → `validator.py`) plus `src/gnn/cross_format_validator.py`
(→ `cross_format.py`).

## Module Structure

```
src/gnn/schema_validator/
├── __init__.py        # Curated public surface (__all__)
├── syntax.py          # GNNParser: regex-based GNN source parser → ParsedGNN
├── validator.py       # GNNValidator + validate_gnn_file: schema, round-trip, semantic checks
├── cross_format.py    # CrossFormatValidator: cross-format consistency validation
├── README.md          # This documentation
└── AGENTS.md          # Agent scaffolding documentation
```

## Usage

```python
from gnn.schema_validator import (
    GNNParser,
    GNNValidator,
    validate_gnn_file,
    validate_cross_format_consistency,
)

# Parse GNN source text into a structured model
parser = GNNParser(enhanced_validation=True)
parsed = parser.parse_file("my_model.gnn")

# Full validation (schema + semantic + optional round-trip)
result = GNNValidator().validate_file("my_model.gnn")
assert result.is_valid

# One-shot convenience
result = validate_gnn_file("my_model.gnn")
```

Notes:

- `FORMAL_PARSER_AVAILABLE` is `False` (the Lark-based formal parser was
  removed); `ROUND_TRIP_AVAILABLE` reflects whether `gnn.parsers` imports
  cleanly. `GNNValidator` degrades accordingly.
- `ValidationLevel`, `ValidationResult`, and `ParsedGNN` live in
  `src/gnn/types.py` and are re-exported here for convenience.

## See Also

- [Package root](../../../README.md)
- [Validation module](../validation/README.md) — semantic/quality validation layered on top
- [Parsers module](../parsers/README.md) — formal parsing system used for round-trip testing
