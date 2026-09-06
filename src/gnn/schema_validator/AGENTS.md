# Schema Validator Module - Agent Scaffolding

## Module Overview

**Purpose**: Syntax-level GNN validation — regex parsing, multi-level schema/semantic validation, and cross-format consistency checking

**Pipeline Step**: Step 2: Parse (2_parse.py) and Step 6: Validation (6_validation.py) both consume this module's surface

**Category**: Validation / Parsing

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-06

---

## Module Structure

Three concerns, one package (split from the former single-file
`src/gnn/schema_validator.py` plus `src/gnn/cross_format_validator.py` in 3.2.0):

1. `syntax.py` — `GNNParser`: regex-based parsing of GNN source text into a
   `ParsedGNN` structure. No validation policy; parsing only.
2. `validator.py` — `GNNValidator` + `validate_gnn_file`: schema validation
   against `schemas/json.json`, semantic checks, optional round-trip testing
   (via `gnn.parsers`), and binary/pickle validation. Imports `GNNParser`
   from `.syntax`.
3. `cross_format.py` — `CrossFormatValidator`,
   `validate_cross_format_consistency`, `validate_schema_consistency`:
   consistency checks across the rendered output formats.

Shared types (`ValidationLevel`, `ValidationResult`, `ParsedGNN`) come from
`src/gnn/types.py`; the package `__init__.py` re-exports them for convenience.

## Agent Guidance

- Import the public surface from `gnn.schema_validator` (see `__all__`), not
  from the submodules, except for genuinely internal use within the package.
- Do not add new validation policy to `syntax.py`; it is parsing only.
- `FORMAL_PARSER_AVAILABLE` is intentionally `False` (no Lark dependency).

## Test Coverage

```
uv run pytest tests/gnn/test_gnn_validation.py tests/gnn/test_gnn_cross_format_validator.py -q
```

Round-trip consumers: `src/gnn/testing/test_round_trip.py`,
`src/gnn/testing/test_integration.py`; pipeline error scenarios:
`tests/pipeline/test_pipeline_error_scenarios.py`.
