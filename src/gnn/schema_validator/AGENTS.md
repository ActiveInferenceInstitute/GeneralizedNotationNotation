# Schema Validator Module - Agent Scaffolding

## Module Overview

**Purpose**: Syntax-level GNN validation — regex parsing, multi-level schema/semantic validation, and cross-format consistency checking

**Pipeline Step**: Step 21 MCP tooling (`21_mcp.py` → `mcp/processors.py`, `mcp/gnn_root.py`) and Steps 8–9 Visualization (`8_visualization.py` / `9_advanced_viz.py` → `visualization/analysis/combined_analysis.py`) consume this module's surface; the `gnn.processing` core (`core_processor.py`) also imports `CrossFormatValidator`

**Category**: Validation / Parsing

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-21

---

## Module Structure

Three concerns, one package (split in 3.2.0 from a former single-file
validator module plus a standalone cross-format module; the two large
modules were decomposed into mixin submodules in 2026-09):

1. `syntax.py` — `GNNParser`: regex-based parsing of GNN source text into a
   `ParsedGNN` structure. No validation policy; parsing only. Facade keeps
   the orchestrator methods (`parse_file`, `parse_content`) and
   `ROUND_TRIP_AVAILABLE`; the section parsers live in `section_parsers.py`
   and format/binary detection in `format_detection.py` (both mixed into
   `GNNParser`).
2. `validator.py` — `GNNValidator` + `validate_gnn_file_comprehensive`: schema validation
   against `schemas/json.json`, semantic checks, optional round-trip testing
   (via `gnn.parsers`), and binary/pickle validation. Imports `GNNParser`
   from `.syntax`. Facade keeps `__init__`, `validate_file`, and the entry
   point; the checks live in mixin submodules: `validation_levels.py`
   (level resolution), `structural_checks.py` (format/structure gates),
   `round_trip_checks.py` (round-trip + cross-format), `semantic_checks.py`
   (semantic/mathematical checks).
3. `cross_format.py` — `CrossFormatValidator`,
   `validate_cross_format_consistency`, `validate_schema_consistency`:
   consistency checks across the rendered output formats.

Shared types (`ValidationLevel`, `ValidationResult`, `ParsedGNN`) come from
`src/gnn/types/` (definitions in `definitions.py`); the package `__init__.py`
re-exports them for convenience.

## Agent Guidance

- Import the public surface from `gnn.schema_validator` (see `__all__`), not
  from the submodules, except for genuinely internal use within the package.
- Do not add new validation policy to `syntax.py`; it is parsing only.
- `FORMAL_PARSER_AVAILABLE` is intentionally `False` (no Lark dependency).

## Test Coverage

```
uv run pytest tests/gnn/test_gnn_validation.py tests/gnn/test_gnn_cross_format_validator.py -q
```

Round-trip consumers: `src/gnn/testing/round_trip_tester.py` (runner) and
`tests/testing/test_round_trip.py` (pytest cases); pipeline error scenarios:
`tests/pipeline/test_pipeline_error_scenarios.py`.
