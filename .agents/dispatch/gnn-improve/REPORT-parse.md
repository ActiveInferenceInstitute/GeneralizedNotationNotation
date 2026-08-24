# REPORT-parse.md — Parse & Core-Authoring Scope

**Mission:** `.agents/dispatch/gnn-improve/mission-parse.md`
**Scope:** `src/gnn/`, `src/model_registry/`, `src/type_checker/`, `src/validation/`, `src/export/` + mirror tests.
**Status:** Complete. All scoped gates green. No commit / push / stage performed (working tree left dirty as required).

---

## Changes made

### 1. `src/gnn/parsers/markdown_parser_parameter.py` — root-cause fix for parameter comment handling
- Added `ParameterParsingMixin._split_inline_comment(line)`: splits a line at its first `#`
  comment delimiter that lies **outside** any bracket (`{}`/`[]`/`()`) or quote region.
- `_parse_parameter_assignment` now uses that helper instead of the nonstandard `"###"`
  marker, matching the single-`#` delimiter used everywhere else in the parser.

**Bugs fixed (confirmed before/after with targeted probes):**
1. **Matrix + trailing inline comment → Python `set` instead of matrix.**
   `A = { (1.0, 0.0), (0.0, 1.0) }  # transition` previously parsed to the set
   `{(1.0, 0.0), (0.0, 1.0)}` because the trailing `# comment` broke the `{...}` matrix
   branch's `endswith("}")` test and pushed the value through `safe_literal_eval`, which
   reads braces as a set **literal**. A set is unordered, non-JSON-serializable, and broke
   JSON export / round-trip. Now parses to the intended `[[...],[...]]` row-major matrix.
2. **Inline comment leaked into bare token values.** `alpha = scaling  # rise correction`
   previously returned the string `"scaling  # rise correction"`. Now returns `"scaling"`
   with `description="rise correction"`.

The helper is quote- and bracket-aware, so a `#` inside a quoted string or inside a `{...}`
matrix row is preserved as data (matching intent), and multiline braced matrices with
per-row `#` comments still parse correctly.

### 2. `src/tests/gnn/test_gnn_parsing.py` — targeted regression tests (5 added)
New `TestParameterParsing` class pinning the invariants above:
- `test_matrix_with_trailing_comment_stays_a_matrix`
- `test_inline_comment_stripped_from_token_value`
- `test_hash_inside_quoted_string_is_preserved`
- `test_hash_inside_matrix_row_is_preserved`
- `test_json_round_trip_of_matrix_parameter` (full parse + `JSONSerializer` round-trip)

Added `import json` to support the round-trip test.

No other files touched. All other modifications in the working tree belong to sibling
agents on disjoint paths and were intentionally left untouched and uncommitted.

---

## Verification (scoped)

Command of record:
`uv run pytest src/tests/gnn src/tests/model_registry src/tests/type_checker src/tests/validation src/tests/export -q --tb=no -x`

- **pytest:** **438 passed, 0 failed** (baseline 433 → +5 new regression tests).
- **ruff check:** `All checks passed!`
- **ruff format --check:** `116 files already formatted`
- **mypy:** `Success: no issues found in 116 source files`

## Notes
- No commit / push / staging performed.
- Kept public API stable; only internal comment-parsing behavior hardened.
- Extra investigation: the `_detect_format_from_content` ASN.1 branch on
  `"ASN1" in content or "BEGIN" in content and "END" in content` evaluates with standard
  Python precedence (`A or (B and C)`), which matches the apparent author intent; a
  "BEGIN-only" probe was actually a test-harness artifact (str vs Path). No change made.