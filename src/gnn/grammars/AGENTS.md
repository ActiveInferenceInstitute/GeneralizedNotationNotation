# GNN Grammars — agent notes

## Purpose

Static **BNF** and **EBNF** grammar files for the GNN surface syntax. They document token and production rules; runtime parsing of GNN Markdown uses [`../parsers/markdown_parser.py`](../parsers/markdown_parser.py) and related modules. Grammar-shaped interchange uses [`../parsers/grammar_parser.py`](../parsers/grammar_parser.py) and [`../parsers/grammar_serializer.py`](../parsers/grammar_serializer.py) (`GNNFormat.BNF` / `GNNFormat.EBNF` share `GrammarSerializer` for output).

## Files

| File | Role |
|------|------|
| `bnf.bnf` | Backus–Naur Form grammar |
| `ebnf.ebnf` | Extended BNF grammar |

SC-27 decision (2026-09-10): `ebnf.ebnf` is **wired as a functional payload** — served to MCP clients via `gnn.mcp.gnn_root.get_gnn_documentation("grammar")` (resource `gnn://documentation/grammar`). `bnf.bnf` is decorative (no code consumer). Both ship in the wheel via the `src/gnn` package include; `tests/test_grammar_spec_payloads.py` asserts resolution.

See **[README.md](README.md)** for Unicode, comment, and Active Inference variable notes.

## Relationship to the rest of `gnn/`

- **Canonical format counts** (23 parsers, 22 serializers, round-trip scope): **[../SPEC.md](../SPEC.md)**.
- **Default round-trip list** in [`../testing/test_round_trip.py`](../testing/test_round_trip.py) includes **BNF**; **EBNF** uses the same serializer machinery but is not a separate row in the default `test_formats` list (see SPEC).
- **Schemas** ([`../schemas/`](../schemas/)) and **documentation** ([`../documentation/`](../documentation/)) complement these grammars for validation and human-readable structure rules.

## Tests

Grammar-related behavior is covered indirectly via parser tests, e.g.:

```bash
uv run --extra dev python -m pytest tests/gnn/test_gnn_parsing.py tests/gnn/test_gnn_parsers_common.py -q
```
