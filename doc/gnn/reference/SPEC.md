# Specification: GNN Reference Documentation

## Scope
Canonical reference material for the GNN language. This is the normative
source for syntax, file structure, type system, and grammar rules.
Everything outside this subtree treats these files as ground truth.

## Contents
| File | Purpose |
|------|---------|
| `gnn_syntax.md` | **Canonical syntax reference** — section inventory, required sections (v1.6.0), syntax rules |
| `gnn_file_structure_doc.md` | GNN file anatomy: section order, metadata, expected headings |
| `gnn_type_system.md` | Variable type annotations (int, float, bool, matrix, tensor) |
| `gnn_schema.md` | JSON Schema for parsed GNN dicts |
| `gnn_standards.md` | Style and naming conventions |
| `gnn_dsl_manual.md` | DSL symbols and operators table |
| `technical_reference.md` | Deep-dive technical reference |
| `architecture_reference.md` | Pipeline architecture overview |

## Versioning
- GNN language version: tracked in `gnn_syntax.md` header
- Document bundle version: inherits from [`doc/gnn/SPEC.md`](../SPEC.md)
- Package version: v1.6.0 (matches `pyproject.toml`)

## Status
Maintained. Every file in this subtree is referenced from `doc/gnn/README.md`
and should remain consistent with `src/gnn/parsers/markdown_parser.py`.
Changes to the normative syntax MUST update `gnn_syntax.md` AND the parser.
