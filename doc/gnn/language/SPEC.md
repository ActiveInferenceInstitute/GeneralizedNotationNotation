# Specification: GNN Language Grammar

## Scope
Detailed grammar and lexical rules for the GNN language. Complements the
canonical syntax reference in [`../reference/gnn_syntax.md`](../reference/gnn_syntax.md)
by providing low-level grammars for variable declarations, connections,
and quick-reference symbol tables.

## Contents
| File | Purpose |
|------|---------|
| `gnn_variable_grammar.md` | Variable declaration grammar (types, dimensions, subscripts) |
| `gnn_connection_grammar.md` | Connection operator grammar (`>`, `-`, `(...)` tuples) |
| `gnn_syntax_quickref.md` | One-page quick reference for most-used symbols |

## Relationship to `reference/`
The `reference/` subtree is the **canonical normative spec**. Files here
provide **grammatical detail** that's too granular for the main spec.
When they conflict, `reference/gnn_syntax.md` wins.

## Status
Maintained. Grammar updates here must be mirrored in
`src/gnn/parsers/markdown_parser.py` tests.
