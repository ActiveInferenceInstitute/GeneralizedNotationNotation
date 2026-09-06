# Specification: GNN Language Grammar

## Scope
Detailed grammar and lexical rules for the GNN language. Complements the
normative specification in [`../gnn_syntax.md`](../gnn_syntax.md) by providing
low-level grammars for variable declarations, connections, and
quick-reference symbol tables.

## Contents
| File | Purpose |
|------|---------|
| `gnn_variable_grammar.md` | Variable declaration grammar (types, dimensions, subscripts) |
| `gnn_connection_grammar.md` | Connection operator grammar (`>`, `-`, `(...)` tuples) |
| `gnn_syntax_quickref.md` | One-page quick reference for most-used symbols |

## Precedence
[`doc/gnn/gnn_syntax.md`](../gnn_syntax.md) is the **normative specification**
and wins every conflict. The `reference/` subtree expands it with worked
examples and validation commands; files here provide grammatical detail too
granular for the spec itself. Neither overrides the spec.

## Status
Maintained. Grammar updates here must be mirrored in
`src/gnn/parsers/markdown_parser.py` tests.
