# Language Directory Manifest (`AGENTS.md`)

**Role**: Maintains metadata and architectural manifest for the `language` subdirectory.
**Version**: v2.0.0
**Context**: Part of the GNN (Generalized Notation Notation) documentation ecosystem.

## Directory Identity

- **Path**: `doc/gnn/language/`
- **Purpose**: GNN syntax specifications, formal grammars, and language-level documentation.

## File Index

| File | Purpose |
|------|---------|
| `README.md` | Human-readable entrypoint with links to all language docs |
| `AGENTS.md` | Machine-readable manifest (this file) |
| `gnn_syntax_quickref.md` | One-page syntax cheatsheet (file structure, variables, connections, front-matter) |
| `gnn_variable_grammar.md` | Formal EBNF grammar for variable declarations |
| `gnn_connection_grammar.md` | Formal EBNF grammar for connection edge syntax |

## Dependencies

- Canonical syntax spec: `doc/gnn/reference/gnn_syntax.md`
- Parser implementation: `src/gnn/schema.py`
- LSP diagnostics: `src/lsp/__init__.py`

## Integration

This directory's grammars define the formal parsing rules implemented in `src/gnn/schema.py`. The CLI `gnn validate` command and LSP server both execute these grammar checks.
