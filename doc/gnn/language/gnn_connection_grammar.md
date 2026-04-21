# GNN Connection Grammar

**Version**: v1.6.0 Engine (Bundle v2.0.0)  
**Status**: Formal specification

---

## Production Rules

```ebnf
connection   = source edge-op target (":" annotation)? comment? ;
source       = identifier ;
target       = identifier ;
edge-op      = ">" | "-" ;
annotation   = (letter | digit | "_")+ ;
identifier   = (letter | "_" | "π" | "'") (letter | digit | "_" | "'" )* ;
comment      = "#" any-char* ;
```

## Edge Types

| Operator | Meaning | Example | Semantics |
|----------|---------|---------|-----------|
| `>` | Directed | `A>B` | A causally influences B |
| `-` | Undirected | `A-B` | Bidirectional association |

## Annotations (v1.1)

Annotations are optional labels after a colon:

```gnn
D>s:prior_initialization     # prior belief over hidden states
A-o:observation_mapping       # likelihood mapping
G>π:policy_selection          # expected free energy → policy
s>o:emission                  # state to observation emission
```

### Annotation Rules

1. Annotations must match `[a-zA-Z0-9_]+` (alphanumeric + underscore).
2. Parsers **must** preserve annotations but **may** ignore them for structural validation.
3. Annotations serve as labels for rendering, documentation, and editor hover.

## Validation Rules

| Rule | Error Code | Description |
|------|------------|-------------|
| Known source | `GNN-E003` | Source variable must be declared in `StateSpaceBlock` |
| Known target | `GNN-E003` | Target variable must be declared in `StateSpaceBlock` |
| Parseable syntax | `GNN-E005` | Line must match connection grammar |
| Undeclared ref | `GNN-W002` | Warning if variable appears only in connections |

## Implementation

- Parser: [`src/gnn/schema.py :: parse_connections()`](../../../src/gnn/schema.py)
- LSP diagnostics: [`src/lsp/__init__.py`](../../../src/lsp/__init__.py) — real-time connection error highlighting
- CLI: `gnn validate <file.md>` runs all connection grammar checks

---

*See [GNN v1.1 Syntax Specification](../gnn_syntax.md) for the canonical reference.*
