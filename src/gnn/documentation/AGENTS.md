# GNN Documentation Module

## Purpose

This module contains the **user-facing reference documentation** for the Generalized Notation Notation (GNN) file format. It provides the authoritative specification of GNN syntax and file structure.

## Core Files

| File | Description |
|------|-------------|
| `file_structure.md` | Defines the required and optional sections of a GNN file (`ModelName`, `StateSpaceBlock`, `Connections`, `InitialParameterization`, `Equations`, `Time`, etc.) |
| `punctuation.md` | Reference for all GNN syntax symbols: superscripts (`^`), subscripts (`_`), comments (`#`), edges (`->`), variable definitions (`=`), grouping (`()`, `[]`, `{}`), and operators (`+`, `*`, `/`, `|`) |

## Integration Points

- **Parser** (`gnn/parser.py`, `gnn/schema_validator.py`): Implements the section structure and punctuation rules defined here
- **Parsers** (`gnn/parsers/markdown_parser.py`): Uses section names from `file_structure.md` for Markdown GNN parsing
- **Validation** (`gnn/validation.py`): Checks files against the structural requirements documented here

## For AI Agents

When working with GNN files:

1. **Consult `file_structure.md`** to understand which sections are required vs optional
2. **Consult `punctuation.md`** for the meaning of any GNN symbol
3. **Cross-reference** with `gnn/gnn_examples/actinf_pomdp_agent.md` for a concrete example of proper GNN formatting
