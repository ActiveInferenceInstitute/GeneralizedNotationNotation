# GNN Documentation Module — Specification

## Architecture

This module follows the **static reference** pattern: it contains Markdown files that are read by the parser and validator modules at runtime, rather than executing code itself.

## Components

| Component | Type | Purpose |
|-----------|------|---------|
| `file_structure.md` | Reference | Defines GNN file sections: `GNNVersionAndFlags`, `ModelName`, `StateSpaceBlock`, `Connections`, `InitialParameterization`, `Equations`, `Time`, `ActInfOntologyAnnotation`, `Footer` |
| `punctuation.md` | Reference | Defines 15 GNN syntax symbols and their semantics |
| `README.md` | Documentation | Overview of this module |

## Requirements

1. **Completeness**: Every valid GNN section name must be listed in `file_structure.md`
2. **Accuracy**: Punctuation definitions must match the parser implementation in `gnn/parsers/`
3. **Consistency**: Section names here must match those recognized by `schema_validator.py` and `markdown_parser.py`

## Testing

Documentation accuracy is verified indirectly through the parser tests in `src/tests/test_gnn_parsing.py`, which parse content using the section names defined here.
