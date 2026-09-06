# DisCoPy Renderer — Technical Specification

**Version**: 1.6.0

## Purpose

Generates Python code using DisCoPy for categorical diagram construction.

## Code Generation

- Maps GNN model structure to categorical diagrams
- Generates morphism compositions and functorial mappings
- Produces executable DisCoPy Python scripts

## Output

- Python script files using `discopy` API
- Diagram serialization (JSON)

## Architecture

- `translator.py` (1648 lines) — Core GNN-to-DisCoPy translation
- Template-based code generation with parametric diagram construction

## Dependencies

Target: `discopy >= 1.1.0`
