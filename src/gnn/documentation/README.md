# GNN Documentation Directory

## Overview

This directory contains user-facing documentation files that provide essential guidance for understanding GNN (Generalized Notation Notation) syntax, structure, and usage.

## Purpose

The documentation files in this directory serve as reference materials for:
- Understanding GNN file structure and organization
- Learning GNN syntax and punctuation rules
- Reference guides for GNN specification development

## Contents

### Core Documentation Files

- **`file_structure.md`** - Complete GNN file structure specification
  - Describes the organization of GNN files
  - Documents required and optional sections
  - Provides structure templates and examples

- **`punctuation.md`** - GNN syntax punctuation guide
  - Comprehensive guide to GNN punctuation marks
  - Connection symbols and their meanings
  - Syntax rules and conventions

## Usage

These documentation files are referenced by:
- GNN parser implementations
- Syntax validation tools
- User documentation and tutorials
- Development and testing processes

## Integration

This directory is part of the `gnn/` module and provides documentation resources that support:
- GNN file parsing (see `../parser.py`)
- Syntax validation (see `../validation.py`)
- Multi-format processing (see `../multi_format_processor.py`)

## Related Documentation

- Module docs: [`../AGENTS.md`](../AGENTS.md), [`../README.md`](../README.md), [`../SPEC.md`](../SPEC.md)
- [`doc/gnn/reference/gnn_syntax.md`](../../../doc/gnn/reference/gnn_syntax.md) is the **v1.1 normative** syntax specification for parsers and validators.
- [`doc/gnn/reference/gnn_syntax.md`](../../../doc/gnn/reference/gnn_syntax.md) is a **v2 quick reference** with examples; it complements but does not replace the normative doc.
- File structure (doc tree): [`doc/gnn/reference/gnn_file_structure_doc.md`](../../../doc/gnn/reference/gnn_file_structure_doc.md)

MCP exposes `file_structure.md` and `punctuation.md` via `get_gnn_documentation` in [`../mcp.py`](../mcp.py).
