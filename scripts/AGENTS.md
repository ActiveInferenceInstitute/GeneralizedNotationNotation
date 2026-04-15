# Developer Tooling Agents (scripts/)

## Purpose

This folder hosts the explicit, stateless developer workflow validation agents responsible for maintaining continuous documentation compliance across the GNN ecosystem.

## Components

The primary agent deployed in this module is the documentation guardrail:

- `check_gnn_doc_patterns.py` (Pattern Validation Agent): Recursively scans `.md` files to eradicate obsolete architectural references to staled imports (e.g., `gnn.parser`), legacy routing, and deprecated file syntax. Operates defensively via CI implementations `--strict` enforcing non-zero process exits.

## Operational Standards

- Strict adherence to Pythonic PEP validation.
- All scripts must contain highly structured `argparse` implementations mapped perfectly for headless CI environments.
- Must execute deterministically without writing to disk.
