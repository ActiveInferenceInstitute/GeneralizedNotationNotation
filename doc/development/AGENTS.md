# development

## Overview

This directory contains documentation and resources for the development subsystem.

**Version**: 1.0

---

## Purpose

Development workflows, contribution guidelines, and coding standards

This subsystem is part of the broader GNN (Generalized Notation Notation) documentation ecosystem, integrated with the 25-step processing pipeline.

## Contents

**Files**: `README.md`, `AGENTS.md`, `agents_readme_review.md`, `agents_readme_triple_review.md`, `thin_orchestrator_pattern.md`, `docs_audit.py`, `rewrite_gnn_doc_links.py`, `docs_audit_report.md` (generated) | **Subdirectories**: none

## Quick Navigation

- **Docs audit**: `uv run --extra dev python doc/development/docs_audit.py` from repo root (`--strict` for CI; failures print per-issue lines on stderr by default, `-q` for summary only); report includes `doc/**/AGENTS.md` orientation checks. See [README.md](README.md#documentation-audit-tooling), [agents_readme_review.md](agents_readme_review.md), [agents_readme_triple_review.md](agents_readme_triple_review.md)
- **README.md**: [Directory overview](README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Main Documentation**: [doc/README.md](../README.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
2. **Simulation** (Steps 10-16): Model execution and analysis  
3. **Integration** (Steps 17-24): System coordination and output

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

## Related Resources

**Main GNN Documentation**: [gnn/](../gnn/)
- [GNN Overview](../gnn/gnn_overview.md)
- [GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)
- [GNN Examples](../gnn/tutorials/gnn_examples_doc.md)

**Pipeline Architecture**: [src/](../../src/)
- [Pipeline AGENTS](../../src/AGENTS.md)
- [Pipeline README](../../src/README.md)

