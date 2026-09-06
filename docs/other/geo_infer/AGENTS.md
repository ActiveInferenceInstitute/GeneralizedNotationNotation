# geo_infer (docs/other archive entry)

## Overview

**Status**: active documentation program (draft) | **Version**: 0.1 | **Last
Updated**: 2026-09-06

## Purpose

Document, for GNN-side readers and agents, the artifact-level interchange
with the sibling spatial active-inference monorepo `GEO-INFER`: how GNN
models are exported as versioned JSON artifacts, and how GEO-INFER consumes
and validates them. This folder is the GNN-side mirror of the canonical
interchange documentation at `../GEO-INFER/GEO-INFER-ACT/docs/gnn_interchange.md`
(cross-repo references are inline code paths, never markdown links). The
sibling is a foreign git repository: never commit into it from GNN tooling.

## Contents

| File | Role |
| --- | --- |
| [README.md](README.md) | Reader-facing entry point: contracts, export command, validation harness |

## Key paths (post-reorg)

- Producers: `src/gnn/export/geo_infer.py` (categorical),
  `src/gnn/export/geo_infer_gaussian.py` (Gaussian),
  `src/gnn/export/geo_infer_factored.py` (factored).
- Normative format docs: `src/gnn/export/geo_infer_contract.md` and
  `src/gnn/export/geo_infer_factored_contract.md`.
- Export entry point: `uv run python -m gnn.export.geo_infer ...`.
- Consumer boundary (GEO-INFER side): `geo_infer_act.core.gnn_contract`
  (`GNNArtifact`, `run_gnn_inference`).
- Validation harness (GEO-INFER side):
  `GEO-INFER-TEST/validate_gnn_interchange.py`.

## Version facts

- Contracts: `gnn-geo-infer/1` (categorical), `gnn-geo-infer/2` (Gaussian
  `F`/`G`/`H`/`Q`/`R`), `gnn-geo-infer/factored/1` (factored).
- Integration is artifact-level only: separate environments, JSON artifacts
  with SHA-256 provenance, no cross-checkout imports (the GEO-INFER contract
  forbids it).
- H3 support is optional (`--extra geo-infer`).

## Editing rules

- Keep all links relative and verify targets exist; the repository-wide
  documentation audit (`uv run python
  docs/development/docs_audit.py --strict --check-anchors --no-write`) checks
  them for every markdown file.
- Follow [style_guide.md](../../style_guide.md): `uv run python` command
  spellings, H1 title, metadata block, honest claims without embedded
  ungenerated counts.
- Update this folder together with the producer modules under
  `src/gnn/export/`; contract changes bump the contract versions named here.

---

**Status**: active | **Maintenance**: update together with the export
modules under `src/gnn/export/` | **Last Updated**: 2026-09-06
