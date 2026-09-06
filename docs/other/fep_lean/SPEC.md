# Specification: fep_lean documentation module

## Design Requirements

- Document the sibling Lean 4 catalogue `fep_lean` and the formal
  GNN-fep_lean collaboration program for GNN readers, without duplicating
  canonical sources owned by either repository.
- Keep the mirrored bridge contract synchronized with its canonical copy;
  cross-repo references are inline code paths, never markdown links.
- All commands use the `uv run` spelling; all links are relative and
  verified.

## Components

- `README.md` — reader-facing entry point.
- `AGENTS.md` — agent scaffolding and editing rules.
- `SPEC.md` — this specification.
- `fep_lean.md` — catalogue and evidence-plane overview.
- `bridge-contract.md` — mirror of the canonical cross-repo contract.
- `fep_lean_gnn.md` — GNN-side collaboration program.
