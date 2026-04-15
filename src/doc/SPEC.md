# Specification: Doc

## Design Requirements

The `src/doc/` directory is an **internal documentation subtree** providing source-adjacent reference material supporting `src/` implementation and maintenance.

## Interface Mapping

This module does not export runtime Python functionality. It contains:

- `CHANGELOG.md`: Documentation subtree change history tracking additions, removals, and structural changes
- `QUICK_REFERENCE.md`: Developer quick-reference for common pipeline operations, CLI commands, and configuration patterns
- `cognitive_phenomena/`: Reference material documenting cognitive modeling concepts used in Active Inference GNN models

## Scope Boundaries

| In Scope | Out of Scope |
|----------|-------------|
| Hand-maintained implementation notes | Generated output artifacts |
| Developer quick-reference guides | Cache files or temporary data |
| Cognitive modeling reference docs | Runtime pipeline configuration |
| Change history for the doc subtree | Top-level `doc/` documentation |

## Standards

- Content must complement (not duplicate) top-level `doc/` documentation
- Documentation coverage policy applies to maintained source/doc folders only
- Generated outputs, caches, and transient artifacts are excluded from coverage expectations
