# Utility module boundaries

The current utility package is documented in
[its README](../../src/gnn/utils/README.md),
[agent guide](../../src/gnn/utils/AGENTS.md) and
[specification](../../src/gnn/utils/SPEC.md). Those owners describe public exports
and implementation boundaries; this page explains how to approach further changes.

## Follow existing owners

Import public APIs through `gnn.*`. Inspect the current facade and consumer tests
before changing an export. Put behavior in its owning concern package and keep
numbered orchestration in the shared script factory. The
[module-pattern guide](../../.agent_rules/module_patterns.md) owns these rules.

## Verify a boundary change

Exercise real public consumers, error handling and import pressure. Run the
declared [quality checks](../../.agent_rules/quality.md) and
[test selections](../../.agent_rules/testing.md); preserve lazy import behavior
and supported signatures. A split is justified by ownership and behavior, not
by a historical line-count target.

Changes to scientific source or manuscript inputs follow the
[paired and publication procedure](../../.agent_rules/release.md). New work
belongs in [TO-DO.md](../../TO-DO.md), with bounded acceptance criteria.
