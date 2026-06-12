# Specification: Operations Documentation

## Scope
Operational runbooks and tooling references: what to check when something
breaks, how to diagnose, which artifacts the pipeline produces, and how
MCP tools surface pipeline capabilities.

## Contents
| File | Purpose |
|------|---------|
| `gnn_tools.md` | Operator-facing tool index (131 MCP tools + pipeline scripts) |
| `gnn_troubleshooting.md` | Common failure modes + remediation |
| `REPO_COHERENCE_CHECK.md` | Cross-module consistency audit procedure |
| `resource_metrics.md` | Resource estimation from Step 5 type checker |
| `improvement_analysis.md` | Longitudinal pipeline improvement tracker |

## Status
Maintained. When adding a new MCP tool, update `gnn_tools.md`; when fixing
a recurring user-reported issue, append to `gnn_troubleshooting.md`.
