# gnn-improve dispatch container

This directory holds per-agent mission charters for a coordinated GNN
repository improvement campaign.

## Charters

- mission-parse.md          — gnn/core-authoring (parse, registry, type, validate, export)
- mission-render-execute.md — render + execute lifecycle
- mission-analysis-viz.md   — analysis + visualization
- mission-integration.md    — integration, mcp, api, cli, gui, website
- mission-infra.md          — setup/template/utils/pipeline/lsp/sapf + docs
- mission-science.md        — ontology, llm, audio, ml, security, report, research

## Reports

Each agent writes REPORT-<scope>.md into this folder on completion. The
coordinator reads these to consolidate, then runs the integration gates
itself (ruff, mypy, full pytest, docs audit) on the combined change set
before committing to main.