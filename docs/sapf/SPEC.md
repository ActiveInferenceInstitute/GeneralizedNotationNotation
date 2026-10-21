# Specification: Sapf

## Design Requirements
Role: `sapf` is an audio sonification capability at Step 15 (`src/gnn/audio/sapf/`; Step 15 audio backend) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

## Documentation contract

[README.md](README.md) owns the topic entrypoints; [AGENTS.md](AGENTS.md) guides
maintenance. Follow the [documentation rules](../../.agent_rules/documentation.md) for links, executable
examples and source-bound claims. Research scripts and quoted upstream code have
their own prerequisites; their presence does not register a pipeline backend.
