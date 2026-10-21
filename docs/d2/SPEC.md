# Specification: D2

## Design Requirements
Role: `d2` is a visualization capability (D2 diagrams from the advanced visualization module, `src/gnn/advanced_visualization/d2_visualizer.py`, Step 9) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

## Documentation contract

[README.md](README.md) owns the topic entrypoints; [AGENTS.md](AGENTS.md) guides
maintenance. Follow the [documentation rules](../../.agent_rules/documentation.md) for links, executable
examples and source-bound claims. Research scripts and quoted upstream code have
their own prerequisites; their presence does not register a pipeline backend.
