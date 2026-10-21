# Specification: Pedalboard

## Design Requirements
Role: `pedalboard` is an audio post-processing dependency in the Step 15 audio-sonification path (`src/gnn/audio/`, optional backend alongside soundfile) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

## Documentation contract

[README.md](README.md) owns the topic entrypoints; [AGENTS.md](AGENTS.md) guides
maintenance. Follow the [documentation rules](../../.agent_rules/documentation.md) for links, executable
examples and source-bound claims. Research scripts and quoted upstream code have
their own prerequisites; their presence does not register a pipeline backend.
