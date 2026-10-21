# Specification: Cognitive Phenomena

## Purpose

Document how cognitive phenomena (attention, memory, metacognition, and related themes) are represented in this tree for cross-linking with GNN models and pipeline outputs. This folder is **documentation-only**; executable code lives under `src/`.

## Scope

- Conceptual notes, glossaries, and experiment-style subtrees that relate Active Inference constructs to cognitive science vocabulary.
- Relative links into `docs/gnn/`, `docs/active_inference/`, and framework docs where relevant.

## Non-goals

- Normative GNN syntax (see [docs/gnn/reference/gnn_syntax.md](../gnn/reference/gnn_syntax.md)).
- Substituting for peer-reviewed sources; cite primary literature where claims are substantive.

## Design Requirements

Role: `cognitive_phenomena` is a research-area documentation module (docs-only; no executable code) — not a render/execution framework; no entry in `src/gnn/render/framework_registry.py`, no Step 12 executor.

Leaf `SPEC.md` files link back here under **Parent specification** for a single canonical policy.

## Documentation contract

[README.md](README.md) owns the topic entrypoints; [AGENTS.md](AGENTS.md) guides
maintenance. Follow the [documentation rules](../../.agent_rules/documentation.md) for links, executable
examples and source-bound claims. Research scripts and quoted upstream code have
their own prerequisites; their presence does not register a pipeline backend.
