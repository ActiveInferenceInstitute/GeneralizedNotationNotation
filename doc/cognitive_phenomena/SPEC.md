# Specification: Cognitive Phenomena

## Purpose

Document how cognitive phenomena (attention, memory, metacognition, and related themes) are represented in this tree for cross-linking with GNN models and pipeline outputs. This folder is **documentation-only**; executable code lives under `src/`.

## Scope

- Conceptual notes, glossaries, and experiment-style subtrees that relate Active Inference constructs to cognitive science vocabulary.
- Relative links into `doc/gnn/`, `doc/active_inference/`, and framework docs where relevant.

## Non-goals

- Normative GNN syntax (see [doc/gnn/reference/gnn_syntax.md](../gnn/reference/gnn_syntax.md)).
- Substituting for peer-reviewed sources; cite primary literature where claims are substantive.

## Design Requirements

This module (`cognitive_phenomena`) maps structural logic to the overall execution graph.
It ensures that `Cognitive Phenomena` tasks resolve without runtime dependency loops.

Leaf `SPEC.md` files link back here under **Parent specification** for a single canonical policy.

## Components

Expected available types: No specific classes exported.
