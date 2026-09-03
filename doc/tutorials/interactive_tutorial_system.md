# GNN Interactive Tutorial System (Proposed)

> **Status**: Proposal only. No code under `src/tutorials/` exists; the
> commands and tools once sketched on this page were design targets and have been
> removed so they are not mistaken for runnable instructions.

This page records the proposal so future work has a starting point.

## Idea

A guided, interactive layer over the existing tutorials:

- Step-by-step model building with live syntax validation (Step 5 feedback).
- Visual model construction and connection editing, exporting valid GNN files.
- Framework comparison walkthroughs driven by the real Step 11/12 pipeline.
- Skill checkpoints keyed to the [learning paths](../learning_paths.md).

## What exists today

The maintained, runnable tutorial surface is:

- [Quickstart tutorial](../gnn/tutorials/quickstart_tutorial.md) — build, validate,
  and run a complete model.
- [Examples and model progression](../gnn/tutorials/gnn_examples_doc.md).
- [Templates](../templates/README.md) for common model families.

Any implementation of the proposal would build on the pipeline entry points
documented in [src/AGENTS.md](../../src/AGENTS.md) and be tracked through the
maintained roadmap, not this page.
