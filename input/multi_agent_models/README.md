# Multi-Agent Model Fixtures

This directory holds one compact multi-agent GNN fixture,
`multi_agent_coordination.md` (3-agent clustered mean-field topology). The
maintained examples under `input/gnn_files/multiagent/` remain the canonical
authored models.

It is a hand-runnable target: `uv run python src/main.py --target-dir
input/multi_agent_models` reaches exactly this one model. `gnn.discovery.
is_model_source_path` accepts it, and it carries the `## GNNSection` and
`## GNNVersionAndFlags` headers the syntax reference marks Required.

**No committed command or test targets this directory.** Every verification
command in `TO-DO.md` points at `input/gnn_files`, and
`scripts/check_capability_contracts.py` asserts this path exists only while
`TO-DO.md` names it — which it does not today. It is therefore outside every
count the manuscript publishes: `GNN_EXAMPLE_COUNT` scans `input/gnn_files`
alone, and `GNN_OUTSIDE_CORPUS_NOTE` (`src/manuscript_variables.py`) is the
token that names this directory and its model count so the omission is stated
rather than silent.
