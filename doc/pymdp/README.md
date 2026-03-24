# PyMDP Documentation

**Signposts:** [AGENTS.md](AGENTS.md) · [doc/INDEX.md](../INDEX.md)

This folder documents how this repository integrates PyMDP in render/execute
pipeline steps.

## Start Here

- `gnn_pymdp.md` - practical GNN → PyMDP integration contract used locally
- `pymdp_1_0_0_alignment_matrix.md` - upstream 1.0.0 claim mapping and local status
- `pymdp_advanced_tutorials.md` - additional examples (not all are pipeline contracts)
- `pymdp_performance_guide.md` - performance notes
- `pymdp_pomdp/README.md` - legacy/reference folder status and boundaries

## Version Scope

Local docs are maintained against repository-tested behavior, plus upstream
release context from:

- <https://github.com/infer-actively/pymdp>
- <https://github.com/infer-actively/pymdp/releases/tag/v1.0.0>

When an upstream feature is not wired into this repository, it is documented as
upstream context, not as a local guarantee.

## Local Conventions

- Install guidance: `uv pip install inferactively-pymdp`
- Import style: `from pymdp.agent import Agent`
- Inference policy outputs should use tuple unpacking:
  - `q_pi, neg_efe = agent.infer_policies(...)` (upstream docstring names the second value `G`)

## Upstream Agent API tests

- `src/tests/test_pymdp_1_0_0_upstream_api.py` — asserts the installed wheel’s
  `Agent` / `utils` behavior used by `execute/pymdp/simple_simulation.py`
  (no mocks).
