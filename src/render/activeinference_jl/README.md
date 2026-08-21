# ActiveInference.jl Render Module

This module renders validated GNN POMDP specifications to executable ActiveInference.jl scripts.

## Public Surface

- `render_gnn_to_activeinference_jl(gnn_spec, output_path, options=None)`
- Step 11 calls this renderer through `render.pomdp_processor.POMDPRenderProcessor`.
- `render_gnn_spec(..., "activeinference_jl", ...)` routes to the same canonical renderer for POMDP specs.

## Contract

The renderer consumes `canonical_pomdp_v1` data:

- `A`: observation likelihood, shape `(observation, state)`
- `B`: transition tensor, shape `(next_state, previous_state, action)`
- `C`: observation preferences
- `D`: initial hidden-state prior
- `E`: policy prior when present
- matrix provenance and runtime metadata

Generated scripts import ActiveInference.jl and write `simulation_results.json` with schema `activeinference_jl_simulation_v1`.

## Native multi-agent (stigmergic) rendering

When a spec declares two or more complete agent groups (`A_agentN` /
`B_agentN` / `C_agentN` / `D_agentN` in `structured_pomdp.matrices`), the
renderer switches to the native multi-agent path (roadmap MAJ-03): one
per-agent simulation (per-agent state spaces — no joint state-space
expansion) coupled through the shared environment affordance. Each agent
deposits signal at its MAP position each timestep; the shared
`env_signal` trace decays by `signal_decay` per timestep. Results are
written with schema `activeinference_jl_stigmergic_swarm_v1`, `model_kind:
multi_agent`, per-agent beliefs/actions/EFE, and the `env_signal_trace`.
Detection helpers live in `render.multi_agent_common`; flat specs keep the
canonical single-agent path unchanged.

## Outputs

```text
output/11_render_output/<model>/activeinference_jl/
├── <model>_activeinference.jl
└── README.md
```

Step 12 collects runtime outputs into:

```text
output/12_execute_output/<model>/activeinference_jl/simulation_data/simulation_results.json
```

## Verification

```bash
julia --startup-file=no -e 'using ActiveInference, JSON, Distributions, StatsBase'
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
