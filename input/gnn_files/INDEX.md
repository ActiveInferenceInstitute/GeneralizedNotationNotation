# GNN Examples Index

Cold-start index of the exemplar GNN spec files under `input/gnn_files/`. Each
entry is a runnable Active Inference generative model spec: parse it, render it,
execute it through the 25-step pipeline. For syntax and file-structure rules see
[normative syntax](../../../doc/gnn/gnn_syntax.md) and the tutorials in
[doc/gnn/tutorials/](../../../doc/gnn/tutorials/).

**Counts (measured 2026-08-31):** 32 runnable `.md` spec files across 9 task
folders (plus non-spec `AGENTS.md`/`README.md` scaffolds in
`pomdp_gridworld/` and `pymdp_scaling_study/`). `../README.md` records
29/29 render+execute under RxInfer.jl for the core exemplar set (the 3 scaling
study files are load variants of the same model family).

## Choosing an example

| If you want to… | Start with |
| --- | --- |
| Learn GNN syntax from scratch | `basics/static_perception.md` → `basics/dynamic_perception.md` |
| Run a minimal discrete-state agent | `discrete/simple_mdp.md` → `discrete/tmaze_epistemic.md` |
| See a canonical full Active Inference agent | `discrete/actinf_pomdp_agent.md` |
| Compare render targets / scaling | `pymdp_scaling_study/pymdp_scaling_N4_T100.md` (then N8…N64) |
| Continuous-time / predictive coding | `continuous/predictive_coding_agent.md` |
| Multi-agent & stigmergy (v3+ features) | `multiagent/stigmergic_swarm.md` |
| Hierarchical / deep temporal models | `hierarchical/hierarchical_pomdp.md` |
| Parameter learning | `learning/dirichlet_likelihood_learning.md` |
| Precision & curiosity mechanisms | `precision/precision_weighted.md`, `precision/curiosity_driven_agent.md` |
| Causal models (bnlearn export) | `discrete/bnlearn_causal_model.md` |

## Full exemplar set

### basics/
- [dynamic_perception.md](basics/dynamic_perception.md)
- [static_perception.md](basics/static_perception.md)

### continuous/
- [continuous_navigation.md](continuous/continuous_navigation.md)
- [predictive_coding_agent.md](continuous/predictive_coding_agent.md)
- [stochastic_dynamics.md](continuous/stochastic_dynamics.md)

### discrete/
- [actinf_pomdp_agent.md](discrete/actinf_pomdp_agent.md)
- [bnlearn_causal_model.md](discrete/bnlearn_causal_model.md)
- [deep_planning_horizon.md](discrete/deep_planning_horizon.md)
- [hmm_baseline.md](discrete/hmm_baseline.md)
- [markov_chain.md](discrete/markov_chain.md)
- [multi_armed_bandit.md](discrete/multi_armed_bandit.md)
- [simple_mdp.md](discrete/simple_mdp.md)
- [time_varying_dynamics.md](discrete/time_varying_dynamics.md)
- [tmaze_epistemic.md](discrete/tmaze_epistemic.md)
- [two_state_bistable.md](discrete/two_state_bistable.md)

### hierarchical/
- [hierarchical_pomdp.md](hierarchical/hierarchical_pomdp.md)
- [temporal_hierarchy.md](hierarchical/temporal_hierarchy.md)

### learning/
- [dirichlet_likelihood_learning.md](learning/dirichlet_likelihood_learning.md)

### multiagent/
- [multi_agent_coordination.md](multiagent/multi_agent_coordination.md)
- [stigmergic_swarm.md](multiagent/stigmergic_swarm.md)

### pomdp_gridworld/
- [pomdp_gridworld_3x3.md](pomdp_gridworld/pomdp_gridworld_3x3.md)
- folder docs: [AGENTS.md](pomdp_gridworld/AGENTS.md), [README.md](pomdp_gridworld/README.md)

### precision/
- [curiosity_driven_agent.md](precision/curiosity_driven_agent.md)
- [precision_weighted.md](precision/precision_weighted.md)

### pymdp_scaling_study/
- [pymdp_scaling_N4_T100.md](pymdp_scaling_study/pymdp_scaling_N4_T100.md)
- [pymdp_scaling_N8_T100.md](pymdp_scaling_study/pymdp_scaling_N8_T100.md)
- [pymdp_scaling_N16_T100.md](pymdp_scaling_study/pymdp_scaling_N16_T100.md)
- [pymdp_scaling_N32_T100.md](pymdp_scaling_study/pymdp_scaling_N32_T100.md)
- [pymdp_scaling_N64_T100.md](pymdp_scaling_study/pymdp_scaling_N64_T100.md)
- folder docs: [README.md](pymdp_scaling_study/README.md)

### structured/
- [factorized_posterior.md](structured/factorized_posterior.md)

## Running an example

```bash
uv run python src/main.py --target-dir input/gnn_files/discrete --output-dir output
```
