# GNN Framework Implementations

Comprehensive technical documentation for each Active Inference framework implementation in the GNN pipeline. See [AGENTS.md](AGENTS.md) for data flow architecture, cross-framework improvement assessment, and source code connection maps.

## Framework Index

| Framework | Language | Type | Documentation |
|---|---|---|---|
| **PyMDP** | Python | Numerical Simulation | [pymdp.md](pymdp.md) |
| **JAX** | Python / XLA | High-Performance Numerical | [jax.md](jax.md) |
| **RxInfer.jl** | Julia | Probabilistic Programming | [rxinfer.md](rxinfer.md) |
| **ActiveInference.jl** | Julia | Discrete Active Inference | [activeinference_jl.md](activeinference_jl.md) |
| **DisCoPy** | Python | Categorical Semantics | [discopy.md](discopy.md) |
| **CatColab** | Julia / Web | Categorical / Structural | [catcolab.md](catcolab.md) |
| **PyTorch** | Python / CUDA | Neural Active Inference | [pytorch.md](pytorch.md) |
| **NumPyro** | Python / JAX | Probabilistic Programming | [numpyro.md](numpyro.md) |

## Architecture Overview

All five frameworks share a common pipeline flow:

```
GNN Specification (JSON)
        │
        ▼
┌─────────────────────┐
│  Parameter Parsing   │  pomdp_processor.py
│  (A, B, C, D, E)    │  Extract matrices from GNN JSON
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Framework Renderer  │  {framework}_renderer.py
│  Code Generation     │  Generate target-language script
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Script Execution    │  {framework}_runner.py
│  Subprocess Launch   │  Execute generated script
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Telemetry Export    │  simulation_results.json
│  (JSON Artifacts)    │  Beliefs, actions, EFE, observations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cross-Framework     │  analyzer.py / visualizations.py
│  Comparison          │  Correlation, dashboards, reports
└─────────────────────┘
```

## Unified POMDP Generative Environment

All numerical frameworks (PyMDP, JAX, RxInfer, ActiveInference.jl, PyTorch, NumPyro) implement an identical true POMDP generative process:

1. **Initialize true state**: `true_state ~ Categorical(D)`
2. **Generate observation**: `observation ~ Categorical(A[:, true_state])`
3. **Agent inference**: Framework-specific belief update and action selection
4. **Transition true state**: `true_state ~ Categorical(B[:, true_state, action])`

This ensures that all frameworks experience the same type of environmental dynamics, enabling meaningful cross-framework comparisons.

## GNN Matrix Definitions

| Matrix | Name | Shape | Role |
|---|---|---|---|
| **A** | Likelihood | `[num_obs, num_states]` | `P(observation \| hidden_state)` |
| **B** | Transition | `[num_states, num_states, num_actions]` | `P(next_state \| current_state, action)` |
| **C** | Preference | `[num_obs]` | Target observation distribution |
| **D** | Prior | `[num_states]` | Initial state belief |
| **E** | Policy Prior | `[num_policies]` | Prior over action policies (ActiveInference.jl only) |

## Telemetry Schema

All numerical frameworks export `simulation_results.json` with a common schema:

| Field | Shape | Frameworks |
|---|---|---|
| `beliefs` | `[T, S]` | PyMDP ✅, JAX ✅, RxInfer ✅, ActiveInference.jl ✅ |
| `actions` | `[T]` | PyMDP ✅, JAX ✅, RxInfer ✅, ActiveInference.jl ✅ |
| `observations` | `[T]` | PyMDP ✅, JAX ✅, RxInfer ✅, ActiveInference.jl ✅ |
| `efe_history` | `[T, A]` or `[T]` | PyMDP ✅, JAX ✅, RxInfer ✅, ActiveInference.jl ✅ |
| `validation` | `dict` | PyMDP ✅, JAX ✅, RxInfer ✅, ActiveInference.jl ✅ |

## Cross-Framework Correlation Results

Latest pipeline run confidence correlations:

| Pair | Correlation |
|---|---|
| PyMDP ↔ JAX | **1.0000** |
| PyMDP ↔ RxInfer | **1.0000** |
| JAX ↔ RxInfer | **1.0000** |
| PyMDP ↔ PyTorch | ~1.0000 (identical matrices) |
| PyMDP ↔ NumPyro (posterior mean) | ~1.0000 |
| PyMDP ↔ ActiveInference.jl | *pending* |
| JAX ↔ ActiveInference.jl | *pending* |
| RxInfer ↔ ActiveInference.jl | *pending* |

> DisCoPy and CatColab are excluded from numerical correlation — they provide structural output only.
