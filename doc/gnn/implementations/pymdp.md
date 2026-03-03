# PyMDP Framework Implementation

> **GNN Integration Layer**: Python
> **Framework Base**: `pymdp` (Python Package)
> **Simulation Architecture**: Online True POMDP Generative Model
> **Documentation Version**: 1.3.0

## Overview

The Generalized Notation Notation (GNN) pipeline translates theoretical model specifications into executable Python code natively utilizing the `pymdp` framework. As the primary reference implementation within the Active Inference ecosystem, PyMDP operates as the mathematical ground truth for multi-framework correlation auditing.

This document details the exact mechanisms through which a GNN JSON specification is extracted, built into a PyMDP `Agent` node, evaluated continuously within an independent generative POMDP environment, and serialized into high-fidelity telemetry artifacts.

## Architecture

The PyMDP implementation relies on three interconnected architectures:

1. **Parameter Parsing**: `pomdp_processor.py` (Translating GNN variable states into multidimensional numpy matrices)
2. **Generative Loop Generation**: `pymdp_renderer.py` (Building the target python script wrapping the agent inside a mathematical observation loop)
3. **Execution Context**: `pymdp_runner.py` (Executing the generated python script and managing filesystem persistence)

### Model Initialization & Variable Extraction

Unlike strict typing systems, PyMDP initializes parameter matrices primarily using pure Numpy objects.

GNN maps structural definitions dynamically into `pymdp.agent.Agent` instantiation endpoints:

- **State Space (`num_states`)**: Inferred dynamically from the row dimensionality of the `B` transition matrix (e.g. `B_matrix.shape[1]`)
- **Action Space (`num_actions`)**: Inferred dynamically from the depth dimensionality of the `B` control matrix (e.g. `B_matrix.shape[2]`)
- **Observation Space (`num_obs`)**: Inferred dynamically from the row dimensionality of the `A` likelihood matrix (e.g. `A_matrix.shape[0]`)

```python
# Extraction logic utilized by GNN to safely route specifications
agent = Agent(
    A=A_matrix,           # Likelihood mapping of Observation = f(State)
    B=B_matrix,           # Transition mapping of State(t+1) = f(State(t), Action)
    C=C_vector,           # Target Preference dist (Risk/Reward)
    D=D_vector,           # Prior initialization density
    policy_len=1,         # Depth 1 horizons by default in standard evaluation
    inference_algo="MMP", # Marginal Message Passing
    use_utility=True      # Essential for Expected Free Energy activation
)
```

## Perception-Action Loop (The Generative Process)

A critical mandate of the cross-framework environment comparison is isolating the belief structure *from* the generative process entirely. PyMDP accomplishes this correctly by implementing a purely stochastic mathematical simulation that wraps around the agent instance itself.

The execution template builds the following sequence:

1. **Initialize Environmental True State**:
   The true environment begins by sampling the Prior definition `D`.

   ```python
   # Stochastic draw from Prior distribution
   true_state = utils.sample(D_vector)
   ```

2. **Step-wise Environment Generation**:
   The environment draws an observation exclusively from the true state using the Likelihood `A` matrix, shielding the agent from oracle knowledge.

   ```python
   # Stochastic draw from A matrix conditioned on current True State
   observation = utils.sample(A_matrix[:, true_state])
   ```

3. **Inference and Control Calculation (Agent)**:
   The PyMDP agent ingests the singular step observation to update its belief and calculate EFE.

   ```python
   qs = agent.infer_states([observation])
   q_pi, efe = agent.infer_policies()
   action = agent.sample_action()
   ```

4. **Environment Transition**:
   The `true_state` physically shifts through spacetime based on the transition likelihood of the `B` matrix, conditioned by the generated $Action$.

   ```python
   # Stochastic environment decay/shift
   true_state = utils.sample(B_matrix[:, true_state, int(action[0])])
   ```

## Expected Free Energy Mechanics

PyMDP mathematically factors Expected Free Energy (EFE), stored internally as `neg_efe`, incorporating two principal components:

1. **Ambiguity** (Expected Uncertainty): The entropy of the likelihood matrix `A` conditioned on predicted states.
2. **Risk** (Expected Divergence): The Kullback-Leibler (KL) divergence between the predicted observation distribution and the pre-defined target preference vector `C`.

*Important Tracking Detail*: By mathematical convention within PyMDP, EFE is minimized as a negative quantity (`neg_efe`). Therefore, more optimal action selections natively resolve to *higher* (closer to 0) EFE values, conversely represented against frameworks that minimize positive EFE vectors.

## Telemetry & Logging Output

At runtime, the PyMDP orchestrator records a full state continuum into memory and eventually dumps the structure out to the `12_execute` block as `simulation_results.json`.

**Logged Vectors:**

- `beliefs`: Array depth `[num_timesteps, num_states]` recording posterior $Q(S|O)$.
- `actions`: Array length `[num_timesteps]` corresponding to index arrays of integer actions executed.
- `observations`: Array length `[num_timesteps]` logging the exact stochastic emission generated mathematically by the true environment.
- `efe_history`: **2-Dimensional Tensor** `[num_timesteps, num_policies]`. While previously flattened dynamically, GNN extracts the comprehensive EFE vectors for *all possible decisions* evaluated at that timestep to allow high-fidelity plotting in downstream analysis engines.

```json
"simulation_trace": {
  "observations": [1, 1, 1, 0, 2],
  "beliefs": [[0.05, 0.89, 0.05], [0.003, 0.99, 0.003], ...],
  "actions": [0, 0, 0, 1, 2],
  "efe_history": [
    [-1.19, -1.19, -1.96], 
    [-0.68, -0.68, -1.45], 
    ...
  ] 
}
```

The telemetry package operates with extremely high resilience, guaranteeing zero missing timesteps and zero floating-point `NaN` propagation across pipeline stages during standard tests.

---

## Source Code Connections

| Pipeline Stage | Module | Key Function | Lines |
|---|---|---|---|
| Rendering | [pymdp_renderer.py](../../../src/render/pymdp/pymdp_renderer.py) | `render_gnn_to_pymdp()` | Entry point |
| Execution | [pymdp_runner.py](../../../src/execute/pymdp/pymdp_runner.py) | `execute_pymdp_script_with_outputs()` | L77-220 |
| Pre-validation | [pymdp_runner.py](../../../src/execute/pymdp/pymdp_runner.py) | `validate_and_clean_pymdp_script()` | L22-75 |
| Trace Generation | [pymdp_runner.py](../../../src/execute/pymdp/pymdp_runner.py) | `generate_simulation_trace()` | L264-304 |
| Analysis | [analyzer.py](../../../src/analysis/pymdp/analyzer.py) | `generate_analysis_from_logs()` | — |
| Cross-Framework | [visualizations.py](../../../src/analysis/visualizations.py) | `generate_unified_comparison()` | — |

---

## Improvement Opportunities

| ID | Area | Description | Impact |
|---|---|---|---|
| P-1 | Rendering | Matrix normalization is inlined — could be extracted into a shared utility | Low |
| P-2 | Execution | ~~The runner had duplicate `env = os.environ.copy()` lines~~ | ✅ FIXED |
| P-3 | Analysis | PyMDP analyzer imports `PyMDPVisualizer` which could fail silently | Low |
| P-4 | Telemetry | `simple_simulation.py` could validate A/B/C/D shapes before simulation | Medium |

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/cross_framework_methodology.md)**: Details on the correlation methodology and benchmarking metrics.
- **[Architecture Reference](../reference/architecture_reference.md)**: Deep dive into the pipeline orchestrator and module integration.
- **[GNN Implementations Index](README.md)**: Return to the master framework implementer manifest.
- **[Back to GNN START_HERE](../../START_HERE.md)**
